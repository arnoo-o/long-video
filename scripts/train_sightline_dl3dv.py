"""Geometry-free, causal Sightline training with one autograd chunk per step."""
from __future__ import annotations
import argparse, hashlib, json, random, sys, time
from dataclasses import asdict
from pathlib import Path
import torch
from long_video.config import load_sightline_config
from long_video.training.flow_matching_exact import exact_flow_matching_items
from long_video.training.sightline import SightlineTrainable, install_lora, curriculum_phase, select_train_chunk, run_single_graph_chunks, native_history_16_2_1
from long_video.training.sightline_data import load_sightline_manifest, load_latent_tensor
from long_video.training.sightline_checkpoint import save_runtime_checkpoint, restore_runtime_checkpoint
from long_video.sightline.helios_integration import SightlineRayProvider, install_sightline_attention
from long_video.sightline.history import CameraHistoryState
from long_video.sightline.pipeline import SightlinePipeline

def _prompt(pipe,text,device):
    result=pipe._get_t5_prompt_embeds(text,device=device,dtype=torch.bfloat16)
    if not isinstance(result,(tuple,list)) or len(result)!=2: raise RuntimeError('pinned Helios prompt API must return (embeds, mask)')
    embeds,mask=result
    if mask.ndim!=2 or mask.shape[:2]!=embeds.shape[:2]: raise RuntimeError('pinned Helios prompt mask shape mismatch')
    return embeds,mask

def _history_for_shape(history,shape):
    result={}
    for name,(latent,indices) in history.items():
        if latent.shape[-2:]!=shape[-2:]:
            b,c,t,h,w=latent.shape; flat=latent.permute(0,2,1,3,4).reshape(b*t,c,h,w)
            flat=torch.nn.functional.interpolate(flat,size=shape[-2:],mode='bilinear')
            latent=flat.reshape(b,t,c,*shape[-2:]).permute(0,2,1,3,4)
        result[name]=(latent,indices)
    return result

def _model_prediction(pipe,noisy,item,prompt_embeds,history,current_start):
    indices=torch.arange(current_start,current_start+noisy.shape[2],device=noisy.device).view(1,-1)
    output=pipe.transformer(hidden_states=noisy,timestep=item['timesteps'],encoder_hidden_states=prompt_embeds,
        indices_hidden_states=indices,latents_history_long=history['long'][0],indices_latents_history_long=history['long'][1],
        latents_history_mid=history['mid'][0],indices_latents_history_mid=history['mid'][1],
        latents_history_short=history['short'][0],indices_latents_history_short=history['short'][1],attention_kwargs={'current_chunk':current_start//8})
    prediction=output[0] if isinstance(output,(tuple,list)) else getattr(output,'sample',output)
    if prediction.shape!=noisy.shape: raise RuntimeError(f'prediction shape {prediction.shape} != {noisy.shape}')
    return prediction

def _generate_detached_chunk(pipe,source,history,prompt_embeds,cfg,chunk):
    """Native Helios autoregressive inference from noise; no target argument exists."""
    noise=torch.randn((source.shape[0],source.shape[1],9,source.shape[-2],source.shape[-1]),device=source.device,dtype=source.dtype)
    indices=torch.arange(chunk*8,chunk*8+9,device=source.device).view(1,-1)
    class Progress:
        def update(self): pass
    return pipe.stage2_sample(latents=noise,pyramid_num_stages=3,pyramid_num_inference_steps_list=list(cfg.pyramid_steps),
        prompt_embeds=prompt_embeds,negative_prompt_embeds=None,guidance_scale=1.0,indices_hidden_states=indices,
        latents_history_long=history['long'][0],indices_latents_history_long=history['long'][1],
        latents_history_mid=history['mid'][0],indices_latents_history_mid=history['mid'][1],
        latents_history_short=history['short'][0],indices_latents_history_short=history['short'][1],
        attention_kwargs={'current_chunk':chunk},device=source.device,transformer_dtype=source.dtype,progress_bar=Progress())

def _load_correspondence(path):
    payload=json.loads(Path(path).read_text())
    if payload.get('schema_version')!='sightline-correspondence-v1': raise RuntimeError('stale correspondence cache')
    return payload['rows']

def _mapped_correspondences(processor,rows,chunk):
    q,k=processor.last_q,processor.last_k
    if q is None or k is None: raise RuntimeError('correspondence processor did not capture Q/K')
    current=processor.last_current_length; memory_count=processor.last_attention_meta.get('memory_tokens',0); native_k=k.shape[1]-memory_count
    selected=[]; positives=[]; weights=[]
    for row in rows:
        if int(row['query_chunk'])!=chunk: continue
        qi=q.shape[1]-current+int(row['query_attention_index'])
        if int(row['key_chunk'])==chunk: ki=native_k-current+int(row['key_attention_index'])
        else:
            wanted=(int(row['key_chunk']),int(row['key_latent_temporal']),int(row['key_y'])//2,int(row['key_x'])//2)
            index=next((i for i,t in enumerate(processor.memory.tokens) if (t.chunk_index,t.temporal,t.pooled_y,t.pooled_x)==wanted),None)
            if index is None: continue
            ki=native_k+index
        if 0<=qi<q.shape[1] and 0<=ki<k.shape[1]: selected.append(qi); positives.append(ki); weights.append(float(row['weight']))
    if not selected: raise RuntimeError('correspondence identities do not map to real attention axes')
    return selected,positives,weights

def _corr_loss(trainable,processors,rows,chunk,layers):
    if not layers: raise RuntimeError('correspondence is enabled but correspondence_layers is empty')
    missing=[layer for layer in layers if layer not in processors]
    if missing: raise RuntimeError(f'correspondence layers have no Sightline processor: {missing}')
    processor=processors[layers[0]]; selected,positives,weights=_mapped_correspondences(processor,rows,chunk)
    logits=torch.einsum('bqhd,bkhd->bhqk',processor.last_q,processor.last_k)*(processor.last_q.shape[-1]**-.5)
    sampled=logits[:,:,torch.tensor(selected,device=logits.device)]
    positive=torch.tensor(positives,device=logits.device).view(1,-1).expand(sampled.shape[0],-1)
    weight=torch.tensor(weights,device=logits.device).view(1,-1).expand_as(positive)
    return trainable.correspondence(sampled,positive,weight)

def _reset_sequence(runner):
    runner.camera_history=CameraHistoryState(); runner._pending_camera_chunk=None
    for bank in runner.memory.banks.values(): bank.tokens.clear()
    for processor in runner.helios.transformer._sightline_processors.values():
        processor.last_q=processor.last_k=processor.last_hidden_states=None

def main():
    p=argparse.ArgumentParser(); p.add_argument('--config',default='configs/sightline.yaml'); p.add_argument('--model',required=True); p.add_argument('--helios-root',required=True); p.add_argument('--manifest',required=True)
    p.add_argument('--expected-records',type=int,default=100); p.add_argument('--max-steps',type=int,default=2500); p.add_argument('--resume'); p.add_argument('--output-dir',required=True); p.add_argument('--save-every',type=int,default=80)
    p.add_argument('--prompt',default='A stable realistic view of the same scene.'); p.add_argument('--probe-only',action='store_true'); p.add_argument('--probe-capture'); p.add_argument('--probe-step',type=int,default=1000); p.add_argument('--alpha-zero-baseline',action='store_true'); p.add_argument('--record-index',type=int); p.add_argument('--train-chunk',type=int); p.add_argument('--train',action='store_true'); args=p.parse_args()
    if not (args.train or args.probe_only) or args.train==args.probe_only: raise ValueError('select exactly one of --train or --probe-only')
    cfg=load_sightline_config(args.config); records=load_sightline_manifest(args.manifest,expected_count=args.expected_records)
    sys.path.insert(0,args.helios_root)
    from helios.diffusers_version.pipeline_helios_diffusers import HeliosPipeline
    import helios.diffusers_version.transformer_helios_diffusers as helios_source
    source_file=Path(args.helios_root)/'helios/diffusers_version/transformer_helios_diffusers.py'; fingerprint=hashlib.sha256(source_file.read_bytes()).hexdigest()
    pipe=HeliosPipeline.from_pretrained(args.model,torch_dtype=torch.bfloat16).to('cuda'); heads=int(pipe.transformer.config.num_attention_heads); inner=int(pipe.transformer.config.attention_head_dim*heads)
    trainable=SightlineTrainable(inner,heads=heads).to('cuda',dtype=torch.bfloat16)
    for parameter in pipe.transformer.parameters(): parameter.requires_grad_(False)
    install_lora(pipe.transformer,cfg.lora_layers,rank=cfg.lora_rank) if cfg.lora_layers else None
    provider=SightlineRayProvider(source_height=cfg.source_height,source_width=cfg.source_width); runner=SightlinePipeline(pipe,config=cfg,conditioner=trainable.conditioner,ray_provider=provider)
    install_sightline_attention(pipe.transformer,trainable.conditioner,provider,layers=cfg.sightline_layers,helios_module=helios_source,memory=runner.memory)
    lora_params=[p for n,p in pipe.transformer.named_parameters() if 'lora_' in n]
    optimizer=torch.optim.AdamW([{'params':list(trainable.parameters())+list(runner.memory.parameters()),'lr':cfg.learning_rate},{'params':lora_params,'lr':cfg.lora_learning_rate}],weight_decay=.01)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=args.max_steps)
    prompt_embeds,_=_prompt(pipe,args.prompt,'cuda'); config=asdict(cfg); memory_config={'layers':list(cfg.memory_layers),'pool':cfg.memory_pool,'budget':cfg.memory_budget}
    start_step=args.probe_step if args.probe_only else 0
    if args.resume:
        payload=torch.load(args.resume,map_location='cpu'); completed_step=restore_runtime_checkpoint(payload,trainable,runner.memory,pipe.transformer,config=config,helios_fingerprint=fingerprint,layers=cfg.sightline_layers,memory_config=memory_config,optimizer=optimizer,scheduler=scheduler,restore_rng=True); start_step=completed_step+1
    elif args.alpha_zero_baseline:
        trainable.conditioner.alpha.data.zero_(); runner.memory.timestamp.weight.data.zero_()
    output=Path(args.output_dir); output.mkdir(parents=True,exist_ok=True); metrics=output/'metrics.jsonl'
    stop=args.max_steps if args.train else min(args.max_steps,start_step+1)
    for step in range(start_step,stop):
        phase=curriculum_phase(step); index=args.record_index if args.record_index is not None else random.randrange(len(records)); record=records[index]
        latent_key='gt_latent_cache' if 'gt_latent_cache' in record.raw else 'latent_cache'; latents=load_latent_tensor(record.path(latent_key)).to('cuda',dtype=torch.bfloat16)
        if latents.shape[2]<49: raise ValueError('six chunks require at least 49 latent frames')
        source=latents[:,:,:1]; c2w_np,K_np=record.load_cameras(); c2w=torch.from_numpy(c2w_np).to('cuda',dtype=torch.float32).unsqueeze(0); K=torch.from_numpy(K_np).to('cuda',dtype=torch.float32).unsqueeze(0)
        _reset_sequence(runner); runner._trajectory_c2w=c2w; runner._trajectory_K=K; runner._source_camera=c2w[:,0]; runner._source_intrinsics=K[:,0]; runner.memory.set_enabled(phase['memory'])
        for name,parameter in pipe.transformer.named_parameters():
            if 'lora_' in name: parameter.requires_grad_(phase['lora'])
        if phase['correspondence']:
            if not cfg.correspondence_layers or any(layer not in pipe.transformer._sightline_processors for layer in cfg.correspondence_layers): raise RuntimeError('configured correspondence layers are not installed Sightline layers')
            for layer in cfg.correspondence_layers: pipe.transformer._sightline_processors[layer].capture_diagnostics=True
            corr_rows=_load_correspondence(record.path('correspondence_cache'))
        else: corr_rows=None
        train_chunk=args.train_chunk if args.train_chunk is not None else select_train_chunk(phase['max_chunks'])
        if not 0<=train_chunk<phase['max_chunks']: raise ValueError('train_chunk outside curriculum')
        completed=[]; completed_ids=[]; losses={}; probe_payload={}; optimizer.zero_grad(set_to_none=True); started=time.perf_counter()
        def forward_chunk(chunk,keep_graph):
            history=native_history_16_2_1(completed,completed_ids,source)
            if not keep_graph:
                template=torch.empty((source.shape[0],source.shape[1],9,*source.shape[-2:]),device=source.device,dtype=source.dtype); runner._prepare_chunk(chunk,template,{})
                generated=_generate_detached_chunk(pipe,source,history,prompt_embeds,cfg,chunk).detach()
            else:
                target=latents[:,:,chunk*8:chunk*8+9]  # The sole non-source GT read in this step.
                items=exact_flow_matching_items(pipe,target,stage_steps=cfg.pyramid_steps,device=target.device); runner._prepare_chunk(chunk,items[-1]['noisy_latents'],{})
                stage_losses=[]; final_prediction=None
                for item in items:
                    prediction=_model_prediction(pipe,item['noisy_latents'],item,prompt_embeds,_history_for_shape(history,item['noisy_latents'].shape),chunk*8); final_prediction=prediction
                    stage_losses.append((prediction.float()-item['target'].float()).square().mean())
                fm=torch.stack(stage_losses).mean(); corr=_corr_loss(trainable,pipe.transformer._sightline_processors,corr_rows,chunk,cfg.correspondence_layers) if corr_rows else fm.new_zeros(())
                losses.update(fm=fm,corr=corr,total=fm+trainable.lambda_corr(step/args.max_steps)*corr,stage=stage_losses)
                final=items[-1]; generated=(final['noisy_latents']-final['sigmas']*final_prediction).detach()
                if args.probe_capture:
                    layer=cfg.correspondence_layers[0]; processor=pipe.transformer._sightline_processors[layer]; selected,positives,_=_mapped_correspondences(processor,corr_rows,chunk)
                    memory_count=processor.last_attention_meta.get('memory_tokens',0); base_q=processor.last_q.detach().cpu(); base_k=processor.last_k.detach().cpu(); base_context=dict(provider.context)
                    with torch.no_grad():
                        provider.context=dict(base_context); provider.context['c2w']=base_context['c2w'].flip(1)
                        wrong=_model_prediction(pipe,final['noisy_latents'],final,prompt_embeds,_history_for_shape(history,final['noisy_latents'].shape),chunk*8)
                        runner.memory.set_enabled(False); provider.context=base_context
                        zero=_model_prediction(pipe,final['noisy_latents'],final,prompt_embeds,_history_for_shape(history,final['noisy_latents'].shape),chunk*8)
                        runner.memory.set_enabled(True); originals={layer:[token.hidden for token in bank.tokens] for layer,bank in runner.memory.banks.items()}
                        for bank in runner.memory.banks.values():
                            shuffled=list(reversed([token.hidden for token in bank.tokens]))
                            for token,hidden in zip(bank.tokens,shuffled): token.hidden=hidden
                        shuffled_prediction=_model_prediction(pipe,final['noisy_latents'],final,prompt_embeds,_history_for_shape(history,final['noisy_latents'].shape),chunk*8)
                        for bank_layer,hiddens in originals.items():
                            for token,hidden in zip(runner.memory.banks[bank_layer].tokens,hiddens): token.hidden=hidden
                        provider.context=base_context
                    probe_payload.update(source='real_helios_forward',layer=layer,sigma=float(final['sigmas'].mean()),q=base_q[:,selected],k=base_k,positive_key=torch.tensor(positives).view(1,-1).expand(base_q.shape[0],-1),k_memory=base_k[:,-memory_count:] if memory_count else base_k[:,:0],fm_loss=float(fm.detach()),wrong_ray_loss=float((wrong.float()-final['target'].float()).square().mean()),memory_zero_loss=float((zero.float()-final['target'].float()).square().mean()),memory_shuffle_loss=float((shuffled_prediction.float()-final['target'].float()).square().mean()),corr_loss=float(corr.detach()),vram_gb=float(torch.cuda.max_memory_allocated()/2**30))
            for local in range(1,generated.shape[2]): completed.append(generated[:,:,local:local+1]); completed_ids.append(chunk*8+local)
            runner._finalize_chunk(chunk)
            for processor in pipe.transformer._sightline_processors.values(): processor.last_q=processor.last_k=processor.last_hidden_states=None
            return generated
        _,policies=run_single_graph_chunks(phase['max_chunks'],train_chunk,forward_chunk)
        if args.train:
            losses['total'].backward(); grad_norm=torch.nn.utils.clip_grad_norm_([p for group in optimizer.param_groups for p in group['params'] if p.grad is not None],cfg.grad_clip); optimizer.step(); scheduler.step()
        else: grad_norm=torch.tensor(0.)
        row={'step':step,'record':record.trajectory_id,'phase':phase['name'],'max_chunks':phase['max_chunks'],'train_chunk':train_chunk,'policies':policies,'flow_loss':float(losses['fm'].detach()),'corr_loss':float(losses['corr'].detach()),'stage_losses':[float(x.detach()) for x in losses['stage']],'grad_norm':float(grad_norm),'lr':scheduler.get_last_lr()[0],'seconds':time.perf_counter()-started,'uses_future_gt':False}
        with metrics.open('a') as handle: handle.write(json.dumps(row)+'\n')
        if args.probe_capture:
            probe_payload['step_time_sec']=row['seconds']; Path(args.probe_capture).parent.mkdir(parents=True,exist_ok=True); torch.save(probe_payload,args.probe_capture)
        if args.train and ((step+1)%args.save_every==0 or step+1==args.max_steps): save_runtime_checkpoint(output/f'checkpoint-{step:06d}.pt',trainable,runner.memory,pipe.transformer,optimizer,scheduler,step,config=config,helios_fingerprint=fingerprint,layers=cfg.sightline_layers,memory_config=memory_config)

if __name__=='__main__': main()
