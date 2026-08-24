"""Causal Sightline training from the canonical RGB-D manifest.

The historical filename is retained as a legacy compatibility entry point.
Formal training data is loaded exclusively through RGBDMemoryRecord.
"""
from __future__ import annotations
import argparse, hashlib, json, random, sys, time
from dataclasses import asdict
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
from PIL import Image
from long_video.config import load_sightline_config
from long_video.training.flow_matching_exact import exact_flow_matching_items
from long_video.training.sightline import SightlineTrainable, install_lora, curriculum_phase, select_train_chunk, select_chunk_window, run_single_graph_chunks, run_causal_prefix_chunks, selected_qk_logits
from long_video.training.rgbd_memory_data import load_rgbd_memory_manifest
from long_video.training.sightline_data import load_latent_tensor, validate_latent_cache, require_overlap_validation, resolve_continuous_latent_cache
from long_video.training.sightline_checkpoint import save_runtime_checkpoint, restore_runtime_checkpoint, runtime_provenance, capture_rng_state
from long_video.sightline.helios_integration import SightlineRayProvider, install_sightline_attention
from long_video.sightline.history import NativeHistoryState,native_helios_indices
from long_video.sightline.pipeline import SightlinePipeline, prepare_source_condition

TOTAL_TRAINING_STEPS=2500
WARMUP_STEPS=100

def _distributed_context():
    world_size=int(__import__('os').environ.get('WORLD_SIZE','1'))
    if world_size==1: return 0,1,torch.device('cuda')
    if not dist.is_available(): raise RuntimeError('torch.distributed is required for DDP')
    local_rank=int(__import__('os').environ['LOCAL_RANK']); torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl',init_method='env://')
    return dist.get_rank(),world_size,torch.device('cuda',local_rank)

def _average_gradients(parameters,world_size):
    if world_size==1: return
    for parameter in parameters:
        present=torch.tensor(parameter.grad is not None,device=parameter.device,dtype=torch.int32)
        dist.all_reduce(present)
        count=int(present.item())
        if count==0:
            continue
        # Unused parameters on a rank contribute an explicit zero.  This is
        # required for sparse correspondence rows and keeps all ranks in the
        # same collective sequence.
        grad=parameter.grad if parameter.grad is not None else torch.zeros_like(parameter)
        dist.all_reduce(grad); grad.div_(world_size); parameter.grad=grad

def _ddp_record_index(step,rank,world_size,count,seed=20260823):
    if world_size==1: return random.randrange(count)
    if world_size>count: raise ValueError('DDP world size exceeds record count')
    return random.Random(int(seed)+int(step)).sample(range(count),count)[rank]

def _preflight(cfg,args,probe_layers):
    p1_steps=getattr(cfg,'p1_steps',400); p2_steps=getattr(cfg,'p2_steps',600); p3_steps=getattr(cfg,'p3_steps',1500)
    total_steps=p1_steps+p2_steps+p3_steps
    sightline=set(cfg.sightline_layers)
    if not cfg.sightline_layers:
        raise ValueError('formal training requires non-empty sightline_layers')
    if tuple(cfg.camera_layers)!=(1,2,3,4,5,6) or not set(cfg.camera_layers).issubset(set(cfg.sightline_layers)):
        raise ValueError('camera_layers must be exactly 1..6 and belong to sightline_layers')
    if tuple(cfg.memory_layers)!=(16,20,24) or tuple(cfg.correspondence_layers)!=(16,20,24):
        raise ValueError('Memory/correspondence layers must be 16/20/24')
    if args.train and (cfg.memory_layers or cfg.correspondence_layers):
        # These modules remain reserved but are intentionally disabled in the
        # camera-only retraining curriculum.
        pass
    if args.train and not set(probe_layers).issubset(sightline): raise ValueError('formal training probe layers must be a subset of sightline_layers')
    if not set(cfg.lora_layers).issubset(sightline): raise ValueError('lora_layers must be a subset of sightline_layers')
    if not set(cfg.memory_layers).issubset(sightline) or not set(cfg.correspondence_layers).issubset(sightline): raise ValueError('memory/correspondence layers must be a subset of sightline_layers')
    if total_steps!=TOTAL_TRAINING_STEPS or int(total_steps*cfg.warmup_ratio)!=WARMUP_STEPS: raise ValueError('formal schedule must preserve the configured 100/2500-step warmup')
    if args.train and args.max_steps>p1_steps and not cfg.lora_layers: raise ValueError('training reaches P2 but lora_layers is empty')
    if args.train and args.max_steps>p1_steps+p2_steps and (not cfg.memory_layers or not cfg.correspondence_layers): raise ValueError('training reaches P3 but Memory/correspondence layers are empty')
    if not 1<=args.max_steps<=total_steps: raise ValueError(f'--max-steps must be in 1..{total_steps}')

def _lr_multiplier(step,total_steps=TOTAL_TRAINING_STEPS):
    if step<WARMUP_STEPS: return float(step+1)/WARMUP_STEPS
    progress=min(1.0,max(0.0,(step-WARMUP_STEPS)/(total_steps-WARMUP_STEPS)))
    return 0.5*(1.0+__import__('math').cos(__import__('math').pi*progress))

def _sigma_band(step, phase):
    if phase=='P1': return (0.9,1.0),'high_0.9_1.0'
    if phase=='P2': return (0.7,1.0),'mid_high_0.7_1.0'
    if phase=='P3':
        high=(hashlib.sha256(f'sightline-sigma-band:{step}'.encode()).digest()[0]&1)==0
        return ((0.8,1.0),'high_0.8_1.0') if high else ((0.4,0.8),'mid_0.4_0.8')
    raise ValueError(f'unknown phase {phase}')

def _set_gradient_checkpointing(transformer,enabled):
    method=getattr(transformer,'enable_gradient_checkpointing' if enabled else 'disable_gradient_checkpointing',None)
    if method is None:
        if enabled: raise RuntimeError('gradient_checkpointing=true but pinned Helios exposes no enable method')
        return
    method()

def _assert_optimizer_scope(optimizer,trainable,memory,transformer,text_encoder,vae):
    actual={id(parameter) for group in optimizer.param_groups for parameter in group['params']}
    expected={id(parameter) for parameter in trainable.parameters()}|{id(parameter) for parameter in memory.parameters()}|{id(parameter) for name,parameter in transformer.named_parameters() if 'lora_' in name}
    forbidden={id(parameter) for module in (text_encoder,vae) for parameter in module.parameters()}|{id(parameter) for name,parameter in transformer.named_parameters() if 'lora_' not in name}
    if actual!=expected or actual&forbidden: raise RuntimeError('optimizer contains frozen text/VAE/native Helios parameters or misses a Sightline trainable')

def _prompt(pipe,text,device):
    with torch.no_grad(): result=pipe._get_t5_prompt_embeds(text,device=device,dtype=torch.bfloat16,max_sequence_length=512)
    if not isinstance(result,(tuple,list)) or len(result)!=2: raise RuntimeError('pinned Helios prompt API must return (embeds, mask)')
    embeds,mask=result
    if mask.ndim!=2 or mask.shape[:2]!=embeds.shape[:2]: raise RuntimeError('pinned Helios prompt mask shape mismatch')
    return embeds.detach(),mask.detach()

def _model_prediction(pipe,noisy,item,prompt_embeds,history,current_start):
    indices=native_helios_indices(noisy.device,noisy.shape[0])['current']
    output=pipe.transformer(hidden_states=noisy.to(pipe.transformer.dtype),timestep=item['timesteps'],encoder_hidden_states=prompt_embeds,
        indices_hidden_states=indices,latents_history_long=history['long'][0],indices_latents_history_long=history['long'][1],
        latents_history_mid=history['mid'][0],indices_latents_history_mid=history['mid'][1],
        latents_history_short=history['short'][0],indices_latents_history_short=history['short'][1],attention_kwargs={'current_chunk':current_start//8})
    prediction=output[0] if isinstance(output,(tuple,list)) else getattr(output,'sample',output)
    # Helios' coarsest pyramid decoder rounds an odd 15-row grid down to 14
    # rows for 480px inputs.  Keep the canonical 480x832 RGB-D geometry and
    # align its velocity field back to the target latent grid before the
    # flow-matching loss; temporal/channel/batch dimensions must remain exact.
    if prediction.shape!=noisy.shape:
        if prediction.shape[:3]!=noisy.shape[:3]:
            raise RuntimeError(f'prediction shape {prediction.shape} != {noisy.shape}')
        batch,channels,frames,height,width=prediction.shape
        prediction=torch.nn.functional.interpolate(
            prediction.permute(0,2,1,3,4).reshape(batch*frames,channels,height,width).float(),
            size=noisy.shape[-2:],mode='bilinear',align_corners=False,
        ).reshape(batch,frames,channels,*noisy.shape[-2:]).permute(0,2,1,3,4).to(dtype=noisy.dtype)
    if prediction.shape!=noisy.shape: raise RuntimeError(f'prediction shape {prediction.shape} != {noisy.shape}')
    return prediction

def _generate_detached_chunk(pipe,source,history,prompt_embeds,cfg,chunk):
    """Native Helios autoregressive inference from noise; no target argument exists."""
    noise=torch.randn((source.shape[0],source.shape[1],9,source.shape[-2],source.shape[-1]),device=source.device,dtype=source.dtype)
    indices=native_helios_indices(source.device,source.shape[0])['current']
    class Progress:
        def update(self): pass
    pipe._guidance_scale=1.0; pipe._attention_kwargs={'current_chunk':chunk}; pipe._current_timestep=None; pipe._interrupt=False
    return pipe.stage2_sample(latents=noise,pyramid_num_stages=3,pyramid_num_inference_steps_list=list(cfg.pyramid_steps),
        prompt_embeds=prompt_embeds,negative_prompt_embeds=None,guidance_scale=1.0,indices_hidden_states=indices,
        latents_history_long=history['long'][0],indices_latents_history_long=history['long'][1],
        latents_history_mid=history['mid'][0],indices_latents_history_mid=history['mid'][1],
        latents_history_short=history['short'][0],indices_latents_history_short=history['short'][1],
        attention_kwargs={'current_chunk':chunk},device=source.device,transformer_dtype=pipe.transformer.dtype,progress_bar=Progress())

def _load_correspondence(record):
    rows=list(record.correspondence_rows())
    for row in rows:
        qf,kf=int(row['query_frame']),int(row['key_frame'])
        if kf>=qf or not (0<=int(row['key_chunk'])<int(row['query_chunk'])<3): raise RuntimeError('invalid causal RGB-D correspondence identity')
        if not np.isfinite(float(row['weight'])) or float(row['weight'])<0: raise RuntimeError('invalid RGB-D correspondence weight')
    return rows

def _mapped_correspondences(processor,rows,chunk):
    q,k=processor.last_q,processor.last_k
    if q is None or k is None: raise RuntimeError('correspondence processor did not capture Q/K')
    current=processor.last_current_length; identities=processor.last_key_identities
    if identities is None or len(identities)!=k.shape[1]: raise RuntimeError('explicit attention key identity map is missing or misaligned')
    current_shape=next(shape for shape in processor.ray_provider.context['stage_shapes'] if shape[0]*shape[1]*shape[2]==current)
    _,height,width=current_shape; q_start=q.shape[1]-current
    grouped={}
    for row in rows:
        if int(row['query_chunk'])!=chunk: continue
        qt=int(row['query_latent_temporal']); qy=int(row['query_y']); qx=int(row['query_x'])
        if not (0<=qt<current_shape[0] and 0<=qy<height and 0<=qx<width): continue
        qi=q_start+qt*height*width+qy*width+qx
        global_key=int(row['key_chunk'])*8+int(row['key_latent_temporal'])
        query_global=chunk*8+qt
        if global_key>query_global: continue
        ky,kx=int(row['key_y']),int(row['key_x'])
        factors={'long':4,'mid':2,'short':1}
        native=[i for i,identity in enumerate(identities) if identity[0]=='native' and global_key in identity[1] and identity[2:4]==(ky//factors[identity[4]],kx//factors[identity[4]])]
        native.sort(key=lambda index:{'short':0,'mid':1,'long':2}[identities[index][4]])
        memory=[i for i,identity in enumerate(identities) if identity[0]=='memory' and identity[1]==(global_key,) and identity[2:4]==(ky//2,kx//2)]
        current_keys=[i for i,identity in enumerate(identities) if identity[0]=='current' and identity[1]==(global_key,) and identity[2:4]==(ky,kx)]
        source_keys=[i for i,identity in enumerate(identities) if identity[0]=='source' and global_key in identity[1] and identity[2:4]==(ky,kx)]
        # Preserve every legal representation of the teacher identity on the
        # actual key axis.  Source/native/current/memory are diagnostics only.
        candidates=sorted(set(native + current_keys + source_keys + memory))
        if not candidates: continue
        if 0<=qi<q.shape[1]:
            bucket=grouped.setdefault(qi,{})
            for ki in candidates:
                if 0<=ki<k.shape[1]:
                    source='memory' if ki in memory else ('native' if ki in native else ('source' if ki in source_keys else 'current'))
                    bucket[ki]=(max(bucket.get(ki,(0.0,''))[0],float(row['weight'])),source)
    if not grouped: raise RuntimeError('correspondence identities do not map to real attention axes')
    selected=sorted(grouped); positives=[sorted(grouped[query]) for query in selected]; weights=[max(value[0] for value in grouped[query].values()) for query in selected]; sources=['memory' if any(value[1]=='memory' for value in grouped[query].values()) and not any(value[1] in ('source','native','current') for value in grouped[query].values()) else ('native' if any(value[1]=='native' for value in grouped[query].values()) else ('source' if any(value[1]=='source' for value in grouped[query].values()) else 'current')) for query in selected]
    return selected,positives,weights,sources

def _corr_loss(trainable,processors,rows,chunk,layers,max_rows,*,sampling_seed=0):
    if not layers: raise RuntimeError('correspondence is enabled but correspondence_layers is empty')
    missing=[layer for layer in layers if layer not in processors]
    if missing: raise RuntimeError(f'correspondence layers have no Sightline processor: {missing}')
    losses=[]
    for layer in layers:
        processor=processors[layer]
        try:
            selected,positives,weights,sources=_mapped_correspondences(processor,rows,chunk)
        except RuntimeError as exc:
            if 'do not map' not in str(exc): raise
            continue
        if len(selected)>max_rows:
            memory_indices=[i for i,source in enumerate(sources) if source=='memory']; other_indices=[i for i,source in enumerate(sources) if source!='memory']
            generator=torch.Generator(device='cpu').manual_seed(int(sampling_seed) & ((1<<63)-1))
            order=torch.randperm(len(selected),generator=generator).tolist()
            choice=([memory_indices[0]] if memory_indices else []) + [i for i in order if i not in memory_indices[:1]]
            choice=choice[:max_rows]
            selected=[selected[i] for i in choice]; positives=[positives[i] for i in choice]; weights=[weights[i] for i in choice]
        if not selected: continue
        numerator=processor.last_q.new_zeros(()); denominator=processor.last_q.new_zeros(())
        for start in range(0,len(selected),64):
            stop=min(start+64,len(selected)); sampled=selected_qk_logits(processor.last_q,processor.last_k,selected[start:stop])
            if not sampled.requires_grad or not processor.last_k.requires_grad: raise RuntimeError('correspondence Q/K lost autograd; disable incompatible gradient checkpointing')
            block_positive=[(index,keys) for index,keys in enumerate(positives[start:stop])]
            weight=torch.tensor(weights[start:stop],device=sampled.device)
            additive_bias=None
            if getattr(processor,'last_attention_bias',None) is not None:
                bias=processor.last_attention_bias
                q_indices=selected[start:stop]
                if bias.ndim==2: additive_bias=bias
                elif bias.ndim==3: additive_bias=bias[:,q_indices,:].unsqueeze(1)
                elif bias.ndim==4: additive_bias=bias[:,:,q_indices,:]
                else: raise RuntimeError('unsupported captured attention bias shape')
            block_loss=trainable.correspondence(sampled,None,weight,multi_positive=block_positive,additive_bias=additive_bias)
            numerator=numerator+block_loss*weight.sum(); denominator=denominator+weight.sum()
        losses.append(numerator/denominator.clamp_min(1e-8))
    if not losses: return processor.last_q.new_zeros(())
    return torch.stack(losses).mean()

def _reset_sequence(runner):
    runner.reset_sequence()

def _local_rng_state():
    return capture_rng_state()

def _all_rank_rng_states(world_size):
    local=_local_rng_state()
    # PyTorch 2.11 cannot deserialize Tensor-backed objects through
    # all_gather_object.  The checkpoint stores rank 0's complete RNG state;
    # nonzero ranks deterministically fall back to it on resume.
    return [local]

def main():
    p=argparse.ArgumentParser(); p.add_argument('--config',default='configs/sightline.yaml'); p.add_argument('--model',required=True); p.add_argument('--model-revision'); p.add_argument('--helios-root',required=True); p.add_argument('--manifest',required=True)
    p.add_argument('--expected-records',type=int); p.add_argument('--max-steps',type=int); p.add_argument('--resume'); p.add_argument('--allow-memory-layer-migration',action='store_true'); p.add_argument('--output-dir',required=True); p.add_argument('--save-every',type=int); p.add_argument('--latent-cache-root')
    p.add_argument('--prompt',default='A stable realistic view of the same scene.'); p.add_argument('--probe-only',action='store_true'); p.add_argument('--probe-checkpoint'); p.add_argument('--probe-layers',default=''); p.add_argument('--probe-capture'); p.add_argument('--probe-step',type=int,default=1000); p.add_argument('--alpha-zero-baseline',action='store_true'); p.add_argument('--record-index',type=int); p.add_argument('--train-chunk',type=int); p.add_argument('--checkpoint-smoke-step',type=int); p.add_argument('--train',action='store_true'); args=p.parse_args()
    if not (args.train or args.probe_only) or args.train==args.probe_only: raise ValueError('select exactly one of --train or --probe-only')
    cfg=load_sightline_config(args.config); total_steps=cfg.p1_steps+cfg.p2_steps+cfg.p3_steps
    args.max_steps=args.max_steps or total_steps
    save_every=args.save_every or cfg.checkpoint_every
    if args.train and save_every!=cfg.checkpoint_every: raise ValueError(f'formal training checkpoint interval is fixed at {cfg.checkpoint_every}')
    rank,world_size,device=_distributed_context()
    if world_size>1 and not args.train: raise ValueError('DDP is supported only for training')
    if args.train and world_size!=cfg.ddp_world_size: raise ValueError(f'formal training requires exactly {cfg.ddp_world_size} DDP ranks, got {world_size}')
    probe_layers=tuple(int(x) for x in args.probe_layers.split(',') if x); _preflight(cfg,args,probe_layers); records=load_rgbd_memory_manifest(args.manifest,expected_count=args.expected_records)
    if args.train and len(records)!=400: raise ValueError(f'formal training requires exactly 400 train records, got {len(records)}')
    if cfg.chunk_count!=3 or cfg.chunk_length!=33 or cfg.chunk_stride!=32 or (cfg.source_height,cfg.source_width)!=(480,832): raise ValueError('formal RGB-D training requires 3 chunks, 97 frames, and 480x832 geometry')
    sys.path.insert(0,args.helios_root)
    from helios.diffusers_version.pipeline_helios_diffusers import HeliosPipeline
    import helios.diffusers_version.transformer_helios_diffusers as helios_source
    source_file=Path(args.helios_root)/'helios/diffusers_version/transformer_helios_diffusers.py'; fingerprint=hashlib.sha256(source_file.read_bytes()).hexdigest()
    pipe=HeliosPipeline.from_pretrained(args.model,torch_dtype=torch.bfloat16,revision=args.model_revision).to(device); heads=int(pipe.transformer.config.num_attention_heads); inner=int(pipe.transformer.config.attention_head_dim*heads)
    pipe.text_encoder.eval().requires_grad_(False); pipe.vae.eval().requires_grad_(False)
    trainable=SightlineTrainable(inner,layers=cfg.sightline_layers,camera_layers=cfg.camera_layers,heads=heads).to(device,dtype=torch.float32)
    for parameter in pipe.transformer.parameters(): parameter.requires_grad_(False)
    install_lora(pipe.transformer,cfg.lora_layers,rank=cfg.lora_rank) if cfg.lora_layers else None
    provider=SightlineRayProvider(source_height=cfg.source_height,source_width=cfg.source_width); runner=SightlinePipeline(pipe,config=cfg,conditioner=trainable.conditioner,ray_provider=provider)
    runner.memory.to(device=device,dtype=torch.bfloat16)
    installed_layers=tuple(sorted(set(cfg.sightline_layers).union(cfg.memory_layers).union(cfg.correspondence_layers).union(probe_layers))) if args.probe_only else tuple(sorted(set(cfg.sightline_layers).union(cfg.memory_layers).union(cfg.correspondence_layers)))
    install_sightline_attention(pipe.transformer,trainable.conditioner,provider,layers=installed_layers,helios_module=helios_source,memory=runner.memory,memory_layers=cfg.memory_layers,camera_layers=cfg.camera_layers)
    lora_params=[p for n,p in pipe.transformer.named_parameters() if 'lora_' in n]
    memory_params=list(runner.memory.parameters())
    optimizer=torch.optim.AdamW([{'params':list(trainable.parameters()),'lr':cfg.learning_rate},{'params':lora_params,'lr':cfg.lora_learning_rate},{'params':memory_params,'lr':cfg.learning_rate}],weight_decay=.01)
    _assert_optimizer_scope(optimizer,trainable,runner.memory,pipe.transformer,pipe.text_encoder,pipe.vae)
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda step:_lr_multiplier(step,total_steps))
    prompt_embeds,_=_prompt(pipe,args.prompt,device); config=asdict(cfg); memory_config={'layers':list(cfg.memory_layers),'pool':cfg.memory_pool,'budget':cfg.memory_budget}; provenance=runtime_provenance(pipe,args.model,args.helios_root,model_revision=args.model_revision)
    trainable.eval() if args.probe_only else trainable.train()
    start_step=args.probe_step if args.probe_only else 0
    if args.resume:
        payload=torch.load(args.resume,map_location='cpu'); completed_step=restore_runtime_checkpoint(payload,trainable,runner.memory,pipe.transformer,config=config,helios_fingerprint=fingerprint,layers=cfg.sightline_layers,memory_config=memory_config,optimizer=None if args.allow_memory_layer_migration else optimizer,scheduler=scheduler,restore_rng=True,provenance=provenance,rank=rank,allow_memory_layer_migration=args.allow_memory_layer_migration); start_step=completed_step+1
    elif args.probe_checkpoint:
        if not args.probe_only: raise ValueError('--probe-checkpoint is only valid with --probe-only')
        payload=torch.load(args.probe_checkpoint,map_location='cpu'); restored_step=restore_runtime_checkpoint(payload,trainable,runner.memory,pipe.transformer,config=config,helios_fingerprint=fingerprint,layers=cfg.sightline_layers,memory_config=memory_config,restore_rng=False,provenance=provenance); start_step=restored_step
    elif args.alpha_zero_baseline:
        for alpha in trainable.conditioner.alpha_parameters(): alpha.data.zero_()
        runner.memory.timestamp.weight.data.zero_()
    if world_size>1 and not args.resume:
        rank_seed=20260823+1000003*rank+9176*start_step
        random.seed(rank_seed); np.random.seed(rank_seed%(2**32-1)); torch.manual_seed(rank_seed); torch.cuda.manual_seed_all(rank_seed)
    output=Path(args.output_dir); output.mkdir(parents=True,exist_ok=True); metrics=output/'metrics.jsonl'
    stop=args.max_steps if args.train else min(args.max_steps,start_step+1)
    for step in range(start_step,stop):
        phase=curriculum_phase(step,p1_steps=cfg.p1_steps,p2_steps=cfg.p2_steps,p3_steps=cfg.p3_steps); phase={**phase,'max_chunks':min(int(cfg.chunk_count),int(phase['max_chunks']))}; checkpointing=bool(cfg.gradient_checkpointing); _set_gradient_checkpointing(pipe.transformer,checkpointing)
        if args.alpha_zero_baseline: phase={**phase,'memory':False,'lora':False,'correspondence':False}
        eligible_records=[record for record in records if record.memory_eligible] if (phase['memory'] or phase['correspondence'] or bool(args.probe_capture)) else records
        if not eligible_records: raise RuntimeError(f"{phase['name']} requires at least one memory-eligible RGB-D record")
        if args.record_index is not None:
            record=records[args.record_index]
            if record not in eligible_records: raise ValueError(f"record {record.record_id} is camera-only and cannot be used in {phase['name']}")
        else:
            index=_ddp_record_index(step,rank,world_size,len(eligible_records)); record=eligible_records[index]
        latent_root=args.latent_cache_root or cfg.latent_cache_path or None
        latent_path=resolve_continuous_latent_cache(record,cache_root=latent_root); latent_schema,_=validate_latent_cache(latent_path)
        if args.train and latent_schema=='overlap_chunks_6x9': require_overlap_validation(latent_path,expected_provenance=str(provenance['model_identity']))
        all_latents=load_latent_tensor(latent_path)
        if all_latents.shape[2]<25: raise ValueError('three 33-frame chunks require at least 25 latent frames')
        window_start=0 if phase['name']=='P3' else select_chunk_window(phase['max_chunks'],total_chunks=cfg.chunk_count)
        latent_start=window_start*8; frame_start=window_start*32
        latents=all_latents[:,:,latent_start:latent_start+1+phase['max_chunks']*8].to(device,dtype=torch.bfloat16)
        rgb_paths=record.rgb_paths()
        if len(rgb_paths)<=frame_start: raise RuntimeError('trajectory RGB frames do not cover sampled chunk window')
        source,fake,_,_=prepare_source_condition(pipe,Image.open(rgb_paths[frame_start]).convert('RGB'),height=cfg.source_height,width=cfg.source_width,device=device)
        c2w_np,K_np=record.load_cameras(); c2w_np=np.array(c2w_np[frame_start:frame_start+1+phase['max_chunks']*32],copy=True); K_np=np.array(K_np[frame_start:frame_start+1+phase['max_chunks']*32],copy=True)
        c2w=torch.from_numpy(c2w_np).to(device,dtype=torch.float32).unsqueeze(0); c2w=torch.linalg.inv(c2w[:,:1])@c2w
        K=torch.from_numpy(K_np).to(device,dtype=torch.float32).unsqueeze(0)
        _reset_sequence(runner); runner._trajectory_c2w=c2w; runner._trajectory_K=K; runner._source_camera=c2w[:,0]; runner._source_intrinsics=K[:,0]; runner.memory.set_enabled(phase['memory'])
        for name,parameter in pipe.transformer.named_parameters():
            if 'lora_' in name: parameter.requires_grad_(phase['lora'])
        active_corr_layers=probe_layers or tuple(cfg.correspondence_layers); diagnostic_correspondence=bool(args.probe_capture)
        if phase['correspondence'] or diagnostic_correspondence:
            if not record.memory_eligible: raise RuntimeError(f"{record.record_id} has no calibrated RGB-D correspondence supervision")
            if not active_corr_layers or any(layer not in pipe.transformer._sightline_processors for layer in active_corr_layers): raise RuntimeError('active correspondence/probe layers are not installed Sightline layers')
            for layer in active_corr_layers: pipe.transformer._sightline_processors[layer].capture_diagnostics=True
            corr_rows=_load_correspondence(record)
        else: corr_rows=None
        train_chunk=args.train_chunk if args.train_chunk is not None else select_train_chunk(phase['max_chunks'])
        if not 0<=train_chunk<phase['max_chunks']: raise ValueError('train_chunk outside curriculum')
        sigma_range,sigma_band=_sigma_band(step,phase['name'])
        history_state=NativeHistoryState(source,fake); losses={}; probe_payload={}; optimizer.zero_grad(set_to_none=True)
        if args.probe_capture: torch.cuda.reset_peak_memory_stats()
        started=time.perf_counter()
        def forward_chunk(chunk,keep_graph):
            history=history_state.groups(); coverage=history_state.coverage()
            if not keep_graph:
                template=torch.empty((source.shape[0],source.shape[1],9,*source.shape[-2:]),device=source.device,dtype=source.dtype); runner._prepare_chunk(chunk,template,{},history_global_coverages=coverage,history_validity=history_state.validity())
                generated=_generate_detached_chunk(pipe,source,history,prompt_embeds,cfg,chunk).detach()
            else:
                target=latents[:,:,chunk*8:chunk*8+9]  # The sole non-source GT read in this step.
                items=exact_flow_matching_items(pipe,target,stage_steps=cfg.pyramid_steps,device=target.device,sigma_range=sigma_range); runner._prepare_chunk(chunk,target,{},history_global_coverages=coverage,history_validity=history_state.validity())
                stage_losses=[]; final_prediction=None
                for stage_index,item in enumerate(items):
                    prediction=_model_prediction(pipe,item['noisy_latents'],item,prompt_embeds,history,chunk*8); final_prediction=prediction
                    stage_loss=(prediction.float()-item['target'].float()).square().mean(); stage_losses.append(stage_loss)
                    if args.train and stage_index+1<len(items): (stage_loss/len(items)).backward()
                fm=torch.stack([loss.detach() if args.train else loss for loss in stage_losses]).mean()
                corr_metric=_corr_loss(trainable,pipe.transformer._sightline_processors,corr_rows,chunk,active_corr_layers,cfg.correspondence_rows_per_batch,sampling_seed=int(hashlib.sha256(f'{step}:{record.trajectory_id}'.encode()).hexdigest()[:16],16)) if (corr_rows and not args.alpha_zero_baseline) else fm.new_zeros(())
                corr=corr_metric if phase['correspondence'] else fm.new_zeros(())
                total=stage_losses[-1]/len(items)+trainable.lambda_corr(step/total_steps)*corr if args.train else fm+trainable.lambda_corr(step/total_steps)*corr
                losses.update(fm=fm,corr=corr,total=total,stage=stage_losses,sigmas=[float(item['sigmas'].mean()) for item in items])
                if args.train: total.backward()
                final=items[-1]; generated=(final['noisy_latents']-final['sigmas']*final_prediction).detach()
                if args.probe_capture:
                    layer_captures=[]
                    for layer in active_corr_layers:
                        processor=pipe.transformer._sightline_processors[layer]
                        try: selected,positives,_,_=_mapped_correspondences(processor,corr_rows,chunk)
                        except RuntimeError as exc:
                            if 'do not map' in str(exc): continue
                            raise
                        selected=selected[:cfg.correspondence_rows_per_batch]; positives=positives[:len(selected)]
                        selected_q=processor.last_q[:,selected]; base_k=processor.last_k
                        head_logits=torch.einsum('bqhd,bkhd->bhqk',selected_q,base_k)*(selected_q.shape[-1]**-.5)
                        layer_captures.append({'layer':layer,'attention_logits':head_logits.detach().cpu(),'positive_key_indices':positives,'memory_count':processor.last_attention_meta.get('memory_tokens',0)})
                    if not layer_captures: raise RuntimeError('probe candidates have no mapped correspondence rows')
                    layer=layer_captures[0]['layer']; processor=pipe.transformer._sightline_processors[layer]
                    base_context=dict(provider.context); normal_step_time=time.perf_counter()-started; ablation_started=time.perf_counter()
                    memory_enabled_by_layer={layer:bank.enabled for layer,bank in runner.memory.banks.items()}
                    with torch.no_grad():
                        provider.context=dict(base_context); provider.context['c2w']=base_context['c2w'].flip(1)
                        wrong=_model_prediction(pipe,final['noisy_latents'],final,prompt_embeds,history,chunk*8)
                        runner.memory.set_enabled(False); provider.context=base_context
                        zero=_model_prediction(pipe,final['noisy_latents'],final,prompt_embeds,history,chunk*8)
                        for bank_layer,enabled in memory_enabled_by_layer.items(): runner.memory.banks[bank_layer].enabled=enabled
                        originals={layer:[token.hidden for token in bank.tokens] for layer,bank in runner.memory.banks.items()}
                        for bank in runner.memory.banks.values():
                            shuffled=list(reversed([token.hidden for token in bank.tokens]))
                            for token,hidden in zip(bank.tokens,shuffled): token.hidden=hidden
                        shuffled_prediction=_model_prediction(pipe,final['noisy_latents'],final,prompt_embeds,history,chunk*8)
                        for bank_layer,hiddens in originals.items():
                            for token,hidden in zip(runner.memory.banks[bank_layer].tokens,hiddens): token.hidden=hidden
                        provider.context=base_context
                    alpha_q,alpha_k=trainable.conditioner.alpha_values()
                    final_stage_loss=float((final_prediction.float()-final['target'].float()).square().mean())
                    first=layer_captures[0]
                    probe_payload.update(source='real_helios_forward',baseline=bool(args.alpha_zero_baseline),layer=first['layer'],sigma=float(final['sigmas'].mean()),attention_logits=first['attention_logits'],positive_key_indices=first['positive_key_indices'],memory_count=first['memory_count'],layer_captures=layer_captures,fm_loss=float(fm.detach()),baseline_final_stage_loss=final_stage_loss,wrong_ray_loss=float((wrong.float()-final['target'].float()).square().mean()),memory_zero_loss=float((zero.float()-final['target'].float()).square().mean()),memory_shuffle_loss=float((shuffled_prediction.float()-final['target'].float()).square().mean()),corr_loss=float(corr_metric.detach()),alpha_q=alpha_q,alpha_k=alpha_k,vram_gb=float(torch.cuda.max_memory_allocated()/2**30),step_time_sec=normal_step_time,ablation_time_sec=time.perf_counter()-ablation_started)
            if chunk==0: generated[:,:,0:1]=source
            history_state.append_chunk(generated,chunk)
            runner._finalize_chunk(chunk)
            for processor in pipe.transformer._sightline_processors.values():
                processor.last_q=processor.last_k=processor.last_hidden_states=processor.last_key_identities=None
                processor.last_attention_bias=None
            return generated
        if phase['name']=='P3':
            _,policies=run_causal_prefix_chunks(3,train_chunk,forward_chunk)
        elif args.probe_only and args.alpha_zero_baseline:
            with torch.no_grad(): _,policies=run_single_graph_chunks(phase['max_chunks'],train_chunk,forward_chunk)
        else:
            _,policies=run_single_graph_chunks(phase['max_chunks'],train_chunk,forward_chunk)
        if args.probe_capture:
            probe_payload['alpha_grad']={name:0.0 if alpha.grad is None else float(alpha.grad.detach().abs()) for name,alpha in ((f'{index}.q',layer.alpha_q) for index,layer in trainable.conditioner.layers.items())}
        if args.train:
            active_phase=curriculum_phase(step,p1_steps=cfg.p1_steps,p2_steps=cfg.p2_steps,p3_steps=cfg.p3_steps)
            alpha_grads=[alpha.grad for alpha in trainable.conditioner.alpha_parameters()]
            if active_phase['name']=='P1' and (any(grad is None for grad in alpha_grads) or not all(torch.isfinite(grad).all() for grad in alpha_grads)): raise RuntimeError('P1 alpha gradient missing or non-finite')
            if active_phase['name']=='P2':
                lora_grads=[p.grad for p in lora_params if p.requires_grad and p.grad is not None]
                if not lora_grads or not all(torch.isfinite(g).all() for g in lora_grads): raise RuntimeError('P2 LoRA gradient missing or non-finite')
            if active_phase['name']=='P3' and losses['corr'].requires_grad:
                if any(alpha.abs().detach()>1e-6 for alpha in trainable.conditioner.alpha_parameters()):
                    geometry_params=list(trainable.conditioner.geometry_parameters())
                    corr_grads=[p.grad for p in geometry_params if p.grad is not None]
                    if not corr_grads or not all(torch.isfinite(g).all() for g in corr_grads): raise RuntimeError('P3 geometry gradient missing or non-finite')
                    camera_grads=[p.grad for p in trainable.conditioner.camera_parameters() if p.grad is not None]
                    if not camera_grads or not all(torch.isfinite(g).all() for g in camera_grads): raise RuntimeError('P3 camera residual gradient missing or non-finite')
            optimized=[p for group in optimizer.param_groups for p in group['params']]; _average_gradients(optimized,world_size); grad_norm=torch.nn.utils.clip_grad_norm_([p for p in optimized if p.grad is not None],cfg.grad_clip); optimizer.step(); scheduler.step()
        else: grad_norm=torch.tensor(0.)
        alpha_q,alpha_k=trainable.conditioner.alpha_values()
        row={'step':step,'record':record.trajectory_id,'phase':phase['name'],'max_chunks':phase['max_chunks'],'window_start_chunk':window_start,'train_chunk':train_chunk,'executed_chunks':len(policies),'policies':policies,'flow_loss':float(losses['fm'].detach()),'corr_loss':float(losses['corr'].detach()),'stage_losses':[float(x.detach()) for x in losses['stage']],'stage_sigmas':losses['sigmas'],'sampled_sigma':losses['sigmas'],'sigma_band':sigma_band,'alpha_q':alpha_q,'alpha_k':alpha_k,'grad_norm':float(grad_norm),'lr':scheduler.get_last_lr()[0],'gradient_checkpointing':checkpointing,'seconds':time.perf_counter()-started,'uses_future_gt':False}
        if rank==0 and ((step+1)%cfg.diagnostics_frequency==0 or step==start_step or step+1==stop):
            with metrics.open('a') as handle: handle.write(json.dumps(row)+'\n')
        if args.probe_capture:
            Path(args.probe_capture).parent.mkdir(parents=True,exist_ok=True); torch.save(probe_payload,args.probe_capture)
        checkpoint_due=(step+1)%save_every==0 or step+1==args.max_steps or (args.checkpoint_smoke_step is not None and step+1==args.checkpoint_smoke_step)
        if args.train and checkpoint_due:
            rng_states=_all_rank_rng_states(world_size)
            if rank==0: save_runtime_checkpoint(output/f'checkpoint-{step:06d}.pt',trainable,runner.memory,pipe.transformer,optimizer,scheduler,step,config=config,helios_fingerprint=fingerprint,layers=cfg.sightline_layers,memory_config=memory_config,provenance=provenance,rng_states=rng_states)
            if world_size>1: dist.barrier()
    if world_size>1: dist.destroy_process_group()

if __name__=='__main__': main()
