"""Construct and dry-run the real geometry-free Sightline training system."""
from __future__ import annotations
import argparse,json,time
from pathlib import Path
import torch
from long_video.config import load_sightline_config
from long_video.training.flow_matching_exact import exact_flow_matching_items
from long_video.training.sightline import SightlineTrainable, install_lora
from long_video.training.sightline import select_train_chunk, chunk_grad_policy, assert_single_backward_chunk, assert_trainable_whitelist, curriculum_max_chunks
from long_video.training.sightline_data import load_sightline_manifest, validate_latent_cache
from long_video.sightline.helios_integration import SightlineRayProvider, install_sightline_attention
def main():
    p=argparse.ArgumentParser(); p.add_argument('--config',default='configs/sightline.yaml'); p.add_argument('--model',required=True); p.add_argument('--target-latents',required=True); p.add_argument('--history-latents',required=True); p.add_argument('--helios-root',required=True); p.add_argument('--manifest'); p.add_argument('--expected-records',type=int,default=100); p.add_argument('--prompt',default='A stable realistic view of the same scene.'); p.add_argument('--dry-run',action='store_true'); p.add_argument('--metrics',default='sightline_metrics.jsonl'); p.add_argument('--inner-dim',type=int); p.add_argument('--step',type=int,default=0); p.add_argument('--curriculum-warmup-steps',type=int,default=100); a=p.parse_args()
    cfg=load_sightline_config(a.config); target=torch.load(a.target_latents,map_location='cuda') if str(a.target_latents).endswith(('.pt','.pth')) else torch.from_numpy(__import__('numpy').load(a.target_latents)).to('cuda'); target=target.to('cuda',dtype=torch.bfloat16)
    if not a.manifest:
        raise SystemExit('--manifest is required: training cannot use placeholder cameras or history')
    records=load_sightline_manifest(a.manifest,expected_count=a.expected_records)
    for record in records: record.validate_teacher_and_latent_caches()
    import sys; sys.path.insert(0,a.helios_root)
    from helios.diffusers_version.pipeline_helios_diffusers import HeliosPipeline
    pipe=HeliosPipeline.from_pretrained(a.model,torch_dtype=torch.bfloat16).to('cuda')
    inner=a.inner_dim or int(getattr(pipe.transformer.config,'attention_head_dim',64)*getattr(pipe.transformer.config,'num_attention_heads',8)); trainable=SightlineTrainable(inner,cfg.correspondence_layers,heads=int(getattr(pipe.transformer.config,'num_attention_heads',16))).to('cuda',dtype=torch.bfloat16)
    for parameter in pipe.transformer.parameters(): parameter.requires_grad_(False)
    installed_lora=install_lora(pipe.transformer,cfg.lora_layers,rank=cfg.lora_rank) if cfg.lora_layers else ()
    optimizer=torch.optim.AdamW([{'params':trainable.parameters(),'lr':cfg.learning_rate},{'params':[p for p in pipe.transformer.parameters() if p.requires_grad],'lr':cfg.lora_learning_rate}],weight_decay=0.01)
    history=torch.load(a.history_latents,map_location='cuda') if str(a.history_latents).endswith(('.pt','.pth')) else torch.from_numpy(__import__('numpy').load(a.history_latents)).to('cuda')
    if history.ndim != 5 or history.shape[2] < 20: raise ValueError('history-latents must be [B,C,T,H,W] with at least source+19 real latent slots')
    items=exact_flow_matching_items(pipe,target,stage_steps=cfg.pyramid_steps,device=target.device); item=items[0]
    trainable.train(); noisy=item['noisy_latents']; p_t,p_h,p_w=tuple(getattr(pipe.transformer.config,'patch_size',(1,2,2))); token_shape=(noisy.shape[2]//p_t,noisy.shape[3]//p_h,noisy.shape[4]//p_w)
    c2w_np,K_np=records[0].load_cameras(); c2w=torch.from_numpy(c2w_np[:33]).to('cuda',dtype=torch.float32).unsqueeze(0); K=torch.from_numpy(K_np[:33]).to('cuda',dtype=torch.float32).unsqueeze(0)
    provider=SightlineRayProvider(c2w,K,token_shape=token_shape,source_height=cfg.source_height,source_width=cfg.source_width)
    source_history=c2w[:, :1].expand(-1,20,-1,-1); source_K=K[:, :1].expand(-1,20,-1,-1)
    provider.set_context(chunk_index=0,c2w=c2w,intrinsics=K,history_cameras=source_history,history_intrinsics=source_K,token_shape=token_shape,stage_shapes=(token_shape,))
    import helios.diffusers_version.transformer_helios_diffusers as helios_source
    install_sightline_attention(pipe.transformer,trainable.conditioner,provider,layers=cfg.sightline_layers,helios_module=helios_source)
    prompt_result=pipe._get_t5_prompt_embeds(a.prompt,device='cuda',dtype=torch.bfloat16)
    if not isinstance(prompt_result,(tuple,list)) or len(prompt_result)!=2:
        raise RuntimeError('pinned Helios _get_t5_prompt_embeds must return (prompt_embeds, prompt_mask)')
    prompt_embeds,prompt_mask=prompt_result
    _,_,latent_t,latent_h,latent_w=noisy.shape
    indices_current=torch.arange(latent_t,device=noisy.device).view(1,-1)
    indices_history=torch.arange(20,device=noisy.device).view(1,-1)
    model_out=pipe.transformer(hidden_states=noisy,timestep=item['timesteps'],encoder_hidden_states=prompt_embeds,
        indices_hidden_states=indices_current, encoder_attention_mask=prompt_mask,
        indices_latents_history_short=indices_history[:,[0,19]], indices_latents_history_mid=indices_history[:,[17,18]], indices_latents_history_long=indices_history[:,:16],
        latents_history_short=history[:,:,:2].to(noisy),latents_history_mid=history[:,:,2:4].to(noisy),latents_history_long=history[:,:,4:20].to(noisy),
        attention_kwargs={'current_chunk':0})
    prediction=model_out[0] if isinstance(model_out,(tuple,list)) else getattr(model_out,'sample',model_out)
    if prediction.shape != noisy.shape: raise RuntimeError(f'Helios prediction shape mismatch {prediction.shape} vs {noisy.shape}')
    started=time.perf_counter(); loss=(prediction.float()-item['target'].float()).square().mean(); loss.backward(); elapsed=time.perf_counter()-started
    alpha_grad=trainable.conditioner.alpha.grad
    if alpha_grad is None or not torch.isfinite(alpha_grad).all(): raise RuntimeError('Sightline alpha gradient is missing/non-finite')
    max_chunks=curriculum_max_chunks(a.step,warmup_steps=a.curriculum_warmup_steps,maximum=cfg.chunk_count)
    train_chunk=select_train_chunk(max_chunks)
    policies=[chunk_grad_policy(i,train_chunk) for i in range(max_chunks)]
    assert_single_backward_chunk(policies,train_chunk)
    assert_trainable_whitelist(trainable)
    record={'total_loss':float(loss.detach()),'flow_loss':float(loss.detach()),'corr_loss':0.0,'lambda_corr':cfg.lambda_corr,'alpha':float(trainable.conditioner.alpha.detach()),'alpha_grad':float(alpha_grad.detach().abs()),'wrong_ray_delta':None,'memory_zero_delta':None,'memory_shuffle_delta':None,'corr_gain':None,'vram_gb':float(torch.cuda.max_memory_allocated()/2**30),'step_time_sec':elapsed,'uses_future_gt':False,'dry_run':True,'optimizer_step':False,'prediction_source':'helios_transformer'}
    Path(a.metrics).write_text(json.dumps(record)+'\n'); print(json.dumps(record))
    if not a.dry_run: print('Training system constructed; formal optimizer loop intentionally not started.')
if __name__=='__main__': main()
