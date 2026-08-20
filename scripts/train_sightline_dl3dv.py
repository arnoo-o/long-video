"""Construct and dry-run the real geometry-free Sightline training system."""
from __future__ import annotations
import argparse,json,time
from pathlib import Path
import torch
from long_video.config import load_sightline_config
from long_video.training.flow_matching_exact import exact_flow_matching_items
from long_video.training.sightline import SightlineTrainable
from long_video.sightline.helios_integration import SightlineRayProvider, install_sightline_attention
def main():
    p=argparse.ArgumentParser(); p.add_argument('--config',default='configs/sightline.yaml'); p.add_argument('--model',required=True); p.add_argument('--target-latents',required=True); p.add_argument('--helios-root',required=True); p.add_argument('--prompt',default='A stable realistic view of the same scene.'); p.add_argument('--dry-run',action='store_true'); p.add_argument('--metrics',default='sightline_metrics.jsonl'); p.add_argument('--inner-dim',type=int); a=p.parse_args()
    cfg=load_sightline_config(a.config); target=torch.load(a.target_latents,map_location='cuda') if str(a.target_latents).endswith(('.pt','.pth')) else torch.from_numpy(__import__('numpy').load(a.target_latents)).to('cuda'); target=target.to('cuda',dtype=torch.bfloat16)
    import sys; sys.path.insert(0,a.helios_root)
    from helios.diffusers_version.pipeline_helios_diffusers import HeliosPipeline
    pipe=HeliosPipeline.from_pretrained(a.model,torch_dtype=torch.bfloat16).to('cuda')
    inner=a.inner_dim or int(getattr(pipe.transformer.config,'attention_head_dim',64)*getattr(pipe.transformer.config,'num_attention_heads',8)); trainable=SightlineTrainable(inner,cfg.correspondence_layers,heads=int(getattr(pipe.transformer.config,'num_attention_heads',16))).to('cuda',dtype=torch.bfloat16)
    for parameter in pipe.transformer.parameters(): parameter.requires_grad_(False)
    items=exact_flow_matching_items(pipe,target,stage_steps=cfg.pyramid_steps,device=target.device); item=items[0]
    trainable.train(); noisy=item['noisy_latents']; p_t,p_h,p_w=tuple(getattr(pipe.transformer.config,'patch_size',(1,2,2))); token_shape=(noisy.shape[2]//p_t,noisy.shape[3]//p_h,noisy.shape[4]//p_w)
    c2w=torch.eye(4,device='cuda',dtype=torch.float32)[None].repeat(1,33,1,1); K=torch.eye(3,device='cuda',dtype=torch.float32)[None].repeat(1,33,1,1); K[:, :, 0,0]=cfg.source_width; K[:, :, 1,1]=cfg.source_height; K[:, :, 0,2]=cfg.source_width/2; K[:, :, 1,2]=cfg.source_height/2
    provider=SightlineRayProvider(c2w,K,token_shape=token_shape,source_height=cfg.source_height,source_width=cfg.source_width)
    source_history=c2w[:, :1].expand(-1,20,-1,-1); source_K=K[:, :1].expand(-1,20,-1,-1)
    provider.set_context(chunk_index=0,c2w=c2w,intrinsics=K,history_cameras=source_history,history_intrinsics=source_K,token_shape=token_shape,stage_shapes=(token_shape,))
    import helios.diffusers_version.transformer_helios_diffusers as helios_source
    install_sightline_attention(pipe.transformer,trainable.conditioner,provider,layers=cfg.sightline_layers,helios_module=helios_source)
    prompt_embeds=pipe._get_t5_prompt_embeds(a.prompt,device='cuda',dtype=torch.bfloat16)
    _,_,latent_t,latent_h,latent_w=noisy.shape
    zeros=lambda n: torch.zeros((noisy.shape[0],noisy.shape[1],n,latent_h,latent_w),device=noisy.device,dtype=noisy.dtype)
    indices_current=torch.arange(latent_t,device=noisy.device).view(1,-1)
    indices_history=torch.arange(20,device=noisy.device).view(1,-1)
    model_out=pipe.transformer(hidden_states=noisy,timestep=item['timesteps'],encoder_hidden_states=prompt_embeds,
        indices_hidden_states=indices_current,
        indices_latents_history_short=indices_history[:,[0,19]], indices_latents_history_mid=indices_history[:,[17,18]], indices_latents_history_long=indices_history[:,:16],
        latents_history_short=zeros(2),latents_history_mid=zeros(2),latents_history_long=zeros(16),
        attention_kwargs={'current_chunk':0})
    prediction=model_out[0] if isinstance(model_out,(tuple,list)) else getattr(model_out,'sample',model_out)
    if prediction.shape != noisy.shape: raise RuntimeError(f'Helios prediction shape mismatch {prediction.shape} vs {noisy.shape}')
    started=time.perf_counter(); loss=(prediction.float()-item['target'].float()).square().mean(); loss.backward(); elapsed=time.perf_counter()-started
    alpha_grad=trainable.conditioner.alpha.grad
    if alpha_grad is None or not torch.isfinite(alpha_grad).all(): raise RuntimeError('Sightline alpha gradient is missing/non-finite')
    record={'total_loss':float(loss.detach()),'flow_loss':float(loss.detach()),'corr_loss':0.0,'lambda_corr':cfg.lambda_corr,'alpha':float(trainable.conditioner.alpha.detach()),'alpha_grad':float(alpha_grad.detach().abs()),'wrong_ray_delta':None,'memory_zero_delta':None,'memory_shuffle_delta':None,'corr_gain':None,'vram_gb':float(torch.cuda.max_memory_allocated()/2**30),'step_time_sec':elapsed,'uses_future_gt':False,'dry_run':True,'optimizer_step':False,'prediction_source':'helios_transformer'}
    Path(a.metrics).write_text(json.dumps(record)+'\n'); print(json.dumps(record))
    if not a.dry_run: print('Training system constructed; formal optimizer loop intentionally not started.')
if __name__=='__main__': main()
