"""Geometry-free Sightline inference using the pinned Helios pipeline."""
from __future__ import annotations
import argparse, hashlib, json, sys
from dataclasses import asdict
from pathlib import Path
import numpy as np
from PIL import Image
import torch

def configure_inference_memory(runner, disabled:bool) -> bool:
    enabled=not bool(disabled)
    runner.memory.set_enabled(enabled)
    return enabled

def configure_sightline_residual_scale(transformer, value:float) -> float:
    value=float(value)
    if not np.isfinite(value) or value<0: raise ValueError('Sightline residual scale must be finite and non-negative')
    processors=getattr(transformer,'_sightline_processors',{})
    if not processors: raise RuntimeError('Sightline processors must be installed before setting residual scale')
    for processor in processors.values(): processor.residual_scale=value
    return value

def resize_source(image, K, height=384, width=640):
    image=image.convert('RGB'); old_w,old_h=image.size; image=image.resize((width,height),Image.Resampling.LANCZOS)
    sx,sy=width/old_w,height/old_h
    K=np.asarray(K,np.float32).copy()
    if K.shape==(3,3): K[0,:]*=sx; K[1,:]*=sy
    elif K.ndim==3 and K.shape[1:]==(3,3): K[:,0,:]*=sx; K[:,1,:]*=sy
    else: raise ValueError('K must be [3,3] or [F,3,3]')
    return image,K

def main():
    p=argparse.ArgumentParser(); p.add_argument('--source',required=True); p.add_argument('--model',required=True); p.add_argument('--model-revision'); p.add_argument('--out',required=True); p.add_argument('--helios-root',required=True); p.add_argument('--config',default='configs/sightline.yaml'); p.add_argument('--checkpoint'); p.add_argument('--alpha-zero-baseline',action='store_true'); p.add_argument('--disable-memory',action='store_true'); p.add_argument('--sightline-residual-scale',type=float,default=1.0); p.add_argument('--near-depth',type=float,default=1.0,help='Trajectory near-depth in metres used to normalize relative translations.'); p.add_argument('--boundary-off-from-chunk',type=int); p.add_argument('--prompt',default=''); p.add_argument('--negative-prompt',default=''); p.add_argument('--intrinsics',required=True); p.add_argument('--c2w'); p.add_argument('--controls'); p.add_argument('--chunks',type=int,default=6); p.add_argument('--steps',type=int); p.add_argument('--layers',default=''); a=p.parse_args()
    if not 1<=a.chunks<=6: raise ValueError('--chunks must be 1..6')
    if bool(a.c2w)==bool(a.controls): raise ValueError('provide exactly one of --c2w or --controls')
    if bool(a.checkpoint)==bool(a.alpha_zero_baseline): raise ValueError('provide --checkpoint, or explicitly select --alpha-zero-baseline')
    sys.path.insert(0,a.helios_root)
    from long_video.config import load_sightline_config
    from long_video.training.sightline import SightlineTrainable, install_lora, configure_alpha_zero_baseline, set_initialization_seed
    from long_video.training.sightline_checkpoint import restore_runtime_checkpoint, runtime_provenance
    from scripts.train_sightline_rgbd import _install_memory_efficient_helios_norm
    from long_video.sightline.helios_integration import SightlineRayProvider, install_sightline_attention
    from long_video.sightline.pipeline import SightlinePipeline
    from long_video.sightline.geometry import padded_size
    from long_video.sightline.rays import canonicalize_c2w, temporal_group_cameras
    source_file=Path(a.helios_root)/'helios/diffusers_version/transformer_helios_diffusers.py'
    if not source_file.is_file(): raise FileNotFoundError(source_file)
    # Match training's canonical provenance while using its algebraically
    # identical low-peak norm implementation at runtime.
    runtime_patch=_install_memory_efficient_helios_norm(source_file)
    source_fingerprint=runtime_patch['original_source_sha256']
    from helios.diffusers_version.pipeline_helios_diffusers import HeliosPipeline
    import helios.diffusers_version.transformer_helios_diffusers as helios_source
    for symbol in ('_get_qkv_projections','apply_rotary_emb_transposed','dispatch_attention_fn','HeliosAttnProcessor'):
        if not hasattr(helios_source,symbol): raise RuntimeError(f'pinned Helios API missing {symbol}')
    cfg=load_sightline_config(a.config); image=Image.open(a.source); K=np.load(a.intrinsics) if a.intrinsics.endswith('.npy') else np.asarray(json.loads(Path(a.intrinsics).read_text()),np.float32)
    image,K=resize_source(image,K,cfg.source_height,cfg.source_width)
    c2w=np.load(a.c2w) if a.c2w and a.c2w.endswith('.npy') else np.asarray(json.loads(Path(a.c2w).read_text()),np.float32) if a.c2w else np.eye(4,dtype=np.float32)[None].repeat(33,0)
    if a.controls:
        from long_video.data.controls import integrate_controls
        controls=json.loads(Path(a.controls).read_text()); c2w=np.concatenate((c2w[:1],integrate_controls(c2w[0],controls)),0)
    needed=1+a.chunks*32
    if c2w.shape[0] < needed: c2w=np.concatenate((c2w,np.repeat(c2w[-1:],needed-c2w.shape[0],0)),0)
    c2w=c2w[:needed]
    if not np.isfinite(a.near_depth) or a.near_depth <= 0: raise ValueError('--near-depth must be finite and positive')
    c2w=canonicalize_c2w(torch.from_numpy(c2w[None]),float(a.near_depth)).to('cuda',dtype=torch.float32)
    if K.ndim==2: K=np.repeat(K[None],needed,axis=0)
    elif K.shape[0]<needed: K=np.concatenate((K,np.repeat(K[-1:],needed-K.shape[0],axis=0)),0)
    K=torch.from_numpy(K[:needed]).to('cuda',dtype=torch.float32)[None]
    pipe=HeliosPipeline.from_pretrained(a.model,torch_dtype=torch.bfloat16,revision=a.model_revision).to('cuda')
    inner=int(getattr(pipe.transformer.config,'attention_head_dim',64)*getattr(pipe.transformer.config,'num_attention_heads',8))
    geometry_layers=tuple(int(x) for x in a.layers.split(',') if x) or tuple(cfg.sightline_layers)
    if not geometry_layers: raise ValueError('select at least one Sightline self-attention layer via --layers or config')
    if tuple(geometry_layers) != tuple(cfg.sightline_layers):
        raise RuntimeError('formal inference requires the configured all-layer geometry set; refusing partial Sightline layer installation')
    layers=tuple(sorted(set(geometry_layers).union(cfg.memory_layers)))
    set_initialization_seed()
    trainable=SightlineTrainable(inner,layers=geometry_layers,heads=int(pipe.transformer.config.num_attention_heads)).to('cuda',dtype=torch.float32); conditioner=trainable.conditioner
    padded_h,padded_w=padded_size(cfg.source_height,cfg.source_width); provider=SightlineRayProvider(c2w,K,source_height=padded_h,source_width=padded_w)
    runner=SightlinePipeline(pipe,config=cfg,conditioner=conditioner,ray_provider=provider)
    runner.memory.to(device='cuda',dtype=torch.bfloat16)
    install_lora(pipe.transformer,cfg.lora_layers,rank=cfg.lora_rank) if cfg.lora_layers else None
    install_sightline_attention(pipe.transformer,conditioner,provider,layers=layers,helios_module=helios_source,memory=runner.memory,memory_layers=cfg.memory_layers)
    if a.checkpoint:
        payload=torch.load(a.checkpoint,map_location='cpu')
        provenance=runtime_provenance(pipe,a.model,a.helios_root,model_revision=a.model_revision,transformer_source_sha256=source_fingerprint,runtime_patch=runtime_patch)
        restore_runtime_checkpoint(payload,trainable,runner.memory,pipe.transformer,config=asdict(cfg),helios_fingerprint=source_fingerprint,layers=geometry_layers,memory_config={'layers':list(cfg.memory_layers),'pool':cfg.memory_pool,'budget':cfg.memory_budget,'tau_pos':cfg.memory_tau_pos,'tau_angle':cfg.memory_tau_angle},provenance=provenance)
    else:
        configure_alpha_zero_baseline(trainable,runner.memory,pipe.transformer)
    memory_enabled=configure_inference_memory(runner,a.alpha_zero_baseline or a.disable_memory)
    residual_scale=configure_sightline_residual_scale(pipe.transformer,a.sightline_residual_scale)
    trainable.eval(); conditioner.eval()
    runner.assert_geometry_free_imports()
    result=runner.generate(prompt=a.prompt,negative_prompt=a.negative_prompt,image=image,height=cfg.source_height,width=cfg.source_width,num_frames=1+a.chunks*32,steps=a.steps,c2w=c2w,intrinsics=K,boundary_off_from_chunk=a.boundary_off_from_chunk)
    frames=np.asarray(getattr(result,'frames',result)); output=Path(a.out).with_suffix('.npy'); output.parent.mkdir(parents=True,exist_ok=True); np.save(output,frames); print(json.dumps({'pipeline':'sightline_helios','chunks':a.chunks,'layers':layers,'memory_enabled':memory_enabled,'sightline_residual_scale':residual_scale,'near_depth':float(a.near_depth),'geometry_sigma_routing':'shared-v2','boundary_off_from_chunk':a.boundary_off_from_chunk,'helios_source_fingerprint':source_fingerprint,'frames':int(frames.shape[1] if frames.ndim>1 else len(frames)),'out':str(output)}))
if __name__=='__main__': main()
