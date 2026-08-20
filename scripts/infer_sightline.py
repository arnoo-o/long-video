"""Geometry-free Sightline inference using the pinned Helios pipeline."""
from __future__ import annotations
import argparse, hashlib, json, sys
from pathlib import Path
import numpy as np
from PIL import Image
import torch

def resize_source(image, K, height=384, width=640):
    image=image.convert('RGB'); old_w,old_h=image.size; image=image.resize((width,height),Image.Resampling.LANCZOS)
    sx,sy=width/old_w,height/old_h
    K=np.asarray(K,np.float32).copy(); K[0]*=sx; K[1]*=sy; return image,K

def main():
    p=argparse.ArgumentParser(); p.add_argument('--source',required=True); p.add_argument('--model',required=True); p.add_argument('--out',required=True); p.add_argument('--helios-root',required=True); p.add_argument('--config',default='configs/sightline.yaml'); p.add_argument('--prompt',default=''); p.add_argument('--negative-prompt',default=''); p.add_argument('--intrinsics',required=True); p.add_argument('--c2w'); p.add_argument('--controls'); p.add_argument('--chunks',type=int,default=6); p.add_argument('--steps',type=int,default=2); p.add_argument('--layers',default=''); a=p.parse_args()
    if not 1<=a.chunks<=6: raise ValueError('--chunks must be 1..6')
    if bool(a.c2w)==bool(a.controls): raise ValueError('provide exactly one of --c2w or --controls')
    sys.path.insert(0,a.helios_root)
    from long_video.config import load_sightline_config
    from long_video.sightline.conditioning import SightlineConditioner
    from long_video.sightline.helios_integration import SightlineRayProvider, install_sightline_attention
    from long_video.sightline.pipeline import SightlinePipeline
    from long_video.sightline.rays import canonicalize_c2w, temporal_group_cameras
    from helios.diffusers_version.pipeline_helios_diffusers import HeliosPipeline
    import helios.diffusers_version.transformer_helios_diffusers as helios_source
    source_file=Path(a.helios_root)/'helios/diffusers_version/transformer_helios_diffusers.py'
    if not source_file.is_file(): raise FileNotFoundError(source_file)
    source_fingerprint=hashlib.sha256(source_file.read_bytes()).hexdigest()
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
    c2w=torch.from_numpy(canonicalize_c2w(torch.from_numpy(c2w[None])).squeeze(0)).to('cuda',dtype=torch.float32)
    if K.ndim==2: K=np.repeat(K[None],needed,axis=0)
    elif K.shape[0]<needed: K=np.concatenate((K,np.repeat(K[-1:],needed-K.shape[0],axis=0)),0)
    K=torch.from_numpy(K[:needed]).to('cuda',dtype=torch.float32)[None]
    pipe=HeliosPipeline.from_pretrained(a.model,torch_dtype=torch.bfloat16).to('cuda')
    inner=int(getattr(pipe.transformer.config,'attention_head_dim',64)*getattr(pipe.transformer.config,'num_attention_heads',8))
    conditioner=SightlineConditioner(inner).to('cuda',dtype=torch.bfloat16); provider=SightlineRayProvider(c2w,K,source_height=cfg.source_height,source_width=cfg.source_width)
    layers=tuple(int(x) for x in a.layers.split(',') if x) or tuple(cfg.memory_layers or cfg.correspondence_layers)
    if not layers: raise ValueError('select at least one Sightline self-attention layer via --layers or config')
    install_sightline_attention(pipe.transformer,conditioner,provider,layers=layers,helios_module=helios_source)
    runner=SightlinePipeline(pipe,config=cfg,conditioner=conditioner); runner.assert_geometry_free_imports()
    result=runner.generate(prompt=a.prompt,negative_prompt=a.negative_prompt,image=image,height=cfg.source_height,width=cfg.source_width,num_frames=1+a.chunks*32,steps=a.steps)
    frames=np.asarray(getattr(result,'frames',result)); output=Path(a.out).with_suffix('.npy'); output.parent.mkdir(parents=True,exist_ok=True); np.save(output,frames); print(json.dumps({'pipeline':'sightline_helios','chunks':a.chunks,'layers':layers,'helios_source_fingerprint':source_fingerprint,'frames':int(frames.shape[1] if frames.ndim>1 else len(frames)),'out':str(output)}))
if __name__=='__main__': main()
