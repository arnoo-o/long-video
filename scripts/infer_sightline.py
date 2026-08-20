"""Geometry-free Sightline inference using the pinned Helios pipeline."""
from __future__ import annotations
import argparse, json, sys
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
    from long_video.sightline.rays import canonicalize_c2w, temporal_group_cameras
    from helios.diffusers_version.pipeline_helios_diffusers import HeliosPipeline
    import helios.diffusers_version.transformer_helios_diffusers as helios_source
    cfg=load_sightline_config(a.config); image=Image.open(a.source); K=np.load(a.intrinsics) if a.intrinsics.endswith('.npy') else np.asarray(json.loads(Path(a.intrinsics).read_text()),np.float32)
    image,K=resize_source(image,K,cfg.source_height,cfg.source_width)
    c2w=np.load(a.c2w) if a.c2w and a.c2w.endswith('.npy') else np.asarray(json.loads(Path(a.c2w).read_text()),np.float32) if a.c2w else np.eye(4,dtype=np.float32)[None].repeat(33,0)
    if a.controls:
        from long_video.data.controls import integrate_controls
        controls=json.loads(Path(a.controls).read_text()); c2w=np.concatenate((c2w[:1],integrate_controls(c2w[0],controls)),0)
    if c2w.shape[0] < 33: c2w=np.concatenate((c2w,np.repeat(c2w[-1:],33-c2w.shape[0],0)),0)
    c2w=torch.from_numpy(canonicalize_c2w(torch.from_numpy(c2w[None])).squeeze(0)).to('cuda',dtype=torch.float32)
    K=torch.from_numpy(K).to('cuda',dtype=torch.float32)[None].expand(1,33,3,3)
    pipe=HeliosPipeline.from_pretrained(a.model,torch_dtype=torch.bfloat16).to('cuda')
    inner=int(getattr(pipe.transformer.config,'attention_head_dim',64)*getattr(pipe.transformer.config,'num_attention_heads',8))
    conditioner=SightlineConditioner(inner).to('cuda',dtype=torch.bfloat16); provider=SightlineRayProvider(c2w,K,source_height=cfg.source_height,source_width=cfg.source_width)
    layers=tuple(int(x) for x in a.layers.split(',') if x) or tuple(cfg.memory_layers or cfg.correspondence_layers)
    if not layers: raise ValueError('select at least one Sightline self-attention layer via --layers or config')
    install_sightline_attention(pipe.transformer,conditioner,provider,layers=layers,helios_module=helios_source)
    result=pipe(prompt=a.prompt,negative_prompt=a.negative_prompt,image=image,height=cfg.source_height,width=cfg.source_width,num_frames=1+a.chunks*32,num_inference_steps=a.steps,history_sizes=list(cfg.history_sizes),num_latent_frames_per_chunk=9,is_enable_stage2=True,pyramid_num_inference_steps_list=list(cfg.pyramid_steps),output_type='np')
    frames=np.asarray(getattr(result,'frames',result)); output=Path(a.out).with_suffix('.npy'); output.parent.mkdir(parents=True,exist_ok=True); np.save(output,frames); print(json.dumps({'pipeline':'sightline_helios','chunks':a.chunks,'layers':layers,'frames':int(frames.shape[1] if frames.ndim>1 else len(frames)),'out':str(output)}))
if __name__=='__main__': main()
