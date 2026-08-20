"""Local copy of pinned Helios exact flow-matching sampling semantics.

This module intentionally does not import the legacy WAH package.  The formulas
match the pinned scheduler's ``start_sigmas/end_sigmas`` and per-stage arrays.
"""
from __future__ import annotations
import math
import torch
import torch.nn.functional as F

def _density(batch, device):
    u=torch.sigmoid(torch.randn((batch,),device=device)); return u

def _upsample(source,target):
    if source.shape[-3:]==target.shape[-3:]: return source
    return F.interpolate(source,size=target.shape[-3:],mode='trilinear',align_corners=False)

def exact_flow_matching_items(pipe, target_latents, *, stage_steps=(2,2,2), device=None):
    device=device or target_latents.device; stages=len(stage_steps)
    scheduler=pipe.scheduler
    if int(scheduler.config.get('stages',stages)) != stages: raise ValueError('scheduler stage count mismatch')
    # The official pyramid latent construction is deterministic downsample by
    # the transformer patch size, then reverse order to native stage order.
    clean=[target_latents]
    for _ in range(stages-1): clean.insert(0,F.avg_pool3d(clean[0],kernel_size=(1,2,2),stride=(1,2,2)))
    noise=torch.randn_like(clean[-1]); items=[]; train_steps=int(scheduler.config.num_train_timesteps)
    for stage in range(stages):
        current=clean[stage]; start=float(scheduler.start_sigmas[stage]); end=float(scheduler.end_sigmas[stage])
        start_point=noise if stage==0 else start*noise+(1-start)*_upsample(clean[stage-1],current)
        end_point=current if stage==stages-1 else end*noise+(1-end)*current
        indices=(_density(target_latents.shape[0],device)*train_steps).long().clamp(0,train_steps-1)
        timesteps=scheduler.timesteps_per_stage[stage][indices].to(device=device)
        sigmas=scheduler.sigmas_per_stage[stage][indices].to(device=device,dtype=start_point.dtype)
        shape=(slice(None),)+(None,)*(start_point.ndim-1); sigma=sigmas[shape]
        noisy=sigma*start_point+(1-sigma)*end_point
        items.append({'stage_id':stage,'noisy_latents':noisy,'timesteps':timesteps,'sigmas':sigmas,'target':start_point-end_point,'start_point':start_point,'end_point':end_point})
    return items
