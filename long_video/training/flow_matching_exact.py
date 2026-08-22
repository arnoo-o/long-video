"""Local copy of pinned Helios exact flow-matching sampling semantics.

This module intentionally does not import the legacy WAH package.  The formulas
match the pinned scheduler's ``start_sigmas/end_sigmas`` and per-stage arrays.
"""
from __future__ import annotations
import torch
import torch.nn.functional as F

def _density(batch, device):
    return torch.rand((batch,),device=device)

def _resize_spatial(latents, height, width, scale=1.0):
    batch,channels,frames,_,_=latents.shape
    flat=latents.permute(0,2,1,3,4).reshape(batch*frames,channels,latents.shape[-2],latents.shape[-1])
    flat=F.interpolate(flat,size=(height,width),mode='bilinear')*float(scale)
    return flat.reshape(batch,frames,channels,height,width).permute(0,2,1,3,4)

def _pyramid_latents(target, stages):
    result=[target.float()]; current=target.float(); height,width=target.shape[-2:]
    for _ in range(stages-1):
        height//=2; width//=2; current=_resize_spatial(current,height,width); result.append(current)
    return list(reversed(result))

def _pyramid_noise(reference, stages, *, generator=None):
    current=torch.randn(reference.shape,device=reference.device,dtype=reference.dtype,generator=generator); result=[current]
    height,width=reference.shape[-2:]
    for _ in range(stages-1):
        height//=2; width//=2; current=_resize_spatial(current,height,width,scale=2.0); result.append(current)
    return list(reversed(result))

def _upsample(source,target): return _resize_spatial(source,target.shape[-2],target.shape[-1])

def _apply_schedule_shift(sigmas,reference,config):
    seq_len=(reference.shape[-1]*reference.shape[-2]*reference.shape[-3])//4
    base_len=float(config.get('base_image_seq_len',256)); max_len=float(config.get('max_image_seq_len',4096))
    base_shift=float(config.get('base_shift',.5)); max_shift=float(config.get('max_shift',1.15))
    mu=seq_len*((max_shift-base_shift)/(max_len-base_len))+(base_shift-(max_shift-base_shift)/(max_len-base_len)*base_len)
    if config.get('time_shift_type','linear')=='exponential': mu=torch.exp(torch.tensor(min(mu,torch.log(torch.tensor(7.)).item()),device=sigmas.device,dtype=sigmas.dtype))
    return (sigmas*mu)/(1+(mu-1)*sigmas)

def exact_flow_matching_items(pipe, target_latents, *, stage_steps=(2,2,2), device=None, generator=None,
                              sigma_range=None):
    device=device or target_latents.device; stages=len(stage_steps)
    scheduler=pipe.scheduler
    if int(scheduler.config.get('stages',stages)) != stages: raise ValueError('scheduler stage count mismatch')
    clean=_pyramid_latents(target_latents,stages); noise=_pyramid_noise(clean[-1],stages,generator=generator)
    items=[]; train_steps=int(scheduler.config.get('num_train_timesteps',1000))
    for stage in range(stages):
        current=clean[stage]; start=float(scheduler.start_sigmas[stage]); end=float(scheduler.end_sigmas[stage])
        start_point=noise[stage] if stage==0 else start*noise[stage]+(1-start)*_upsample(clean[stage-1],current)
        end_point=current if stage==stages-1 else end*noise[stage]+(1-end)*current
        if sigma_range is None:
            indices=(_density(target_latents.shape[0],device)*train_steps).long().clamp(0,train_steps-1)
        else:
            low,high=(float(value) for value in sigma_range)
            stage_sigmas=torch.as_tensor(scheduler.sigmas_per_stage[stage]).flatten()
            eligible=torch.nonzero((stage_sigmas>low)&(stage_sigmas<=high),as_tuple=False).flatten()
            if not len(eligible): raise RuntimeError(f'stage {stage} has no scheduler sigma in ({low}, {high}]')
            picks=torch.randint(len(eligible),(target_latents.shape[0],),generator=generator,device='cpu')
            indices=eligible.index_select(0,picks)
        cpu_indices=indices.detach().cpu()
        timesteps=scheduler.timesteps_per_stage[stage][cpu_indices].to(device=device)
        sigmas=scheduler.sigmas_per_stage[stage][cpu_indices].to(device=device,dtype=start_point.dtype)
        while sigmas.ndim<start_point.ndim: sigmas=sigmas.unsqueeze(-1)
        sigma=sigmas
        noisy=sigma*start_point+(1-sigma)*end_point
        for name,tensor in {'noisy':noisy,'target':start_point-end_point,'start':start_point,'end':end_point}.items():
            if tensor.shape!=current.shape: raise RuntimeError(f'stage {stage} {name} shape mismatch: {tensor.shape} vs {current.shape}')
        items.append({'stage_id':stage,'noisy_latents':noisy,'timesteps':timesteps,'sigmas':sigmas,'target':start_point-end_point,'start_point':start_point,'end_point':end_point,'noise':noise[stage],'use_dynamic_shifting':False})
    return items
