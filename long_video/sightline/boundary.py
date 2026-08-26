"""Helios-native three-stage shared-boundary flow constraints."""
from __future__ import annotations
from dataclasses import dataclass
import torch
import torch.nn.functional as F


def _resize(value: torch.Tensor, spatial: tuple[int, int]) -> torch.Tensor:
    if tuple(value.shape[-2:]) == tuple(spatial): return value
    batch,channels,frames=value.shape[:3]
    flat=value.permute(0,2,1,3,4).reshape(batch*frames,channels,*value.shape[-2:])
    flat=F.interpolate(flat.float(),size=spatial,mode='bilinear',align_corners=False).to(value.dtype)
    return flat.reshape(batch,frames,channels,*spatial).permute(0,2,1,3,4)


@dataclass(frozen=True)
class BoundaryStage:
    index: int
    start: torch.Tensor
    end: torch.Tensor

    def at(self, coefficient) -> torch.Tensor:
        coefficient=torch.as_tensor(coefficient,device=self.start.device,dtype=self.start.dtype)
        while coefficient.ndim<self.start.ndim: coefficient=coefficient.unsqueeze(-1)
        return coefficient*self.start+(1.0-coefficient)*self.end

    @property
    def target(self) -> torch.Tensor: return self.start-self.end


def training_boundary_stages(items: list[dict], clean_boundary: torch.Tensor) -> tuple[BoundaryStage,...]:
    """Apply Helios' native per-stage flow to a replaced temporal0 clean target."""
    stages=[]; shapes=[tuple(item['noisy_latents'].shape[-2:]) for item in items]
    # Match _pyramid_latents exactly: full -> half -> quarter, sequentially,
    # and then consume the list coarse -> mid -> full.
    current=_resize(clean_boundary,shapes[-1]); descending=[current]
    for spatial in reversed(shapes[:-1]):
        current=_resize(current,spatial); descending.append(current)
    clean_pyramid=list(reversed(descending))
    for index,item in enumerate(items):
        spatial=tuple(item['noisy_latents'].shape[-2:]); clean=clean_pyramid[index]
        noise=item['noise'][:,:,:1]
        start_sigma=float(item['stage_start_sigma']); end_sigma=float(item['stage_end_sigma'])
        if index==0:
            start=noise
        else:
            # This is the same definition used for temporal1..8 by
            # exact_flow_matching_items: upsample the previous clean pyramid,
            # never the previous stage's noisy end point.
            previous_clean=_resize(clean_pyramid[index-1],spatial)
            start=start_sigma*noise+(1.0-start_sigma)*previous_clean
        end=clean if index==len(items)-1 else end_sigma*noise+(1.0-end_sigma)*clean
        stages.append(BoundaryStage(index,start,end))
    return tuple(stages)


def constrain_flow_items(items: list[dict], clean_boundary: torch.Tensor) -> list[dict]:
    constrained=[]
    for item,stage in zip(items,training_boundary_stages(items,clean_boundary)):
        noisy=item['noisy_latents'].clone(); target=item['target'].clone()
        noisy[:,:,:1]=stage.at(item['sigmas']); target[:,:,:1]=stage.target
        constrained.append({**item,'noisy_latents':noisy,'target':target,'boundary_stage':stage})
    return constrained


def _scheduler_coefficient(scheduler,timestep,*,after_step:bool) -> torch.Tensor:
    times=scheduler.timesteps; raw=torch.as_tensor(timestep,device=times.device); needle=raw.reshape(-1)[0]
    matches=torch.nonzero(torch.isclose(times.float(),needle.float()),as_tuple=False).flatten()
    # Helios deliberately casts the timestep passed to the Transformer to
    # int64, while scheduler.step receives the original floating-point value.
    # Resolve the pre-hook coordinate through that same legal cast instead of
    # requiring equality with the scheduler's unrounded local timestep.
    if not len(matches) and not torch.is_floating_point(raw):
        matches=torch.nonzero(times.to(dtype=raw.dtype)==needle,as_tuple=False).flatten()
    if not len(matches): raise RuntimeError(f'scheduler timestep {float(needle)} has no local coefficient')
    if len(matches)!=1: raise RuntimeError(f'scheduler timestep {float(needle)} is ambiguous in the local schedule')
    index=int(matches[0])+int(after_step)
    return scheduler.sigmas[min(index,len(scheduler.sigmas)-1)]


class SamplingBoundaryFlow:
    """Consume Helios' observed starts and its actual block-noise samples."""
    def __init__(self,scheduler,clean_boundary:torch.Tensor,full_spatial:tuple[int,int],stage_count:int):
        self.scheduler=scheduler; self.clean=clean_boundary; self.stage_count=stage_count
        self.spatial_to_stage={(full_spatial[0]//(2**(stage_count-1-index)),full_spatial[1]//(2**(stage_count-1-index))):index for index in range(stage_count)}
        full=_resize(clean_boundary,full_spatial); descending=[full]; current=full
        for divisor in (2**index for index in range(1,stage_count)):
            current=_resize(current,(full_spatial[0]//divisor,full_spatial[1]//divisor)); descending.append(current)
        self.clean_pyramid=tuple(reversed(descending))
        self.stages={}; self.native_noise={}

    def record_block_noise(self,noise:torch.Tensor):
        """Record the exact noise Helios sampled for stage1, then stage2."""
        index=len(self.native_noise)+1
        if not 1<=index<self.stage_count: raise RuntimeError('unexpected extra Helios block-noise sample')
        self.native_noise[index]=noise[:,:,:1].detach().clone()

    def stage_for(self,hidden:torch.Tensor) -> BoundaryStage:
        spatial=tuple(hidden.shape[-2:])
        if spatial not in self.spatial_to_stage: raise RuntimeError(f'unexpected Helios boundary stage grid {spatial}')
        index=self.spatial_to_stage[spatial]
        if index in self.stages: return self.stages[index]
        if index and index-1 not in self.stages: raise RuntimeError('Helios boundary stages were entered out of order')
        start=hidden[:,:,:1].detach().clone(); clean=self.clean_pyramid[index]
        # start is the exact native stage start, including the pinned nearest
        # upsample and alpha/beta block-noise transition for stage1/2.
        noise=start if index==0 else self.native_noise.get(index)
        if noise is None: raise RuntimeError(f'Helios stage{index} block noise was not captured before Transformer entry')
        end=clean if index==self.stage_count-1 else (
            float(self.scheduler.end_sigmas[index])*noise+(1.0-float(self.scheduler.end_sigmas[index]))*clean)
        stage=BoundaryStage(index,start,end); self.stages[index]=stage; return stage


def stage2_sample_with_boundary(pipe,*,clean_boundary:torch.Tensor|None,**kwargs):
    """Constrain temporal0 along native Helios stages without replacing transitions."""
    if clean_boundary is None: return pipe.stage2_sample(**kwargs)
    initial=kwargs['latents']; stage_count=int(kwargs.get('pyramid_num_stages',3))
    flow=SamplingBoundaryFlow(pipe.scheduler,clean_boundary,tuple(initial.shape[-2:]),stage_count)
    user_callback=kwargs.pop('callback_on_step_end',None)
    original_block_noise=getattr(pipe,'sample_block_noise',None)
    if not callable(original_block_noise): raise RuntimeError('pinned Helios stage2_sample must expose sample_block_noise')

    def capture_block_noise(*args,**block_kwargs):
        noise=original_block_noise(*args,**block_kwargs)
        flow.record_block_noise(noise)
        return noise

    def pre_hook(_module,args,call_kwargs):
        hidden=call_kwargs.get('hidden_states')
        if hidden is None: return args,call_kwargs
        stage=flow.stage_for(hidden); coefficient=_scheduler_coefficient(pipe.scheduler,call_kwargs['timestep'],after_step=False)
        hidden=hidden.clone(); hidden[:,:,:1]=stage.at(coefficient)
        call_kwargs=dict(call_kwargs); call_kwargs['hidden_states']=hidden
        return args,call_kwargs

    def callback(owner,step,timestep,callback_kwargs):
        if user_callback is not None: callback_kwargs=user_callback(owner,step,timestep,callback_kwargs)
        latents=callback_kwargs['latents'].clone(); stage=flow.stage_for(latents)
        coefficient=_scheduler_coefficient(pipe.scheduler,timestep,after_step=True)
        latents[:,:,:1]=stage.at(coefficient)
        return {**callback_kwargs,'latents':latents}

    handle=pipe.transformer.register_forward_pre_hook(pre_hook,with_kwargs=True)
    pipe.sample_block_noise=capture_block_noise
    try:
        result=pipe.stage2_sample(callback_on_step_end=callback,**kwargs)
    finally:
        pipe.sample_block_noise=original_block_noise
        handle.remove()
    if len(flow.stages)!=stage_count: raise RuntimeError('Helios did not execute every configured boundary stage')
    result=result.clone(); result[:,:,:1]=_resize(clean_boundary,tuple(result.shape[-2:]))
    return result
