"""Original WAH plus Stage2-only renderer RGB consistency clamp."""
from __future__ import annotations
from typing import Any
import torch
try:
    from warp_as_history import WarpAsHistoryPipeline
except ImportError:
    WarpAsHistoryPipeline = object

PYRAMID_INFERENCE_STEPS = (2, 2, 4)
STAGE2_CLAMP_STATES = (1, 1, 0, 0)

def clamp_enabled(stage_id: int, step_id: int) -> bool:
    return int(stage_id) == 2 and 0 <= int(step_id) < 2

def composite_renderer_rgb(model_rgb, warp_rgb, visibility):
    if tuple(model_rgb.shape) != tuple(warp_rgb.shape): raise ValueError("model and warp RGB shapes must match")
    expected=(model_rgb.shape[0],1,*model_rgb.shape[2:])
    if tuple(visibility.shape)!=expected: raise ValueError(f"visibility must be {expected}, got {tuple(visibility.shape)}")
    mask=visibility.to(model_rgb.device,torch.float32)
    if not bool(((mask==0)|(mask==1)).all()): raise ValueError("renderer visibility must be binary")
    return (mask*warp_rgb.float()+(1-mask)*model_rgb.float()).to(model_rgb.dtype)

def posterior_mode_or_mean(posterior):
    posterior=getattr(posterior,"latent_dist",posterior); mode=getattr(posterior,"mode",None)
    if callable(mode):
        value=mode()
        if value is not None: return value
    value=getattr(posterior,"mean",None)
    if value is None: raise TypeError("VAE posterior must expose mode() or mean")
    return value

class RGBClampWarpAsHistoryPipeline(WarpAsHistoryPipeline):
    """Wrap Helios scheduler updates; no model parameters or attention changes."""
    def set_rgb_clamp_context(self,warp_rgb:Any,visibility:Any,*,height:int,width:int):
        device=self._wah_execution_device(); rgb=self._coerce_warp_video_tensor(warp_rgb,height=height,width=width,device=device).to(device=device,dtype=self.vae.dtype)
        mask=torch.as_tensor(visibility,device=device)
        if mask.ndim==3: mask=mask[None,None]
        elif mask.ndim==4: mask=mask[:,None]
        expected=(rgb.shape[0],1,*rgb.shape[2:])
        if tuple(mask.shape)!=expected: raise ValueError(f"renderer visibility must be {expected}, got {tuple(mask.shape)}")
        self._rgb_clamp_context=(rgb.detach(),(mask>0).detach()); self._rgb_clamp_diagnostics=[]
    def clear_rgb_clamp_context(self): self._rgb_clamp_context=None
    def _latent_stats(self,device):
        vae_dtype=next(self.vae.parameters()).dtype
        mean=torch.tensor(self.vae.config.latents_mean,device=device,dtype=vae_dtype).view(1,-1,1,1,1)
        std=1/torch.tensor(self.vae.config.latents_std,device=device,dtype=vae_dtype).view(1,-1,1,1,1)
        return mean,std
    def stage2_sample(self,*args,**kwargs):
        context=getattr(self,"_rgb_clamp_context",None)
        if context is None: return super().stage2_sample(*args,**kwargs)
        scheduler,original_step=self.scheduler,self.scheduler.step; stage_id=-1; stage_start=None
        def clamped_step(model_output,timestep,sample,*step_args,**step_kwargs):
            nonlocal stage_id,stage_start
            step_id=int(step_kwargs.get("cur_sampling_step",0))
            if step_id==0: stage_id+=1; stage_start=sample.detach().clone()
            result=original_step(model_output,timestep,sample,*step_args,**step_kwargs); enabled=clamp_enabled(stage_id,step_id)
            self._rgb_clamp_diagnostics.append({"stage_id":stage_id,"step_id":step_id,"rgb_clamp":int(enabled)})
            if not enabled: return result
            z_raw=result[0]; sigmas=step_kwargs.get("dmd_sigmas",getattr(scheduler,"sigmas",None)); dmd_timesteps=step_kwargs.get("dmd_timesteps",getattr(scheduler,"timesteps",None)); all_timesteps=step_kwargs.get("all_timesteps")
            if sigmas is None or dmd_timesteps is None or all_timesteps is None: raise RuntimeError("Helios scheduler coordinates are required")
            value=torch.as_tensor(timestep,device=model_output.device).item(); batch=torch.full((model_output.shape[0],),value,dtype=torch.long,device=model_output.device)
            clean=scheduler.convert_flow_pred_to_x0(flow_pred=model_output,xt=sample,timestep=batch,sigmas=sigmas,timesteps=dmd_timesteps)
            mean,std=self._latent_stats(z_raw.device); model_rgb=self.vae.decode(clean.to(self.vae.dtype)/std+mean,return_dict=False)[0]
            mixed=composite_renderer_rgb(model_rgb,context[0],context[1]); mixed_clean=(posterior_mode_or_mean(self.vae.encode(mixed.to(self.vae.dtype)))-mean)*std
            next_timestep=torch.full((mixed_clean.shape[0],),all_timesteps[step_id+1],dtype=torch.long,device=mixed_clean.device)
            z_next=scheduler.add_noise(mixed_clean.to(z_raw.dtype),stage_start.to(z_raw.dtype),next_timestep,sigmas=sigmas,timesteps=dmd_timesteps)
            return (z_next,*result[1:])
        scheduler.step=clamped_step
        try:
            sampled=super().stage2_sample(*args,**kwargs); expected=[(s,i) for s,n in enumerate(PYRAMID_INFERENCE_STEPS) for i in range(n)]; actual=[(x["stage_id"],x["step_id"]) for x in self._rgb_clamp_diagnostics]
            if actual!=expected: raise RuntimeError(f"unexpected pyramid scheduler steps: {actual}")
            if [x["rgb_clamp"] for x in self._rgb_clamp_diagnostics if x["stage_id"]==2]!=list(STAGE2_CLAMP_STATES): raise RuntimeError("Stage2 RGB clamp schedule must be [1,1,0,0]")
            return sampled
        finally: scheduler.step=original_step
