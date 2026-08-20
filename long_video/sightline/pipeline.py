"""Formal geometry-free Sightline runtime around native Helios."""
from __future__ import annotations
from dataclasses import dataclass
from types import MethodType
import torch
from .history import CameraHistoryState
from .memory import LayerKVMemoryBank
from .rays import chunk_cameras, temporal_group_cameras

@dataclass
class SightlineRuntimeContext:
    chunk_index:int=0; stage:int=0; sigma:float=0.0; memory_write_allowed:bool=False

class SightlinePipeline:
    def __init__(self, helios_pipeline, *, config, conditioner=None, ray_provider=None):
        self.helios=helios_pipeline; self.config=config; self.conditioner=conditioner; self.ray_provider=ray_provider
        inner_dim=int(conditioner.q_proj[-1].out_features) if conditioner is not None else None
        self.camera_history=CameraHistoryState(); self.memory=LayerKVMemoryBank(getattr(config,'memory_layers',()),config.memory_budget,config.memory_pool,hidden_dim=inner_dim)
        self.runtime=SightlineRuntimeContext(); self._source_initialized=False; self._trajectory_c2w=None; self._trajectory_K=None; self._source_camera=None; self._source_intrinsics=None; self._active_chunk=0; self._hooks_installed=False

    def _stage_shapes(self, latents):
        patch=tuple(int(x) for x in self.helios.transformer.config.patch_size)
        stages=len(self.config.pyramid_steps); h,w=int(latents.shape[-2]),int(latents.shape[-1]); t=int(latents.shape[2]); shapes=[]
        h//=2**(stages-1); w//=2**(stages-1)
        for _ in range(stages):
            if any(x<=0 for x in (t//patch[0],h//patch[1],w//patch[2])): raise RuntimeError('invalid Helios latent/patch shape')
            shapes.append((t//patch[0],h//patch[1],w//patch[2])); h*=2; w*=2
        return tuple(shapes)

    def _prepare_chunk(self, chunk_index, latents, attention_kwargs):
        if self._trajectory_c2w is None or self.ray_provider is None: raise RuntimeError('Sightline trajectory/provider is not bound')
        cameras,K=chunk_cameras(self._trajectory_c2w,self._trajectory_K,chunk_index); reps,repK=temporal_group_cameras(cameras,K)
        frame_ids=(torch.arange(9,device=cameras.device)*4 + chunk_index*32).tolist()
        source_camera=self._source_camera
        source_K=self._source_intrinsics
        history_slots=self.camera_history.slots(source_camera,source_K)
        history_cameras=torch.stack([source_camera]+[camera for camera,_ in history_slots],1)
        history_K=torch.stack([source_K]+[K for _,K in history_slots],1)
        shapes=self._stage_shapes(latents)
        self.ray_provider.set_context(chunk_index=chunk_index,c2w=cameras,intrinsics=K,latent_cameras=reps,history_cameras=history_cameras,history_intrinsics=history_K,stage_shapes=shapes,token_shape=shapes[0])
        self._pending_camera_chunk=(list(reps.unbind(1)),frame_ids,list(repK.unbind(1)))
        attention_kwargs['current_chunk']=chunk_index
        attention_kwargs['sightline_stage_shapes']=shapes

    def _finalize_chunk(self, chunk_index):
        if self.ray_provider is None:
            pending=getattr(self,'_pending_camera_chunk',None)
            if pending is not None: self.camera_history.append_chunk(*pending); self._pending_camera_chunk=None
            return
        for processor in getattr(self.helios.transformer,'_sightline_processors',{}).values():
            if processor.last_hidden_states is None or processor.memory is None: continue
            hidden=processor.last_hidden_states[:, -processor.last_current_length:]
            shape=next((s for s in (self.ray_provider.context.get('stage_shapes') or ()) if s[0]*s[1]*s[2]==processor.last_current_length),None)
            if shape is None: continue
            rays=self.ray_provider.current_rays(shape)
            processor.memory.capture(hidden,rays,chunk_index,grid_shape=shape,ray_recompute=self.ray_provider.current_rays)
            processor.last_hidden_states=None
        pending=getattr(self,'_pending_camera_chunk',None)
        if pending is not None:
            self.camera_history.append_chunk(*pending)
            self._pending_camera_chunk=None

    def _install_chunk_hooks(self):
        if self._hooks_installed: return
        for name in ('stage1_sample','stage2_sample'):
            original=getattr(self.helios,name)
            def wrapped(instance,*args,_original=original,_name=name,**kwargs):
                latents=kwargs.get('latents',args[0] if args else None)
                attention_kwargs=kwargs.get('attention_kwargs') or {}
                kwargs['attention_kwargs']=attention_kwargs; self._prepare_chunk(self._active_chunk,latents,attention_kwargs)
                result=_original(*args,**kwargs); self._finalize_chunk(self._active_chunk); self._active_chunk+=1
                return result
            setattr(self.helios,name,MethodType(wrapped,self.helios))
        self._hooks_installed=True

    def generate(self, *, image, prompt, negative_prompt, height, width, num_frames, steps, c2w, intrinsics, attention_kwargs=None):
        if not self._source_initialized: self._source_initialized=True
        if num_frames<33 or (num_frames-1)%32: raise ValueError('Sightline inference requires 1+32*N frames')
        if c2w.ndim!=4 or intrinsics.ndim!=4 or c2w.shape[:2]!=intrinsics.shape[:2]: raise ValueError('c2w/K must remain [B,F,...]')
        self._trajectory_c2w=c2w; self._trajectory_K=intrinsics; self._source_camera=c2w[:,0]; self._source_intrinsics=intrinsics[:,0]; self._install_chunk_hooks()
        kwargs=dict(prompt=prompt,negative_prompt=negative_prompt,image=image,height=height,width=width,num_frames=num_frames,num_inference_steps=steps,history_sizes=list(self.config.history_sizes),num_latent_frames_per_chunk=9,is_enable_stage2=True,pyramid_num_inference_steps_list=list(self.config.pyramid_steps),attention_kwargs=attention_kwargs or {},output_type='np')
        return self.helios(**kwargs)

    @staticmethod
    def assert_geometry_free_imports():
        import sys
        forbidden=('warp_as_history','long_video.wah','long_video.geometry.point_renderer','long_video.initialization.recal3r','pi3')
        bad=[x for x in sys.modules if any(x==f or x.startswith(f+'.') for f in forbidden)]
        if bad: raise RuntimeError('geometry/legacy modules loaded: '+','.join(bad))
