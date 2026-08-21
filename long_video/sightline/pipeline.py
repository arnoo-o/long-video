"""Formal geometry-free Sightline runtime around native Helios."""
from __future__ import annotations
from dataclasses import dataclass
from types import SimpleNamespace
import torch
from .history import CameraHistoryState,NativeHistoryState,native_helios_indices
from .memory import LayerKVMemoryBank
from .rays import chunk_cameras, temporal_group_cameras

def prepare_source_condition(helios, image, *, height, width, device=None):
    """Use the pinned Helios image-conditioning path for training and inference."""
    device=device or helios._execution_device
    image_tensor=helios.video_processor.preprocess(image,height=height,width=width)
    vae=helios.vae
    mean=torch.tensor(vae.config.latents_mean,device=device,dtype=vae.dtype).view(1,vae.config.z_dim,1,1,1)
    std=(1.0/torch.tensor(vae.config.latents_std,device=device,dtype=vae.dtype)).view(1,vae.config.z_dim,1,1,1)
    with torch.no_grad():
        source,fake=helios.prepare_image_latents(image_tensor,latents_mean=mean,latents_std=std,num_latent_frames_per_chunk=9,dtype=torch.float32,device=device)
    return source.detach(),fake.detach(),mean,std

@dataclass
class SightlineRuntimeContext:
    chunk_index:int=0; stage:int=0; sigma:float=0.0; memory_write_allowed:bool=False

class SightlinePipeline:
    def __init__(self, helios_pipeline, *, config, conditioner=None, ray_provider=None):
        self.helios=helios_pipeline; self.config=config; self.conditioner=conditioner; self.ray_provider=ray_provider
        inner_dim=int(conditioner.q_proj.out_features) if conditioner is not None else None
        self.camera_history=CameraHistoryState(); self.memory=LayerKVMemoryBank(getattr(config,'memory_layers',()),config.memory_budget,config.memory_pool,hidden_dim=inner_dim)
        self.runtime=SightlineRuntimeContext(); self._source_initialized=False; self._trajectory_c2w=None; self._trajectory_K=None; self._source_camera=None; self._source_intrinsics=None; self._active_chunk=0; self.history_state=None

    @staticmethod
    def append_stride32_latents(accumulated, chunk):
        if chunk.ndim!=5 or chunk.shape[2]!=9: raise ValueError('each Sightline chunk must contain 9 latent frames')
        return chunk if accumulated is None else torch.cat((accumulated,chunk[:,:,1:]),dim=2)

    @staticmethod
    def resolve_pyramid_steps(configured, override=None):
        steps=tuple(int(x) for x in configured) if override is None else (int(override),)*3
        if len(steps)!=3 or any(step<1 for step in steps): raise ValueError('Sightline requires three positive pyramid step counts')
        return steps

    def reset_sequence(self):
        self._active_chunk=0; self.runtime=SightlineRuntimeContext(); self.camera_history=CameraHistoryState(); self.history_state=None; self._pending_camera_chunk=None
        self.memory.reset()
        for processor in getattr(self.helios.transformer,'_sightline_processors',{}).values():
            processor.last_q=processor.last_k=processor.last_hidden_states=processor.last_key_identities=None
            processor.last_attention_bias=None
            processor.last_attention_meta={}; processor.last_current_length=None

    def _stage_shapes(self, latents):
        patch=tuple(int(x) for x in self.helios.transformer.config.patch_size)
        stages=len(self.config.pyramid_steps); h,w=int(latents.shape[-2]),int(latents.shape[-1]); t=int(latents.shape[2]); shapes=[]
        h//=2**(stages-1); w//=2**(stages-1)
        for _ in range(stages):
            if any(x<=0 for x in (t//patch[0],h//patch[1],w//patch[2])): raise RuntimeError('invalid Helios latent/patch shape')
            shapes.append((t//patch[0],h//patch[1],w//patch[2])); h*=2; w*=2
        return tuple(shapes)

    def _prepare_chunk(self, chunk_index, latents, attention_kwargs, *, history_global_coverages, history_validity=None, history_token_shapes=None, history_latent_hw=None):
        if self._trajectory_c2w is None or self.ray_provider is None: raise RuntimeError('Sightline trajectory/provider is not bound')
        cameras,K=chunk_cameras(self._trajectory_c2w,self._trajectory_K,chunk_index); reps,repK=temporal_group_cameras(cameras,K)
        frame_ids=(torch.arange(9,device=cameras.device)*4 + chunk_index*32).tolist()
        source_camera=self._source_camera
        source_K=self._source_intrinsics
        def representative(covered):
            identity=covered[-1] if covered else None
            return self.camera_history.camera_for(identity,source_camera,source_K)
        long_slots=[representative(covered) for covered in history_global_coverages['long']]
        mid_slots=[representative(covered) for covered in history_global_coverages['mid']]
        short_slots=[representative(covered) for covered in history_global_coverages['short']]
        history_groups={
            'long':(torch.stack([c for c,_ in long_slots],1),torch.stack([k for _,k in long_slots],1)),
            'mid':(torch.stack([c for c,_ in mid_slots],1),torch.stack([k for _,k in mid_slots],1)),
            'short':(torch.stack([c for c,_ in short_slots],1),torch.stack([k for _,k in short_slots],1)),
        }
        shapes=self._stage_shapes(latents)
        if history_token_shapes is None:
            lh,lw=history_latent_hw or (int(latents.shape[-2]),int(latents.shape[-1]))
            history_token_shapes={'long':(4,(lh+7)//8,(lw+7)//8),'mid':(1,(lh+3)//4,(lw+3)//4),'short':(2,(lh+1)//2,(lw+1)//2)}
        if history_validity is None:
            history_validity={'long':tuple(bool(c) for c in history_global_coverages.get('long',())), 'mid':tuple(bool(c) for c in history_global_coverages.get('mid',())), 'short':tuple(True for _ in history_global_coverages.get('short',()))}
        self.ray_provider.set_context(chunk_index=chunk_index,c2w=cameras,intrinsics=K,latent_cameras=reps,history_groups=history_groups,history_token_shapes=history_token_shapes,history_global_coverages=history_global_coverages,history_validity=history_validity,stage_shapes=shapes,token_shape=shapes[0])
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

    @torch.inference_mode()
    def generate(self, *, image, prompt, negative_prompt, height, width, num_frames, steps=None, c2w, intrinsics, attention_kwargs=None):
        self.reset_sequence()
        if not self._source_initialized: self._source_initialized=True
        if num_frames<33 or (num_frames-1)%32: raise ValueError('Sightline inference requires 1+32*N frames')
        if c2w.ndim!=4 or intrinsics.ndim!=4 or c2w.shape[:2]!=intrinsics.shape[:2]: raise ValueError('c2w/K must remain [B,F,...]')
        self._trajectory_c2w=c2w; self._trajectory_K=intrinsics; self._source_camera=c2w[:,0]; self._source_intrinsics=intrinsics[:,0]
        chunks=(num_frames-1)//32; device=self.helios._execution_device; transformer_dtype=self.helios.transformer.dtype; runtime_kwargs=dict(attention_kwargs or {})
        for module in (self.helios.transformer,self.helios.text_encoder,self.helios.vae): module.eval()
        if self.conditioner is not None: self.conditioner.eval()
        self.helios._guidance_scale=1.0; self.helios._attention_kwargs=runtime_kwargs; self.helios._current_timestep=None; self.helios._interrupt=False
        prompt_embeds,negative_embeds=self.helios.encode_prompt(prompt=prompt,negative_prompt=negative_prompt,do_classifier_free_guidance=False,num_videos_per_prompt=1,max_sequence_length=512,device=device)
        prompt_embeds=prompt_embeds.to(transformer_dtype); negative_embeds=None
        vae=self.helios.vae; source,fake,mean,std=prepare_source_condition(self.helios,image,height=height,width=width,device=device)
        self.history_state=NativeHistoryState(source,fake); accumulated=None; stage_steps=list(self.resolve_pyramid_steps(self.config.pyramid_steps,steps))
        class Progress:
            def update(self): pass
        for chunk in range(chunks):
            history=self.history_state.groups(); coverage=self.history_state.coverage(); validity=self.history_state.validity()
            latent=self.helios.prepare_latents(source.shape[0],self.helios.transformer.config.in_channels,height,width,33,dtype=torch.float32,device=device)
            current_ids=native_helios_indices(device,source.shape[0])['current']
            self._prepare_chunk(chunk,latent,runtime_kwargs,history_global_coverages=coverage,history_validity=validity,history_latent_hw=source.shape[-2:])
            latent=self.helios.stage2_sample(latents=latent,pyramid_num_stages=3,pyramid_num_inference_steps_list=stage_steps,prompt_embeds=prompt_embeds,negative_prompt_embeds=negative_embeds,guidance_scale=1.0,indices_hidden_states=current_ids,indices_latents_history_short=history['short'][1],indices_latents_history_mid=history['mid'][1],indices_latents_history_long=history['long'][1],latents_history_short=history['short'][0],latents_history_mid=history['mid'][0],latents_history_long=history['long'][0],attention_kwargs=runtime_kwargs,device=device,transformer_dtype=transformer_dtype,progress_bar=Progress())
            if chunk==0: latent[:,:,0:1]=source.to(latent)
            accumulated=self.append_stride32_latents(accumulated,latent)
            self.history_state.append_chunk(latent,chunk)
            self._finalize_chunk(chunk); self._active_chunk=chunk+1
        if accumulated is None or accumulated.shape[2]!=1+8*chunks: raise RuntimeError('stride-32 latent assembly failed')
        decoded=vae.decode(accumulated.to(vae.dtype)/std+mean,return_dict=False)[0]
        expected_frames=1+32*chunks; decoded=decoded[:,:,:expected_frames]
        frames=self.helios.video_processor.postprocess_video(decoded,output_type='np')
        self.helios._current_timestep=None
        return SimpleNamespace(frames=frames,latents=accumulated)

    @staticmethod
    def assert_geometry_free_imports():
        import sys
        forbidden=('warp_as_history','long_video.wah','long_video.geometry.point_renderer','long_video.initialization.recal3r','pi3')
        bad=[x for x in sys.modules if any(x==f or x.startswith(f+'.') for f in forbidden)]
        if bad: raise RuntimeError('geometry/legacy modules loaded: '+','.join(bad))
