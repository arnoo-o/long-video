"""Formal geometry-free Sightline runtime around native Helios."""
from __future__ import annotations
from dataclasses import dataclass
from types import MethodType
from types import SimpleNamespace
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

    @staticmethod
    def append_stride32_latents(accumulated, chunk):
        if chunk.ndim!=5 or chunk.shape[2]!=9: raise ValueError('each Sightline chunk must contain 9 latent frames')
        return chunk if accumulated is None else torch.cat((accumulated,chunk[:,:,1:]),dim=2)

    def reset_sequence(self):
        self._active_chunk=0; self.runtime=SightlineRuntimeContext(); self.camera_history=CameraHistoryState(); self._pending_camera_chunk=None
        self.memory.reset()
        for processor in getattr(self.helios.transformer,'_sightline_processors',{}).values():
            processor.last_q=processor.last_k=processor.last_hidden_states=processor.last_key_identities=None
            processor.last_attention_meta={}; processor.last_current_length=None

    def _stage_shapes(self, latents):
        patch=tuple(int(x) for x in self.helios.transformer.config.patch_size)
        stages=len(self.config.pyramid_steps); h,w=int(latents.shape[-2]),int(latents.shape[-1]); t=int(latents.shape[2]); shapes=[]
        h//=2**(stages-1); w//=2**(stages-1)
        for _ in range(stages):
            if any(x<=0 for x in (t//patch[0],h//patch[1],w//patch[2])): raise RuntimeError('invalid Helios latent/patch shape')
            shapes.append((t//patch[0],h//patch[1],w//patch[2])); h*=2; w*=2
        return tuple(shapes)

    def _prepare_chunk(self, chunk_index, latents, attention_kwargs, history_token_shapes=None, history_latent_hw=None):
        if self._trajectory_c2w is None or self.ray_provider is None: raise RuntimeError('Sightline trajectory/provider is not bound')
        cameras,K=chunk_cameras(self._trajectory_c2w,self._trajectory_K,chunk_index); reps,repK=temporal_group_cameras(cameras,K)
        frame_ids=(torch.arange(9,device=cameras.device)*4 + chunk_index*32).tolist()
        source_camera=self._source_camera
        source_K=self._source_intrinsics
        history_slots=self.camera_history.slots(source_camera,source_K)
        long_all=history_slots[:16]; mid_all=history_slots[16:18]
        long_slots=[long_all[i] for i in (3,7,11,15)]; mid_slots=[mid_all[-1]]; short_slots=[(source_camera,source_K),history_slots[18]]
        history_groups={
            'long':(torch.stack([c for c,_ in long_slots],1),torch.stack([k for _,k in long_slots],1)),
            'mid':(torch.stack([c for c,_ in mid_slots],1),torch.stack([k for _,k in mid_slots],1)),
            'short':(torch.stack([c for c,_ in short_slots],1),torch.stack([k for _,k in short_slots],1)),
        }
        latent_ids=self.camera_history.slot_latent_ids(); history_latent_ids={'long':tuple(latent_ids[i] for i in (3,7,11,15)),'mid':(latent_ids[17],),'short':(0,latent_ids[18])}
        shapes=self._stage_shapes(latents)
        if history_token_shapes is None:
            lh,lw=history_latent_hw or (int(latents.shape[-2]),int(latents.shape[-1]))
            history_token_shapes={'long':(4,(lh+7)//8,(lw+7)//8),'mid':(1,(lh+3)//4,(lw+3)//4),'short':(2,(lh+1)//2,(lw+1)//2)}
        self.ray_provider.set_context(chunk_index=chunk_index,c2w=cameras,intrinsics=K,latent_cameras=reps,history_groups=history_groups,history_token_shapes=history_token_shapes,history_latent_ids=history_latent_ids,stage_shapes=shapes,token_shape=shapes[0])
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
        self.reset_sequence()
        if not self._source_initialized: self._source_initialized=True
        if num_frames<33 or (num_frames-1)%32: raise ValueError('Sightline inference requires 1+32*N frames')
        if c2w.ndim!=4 or intrinsics.ndim!=4 or c2w.shape[:2]!=intrinsics.shape[:2]: raise ValueError('c2w/K must remain [B,F,...]')
        self._trajectory_c2w=c2w; self._trajectory_K=intrinsics; self._source_camera=c2w[:,0]; self._source_intrinsics=intrinsics[:,0]
        chunks=(num_frames-1)//32; device=self.helios._execution_device; transformer_dtype=self.helios.transformer.dtype; runtime_kwargs=dict(attention_kwargs or {})
        self.helios._guidance_scale=1.0; self.helios._attention_kwargs=runtime_kwargs; self.helios._current_timestep=None; self.helios._interrupt=False
        prompt_embeds,negative_embeds=self.helios.encode_prompt(prompt=prompt,negative_prompt=negative_prompt,do_classifier_free_guidance=False,num_videos_per_prompt=1,max_sequence_length=512,device=device)
        prompt_embeds=prompt_embeds.to(transformer_dtype); negative_embeds=None
        image_tensor=self.helios.video_processor.preprocess(image,height=height,width=width)
        vae=self.helios.vae; mean=torch.tensor(vae.config.latents_mean,device=device,dtype=vae.dtype).view(1,vae.config.z_dim,1,1,1); std=(1.0/torch.tensor(vae.config.latents_std,device=device,dtype=vae.dtype)).view(1,vae.config.z_dim,1,1,1)
        source,fake=self.helios.prepare_image_latents(image_tensor,latents_mean=mean,latents_std=std,num_latent_frames_per_chunk=9,dtype=torch.float32,device=device)
        completed=[]; completed_ids=[]; accumulated=None
        class Progress:
            def update(self): pass
        for chunk in range(chunks):
            pairs=list(zip(completed_ids,completed))[-19:]; pairs=[(0,source)]*(19-len(pairs))+pairs
            long_pairs=pairs[:16]; mid_pairs=pairs[16:18]; short_pairs=[(0,source),pairs[18]]
            def pack(items):
                return torch.cat([value for _,value in items],2),torch.tensor([identity for identity,_ in items],device=device,dtype=torch.long).view(1,-1)
            long_history,long_ids=pack(long_pairs); mid_history,mid_ids=pack(mid_pairs); short_history,short_ids=pack(short_pairs)
            latent=self.helios.prepare_latents(source.shape[0],self.helios.transformer.config.in_channels,height,width,33,dtype=torch.float32,device=device)
            current_ids=torch.arange(chunk*8,chunk*8+9,device=device,dtype=torch.long).view(1,-1)
            self._prepare_chunk(chunk,latent,runtime_kwargs,history_latent_hw=source.shape[-2:])
            latent=self.helios.stage2_sample(latents=latent,pyramid_num_stages=3,pyramid_num_inference_steps_list=list(self.config.pyramid_steps),prompt_embeds=prompt_embeds,negative_prompt_embeds=negative_embeds,guidance_scale=1.0,indices_hidden_states=current_ids,indices_latents_history_short=short_ids,indices_latents_history_mid=mid_ids,indices_latents_history_long=long_ids,latents_history_short=short_history,latents_history_mid=mid_history,latents_history_long=long_history,attention_kwargs=runtime_kwargs,device=device,transformer_dtype=transformer_dtype,progress_bar=Progress())
            if chunk==0: latent[:,:,0:1]=source.to(latent)
            accumulated=self.append_stride32_latents(accumulated,latent)
            for local in range(1,9): completed.append(latent[:,:,local:local+1].detach()); completed_ids.append(chunk*8+local)
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
