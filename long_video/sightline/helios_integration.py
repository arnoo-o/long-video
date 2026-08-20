"""Pinned-Helios-compatible Sightline self-attention processor."""
from __future__ import annotations
import hashlib
import torch
from .conditioning import SightlineConditioner
from .rays import token_rays_for_shape, plucker_rays

def helio_source_fingerprint(source_text: str) -> str:
    return hashlib.sha256(source_text.encode()).hexdigest()

class SightlineHeliosAttnProcessor:
    def __init__(self, conditioner: SightlineConditioner, ray_provider, *, memory=None,
                 qkv_projection=None, rotary_apply=None, attention_dispatch=None):
        self.conditioner=conditioner; self.ray_provider=ray_provider; self.memory=memory
        self.qkv_projection=qkv_projection; self.rotary_apply=rotary_apply; self.attention_dispatch=attention_dispatch
        self.last_q=None; self.last_k=None; self.last_attention_meta={}
        self.last_hidden_states=None; self.last_current_length=None
    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None,
                 rotary_emb=None, original_context_length=None, original_context_length_list=None, **kwargs):
        if self.qkv_projection is None or self.rotary_apply is None or self.attention_dispatch is None:
            raise RuntimeError("Sightline processor is not bound to pinned Helios primitives")
        query,key,value=self.qkv_projection(attn,hidden_states,encoder_hidden_states)
        query=attn.norm_q(query); key=attn.norm_k(key)
        query=query.unflatten(2,(attn.heads,-1)); key=key.unflatten(2,(attn.heads,-1)); value=value.unflatten(2,(attn.heads,-1))
        if rotary_emb is not None:
            query=self.rotary_apply(query,rotary_emb); key=self.rotary_apply(key,rotary_emb)
        current_len=original_context_length or query.shape[1]
        self.last_hidden_states=hidden_states
        self.last_current_length=current_len
        rays_q,rays_k=self.ray_provider(hidden_states,key_length=key.shape[1],current_length=current_len,**kwargs)
        dq,dk=self.conditioner(rays_q,rays_k,training=self.conditioner.training)
        dq=dq.unflatten(-1,(attn.heads,-1)); dk=dk.unflatten(-1,(attn.heads,-1))
        if dq.shape[:3]!=query.shape[:3] or dk.shape[:3]!=key.shape[:3]: raise RuntimeError(f"Sightline delta shape mismatch q={dq.shape}/{query.shape} k={dk.shape}/{key.shape}")
        query=query+dq; key=key+dk
        history_len=max(0,key.shape[1]-current_len)
        if getattr(attn,'is_amplify_history',False) and history_len:
            scale=1.0+getattr(attn,'max_scale',1.0-1.0)*0.0
            scale=1.0+__import__('torch').sigmoid(attn.history_key_scale)*(attn.max_scale-1.0)
            if getattr(attn,'history_scale_mode','per_head')=='per_head': scale=scale.view(1,1,-1,1)
            key=torch.cat((key[:,:history_len]*scale,key[:,history_len:]),1)
        if getattr(attn,'restrict_self_attn',False) and history_len:
            key=key[:,:history_len]; value=value[:,:history_len]
        if self.memory is not None:
            memory_kwargs=dict(kwargs); memory_kwargs.pop('timestamp_embedding',None)
            key,value,self.last_attention_meta=self.memory.append_native_attention(
                attn,key,value,rotary_emb,self.rotary_apply,
                current_chunk=kwargs.get('current_chunk',0),
                timestamp_embedding=getattr(self.memory,'timestamp',None),
                sightline_projector=self.conditioner,
                **memory_kwargs)
        self.last_q=query; self.last_k=key
        out=self.attention_dispatch(query,key,value,attn_mask=attention_mask,dropout_p=0.0,is_causal=False,backend=None,parallel_config=None)
        if out.ndim!=4: raise RuntimeError(f"pinned Helios attention returned unexpected shape {out.shape}")
        out=out.flatten(2,3).type_as(query)
        return attn.to_out[1](attn.to_out[0](out))

class SightlineRayProvider:
    """Runtime ray provider bound to one actual Helios token grid."""
    def __init__(self, c2w=None, intrinsics=None, *, token_shape=None, source_height, source_width, vae_spatial_factor=8):
        self.token_shape=tuple(token_shape) if token_shape is not None else None; self.source_height=source_height; self.source_width=source_width; self.vae_spatial_factor=vae_spatial_factor
        self.c2w=c2w; self.intrinsics=intrinsics; self.context=None
    def set_context(self, *, chunk_index, c2w, intrinsics, latent_cameras=None, history_rays=None, history_cameras=None, history_intrinsics=None, stage=0, sigma=0.0, token_shape=None, stage_shapes=None):
        if c2w.ndim != 4 or intrinsics.ndim != 4 or c2w.shape[:2] != intrinsics.shape[:2]: raise ValueError("runtime c2w/K must be [B,F,...] with matching shape")
        self.context={'chunk_index':chunk_index,'c2w':c2w,'intrinsics':intrinsics,'latent_cameras':latent_cameras,'history_rays':history_rays,'history_cameras':history_cameras,'history_intrinsics':history_intrinsics,'stage':stage,'sigma':sigma,'token_shape':tuple(token_shape) if token_shape is not None else self.token_shape,'stage_shapes':stage_shapes}
    def __call__(self, hidden_states, *, key_length, current_length, **kwargs):
        B,N,_=hidden_states.shape
        context=self.context
        if context is None: raise RuntimeError("SightlineRayProvider has no current chunk context")
        c2w=context['c2w']; intrinsics=context['intrinsics']; history=kwargs.get('history_rays',context.get('history_rays')); T,H,W=context.get('token_shape') or self.token_shape or (None,None,None)
        candidates=context.get('stage_shapes') or ((T,H,W),)
        match=[shape for shape in candidates if shape[0]*shape[1]*shape[2]==current_length]
        if len(match)==1: T,H,W=match[0]
        if T is None or H is None or W is None: raise RuntimeError("real Helios token_shape is required; refusing to guess")
        current_count=T*H*W
        if current_length != current_count: raise RuntimeError(f"runtime context current_length={current_length} != ray grid tokens={current_count}")
        rays=token_rays_for_shape(c2w,intrinsics,(B,T,H,W,1),source_height=self.source_height,source_width=self.source_width).reshape(B,current_count,7)
        if key_length == current_count: return rays,rays
        history=kwargs.get('history_rays',context.get('history_rays'))
        if history is None and context.get('history_cameras') is not None:
            hc=context['history_cameras']; hk=context.get('history_intrinsics')
            if hk is None: hk=intrinsics[:, :hc.shape[1]]
            if hc.shape[1] < 1: raise RuntimeError("history camera set is empty")
            history=plucker_rays(hc,hk,H,W,source_height=self.source_height,source_width=self.source_width).reshape(B,-1,7)
        if history is None: raise RuntimeError("history rays are required for attention with history tokens")
        if history.shape[:1] != (B,) or history.shape[-1] != 7 or history.shape[1] != key_length-current_count:
            raise RuntimeError(f"history ray count does not match native Helios context: history={tuple(history.shape)}, key={key_length}, current={current_count}, token_shape={(T,H,W)}")
        all_rays=torch.cat((history.to(rays),rays),1)
        return all_rays,all_rays

    def current_rays(self, token_shape):
        T,H,W=token_shape; return token_rays_for_shape(self.context['c2w'],self.context['intrinsics'],(self.context['c2w'].shape[0],T,H,W,1),source_height=self.source_height,source_width=self.source_width).reshape(self.context['c2w'].shape[0],-1,7)

def install_sightline_attention(transformer, conditioner, ray_provider, *, layers, helios_module, memory=None):
    blocks=list(getattr(transformer,'transformer_blocks',getattr(transformer,'blocks',[])))
    if not layers: raise ValueError("Sightline selected layers must be explicit")
    installed=[]
    for index in layers:
        if not isinstance(index,int) or index<0 or index>=len(blocks): raise ValueError(f"invalid Helios layer {index}")
        attn=getattr(blocks[index],'attn1',None)
        if attn is None: raise RuntimeError(f"layer {index} has no self-attention")
        layer_memory=memory.for_layer(index) if memory is not None and hasattr(memory,'for_layer') else memory
        if layer_memory is not None and memory is not None:
            layer_memory.timestamp=memory.timestamp
        processor=SightlineHeliosAttnProcessor(conditioner,ray_provider,memory=layer_memory,qkv_projection=helios_module._get_qkv_projections,rotary_apply=helios_module.apply_rotary_emb_transposed,attention_dispatch=helios_module.dispatch_attention_fn)
        if hasattr(attn,'set_processor'): attn.set_processor(processor)
        else: attn.processor=processor
        installed.append(index)
    transformer._sightline_processors={index: blocks[index].attn1.processor for index in installed}
    return tuple(installed)
