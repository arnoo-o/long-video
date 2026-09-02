"""Pinned-Helios-compatible Sightline self-attention processor."""
from __future__ import annotations
import hashlib
import torch
from .conditioning import SightlineConditioner
from .rays import token_rays_for_shape, plucker_rays
from .history import covered_history_chunk_ids

def helio_source_fingerprint(source_text: str) -> str:
    return hashlib.sha256(source_text.encode()).hexdigest()

class SightlineHeliosAttnProcessor:
    def __init__(self, conditioner: SightlineConditioner, ray_provider, *, memory=None,
                 qkv_projection=None, rotary_apply=None, attention_dispatch=None,
                 attention_backend=None, parallel_config=None):
        self.conditioner=conditioner; self.ray_provider=ray_provider; self.memory=memory
        self.qkv_projection=qkv_projection; self.rotary_apply=rotary_apply; self.attention_dispatch=attention_dispatch
        self.attention_backend=attention_backend; self.parallel_config=parallel_config
        self.residual_scale=1.0
        self.last_q=None; self.last_k=None; self.last_key_identities=None; self.last_attention_meta={}; self.capture_diagnostics=False
        self.capture_numeric_diagnostics=False; self.last_numeric_diagnostics=None
        self.capture_memory_hidden=False; self.last_hidden_states=None; self.last_current_length=None; self.last_attention_bias=None
    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None,
                 rotary_emb=None, original_context_length=None, original_context_length_list=None, **kwargs):
        if self.qkv_projection is None or self.rotary_apply is None or self.attention_dispatch is None:
            raise RuntimeError("Sightline processor is not bound to pinned Helios primitives")
        self.last_attention_bias=None
        query,key,value=self.qkv_projection(attn,hidden_states,encoder_hidden_states)
        query=attn.norm_q(query); key=attn.norm_k(key)
        query=query.unflatten(2,(attn.heads,-1)); key=key.unflatten(2,(attn.heads,-1)); value=value.unflatten(2,(attn.heads,-1))
        if rotary_emb is not None:
            query=self.rotary_apply(query,rotary_emb); key=self.rotary_apply(key,rotary_emb)
        current_len=original_context_length or query.shape[1]
        if self.capture_memory_hidden:
            self.last_hidden_states=hidden_states
        if self.capture_memory_hidden or self.capture_diagnostics:
            self.last_current_length=current_len
        rays_q,rays_k=self.ray_provider(hidden_states,key_length=key.shape[1],current_length=current_len,**kwargs)
        if self.conditioner is None:
            scale_delta=None; dq=torch.zeros_like(query.flatten(2,3)); dk=torch.zeros_like(key.flatten(2,3))
        else:
            condition_dtype=next(self.conditioner.parameters()).dtype
            conditioned_q=rays_q.to(condition_dtype); conditioned_k=rays_k.to(condition_dtype)
            scale_delta=self.conditioner.sample_scale_delta(conditioned_q,self.conditioner.training)
            dq=self.conditioner.project(conditioned_q,kind='q',training=self.conditioner.training,scale_delta=scale_delta)
            dk=self.conditioner.project(conditioned_k,kind='k',training=self.conditioner.training,scale_delta=scale_delta)
            history_len=max(0,key.shape[1]-current_len)
            if history_len:
                pooled_q=self.ray_provider.project_history(self.conditioner,kind='q',scale_delta=scale_delta)
                pooled_k=self.ray_provider.project_history(self.conditioner,kind='k',scale_delta=scale_delta)
                if pooled_q.shape[1]!=history_len or pooled_k.shape[1]!=history_len: raise RuntimeError('pooled history ray embedding count differs from Helios history tokens')
                dq=torch.cat((pooled_q.to(dq),dq[:,-current_len:]),1)
                dk=torch.cat((pooled_k.to(dk),dk[:,-current_len:]),1)
        dq=dq.to(query.dtype).unflatten(-1,(attn.heads,-1)); dk=dk.to(key.dtype).unflatten(-1,(attn.heads,-1))
        if key.shape[1] > current_len:
            validity=self.ray_provider.context.get('history_validity') if self.ray_provider.context is not None else None
            if validity is not None:
                flags=[]; shapes=self.ray_provider.context.get('history_token_shapes') or {}
                for name in ('long','mid','short'):
                    _,h,w=shapes[name]; flags.extend(bool(flag) for flag in validity[name] for _ in range(h*w))
                if len(flags)!=key.shape[1]-current_len: raise RuntimeError('history validity does not match native token count')
                valid_mask=torch.tensor(flags,device=dk.device,dtype=torch.bool).view(1,-1,1,1)
                dk=torch.cat((dk[:,:len(flags)].masked_fill(~valid_mask,0),dk[:,len(flags):]),dim=1)
                dq=torch.cat((dq[:,:len(flags)].masked_fill(~valid_mask,0),dq[:,len(flags):]),dim=1)
        if dq.shape[:3]!=query.shape[:3] or dk.shape[:3]!=key.shape[:3]: raise RuntimeError(f"Sightline delta shape mismatch q={dq.shape}/{query.shape} k={dk.shape}/{key.shape}")
        if self.capture_numeric_diagnostics:
            def rms(value): return float(value.detach().float().square().mean().sqrt().cpu())
            def ratio(delta,native):
                denominator=native.detach().float().norm().clamp_min(1e-30)
                return float((delta.detach().float().norm()/denominator).cpu())
            self.last_numeric_diagnostics={
                'proj_q_rms_before_norm':self.conditioner.last_pre_norm_rms['q'],
                'proj_k_rms_before_norm':self.conditioner.last_pre_norm_rms['k'],
                'delta_q_rms':rms(dq),'delta_k_rms':rms(dk),
                'delta_q_over_q_native':ratio(dq,query),
                'delta_k_over_k_native':ratio(dk,key),
            }
        residual_scale=torch.as_tensor(self.residual_scale,device=query.device,dtype=query.dtype)
        query=query+residual_scale*dq; key=key+residual_scale.to(key.dtype)*dk
        history_len=max(0,key.shape[1]-current_len)
        if getattr(attn,'is_amplify_history',False) and history_len:
            scale=1.0+__import__('torch').sigmoid(attn.history_key_scale)*(attn.max_scale-1.0)
            if getattr(attn,'history_scale_mode','per_head')=='per_head': scale=scale.view(1,1,-1,1)
            key=torch.cat((key[:,:history_len]*scale,key[:,history_len:]),1)
        boost_mask=getattr(attn,'history_key_boost_mask',None)
        boost_scale=float(getattr(attn,'history_key_boost_scale',1.0) or 1.0)
        if boost_mask is not None and boost_scale != 1.0:
            boost_mask=boost_mask.to(device=key.device,dtype=torch.bool)
            if boost_mask.ndim != 1 or boost_mask.shape[0] != key.shape[1]: raise ValueError('native history boost mask does not match key length')
            key=torch.where(boost_mask.view(1,-1,1,1),key*boost_scale,key)
        history_bias=getattr(attn,'history_key_bias',None)
        if history_bias is not None:
            history_bias=history_bias.to(device=query.device,dtype=query.dtype)
            if history_bias.ndim==1: history_bias=history_bias.unsqueeze(0)
            if history_bias.shape != (query.shape[0],key.shape[1]): raise ValueError('native history bias does not match key length')
            additive=history_bias[:,None,None,:]
            self.last_attention_bias=history_bias
            attention_mask=additive if attention_mask is None else attention_mask.to(additive)+additive
        if getattr(attn,'restrict_self_attn',False) and history_len:
            key=key[:,:history_len]; value=value[:,:history_len]
        if self.memory is not None and self.memory.enabled:
            memory_kwargs=dict(kwargs); memory_kwargs.pop('timestamp_embedding',None); memory_kwargs.pop('memory_rotary_emb',None); memory_kwargs.pop('current_chunk',None)
            if self.ray_provider.context is None: raise RuntimeError('memory attention requires active Sightline context')
            active_chunk=int(self.ray_provider.context['chunk_index']); current_global_start=active_chunk*8
            coverages=self.ray_provider.context.get('history_global_coverages') or {}
            native_chunks=covered_history_chunk_ids(coverages,active_chunk)
            key,value,self.last_attention_meta=self.memory.append_native_attention(
                attn,key,value,rotary_emb,self.rotary_apply,
                current_chunk=active_chunk,current_global_start=current_global_start,
                timestamp_embedding=getattr(self.memory,'timestamp',None),
                sightline_projector=self.conditioner,
                scale_delta=scale_delta,
                query_camera_poses=self.ray_provider.context['c2w'],
                native_history_chunk_ids=native_chunks,
                **memory_kwargs)
            mem_count=self.last_attention_meta.get('memory_tokens',0)
            if mem_count and attention_mask is not None:
                old_len=key.shape[1]-mem_count
                if attention_mask.shape[-1] != old_len:
                    raise ValueError('attention mask key axis does not match key length before memory')
                pad=torch.zeros((*attention_mask.shape[:-1],mem_count),device=attention_mask.device,dtype=attention_mask.dtype)
                attention_mask=torch.cat((attention_mask,pad),dim=-1)
            if attention_mask is not None and attention_mask.shape[-1] != key.shape[1]:
                raise RuntimeError('attention mask key axis must equal final K length')
        if self.capture_diagnostics:
            self.last_q=query; self.last_k=key; self.last_key_identities=self.ray_provider.key_identities(current_len,self.memory)
            if len(self.last_key_identities)!=key.shape[1]: raise RuntimeError('key identity map length does not match attention K axis')
        else:
            self.last_q=None; self.last_k=None
        out=self.attention_dispatch(query,key,value,attn_mask=attention_mask,dropout_p=0.0,is_causal=False,backend=self.attention_backend,parallel_config=self.parallel_config)
        if out.ndim!=4: raise RuntimeError(f"pinned Helios attention returned unexpected shape {out.shape}")
        out=out.flatten(2,3).type_as(query)
        return attn.to_out[1](attn.to_out[0](out))

class SightlineRayProvider:
    """Runtime ray provider bound to one actual Helios token grid."""
    def __init__(self, c2w=None, intrinsics=None, *, token_shape=None, source_height, source_width, vae_spatial_factor=8):
        self.token_shape=tuple(token_shape) if token_shape is not None else None; self.source_height=source_height; self.source_width=source_width; self.vae_spatial_factor=vae_spatial_factor
        self.c2w=c2w; self.intrinsics=intrinsics; self.context=None; self._key_identity_cache={}
    def set_context(self, *, chunk_index, c2w, intrinsics, latent_cameras=None, history_rays=None, history_cameras=None, history_intrinsics=None, history_groups=None, history_token_shapes=None, history_global_coverages=None, history_validity=None, stage=0, sigma=0.0, token_shape=None, stage_shapes=None):
        if c2w.ndim != 4 or intrinsics.ndim != 4 or c2w.shape[:2] != intrinsics.shape[:2]: raise ValueError("runtime c2w/K must be [B,F,...] with matching shape")
        self.context={'chunk_index':chunk_index,'c2w':c2w,'intrinsics':intrinsics,'latent_cameras':latent_cameras,'history_rays':history_rays,'history_cameras':history_cameras,'history_intrinsics':history_intrinsics,'history_groups':history_groups,'history_token_shapes':history_token_shapes,'history_global_coverages':history_global_coverages,'history_validity':history_validity,'stage':stage,'sigma':sigma,'token_shape':tuple(token_shape) if token_shape is not None else self.token_shape,'stage_shapes':stage_shapes}
        self._key_identity_cache={}

    def key_identities(self, current_length, memory=None):
        memory_metadata=memory.active_identity_metadata() if memory is not None and memory.enabled else None
        cache_key=(int(current_length),id(memory_metadata) if memory_metadata is not None else None)
        cached=self._key_identity_cache.get(cache_key)
        if cached is not None:return cached
        context=self.context; identities=[]; shapes=context.get('history_token_shapes') or {}; coverages=context.get('history_global_coverages') or {}
        for name in ('long','mid','short'):
            if name not in shapes: continue
            _,h,w=shapes[name]
            if len(coverages.get(name,())) != shapes[name][0]: raise RuntimeError(f'{name} identity coverage does not match temporal tokens')
            for covered in coverages[name]: identities.extend(('native',tuple(int(x) for x in covered),y,x,name) for y in range(h) for x in range(w))
        T,H,W=next(shape for shape in context['stage_shapes'] if shape[0]*shape[1]*shape[2]==current_length)
        base=int(context['chunk_index'])*8
        identities.extend(('current',(base+t,),y,x,'current') for t in range(T) for y in range(H) for x in range(W))
        if memory_metadata is not None:
            identities.extend(('memory',(global_id,),y,x,'memory') for global_id,y,x in memory_metadata)
        result=tuple(identities); self._key_identity_cache[cache_key]=result
        return result
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
        if history is None and context.get('history_groups') is not None:
            groups=context['history_groups']; rays_by_group=[]
            shapes=context.get('history_token_shapes')
            if not shapes: raise RuntimeError('actual native history token shapes are required')
            for name in ('long','mid','short'):
                ht,hh,hw=shapes[name]
                rays_by_group.append(torch.zeros(B,ht*hh*hw,7,device=rays.device,dtype=rays.dtype))
            history=torch.cat(rays_by_group,1)
        if history is None and context.get('history_cameras') is not None:
            hc=context['history_cameras']; hk=context.get('history_intrinsics')
            if hk is None: hk=intrinsics[:, :hc.shape[1]]
            if hc.shape[1] < 1: raise RuntimeError("history camera set is empty")
            history=plucker_rays(hc,hk,H,W,source_height=self.source_height,source_width=self.source_width).reshape(B,-1,7)
        if history is None: raise RuntimeError("history rays are required for attention with history tokens")
        expected=key_length-current_count
        if history.shape[:1] != (B,) or history.shape[-1] != 7:
            raise RuntimeError(f"invalid history ray shape: history={tuple(history.shape)}, key={key_length}, current={current_count}")
        if history.shape[1] != expected:
            raise RuntimeError(f"history ray count does not exactly match native Helios context: history={tuple(history.shape)}, key={key_length}, current={current_count}, token_shape={(T,H,W)}")
        all_rays=torch.cat((history.to(rays),rays),1)
        return all_rays,all_rays

    def project_history(self,conditioner,*,kind,scale_delta):
        """Project each real camera ray first, then pool temporal embeddings."""
        context=self.context; groups=context.get('history_groups'); shapes=context.get('history_token_shapes')
        if groups is None or not shapes: raise RuntimeError('history camera footprints are unavailable')
        factors={'long':4,'mid':2,'short':1}; values=[]
        for name in ('long','mid','short'):
            cameras,K=groups[name]; out_t,height,width=shapes[name]; factor=factors[name]
            if cameras.shape[1]!=out_t*factor: raise RuntimeError(f'{name} camera footprint does not match Helios temporal pooling')
            rays=plucker_rays(cameras,K,height,width,source_height=self.source_height,source_width=self.source_width)
            projected=conditioner.project(rays.to(next(conditioner.parameters()).dtype),kind=kind,training=conditioner.training,scale_delta=scale_delta)
            # Never average poses or raw Plücker rays: only projected embeddings.
            projected=projected.reshape(projected.shape[0],out_t,factor,height,width,-1).mean(2)
            values.append(projected.reshape(projected.shape[0],-1,projected.shape[-1]))
        return torch.cat(values,1)

    def current_rays(self, token_shape):
        T,H,W=token_shape; return token_rays_for_shape(self.context['c2w'],self.context['intrinsics'],(self.context['c2w'].shape[0],T,H,W,1),source_height=self.source_height,source_width=self.source_width).reshape(self.context['c2w'].shape[0],-1,7)

def install_sightline_attention(transformer, conditioner, ray_provider, *, layers, helios_module, memory=None, memory_layers=None):
    blocks=list(getattr(transformer,'transformer_blocks',None) or getattr(transformer,'blocks',()))
    if not layers: raise ValueError("Sightline selected layers must be explicit")
    installed=[]
    if memory is not None and hasattr(memory,'bind_rope'): memory.bind_rope(transformer.rope)
    for index in layers:
        if not isinstance(index,int) or index<0 or index>=len(blocks): raise ValueError(f"invalid Helios layer {index}")
        attn=getattr(blocks[index],'attn1',None)
        if attn is None: raise RuntimeError(f"layer {index} has no self-attention")
        enabled_memory_layers=set(layers if memory_layers is None else memory_layers)
        layer_memory=memory.for_layer(index) if memory is not None and hasattr(memory,'for_layer') and index in enabled_memory_layers else (memory if memory is not None and not hasattr(memory,'for_layer') and index in enabled_memory_layers else None)
        if layer_memory is not None and memory is not None:
            layer_memory.timestamp=memory.timestamp
        layer_conditioner=conditioner.for_layer(index) if hasattr(conditioner,'for_layer') and str(index) in conditioner.layers else None
        native=helios_module.HeliosAttnProcessor()
        processor=SightlineHeliosAttnProcessor(layer_conditioner,ray_provider,memory=layer_memory,qkv_projection=helios_module._get_qkv_projections,rotary_apply=helios_module.apply_rotary_emb_transposed,attention_dispatch=helios_module.dispatch_attention_fn,attention_backend=native._attention_backend,parallel_config=native._parallel_config)
        if hasattr(attn,'set_processor'): attn.set_processor(processor)
        else: attn.processor=processor
        installed.append(index)
    transformer._sightline_processors={index: blocks[index].attn1.processor for index in installed}
    return tuple(installed)
