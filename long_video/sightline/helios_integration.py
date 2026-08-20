"""Minimal real Helios self-attention processor for Sightline.

The processor executes Q/K projection -> QKNorm -> native RoPE -> Sightline
injection -> SDPA.  It is installed only on selected self-attention modules;
cross-attention and all V paths remain native.
"""
from __future__ import annotations
import torch
import torch.nn.functional as F
from .conditioning import SightlineConditioner

def _rope(x, rotary_emb):
    if rotary_emb is None: return x
    # Pinned Helios exports this helper in the module; the installer injects it
    # so this processor does not import Warp-as-History.
    if callable(rotary_emb): return rotary_emb(x)
    fn=getattr(rotary_emb,'apply',None)
    if fn is None: raise RuntimeError('Helios rotary embedding helper is required')
    return fn(x)

class SightlineSelfAttentionProcessor:
    def __init__(self, conditioner: SightlineConditioner, ray_getter, memory=None):
        self.conditioner=conditioner; self.ray_getter=ray_getter; self.memory=memory; self.last_logits=None
    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, rotary_emb=None, original_context_length=None):
        if encoder_hidden_states is not None or getattr(attn,'is_cross_attention',False):
            raise RuntimeError('Sightline processor is self-attention only')
        q=attn.to_q(hidden_states); k=attn.to_k(hidden_states); v=attn.to_v(hidden_states)
        q=attn.norm_q(q); k=attn.norm_k(k)
        q=q.unflatten(2,(attn.heads,-1)); k=k.unflatten(2,(attn.heads,-1)); v=v.unflatten(2,(attn.heads,-1))
        q=_rope(q,rotary_emb); k=_rope(k,rotary_emb)
        rays_q,rays_k=self.ray_getter(hidden_states, key_length=k.shape[1])
        dq=self.conditioner.project(rays_q,kind='q').unflatten(-1,(attn.heads,-1)); dk=self.conditioner.project(rays_k,kind='k').unflatten(-1,(attn.heads,-1))
        q=q+dq; k=k+dk
        if self.memory is not None:
            k,v,meta=self.memory.append_to_attention(k,k,v,attn.to_k,attn.to_v)
        mask=attention_mask
        out=F.scaled_dot_product_attention(q.transpose(1,2),k.transpose(1,2),v.transpose(1,2),attn_mask=mask,dropout_p=0.0,is_causal=False)
        return attn.to_out[1](attn.to_out[0](out.flatten(2,3)).type_as(q))

def install_sightline_attention(transformer, conditioner, ray_getter, *, layers=(), memory=None):
    installed=[]
    blocks=list(getattr(transformer,'transformer_blocks',getattr(transformer,'blocks',[])))
    for index in layers:
        if index<0 or index>=len(blocks): raise ValueError(f'invalid Helios self-attention layer {index}')
        attn=getattr(blocks[index],'attn1',None)
        if attn is None: raise RuntimeError(f'layer {index} has no self-attention attn1')
        processor=SightlineSelfAttentionProcessor(conditioner,ray_getter,memory if index in layers else None); attn.set_processor(processor) if hasattr(attn,'set_processor') else setattr(attn,'processor',processor); installed.append(index)
    return tuple(installed)
