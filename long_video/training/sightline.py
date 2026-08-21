"""Trainable Sightline components and correspondence supervision.

Helios itself is frozen and supplied by the H100 adapter; this module owns only
the trainable ray projections, alpha, timestamp, correspondence head and LoRA.
"""
from __future__ import annotations
import math, torch
from torch import nn
from ..sightline.conditioning import SightlineConditioner
from ..sightline.correspondence import correspondence_loss

def select_train_chunk(max_chunks: int, generator: torch.Generator | None = None) -> int:
    if not 1 <= max_chunks <= 6: raise ValueError("max_chunks must be in 1..6")
    return int(torch.randint(max_chunks,(1,),generator=generator).item())

def assert_trainable_whitelist(module: nn.Module) -> None:
    allowed=("conditioner.","corr_head.","memory.timestamp.","lora_")
    bad=[name for name,p in module.named_parameters() if p.requires_grad and not name.startswith(allowed)]
    if bad: raise RuntimeError(f"Sightline trainable whitelist violation: {bad[:8]}")

def chunk_grad_policy(chunk_index: int, train_chunk: int):
    if chunk_index < train_chunk: return "forward_detached"
    if chunk_index == train_chunk: return "backward"
    return "rollout_detached"

def curriculum_max_chunks(step: int, *, warmup_steps: int, maximum: int = 6) -> int:
    """Monotonic 1..6 chunk curriculum, kept independent of data semantics."""
    if step < 0 or warmup_steps < 1 or not 1 <= maximum <= 6:
        raise ValueError("invalid curriculum arguments")
    return min(maximum, 1 + step // warmup_steps)

def curriculum_phase(step: int):
    """Exact 2500-step Sightline curriculum requested by the training spec."""
    if step < 0: raise ValueError("step must be non-negative")
    if step < 300: return {"name":"P1","max_chunks":1,"lora":False,"correspondence":False,"memory":False}
    if step < 1000: return {"name":"P2","max_chunks":1,"lora":True,"correspondence":False,"memory":False}
    bounds=((1500,2),(1800,3),(2100,4),(2300,5))
    max_chunks=next((chunks for end,chunks in bounds if step < end),6)
    return {"name":"P3","max_chunks":max_chunks,"lora":True,"correspondence":True,"memory":True}

def assert_single_backward_chunk(policies, train_chunk: int) -> None:
    if sum(policy == "backward" for policy in policies) != 1 or policies[train_chunk] != "backward":
        raise RuntimeError("exactly one train chunk may retain autograd")

def causal_chunk_plan(max_chunks: int, train_chunk: int):
    """Return the only permitted per-chunk autograd policy for one sample."""
    if not 0 <= train_chunk < max_chunks <= 6:
        raise ValueError("invalid train chunk")
    policies=tuple(chunk_grad_policy(i,train_chunk) for i in range(max_chunks))
    assert_single_backward_chunk(policies,train_chunk)
    return policies

def run_single_graph_chunks(max_chunks, train_chunk, forward_chunk):
    """Execute all causal chunks while retaining exactly one autograd graph."""
    policies=causal_chunk_plan(max_chunks,train_chunk); outputs=[]
    for chunk,policy in enumerate(policies):
        if policy=="backward": output=forward_chunk(chunk,True)
        else:
            with torch.no_grad(): output=forward_chunk(chunk,False)
            if isinstance(output,torch.Tensor): output=output.detach()
        outputs.append(output)
    return outputs,policies

def selected_qk_logits(query, key, query_indices):
    """Compute attention logits only for selected queries, never full Q x K."""
    if query.ndim!=4 or key.ndim!=4: raise ValueError('Q/K must be [B,N,H,D]')
    indices=torch.as_tensor(query_indices,device=query.device,dtype=torch.long)
    selected=query.index_select(1,indices)
    return torch.einsum('bqhd,bkhd->bhqk',selected,key)*(selected.shape[-1]**-.5)

class SightlineTrainable(nn.Module):
    def __init__(self, inner_dim, layers=(), timestamp_buckets=64, heads=16):
        super().__init__(); self.conditioner=SightlineConditioner(inner_dim); self.corr_head=nn.Sequential(nn.Linear(heads,heads),nn.SiLU(),nn.Linear(heads,1))
    def correspondence(self, logits, positives, weights=None, multi_positive=None):
        if logits.ndim!=4: raise ValueError('logits must be [B,H,Q,K]')
        z=self.corr_head(logits.permute(0,2,3,1)).squeeze(-1); z=z.log_softmax(-1)
        if multi_positive is not None:
            rows=[]
            for q,keys in multi_positive: rows.append(-torch.logsumexp(z[:,q,keys],-1).mean())
            if not rows: return z.new_zeros(())
            values=torch.stack(rows)
            if weights is not None:
                weights=weights.to(device=values.device,dtype=values.dtype); return (values*weights).sum()/weights.sum().clamp_min(1e-8)
            return values.mean()
        return correspondence_loss(z.reshape(-1,z.shape[-1]),positives.reshape(-1),weights)
    @staticmethod
    def lambda_corr(progress, start=.4, initial=.02, final=.005):
        if progress <= start: return initial
        return initial+(final-initial)*min(1.,(progress-start)/(1-start))
    def diagnostics(self):
        alpha=self.conditioner.alpha.detach(); grad=self.conditioner.alpha.grad
        return {'alpha':float(alpha),'alpha_grad':0.0 if grad is None else float(grad.detach().abs()),'eq_grad_norm':float(self.conditioner.q_proj[0].weight.grad.norm()) if self.conditioner.q_proj[0].weight.grad is not None else 0.0,'ek_grad_norm':float(self.conditioner.k_proj[0].weight.grad.norm()) if self.conditioner.k_proj[0].weight.grad is not None else 0.0}

class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank=8, scale=None):
        super().__init__(); self.base=base; self.rank=rank; self.scale=float(scale if scale is not None else 1.0/rank)
        factory={'device':base.weight.device,'dtype':base.weight.dtype}
        self.lora_down=nn.Linear(base.in_features,rank,bias=False,**factory); self.lora_up=nn.Linear(rank,base.out_features,bias=False,**factory)
        nn.init.kaiming_uniform_(self.lora_down.weight,a=math.sqrt(5)); nn.init.zeros_(self.lora_up.weight)
        for parameter in self.base.parameters(): parameter.requires_grad_(False)
    def forward(self,x): return self.base(x)+self.lora_up(self.lora_down(x))*self.scale

def install_lora(transformer: nn.Module, layers, rank=8):
    """Wrap only Q/K/V/O of explicitly selected self-attention blocks."""
    if rank not in (8,16): raise ValueError("LoRA rank must be 8 or 16")
    blocks=list(getattr(transformer,"transformer_blocks",getattr(transformer,"blocks",[]))); installed=[]
    for index in layers:
        if not 0 <= int(index) < len(blocks): raise ValueError(f"invalid LoRA layer {index}")
        attn=getattr(blocks[int(index)],"attn1",None)
        if attn is None: raise RuntimeError(f"layer {index} has no self-attention")
        if hasattr(attn,'unfuse_projections'): attn.unfuse_projections()
        elif hasattr(attn,'fuse_projections'): attn.fuse_projections(fuse=False)
        else: raise RuntimeError(f'layer {index} cannot explicitly disable fused projections')
        attn.to_qkv=None
        if getattr(attn,'fused_projections',False):
            raise RuntimeError(f"layer {index} remained fused after unfuse_projections(); LoRA would be bypassed")
        for name in ("to_q","to_k","to_v"):
            module=getattr(attn,name,None)
            if not isinstance(module,nn.Linear): raise RuntimeError(f'layer {index} missing unfused {name}')
            if not isinstance(module,LoRALinear): setattr(attn,name,LoRALinear(module,rank))
        output=getattr(attn,"to_out",None)
        if output is not None and isinstance(output[0],nn.Linear): output[0]=LoRALinear(output[0],rank)
        installed.append(int(index))
    return tuple(installed)
