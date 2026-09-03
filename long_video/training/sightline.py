"""Trainable Sightline components and correspondence supervision.

Helios itself is frozen and supplied by the H100 adapter; this module owns only
the trainable ray projections, alpha, timestamp and LoRA.
"""
from __future__ import annotations
from dataclasses import dataclass
import hashlib, math, random
import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from ..sightline.conditioning import LayeredSightlineConditioner
from ..sightline.correspondence import correspondence_loss

@dataclass(frozen=True)
class CorrespondencePlan:
    query_indices: torch.Tensor
    positive_indices: torch.Tensor
    positive_mask: torch.Tensor
    weights: torch.Tensor
    identities: tuple
    flags: tuple

def _streaming_correspondence_value(q, k, positive_indices, positive_mask, weights, bias, key_block):
    """Exact dense-reference value using bounded CUDA K blocks."""
    scale=q.shape[-1]**-0.5; b,nq,h=q.shape[:3]; nk=k.shape[1]
    running_max=torch.full((b,h,nq),-torch.inf,device=q.device,dtype=torch.float32)
    running_sum=torch.zeros_like(running_max)
    has_bias=bias.numel()!=0
    for start in range(0,nk,int(key_block)):
        stop=min(nk,start+int(key_block))
        logits=torch.einsum('bqhd,bkhd->bhqk',q,k[:,start:stop]).float()*scale
        if has_bias:
            logits=logits+(bias[:,None,None,start:stop] if bias.ndim==2 else bias[...,start:stop]).float()
        block_max=logits.amax(-1); new_max=torch.maximum(running_max,block_max)
        running_sum=running_sum*torch.exp(running_max-new_max)+torch.exp(logits-new_max[...,None]).sum(-1)
        running_max=new_max
    log_denom=running_max+running_sum.log()
    safe=positive_indices.clamp_min(0)
    gathered=k[:,safe.reshape(-1)].reshape(b,nq,safe.shape[1],h,k.shape[-1])
    pos_logits=torch.einsum('bqhd,bqphd->bhqp',q,gathered).float()*scale
    if has_bias:
        if bias.ndim==2:
            pos_bias=bias[:,safe]
            pos_logits=pos_logits+pos_bias[:,None].float()
        else:
            gather_index=safe[None,None].expand(b,h,-1,-1)
            pos_logits=pos_logits+torch.gather(bias,3,gather_index).float()
    log_prob=(pos_logits-log_denom[...,None]).masked_fill(~positive_mask[None,None],-torch.inf)
    log_mass=torch.logsumexp(log_prob,dim=(1,3))-math.log(h)
    row_loss=-log_mass.mean(0)
    w=weights.float(); return (row_loss*w).sum()/w.sum().clamp_min(1e-8)

class _StreamingCorrespondence(torch.autograd.Function):
    @staticmethod
    def forward(ctx,q,k,positive_indices,positive_mask,weights,bias,key_block):
        ctx.save_for_backward(q,k,positive_indices,positive_mask,weights,bias); ctx.key_block=int(key_block)
        return _streaming_correspondence_value(q,k,positive_indices,positive_mask,weights,bias,ctx.key_block)
    @staticmethod
    def backward(ctx,grad_output):
        q,k,positive_indices,positive_mask,weights,bias=ctx.saved_tensors
        with torch.enable_grad():
            qr=q.detach().requires_grad_(True); kr=k.detach().requires_grad_(True)
            value=_streaming_correspondence_value(qr,kr,positive_indices,positive_mask,weights,bias,ctx.key_block)
            dq,dk=torch.autograd.grad(value,(qr,kr),grad_output)
        return dq,dk,None,None,None,None,None

def select_train_chunk(max_chunks: int, generator: torch.Generator | None = None, *, minimum: int = 0) -> int:
    if not 1 <= max_chunks <= 6: raise ValueError("max_chunks must be in 1..6")
    if not 0<=minimum<max_chunks: raise ValueError('minimum train chunk must be inside the rollout')
    return int(torch.randint(minimum,max_chunks,(1,),generator=generator).item())

def assert_trainable_whitelist(module: nn.Module) -> None:
    allowed=("conditioner.","memory.timestamp.","memory.memory_type_embedding","lora_")
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

def curriculum_phase(step: int, *, p1_steps: int = 400, p2_steps: int = 600, p3_steps: int = 1500):
    """Formal 400/600/1500 curriculum with a 1→6 chunk P3 rollout."""
    if step < 0 or min(p1_steps, p2_steps, p3_steps) < 1:
        raise ValueError("invalid curriculum schedule")
    if step < p1_steps - 100:
        return {"name":"P1","max_chunks":1,"lora":False,"correspondence":False,"memory":False}
    if step < p1_steps:
        return {"name":"P1","max_chunks":2,"lora":False,"correspondence":False,"memory":False}
    if step < p1_steps + p2_steps:
        return {"name":"P2","max_chunks":2,"lora":True,"correspondence":False,"memory":False}
    p3_step = step - p1_steps - p2_steps
    if p3_step < p3_steps:
        # Fixed global-step boundaries keep checkpoint resumes deterministic.
        if p3_step < 500: chunks = 2
        elif p3_step < 800: chunks = 3
        elif p3_step < 1100: chunks = 4
        elif p3_step < 1300: chunks = 5
        else: chunks = 6
        return {"name":"P3","max_chunks":chunks,"lora":True,"correspondence":True,"memory":True}
    raise ValueError("step is outside the configured training schedule")

INIT_SEED = 20260826

def set_initialization_seed(seed: int = INIT_SEED) -> None:
    random.seed(seed); np.random.seed(seed % (2**32-1)); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def set_rank_runtime_seed(rank: int, step: int = 0) -> int:
    seed=INIT_SEED+1000003*int(rank)+9176*int(step)
    random.seed(seed); np.random.seed(seed%(2**32-1)); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    return seed

def synchronized_trainable_parameters(trainable, memory, transformer):
    values=list(trainable.named_parameters())+[(f'memory.{n}',p) for n,p in memory.named_parameters()]
    values += [(f'transformer.{n}',p) for n,p in transformer.named_parameters() if 'lora_' in n]
    return sorted(values,key=lambda item:item[0])

def parameter_digest(named_parameters) -> str:
    digest=hashlib.sha256()
    for name,parameter in named_parameters:
        digest.update(name.encode()); digest.update(parameter.detach().float().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()

def broadcast_and_assert_trainables(trainable, memory, transformer, world_size: int) -> str:
    named=synchronized_trainable_parameters(trainable,memory,transformer)
    if world_size>1:
        if not dist.is_initialized(): raise RuntimeError('distributed parameter synchronization requires an initialized process group')
        for _,parameter in named: dist.broadcast(parameter.data,src=0)
    if any(not torch.isfinite(parameter).all() for _,parameter in named): raise RuntimeError('non-finite trainable parameter before step0')
    digest=parameter_digest(named); signature=(sum(parameter.numel() for _,parameter in named),sum(float(parameter.detach().double().sum()) for _,parameter in named),sum(float(parameter.detach().double().square().sum()) for _,parameter in named)); gathered=[None]*world_size; signatures=[None]*world_size
    if world_size>1:
        dist.all_gather_object(gathered,digest); dist.all_gather_object(signatures,signature)
    else:
        gathered[0]=digest; signatures[0]=signature
    if len(set(gathered))!=1: raise RuntimeError(f'trainable parameters differ before step0: {gathered}')
    if len(set(signatures))!=1: raise RuntimeError(f'trainable numeric signatures differ before step0: {signatures}')
    return digest

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

def run_causal_prefix_chunks(max_chunks, train_chunk, forward_chunk):
    """Run chunk 0 through the sole backward chunk, never any future chunk."""
    policies=causal_chunk_plan(max_chunks,train_chunk)[:train_chunk+1]
    outputs=[]
    for chunk,policy in enumerate(policies):
        if policy=="backward": output=forward_chunk(chunk,True)
        else:
            with torch.no_grad(): output=forward_chunk(chunk,False)
            if isinstance(output,torch.Tensor): output=output.detach()
        outputs.append(output)
    assert_single_backward_chunk(policies,train_chunk)
    return outputs,policies

def prefix_chunk_should_capture_memory(chunk_index:int, train_chunk:int) -> bool:
    """Every completed prefix may serve the next query; the backward chunk cannot."""
    return 0<=int(chunk_index)<int(train_chunk)

def correspondence_capture_for_stage(stage_index:int,stage_count:int,enabled:bool) -> bool:
    if not 0<=int(stage_index)<int(stage_count): raise ValueError('stage index outside flow')
    return bool(enabled and int(stage_index)+1==int(stage_count))

def selected_qk_logits(query, key, query_indices):
    """Compute attention logits only for selected queries, never full Q x K."""
    if query.ndim!=4 or key.ndim!=4: raise ValueError('Q/K must be [B,N,H,D]')
    indices=torch.as_tensor(query_indices,device=query.device,dtype=torch.long)
    selected=query.index_select(1,indices)
    return torch.einsum('bqhd,bkhd->bhqk',selected,key)*(selected.shape[-1]**-.5)

class SightlineTrainable(nn.Module):
    def __init__(self, inner_dim, layers=(0,), timestamp_buckets=64, heads=16):
        super().__init__(); self.conditioner=LayeredSightlineConditioner(inner_dim,layers)
    def correspondence(self, logits, positives=None, weights=None, multi_positive=None, additive_bias=None):
        if logits.ndim!=4: raise ValueError('logits must be [B,H,Q,K]')
        if additive_bias is not None:
            if additive_bias.ndim not in (2,4): raise ValueError('additive bias must be [B,K] or [B,H,Q,K]')
            if additive_bias.ndim==2: additive_bias=additive_bias[:,None,None,:]
            logits=logits+additive_bias
        z=torch.logsumexp(logits.log_softmax(-1),dim=1)-math.log(logits.shape[1])
        if multi_positive is not None:
            rows=[]
            for q,keys in multi_positive: rows.append(-torch.logsumexp(z[:,q,keys],-1).mean())
            if not rows: return z.new_zeros(())
            values=torch.stack(rows)
            if weights is not None:
                weights=weights.to(device=values.device,dtype=values.dtype); return (values*weights).sum()/weights.sum().clamp_min(1e-8)
            return values.mean()
        if positives is None: raise ValueError('positives are required without multi_positive')
        return correspondence_loss(z.reshape(-1,z.shape[-1]),positives.reshape(-1),weights)

    def correspondence_streaming(self, selected_query, key, plan, additive_bias=None, key_block=256):
        if selected_query.ndim!=4 or key.ndim!=4: raise ValueError('Q/K must be [B,N,H,D]')
        bias=selected_query.new_empty(0) if additive_bias is None else additive_bias
        return _StreamingCorrespondence.apply(selected_query,key,plan.positive_indices,plan.positive_mask,plan.weights,bias,int(key_block))
    @staticmethod
    def lambda_corr(progress, start=.4, initial=.02, final=.005):
        if progress <= start: return initial
        return initial+(final-initial)*min(1.,(progress-start)/(1-start))
    def diagnostics(self):
        alpha_q,alpha_k=self.conditioner.alpha_values()
        alpha_grads={name:0.0 if parameter.grad is None else float(parameter.grad.detach().abs()) for name,parameter in self.conditioner.layers.items() for name,parameter in ((f'{name}.q',parameter.alpha_q),(f'{name}.k',parameter.alpha_k))}
        qgrads=[layer.q_proj.weight.grad for layer in self.conditioner.layers.values()]; kgrads=[layer.k_proj.weight.grad for layer in self.conditioner.layers.values()]
        qnorm=sum(float(value.norm()) for value in qgrads if value is not None); knorm=sum(float(value.norm()) for value in kgrads if value is not None)
        return {'alpha_q':alpha_q,'alpha_k':alpha_k,'alpha_grad':alpha_grads,'eq_grad_norm':qnorm,'ek_grad_norm':knorm}

class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank=8, scale=None):
        super().__init__(); self.base=base; self.rank=rank; self.scale=float(scale if scale is not None else 1.0/rank)
        factory={'device':base.weight.device,'dtype':base.weight.dtype}
        self.lora_down=nn.Linear(base.in_features,rank,bias=False,**factory); self.lora_up=nn.Linear(rank,base.out_features,bias=False,**factory); self.enabled=True
        nn.init.kaiming_uniform_(self.lora_down.weight,a=math.sqrt(5)); nn.init.zeros_(self.lora_up.weight)
        for parameter in self.base.parameters(): parameter.requires_grad_(False)
    def forward(self,x): return self.base(x) if not self.enabled else self.base(x)+self.lora_up(self.lora_down(x))*self.scale

def set_lora_enabled(transformer: nn.Module, enabled: bool) -> None:
    for module in transformer.modules():
        if isinstance(module,LoRALinear): module.enabled=bool(enabled)

def configure_alpha_zero_baseline(trainable, memory, transformer) -> None:
    """Disable every Sightline modification while retaining native Helios V."""
    for alpha in trainable.conditioner.alpha_parameters(): alpha.data.zero_()
    memory.set_enabled(False)
    set_lora_enabled(transformer,False)

def install_lora(transformer: nn.Module, layers, rank=8):
    """Wrap only Q/K/V/O of explicitly selected self-attention blocks."""
    if rank not in (8,16): raise ValueError("LoRA rank must be 8 or 16")
    blocks=list(getattr(transformer,"transformer_blocks",None) or getattr(transformer,"blocks",())); installed=[]
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
