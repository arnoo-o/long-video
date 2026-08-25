"""K/V-only long-term memory with bounded oldest-first eviction."""
from dataclasses import dataclass
import torch
from torch import nn
@dataclass
class MemoryToken:
    hidden: torch.Tensor; ray: torch.Tensor; chunk_index: int; token_index: int
    temporal: int; pooled_y: int; pooled_x: int
class LongTermKVMemory:
    def __init__(self, budget=2160*6, pool=2):
        self.budget=budget; self.pool=pool; self.tokens=[]; self.timestamp=None; self.rope=None; self.enabled=True
    def capture(self, hidden, rays, chunk_index, *, grid_shape=None, ray_recompute=None):
        if not self.enabled: return
        if hidden.ndim<3: raise ValueError("hidden must be [B,N,C]")
        # caller supplies already aligned rays; pool spatially by simple grouping
        if rays.shape[:2] != hidden.shape[:2]: raise ValueError("memory hidden/ray token count mismatch")
        n=hidden.shape[1]
        if grid_shape is None: raise ValueError("memory capture requires real (T,H,W) grid metadata")
        T,H,W=grid_shape
        if T*H*W != n or rays.shape[1] != n: raise ValueError("memory grid does not match hidden/rays")
        hh=hidden.reshape(hidden.shape[0],T,H,W,-1); rr=rays.reshape(rays.shape[0],T,H,W,-1)
        if H%self.pool or W%self.pool: raise ValueError("memory grid must be divisible by 2x2 pooling")
        hh=hh.reshape(hh.shape[0],T,H//self.pool,self.pool,W//self.pool,self.pool,-1).mean((3,5))
        rr=rr.reshape(rr.shape[0],T,H//self.pool,self.pool,W//self.pool,self.pool,-1).mean((3,5))
        if ray_recompute is not None:
            rr=ray_recompute((T,H//self.pool,W//self.pool)).reshape(rr.shape[0],T,H//self.pool,W//self.pool,-1)
        hh=hh.reshape(hh.shape[0],-1,hidden.shape[-1]); rr=rr.reshape(rr.shape[0],-1,rays.shape[-1])
        # Re-normalize pooled direction and moment components; callers may
        # replace this with exact cell-centre rays from the camera/K provider.
        rr[..., :3] = rr[..., :3] / rr[..., :3].norm(dim=-1, keepdim=True).clamp_min(1e-6)
        rr[..., 3:6] = rr[..., 3:6] / rr[..., 3:6].norm(dim=-1, keepdim=True).clamp_min(1e-6)
        pooled_h,pooled_w=H//self.pool,W//self.pool
        for i in range(hh.shape[1]):
            t=i//(pooled_h*pooled_w); rem=i%(pooled_h*pooled_w); y,x=divmod(rem,pooled_w)
            # temporal0 is always a shared/source boundary.  It is already
            # represented by the preceding clean latent (or source image), so
            # storing it would duplicate identity with an incorrect hidden.
            if t==0: continue
            self.tokens.append(MemoryToken(hh[:,i:i+1].detach(),rr[:,i:i+1].detach(),chunk_index,i,t,y,x))
        if len(self.tokens)>self.budget: self.tokens=self.tokens[-self.budget:]
    def active_tokens(self,current_global_start=None):
        if current_global_start is None: return list(self.tokens)
        return [token for token in self.tokens if token.chunk_index*8+token.temporal < current_global_start]
    def get(self,current_global_start=None):
        tokens=self.active_tokens(current_global_start)
        if not tokens: return None,None
        return torch.cat([t.hidden for t in tokens],1), torch.cat([t.ray for t in tokens],1)
    def __len__(self): return len(self.tokens)

    def position_metadata(self, device, current_global_start):
        tokens=self.active_tokens(current_global_start)
        if not tokens: return None
        global_ids=[token.chunk_index*8+token.temporal for token in tokens]
        frame=torch.tensor([max(1,min(18,19-(current_global_start-global_id))) for global_id in global_ids],device=device,dtype=torch.float32)
        offset=(self.pool-1)/2
        y=torch.tensor([token.pooled_y*self.pool+offset for token in tokens],device=device,dtype=torch.float32)
        x=torch.tensor([token.pooled_x*self.pool+offset for token in tokens],device=device,dtype=torch.float32)
        return frame.view(1,-1,1,1),y.view(1,-1,1,1),x.view(1,-1,1,1)
    def memory_rotary_emb(self, device, current_global_start):
        if self.rope is None: raise RuntimeError("Helios native RoPE module is not bound to memory")
        positions=self.position_metadata(device,current_global_start)
        if positions is None: return None
        rotary=self.rope.forward_with_positions(*positions,device=device)
        return rotary.flatten(2).transpose(1,2)

    def append_native_attention(self, attn, key, value, rotary_emb, rotary_apply, *, current_chunk, current_global_start, timestamp_embedding=None, sightline_projector=None, scale_delta=None, **kwargs):
        tokens=self.active_tokens(current_global_start); hidden,rays=self.get(current_global_start)
        if not self.enabled or hidden is None: return key,value,{"memory_tokens":0}
        if hidden.shape[0] != key.shape[0]:
            if hidden.shape[0] == 1: hidden=hidden.expand(key.shape[0],-1,-1)
            else: raise ValueError("memory batch differs from attention batch")
        mem_k=attn.to_k(hidden); mem_v=attn.to_v(hidden)
        mem_k=attn.norm_k(mem_k).unflatten(2,(attn.heads,-1)); mem_v=mem_v.unflatten(2,(attn.heads,-1))
        memory_rotary_emb=self.memory_rotary_emb(hidden.device,current_global_start)
        memory_tokens=mem_k.shape[1]
        if memory_rotary_emb.ndim < 2 or memory_rotary_emb.shape[1] != memory_tokens:
            raise RuntimeError(f"memory rotary embedding must have exactly {memory_tokens} positions")
        memory_rotary=memory_rotary_emb.to(device=hidden.device,dtype=mem_k.dtype)
        mem_k=rotary_apply(mem_k,memory_rotary)
        if sightline_projector is not None:
            delta=sightline_projector.project(rays.to(hidden),kind='k',training=sightline_projector.training,scale_delta=scale_delta)
            mem_k=mem_k+delta.unflatten(-1,(attn.heads,-1))
        if timestamp_embedding is not None:
            ages=torch.tensor([max(0,current_chunk-t.chunk_index) for t in tokens],device=hidden.device,dtype=torch.long).clamp_max(timestamp_embedding.num_embeddings-1)
            if timestamp_embedding.weight.device != hidden.device or timestamp_embedding.weight.dtype != hidden.dtype:
                raise RuntimeError('memory timestamp embedding must be moved to the runtime device/dtype before forward')
            age=timestamp_embedding(ages).to(mem_k.dtype).unsqueeze(0).unflatten(2,(attn.heads,-1))
            mem_k=mem_k+age
        return torch.cat((key,mem_k),1), torch.cat((value,mem_v),1), {"memory_tokens":mem_k.shape[1],"memory_chunk_count":len({t.chunk_index for t in tokens}),"memory_global_ids":[t.chunk_index*8+t.temporal for t in tokens]}

    def _memory_rays(self):
        return torch.cat([t.ray for t in self.tokens],1)

class LayerKVMemoryBank(nn.Module):
    """Independent memory banks; hidden representations never cross layers."""
    def __init__(self, layers, budget=2160*6, pool=2, hidden_dim=None):
        super().__init__()
        self.banks={int(layer): LongTermKVMemory(budget=budget,pool=pool) for layer in layers}
        self.timestamp=nn.Embedding(64,hidden_dim) if hidden_dim is not None else None
        if self.timestamp is not None:
            # Timestamp is a trainable age signal, but must start neutral.
            nn.init.zeros_(self.timestamp.weight)
    def for_layer(self, layer):
        if int(layer) not in self.banks: raise KeyError(f"memory layer {layer} is not selected")
        return self.banks[int(layer)]

    def bind_rope(self, rope):
        for bank in self.banks.values(): bank.rope=rope; bank.timestamp=self.timestamp

    def set_enabled(self, enabled):
        for bank in self.banks.values(): bank.enabled=bool(enabled)

    def reset(self):
        for bank in self.banks.values(): bank.tokens.clear()

    def append_to_attention(self, *args, **kwargs):
        raise RuntimeError("select a layer bank with for_layer(); memory cannot be shared across layers")
