"""K/V-only long-term memory with bounded oldest-first eviction."""
from dataclasses import dataclass
import torch
from torch import nn
@dataclass
class MemoryToken:
    hidden: torch.Tensor; ray: torch.Tensor; chunk_index: int; token_index: int
class LongTermKVMemory:
    def __init__(self, budget=2160*6, pool=2, timestamp_buckets=64, hidden_dim=None):
        self.budget=budget; self.pool=pool; self.tokens=[]; self.timestamp=nn.Embedding(timestamp_buckets,hidden_dim) if hidden_dim else None
    def capture(self, hidden, rays, chunk_index, *, grid_shape=None):
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
        hh=hh.reshape(hh.shape[0],-1,hidden.shape[-1]); rr=rr.reshape(rr.shape[0],-1,rays.shape[-1])
        # Re-normalize pooled direction and moment components; callers may
        # replace this with exact cell-centre rays from the camera/K provider.
        rr[..., :3] = rr[..., :3] / rr[..., :3].norm(dim=-1, keepdim=True).clamp_min(1e-6)
        rr[..., 3:6] = rr[..., 3:6] / rr[..., 3:6].norm(dim=-1, keepdim=True).clamp_min(1e-6)
        for i in range(hh.shape[1]): self.tokens.append(MemoryToken(hh[:,i:i+1].detach(),rr[:,i:i+1].detach(),chunk_index,i))
        if len(self.tokens)>self.budget: self.tokens=self.tokens[-self.budget:]
    def get(self):
        if not self.tokens: return None,None
        return torch.cat([t.hidden for t in self.tokens],1), torch.cat([t.ray for t in self.tokens],1)
    def __len__(self): return len(self.tokens)

    def append_native_attention(self, attn, key, value, rotary_emb, rotary_apply, *, current_chunk=0, **kwargs):
        hidden,rays=self.get()
        if hidden is None: return key,value,{"memory_tokens":0}
        if hidden.shape[0] != key.shape[0]:
            if hidden.shape[0] == 1: hidden=hidden.expand(key.shape[0],-1,-1)
            else: raise ValueError("memory batch differs from attention batch")
        mem_k=attn.to_k(hidden); mem_v=attn.to_v(hidden)
        mem_k=attn.norm_k(mem_k).unflatten(2,(attn.heads,-1)); mem_v=mem_v.unflatten(2,(attn.heads,-1))
        if rotary_emb is not None:
            # Memory uses a fixed temporal band and pooled spatial identity; the
            # caller can pass a dedicated memory rotary tensor via kwargs.
            memory_rotary=kwargs.get('memory_rotary_emb')
            if memory_rotary is not None: mem_k=rotary_apply(mem_k,memory_rotary)
        return torch.cat((key,mem_k),1), torch.cat((value,mem_v),1), {"memory_tokens":mem_k.shape[1],"memory_chunk_count":len({t.chunk_index for t in self.tokens})}

class LayerKVMemoryBank:
    """Independent memory banks; hidden representations never cross layers."""
    def __init__(self, layers, budget=2160*6, pool=2):
        self.banks={int(layer): LongTermKVMemory(budget=budget,pool=pool) for layer in layers}
    def for_layer(self, layer):
        if int(layer) not in self.banks: raise KeyError(f"memory layer {layer} is not selected")
        return self.banks[int(layer)]

    def append_to_attention(self, query, key, value, key_projection, value_projection, timestamp_embedding=None):
        """Append memory only on K/V axes; query/output length is unchanged."""
        hidden,rays=self.get()
        if hidden is None: return key,value,{"memory_tokens":0}
        mem_k=key_projection(hidden); mem_v=value_projection(hidden)
        if timestamp_embedding is not None:
            ages=torch.tensor([max(0, self.tokens[-1].chunk_index-t.chunk_index) for t in self.tokens],device=hidden.device)
            mem_k=mem_k+timestamp_embedding(ages).unsqueeze(0)
        return torch.cat((key,mem_k),1),torch.cat((value,mem_v),1),{"memory_tokens":mem_k.shape[1]}
