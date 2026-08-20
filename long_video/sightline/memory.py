"""K/V-only long-term memory with bounded oldest-first eviction."""
from dataclasses import dataclass
import torch
@dataclass
class MemoryToken:
    hidden: torch.Tensor; ray: torch.Tensor; chunk_index: int; token_index: int
class LongTermKVMemory:
    def __init__(self, budget=2160*6, pool=2): self.budget=budget; self.pool=pool; self.tokens=[]
    def capture(self, hidden, rays, chunk_index):
        if hidden.ndim<3: raise ValueError("hidden must be [B,N,C]")
        # caller supplies already aligned rays; pool spatially by simple grouping
        n=hidden.shape[1]; step=max(1,self.pool*self.pool); ids=list(range(0,n,step))
        for i in ids: self.tokens.append(MemoryToken(hidden[:,i:i+1].detach(),rays[:,i:i+1].detach(),chunk_index,i))
        if len(self.tokens)>self.budget: self.tokens=self.tokens[-self.budget:]
    def get(self):
        if not self.tokens: return None,None
        return torch.cat([t.hidden for t in self.tokens],1), torch.cat([t.ray for t in self.tokens],1)
    def __len__(self): return len(self.tokens)
