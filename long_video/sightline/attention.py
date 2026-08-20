"""Post-QKNorm/post-RoPE Sightline processor and correspondence head."""
import torch
from torch import nn
class SightlineAttentionBinding(nn.Module):
    def __init__(self, inner_dim, rays_dim=7): super().__init__(); self.conditioner=nn.Linear(rays_dim,inner_dim); self.alpha=nn.Parameter(torch.zeros(()))
    def inject(self,q,k,rays_q,rays_k):
        dq=self.alpha*self.conditioner(rays_q); dk=self.alpha*self.conditioner(rays_k)
        return q+dq,k+dk

def post_qknorm_rope_inject(q_native, k_native, rays_q, rays_k, binding):
    """Explicit ordering contract: projection -> QKNorm -> RoPE -> this hook."""
    if not isinstance(binding, SightlineAttentionBinding): raise TypeError("invalid sightline binding")
    return binding.inject(q_native,k_native,rays_q,rays_k)
class CorrespondenceProjection(nn.Module):
    def __init__(self, heads): super().__init__(); self.proj=nn.Sequential(nn.Linear(heads,heads),nn.SiLU(),nn.Linear(heads,1))
    def forward(self, logits): return self.proj(logits.transpose(-1,-2)).squeeze(-1)
