"""Sightline Q/K projections and the single global zero-initialized alpha."""
import torch
from torch import nn

class SightlineConditioner(nn.Module):
    def __init__(self, inner_dim: int, hidden_dim: int = 128, eps: float = 1e-6,
                 scale_aug_prob: float = .3, scale_aug_range=(-1.2, 1.6)):
        super().__init__(); self.eps=eps; self.scale_aug_prob=scale_aug_prob; self.scale_aug_range=scale_aug_range
        self.q_proj=nn.Linear(7,inner_dim); self.rms_norm_q=nn.RMSNorm(inner_dim,eps=eps)
        self.k_proj=nn.Linear(7,inner_dim); self.rms_norm_k=nn.RMSNorm(inner_dim,eps=eps)
        self.gate=nn.Linear(1,inner_dim)
        self.alpha=nn.Parameter(torch.zeros(()))
        nn.init.normal_(self.q_proj.weight,std=1e-3); nn.init.zeros_(self.q_proj.bias); nn.init.normal_(self.k_proj.weight,std=1e-3); nn.init.zeros_(self.k_proj.bias); nn.init.zeros_(self.gate.weight); nn.init.zeros_(self.gate.bias)
    def sample_scale_delta(self, rays, training=None):
        if training is None: training=self.training
        if training and torch.rand((),device=rays.device) < self.scale_aug_prob:
            return torch.empty((),device=rays.device,dtype=rays.dtype).uniform_(*self.scale_aug_range)
        return None
    def project(self, rays, *, kind: str, training=None, scale_delta=None):
        if rays.shape[-1] != 7: raise ValueError("rays must contain d, m_hat, log_norm")
        if training is None: training=self.training
        s=rays[...,6:7]; s_in=s if scale_delta is None else s+scale_delta
        # Scale augmentation exclusively affects this gate. E_q/E_k always see
        # true Plücker scale s, never the perturbed value.
        g=torch.sigmoid(self.gate(s_in))
        if kind not in ('q','k'): raise ValueError("kind must be q or k")
        q_in=torch.cat((rays[...,:3],rays[...,3:6],s),-1)
        k_in=torch.cat((rays[...,3:6],rays[...,:3],s),-1)
        dim=self.q_proj.out_features
        value=self.q_proj(q_in) if kind=='q' else self.k_proj(k_in)
        return self.alpha*g*(self.rms_norm_q if kind=='q' else self.rms_norm_k)(value)
    def forward(self, rays_q, rays_k=None, *, training=None, scale_delta=None):
        rays_k = rays_q if rays_k is None else rays_k
        if training is None: training=self.training
        if scale_delta is None: scale_delta=self.sample_scale_delta(rays_q,training)
        return (self.project(rays_q,kind='q',training=training,scale_delta=scale_delta), self.project(rays_k,kind='k',training=training,scale_delta=scale_delta))
