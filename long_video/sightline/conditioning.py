"""Sightline Q/K projections and the single global zero-initialized alpha."""
import torch
from torch import nn

class SightlineConditioner(nn.Module):
    def __init__(self, inner_dim: int, hidden_dim: int = 128, eps: float = 1e-6,
                 scale_aug_prob: float = .3, scale_aug_range=(-1.2, 1.6)):
        super().__init__(); self.eps=eps; self.scale_aug_prob=scale_aug_prob; self.scale_aug_range=scale_aug_range
        self.q_proj=nn.Sequential(nn.Linear(7,hidden_dim),nn.SiLU(),nn.Linear(hidden_dim,inner_dim))
        self.k_proj=nn.Sequential(nn.Linear(7,hidden_dim),nn.SiLU(),nn.Linear(hidden_dim,inner_dim))
        self.gate=nn.Sequential(nn.Linear(1,32),nn.SiLU(),nn.Linear(32,1))
        self.alpha=nn.Parameter(torch.zeros(()))
        for seq in (self.q_proj,self.k_proj): nn.init.normal_(seq[-1].weight, std=1e-3); nn.init.zeros_(seq[-1].bias)
    def project(self, rays, *, kind: str, training=None, s_aug=None):
        if rays.shape[-1] != 7: raise ValueError("rays must contain d, m_hat, log_norm")
        if training is None: training=self.training
        s=rays[...,6:7]; s_in=s if s_aug is None else s_aug
        if s_aug is None and training and torch.rand((),device=rays.device) < self.scale_aug_prob:
            s_in=s + torch.empty((),device=rays.device,dtype=rays.dtype).uniform_(*self.scale_aug_range)
        # Scale augmentation exclusively affects this gate. E_q/E_k always see
        # true Plücker scale s, never the perturbed value.
        g=torch.sigmoid(self.gate(s_in))
        if kind not in ('q','k'): raise ValueError("kind must be q or k")
        q_in=torch.cat((rays[...,:3],rays[...,3:6],s),-1)
        k_in=torch.cat((rays[...,3:6],rays[...,:3],s),-1)
        dim=self.q_proj[-1].out_features
        value=self.q_proj(q_in) if kind=='q' else self.k_proj(k_in)
        return self.alpha*g*torch.nn.functional.rms_norm(value, (dim,), eps=self.eps)
    def forward(self, rays_q, rays_k=None, *, training=None):
        rays_k = rays_q if rays_k is None else rays_k
        s = rays_q[..., 6:7]
        if training is None: training=self.training
        s_aug=s
        if training and torch.rand((),device=s.device) < self.scale_aug_prob:
            delta=torch.empty((),device=s.device,dtype=s.dtype).uniform_(*self.scale_aug_range)
            s_aug=s+delta
        return (self.project(rays_q,kind='q',training=training,s_aug=s_aug), self.project(rays_k,kind='k',training=training,s_aug=s_aug))
