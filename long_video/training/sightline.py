"""Trainable Sightline components and correspondence supervision.

Helios itself is frozen and supplied by the H100 adapter; this module owns only
the trainable ray projections, alpha, timestamp, correspondence head and LoRA.
"""
from __future__ import annotations
import math, torch
from torch import nn
from ..sightline.conditioning import SightlineConditioner
from ..sightline.correspondence import correspondence_loss

class SightlineTrainable(nn.Module):
    def __init__(self, inner_dim, layers=(), timestamp_buckets=64, heads=16):
        super().__init__(); self.conditioner=SightlineConditioner(inner_dim); self.timestamp=nn.Embedding(timestamp_buckets,inner_dim); self.corr_head=nn.Sequential(nn.Linear(heads,heads),nn.SiLU(),nn.Linear(heads,1))
    def correspondence(self, logits, positives, weights=None, multi_positive=None):
        if logits.ndim!=4: raise ValueError('logits must be [B,H,Q,K]')
        z=self.corr_head(logits.permute(0,2,3,1)).squeeze(-1); z=z.log_softmax(-1)
        if multi_positive is not None:
            rows=[]
            for q,keys in multi_positive: rows.append(-torch.logsumexp(z[:,q,keys],-1).mean())
            return torch.stack(rows).mean() if rows else z.new_zeros(())
        return correspondence_loss(z.reshape(-1,z.shape[-1]),positives.reshape(-1),weights)
    @staticmethod
    def lambda_corr(progress, start=.4, initial=.02, final=.005):
        if progress <= start: return initial
        return initial+(final-initial)*min(1.,(progress-start)/(1-start))
    def diagnostics(self):
        alpha=self.conditioner.alpha.detach(); grad=self.conditioner.alpha.grad
        return {'alpha':float(alpha),'alpha_grad':0.0 if grad is None else float(grad.detach().abs()),'eq_grad_norm':float(self.conditioner.q_proj[0].weight.grad.norm()) if self.conditioner.q_proj[0].weight.grad is not None else 0.0,'ek_grad_norm':float(self.conditioner.k_proj[0].weight.grad.norm()) if self.conditioner.k_proj[0].weight.grad is not None else 0.0}
