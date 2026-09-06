"""Per-layer zero-initialized Sightline Q/K geometry projections."""
from __future__ import annotations
import math
import torch
from torch import nn
from .bounded_ops import DEFAULT_TOKEN_TILE, token_blocked_sightline_fixed_rms_project

MAX_DIAGNOSTIC_QUANTILE_VALUES = 262_144
def _bounded_quantile_sample(values: torch.Tensor) -> torch.Tensor:
    flat=values.reshape(-1)
    return flat if flat.numel()<=MAX_DIAGNOSTIC_QUANTILE_VALUES else flat[::(flat.numel()+MAX_DIAGNOSTIC_QUANTILE_VALUES-1)//MAX_DIAGNOSTIC_QUANTILE_VALUES]
def fixed_rms_norm(value: torch.Tensor, geometry_rms_epsilon: float) -> torch.Tensor:
    """Parameter-free RMSNorm: gamma is exactly one and is not trainable."""
    return value*torch.rsqrt(value.float().square().mean(-1,keepdim=True).add(geometry_rms_epsilon)).to(value.dtype)

class SightlineConditioner(nn.Module):
    def __init__(self,inner_dim:int,geometry_rms_epsilon:float=1e-3,scale_aug_prob:float=.3,scale_aug_range=(-1.2,1.6),alpha_init:float=.7):
        super().__init__()
        if not 0.<float(alpha_init)<1.: raise ValueError('alpha_init must be strictly inside (0, 1)')
        if not math.isfinite(float(geometry_rms_epsilon)) or float(geometry_rms_epsilon)<=0.: raise ValueError('geometry_rms_epsilon must be finite and positive')
        self.inner_dim,self.geometry_rms_epsilon=int(inner_dim),float(geometry_rms_epsilon); self.scale_aug_prob,self.scale_aug_range=float(scale_aug_prob),tuple(map(float,scale_aug_range))
        self.token_tile=DEFAULT_TOKEN_TILE
        self.q_proj=nn.Linear(7,self.inner_dim,bias=False); self.k_proj=nn.Linear(7,self.inner_dim,bias=False)
        self.gate=nn.Sequential(nn.Linear(1,self.inner_dim),nn.Sigmoid())
        nn.init.zeros_(self.q_proj.weight); nn.init.zeros_(self.k_proj.weight); nn.init.zeros_(self.gate[0].weight); nn.init.zeros_(self.gate[0].bias)
        beta=math.log(float(alpha_init)/(1.-float(alpha_init))); self.beta_q=nn.Parameter(torch.tensor(beta)); self.beta_k=nn.Parameter(torch.tensor(beta))
        self.capture_numeric_diagnostics=False; self.last_pre_norm_rms={'q':None,'k':None}; self.last_post_norm_rms={'q':None,'k':None}; self.last_gate_stats={'q':None,'k':None}
    @property
    def alpha_q(self): return self.beta_q.sigmoid()
    @property
    def alpha_k(self): return self.beta_k.sigmoid()
    def sample_scale_delta(self,rays,training=None):
        if training is None: training=self.training
        if training and torch.rand((),device=rays.device)<self.scale_aug_prob: return torch.empty((),device=rays.device,dtype=rays.dtype).uniform_(*self.scale_aug_range)
        return None
    @staticmethod
    def _ordered_rays(rays,kind): return torch.cat((rays[...,3:6],rays[...,:3],rays[...,6:7]),-1) if kind=='k' else rays
    def project(self,rays,*,kind:str,training=None,scale_delta=None,detach_alpha:bool=False):
        if rays.shape[-1]!=7 or kind not in ('q','k'): raise ValueError('rays must be 7D and kind must be q or k')
        projection=self.q_proj if kind=='q' else self.k_proj; beta=self.beta_q if kind=='q' else self.beta_k
        output=token_blocked_sightline_fixed_rms_project(
            rays, projection, self.gate, beta.detach() if detach_alpha else beta, kind=kind,
            eps=self.geometry_rms_epsilon, scale_delta=scale_delta,
            token_tile=self.token_tile,
        )
        if self.capture_numeric_diagnostics:
            with torch.no_grad():
                # Diagnostics deliberately use a bounded sample; formal projection never has a dense path.
                flat=self._ordered_rays(rays,kind).reshape(-1,7)
                sample_flat=flat[::max(1,(flat.shape[0]+4095)//4096)].to(projection.weight.dtype)
                raw=torch.nn.functional.linear(sample_flat,projection.weight)
                normalized=fixed_rms_norm(raw,self.geometry_rms_epsilon)
                scale=sample_flat[:,6:7] if scale_delta is None else sample_flat[:,6:7]+scale_delta
                gate=self.gate(scale)
                sample=gate.flatten().float(); sampled=_bounded_quantile_sample(sample)
                self.last_pre_norm_rms[kind]=float(raw.float().square().mean().sqrt().cpu())
                self.last_post_norm_rms[kind]=float(normalized.float().square().mean().sqrt().cpu())
                self.last_gate_stats[kind]={'mean':float(sample.mean().cpu()),'p05':float(torch.quantile(sampled,.05).cpu()),'p50':float(torch.quantile(sampled,.5).cpu()),'p95':float(torch.quantile(sampled,.95).cpu())}
        return output.reshape(*rays.shape[:-1],self.inner_dim)
    def forward(self,rays_q,rays_k=None,*,training=None,scale_delta=None,detach_alpha:bool=False):
        rays_k=rays_q if rays_k is None else rays_k
        if scale_delta is None: scale_delta=self.sample_scale_delta(rays_q,training)
        return self.project(rays_q,kind='q',training=training,scale_delta=scale_delta,detach_alpha=detach_alpha),self.project(rays_k,kind='k',training=training,scale_delta=scale_delta,detach_alpha=detach_alpha)

class LayeredSightlineConditioner(nn.Module):
    def __init__(self,inner_dim:int,layers,**kwargs):
        super().__init__(); layers=tuple(map(int,layers))
        if not layers or len(set(layers))!=len(layers): raise ValueError('Sightline geometry layers must be non-empty and unique')
        self.inner_dim=int(inner_dim); self.layers=nn.ModuleDict({str(layer):SightlineConditioner(self.inner_dim,**kwargs) for layer in layers})
    def for_layer(self,layer):
        key=str(int(layer))
        if key not in self.layers: raise KeyError(f'layer {layer} has no Sightline geometry conditioner')
        return self.layers[key]
    def geometry_parameters(self):
        for layer in self.layers.values():
            yield from layer.q_proj.parameters(); yield from layer.k_proj.parameters(); yield from layer.gate.parameters()
    def alpha_parameters(self):
        for layer in self.layers.values(): yield layer.beta_q; yield layer.beta_k
    def alpha_values(self): return ({key:float(layer.alpha_q.detach()) for key,layer in self.layers.items()},{key:float(layer.alpha_k.detach()) for key,layer in self.layers.items()})
