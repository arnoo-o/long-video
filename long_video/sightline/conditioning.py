"""Per-layer zero-initialized Sightline Q/K geometry projections."""
from __future__ import annotations
import torch
from torch import nn
from .bounded_ops import DEFAULT_TOKEN_TILE, token_blocked_sightline_project

MAX_DIAGNOSTIC_QUANTILE_VALUES=262_144
def _bounded_quantile_sample(values: torch.Tensor) -> torch.Tensor:
    flat=values.reshape(-1)
    return flat if flat.numel()<=MAX_DIAGNOSTIC_QUANTILE_VALUES else flat[::(flat.numel()+MAX_DIAGNOSTIC_QUANTILE_VALUES-1)//MAX_DIAGNOSTIC_QUANTILE_VALUES]

class SightlineConditioner(nn.Module):
    def __init__(self,inner_dim:int,scale_aug_prob:float=.3,scale_aug_range=(-1.2,1.6),alpha_init:float=.7):
        super().__init__(); self.inner_dim=int(inner_dim); self.scale_aug_prob=float(scale_aug_prob); self.scale_aug_range=tuple(map(float,scale_aug_range)); self.token_tile=DEFAULT_TOKEN_TILE
        self.q_proj=nn.Linear(7,self.inner_dim,bias=True); self.k_proj=nn.Linear(7,self.inner_dim,bias=True); self.gate=nn.Linear(1,self.inner_dim,bias=True)
        self.rms_norm_q=nn.RMSNorm(self.inner_dim,eps=1e-6); self.rms_norm_k=nn.RMSNorm(self.inner_dim,eps=1e-6)
        self.alpha_q=nn.Parameter(torch.tensor(float(alpha_init))); self.alpha_k=nn.Parameter(torch.tensor(float(alpha_init)))
        for projection in (self.q_proj,self.k_proj): nn.init.zeros_(projection.weight); nn.init.zeros_(projection.bias)
        nn.init.zeros_(self.gate.weight); nn.init.zeros_(self.gate.bias)
        self.capture_numeric_diagnostics=False; self.last_pre_norm_rms={'q':None,'k':None}; self.last_post_norm_rms={'q':None,'k':None}; self.last_gate_stats={'q':None,'k':None}
    def sample_scale_delta(self,rays,training=None):
        if training is None: training=self.training
        if training and torch.rand((),device=rays.device)<self.scale_aug_prob: return torch.empty((),device=rays.device,dtype=rays.dtype).uniform_(*self.scale_aug_range)
        return None
    @staticmethod
    def _ordered_rays(rays,kind): return torch.cat((rays[...,3:6],rays[...,:3],rays[...,6:7]),-1) if kind=='k' else rays
    def project(self,rays,*,kind:str,training=None,scale_delta=None,detach_alpha:bool=False):
        if rays.shape[-1]!=7 or kind not in ('q','k'): raise ValueError('rays must be 7D and kind must be q or k')
        projection=self.q_proj if kind=='q' else self.k_proj; norm=self.rms_norm_q if kind=='q' else self.rms_norm_k; alpha=self.alpha_q if kind=='q' else self.alpha_k
        output=token_blocked_sightline_project(rays,projection,self.gate,norm,alpha.detach() if detach_alpha else alpha,kind=kind,scale_delta=scale_delta,token_tile=self.token_tile)
        if self.capture_numeric_diagnostics:
            with torch.no_grad():
                flat=self._ordered_rays(rays,kind).reshape(-1,7); sample_flat=flat[::max(1,(flat.shape[0]+4095)//4096)].to(projection.weight.dtype)
                raw=projection(sample_flat); normalized=norm(raw); scale=sample_flat[:,6:7] if scale_delta is None else sample_flat[:,6:7]+scale_delta; gate=self.gate(scale).sigmoid()
                sample=gate.flatten().float(); sampled=_bounded_quantile_sample(sample)
                self.last_pre_norm_rms[kind]=float(raw.float().square().mean().sqrt().cpu()); self.last_post_norm_rms[kind]=float(normalized.float().square().mean().sqrt().cpu())
                self.last_gate_stats[kind]={'mean':float(sample.mean().cpu()),'p05':float(torch.quantile(sampled,.05).cpu()),'p50':float(torch.quantile(sampled,.5).cpu()),'p95':float(torch.quantile(sampled,.95).cpu())}
        return output
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
            yield layer.alpha_q; yield layer.alpha_k
            yield from layer.q_proj.parameters(); yield from layer.k_proj.parameters(); yield from layer.gate.parameters(); yield from layer.rms_norm_q.parameters(); yield from layer.rms_norm_k.parameters()
    def alpha_parameters(self):
        for layer in self.layers.values(): yield layer.alpha_q; yield layer.alpha_k
    def alpha_values(self): return ({key:float(layer.alpha_q.detach()) for key,layer in self.layers.items()},{key:float(layer.alpha_k.detach()) for key,layer in self.layers.items()})
