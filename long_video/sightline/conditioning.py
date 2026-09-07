"""Per-layer zero-initialized Sightline Q/K geometry projections."""
from __future__ import annotations
import torch
from torch import nn
from .bounded_ops import DEFAULT_TOKEN_TILE, token_blocked_sightline_relative_project

MAX_DIAGNOSTIC_QUANTILE_VALUES=262_144
def _bounded_quantile_sample(values: torch.Tensor) -> torch.Tensor:
    flat=values.reshape(-1)
    return flat if flat.numel()<=MAX_DIAGNOSTIC_QUANTILE_VALUES else flat[::(flat.numel()+MAX_DIAGNOSTIC_QUANTILE_VALUES-1)//MAX_DIAGNOSTIC_QUANTILE_VALUES]

class SightlineConditioner(nn.Module):
    def __init__(self,inner_dim:int,scale_aug_prob:float=.3,scale_aug_range=(-1.2,1.6),rho_init:float=.6):
        super().__init__(); self.inner_dim=int(inner_dim); self.scale_aug_prob=float(scale_aug_prob); self.scale_aug_range=tuple(map(float,scale_aug_range)); self.token_tile=DEFAULT_TOKEN_TILE
        self.q_proj=nn.Linear(7,self.inner_dim,bias=True); self.k_proj=nn.Linear(7,self.inner_dim,bias=True); self.gate=nn.Linear(1,self.inner_dim,bias=True)
        self.rms_norm_q=nn.RMSNorm(self.inner_dim,eps=1e-6); self.rms_norm_k=nn.RMSNorm(self.inner_dim,eps=1e-6)
        if not 0.0 < float(rho_init) < 1.0: raise ValueError('rho_init must be strictly inside (0, 1)')
        beta_init=torch.logit(torch.tensor(float(rho_init)))
        self.beta_q=nn.Parameter(beta_init.clone()); self.beta_k=nn.Parameter(beta_init.clone())
        for projection in (self.q_proj,self.k_proj): nn.init.zeros_(projection.weight); nn.init.zeros_(projection.bias)
        nn.init.zeros_(self.gate.weight); nn.init.zeros_(self.gate.bias)
        self.capture_numeric_diagnostics=False; self.last_pre_norm_rms={'q':None,'k':None}; self.last_post_norm_rms={'q':None,'k':None}; self.last_gate_stats={'q':None,'k':None}
    def sample_scale_delta(self,rays,training=None):
        if training is None: training=self.training
        if training and torch.rand((),device=rays.device)<self.scale_aug_prob: return torch.empty((),device=rays.device,dtype=rays.dtype).uniform_(*self.scale_aug_range)
        return None
    @staticmethod
    def _ordered_rays(rays,kind): return torch.cat((rays[...,3:6],rays[...,:3],rays[...,6:7]),-1) if kind=='k' else rays
    def rho_values(self):
        return torch.sigmoid(self.beta_q), torch.sigmoid(self.beta_k)
    # Compatibility accessors for older runner diagnostics.  They intentionally
    # expose the bounded-rho logits, never a second amplitude parameter.
    @property
    def alpha_q(self): return self.beta_q
    @property
    def alpha_k(self): return self.beta_k
    def project(self,rays,native=None,*,kind:str,training=None,scale_delta=None,detach_rho:bool=False):
        if rays.shape[-1]!=7 or kind not in ('q','k'): raise ValueError('rays must be 7D and kind must be q or k')
        if native is None: native=rays.new_zeros((*rays.shape[:-1],self.inner_dim))
        projection=self.q_proj if kind=='q' else self.k_proj; norm=self.rms_norm_q if kind=='q' else self.rms_norm_k; beta=self.beta_q if kind=='q' else self.beta_k
        output=token_blocked_sightline_relative_project(rays,native,projection,self.gate,norm,beta.detach() if detach_rho else beta,kind=kind,scale_delta=scale_delta,token_tile=self.token_tile,eps=1e-6)
        if self.capture_numeric_diagnostics:
            with torch.no_grad():
                flat=self._ordered_rays(rays,kind).reshape(-1,7); sample_flat=flat[::max(1,(flat.shape[0]+4095)//4096)].to(projection.weight.dtype)
                raw=projection(sample_flat); normalized=norm(raw); scale=sample_flat[:,6:7] if scale_delta is None else sample_flat[:,6:7]+scale_delta; gate=self.gate(scale).sigmoid()
                sample=gate.flatten().float(); sampled=_bounded_quantile_sample(sample)
                self.last_pre_norm_rms[kind]=float(raw.float().square().mean().sqrt().cpu()); self.last_post_norm_rms[kind]=float(normalized.float().square().mean().sqrt().cpu())
                self.last_gate_stats[kind]={'mean':float(sample.mean().cpu()),'p05':float(torch.quantile(sampled,.05).cpu()),'p50':float(torch.quantile(sampled,.5).cpu()),'p95':float(torch.quantile(sampled,.95).cpu())}
        return output
    def forward(self,rays_q,rays_k=None,native_q=None,native_k=None,*,training=None,scale_delta=None,detach_rho:bool=False):
        rays_k=rays_q if rays_k is None else rays_k
        if native_q is None: native_q=rays_q.new_zeros((*rays_q.shape[:-1],self.inner_dim))
        if native_k is None: native_k=rays_k.new_zeros((*rays_k.shape[:-1],self.inner_dim))
        if scale_delta is None: scale_delta=self.sample_scale_delta(rays_q,training)
        return self.project(rays_q,native=native_q,kind='q',training=training,scale_delta=scale_delta,detach_rho=detach_rho),self.project(rays_k,native=native_k,kind='k',training=training,scale_delta=scale_delta,detach_rho=detach_rho)

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
            yield layer.beta_q; yield layer.beta_k
            yield from layer.q_proj.parameters(); yield from layer.k_proj.parameters(); yield from layer.gate.parameters(); yield from layer.rms_norm_q.parameters(); yield from layer.rms_norm_k.parameters()
    def rho_parameters(self):
        for layer in self.layers.values(): yield layer.beta_q; yield layer.beta_k
    def alpha_parameters(self):
        yield from self.rho_parameters()
    def alpha_values(self):
        return self.rho_values()
    def rho_values(self): return ({key:float(layer.rho_values()[0].detach()) for key,layer in self.layers.items()},{key:float(layer.rho_values()[1].detach()) for key,layer in self.layers.items()})
