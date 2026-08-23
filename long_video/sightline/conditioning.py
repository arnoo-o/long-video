"""Sightline Q/K projections and the single global zero-initialized alpha."""
import torch
from torch import nn

class SightlineConditioner(nn.Module):
    def __init__(self, inner_dim: int, hidden_dim: int = 128, eps: float = 1e-6,
                 scale_aug_prob: float = .3, scale_aug_range=(-1.2, 1.6)):
        super().__init__(); self.eps=eps; self.scale_aug_prob=scale_aug_prob; self.scale_aug_range=scale_aug_range
        self.q_proj=nn.Linear(7,inner_dim)
        self.k_proj=nn.Linear(7,inner_dim)
        self.gate=nn.Linear(1,inner_dim)
        self.rms_norm_q=nn.RMSNorm(inner_dim,eps=eps)
        self.rms_norm_k=nn.RMSNorm(inner_dim,eps=eps)
        # Separate per-layer Q/K gains.  The terminal projections start at
        # zero, so alpha=1 still yields an exact native-Helios forward pass.
        self.alpha_q=nn.Parameter(torch.ones(())); self.alpha_k=nn.Parameter(torch.ones(()))
        nn.init.zeros_(self.q_proj.weight); nn.init.zeros_(self.q_proj.bias)
        nn.init.zeros_(self.k_proj.weight); nn.init.zeros_(self.k_proj.bias)
        nn.init.zeros_(self.gate.weight); nn.init.zeros_(self.gate.bias)
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
        value=(self.rms_norm_q if kind=='q' else self.rms_norm_k)(value)
        alpha=self.alpha_q if kind=='q' else self.alpha_k
        return alpha*g*value
    def forward(self, rays_q, rays_k=None, *, training=None, scale_delta=None):
        rays_k = rays_q if rays_k is None else rays_k
        if training is None: training=self.training
        if scale_delta is None: scale_delta=self.sample_scale_delta(rays_q,training)
        return (self.project(rays_q,kind='q',training=training,scale_delta=scale_delta), self.project(rays_k,kind='k',training=training,scale_delta=scale_delta))

class CameraValueResidual(nn.Module):
    """Frame-level camera residual for current V tokens only."""
    def __init__(self, inner_dim: int):
        super().__init__()
        self.proj=nn.Linear(7,inner_dim)
        self.gate=nn.Linear(7,inner_dim)
        nn.init.normal_(self.proj.weight,std=1e-3); nn.init.zeros_(self.proj.bias)
        nn.init.zeros_(self.gate.weight); nn.init.zeros_(self.gate.bias)
    def forward(self, rays, *, temporal_tokens=None):
        if rays.ndim != 3 or rays.shape[-1] != 7: raise ValueError('camera residual rays must be [B,N,7]')
        if temporal_tokens is None or rays.shape[1] % temporal_tokens:
            raise ValueError('camera residual requires the native temporal token count')
        spatial=rays.shape[1]//temporal_tokens
        feature=rays.float().reshape(rays.shape[0],temporal_tokens,spatial,7).mean(dim=2)
        value=self.proj(feature)*torch.sigmoid(self.gate(feature))
        return value.repeat_interleave(spatial,dim=1)

class LayeredSightlineConditioner(nn.Module):
    """Independent Q/K geometry projections and gains for every selected layer."""
    def __init__(self, inner_dim: int, layers, camera_layers=(), **conditioner_kwargs):
        super().__init__(); layers=tuple(int(layer) for layer in layers)
        if not layers or len(set(layers))!=len(layers): raise ValueError('Sightline geometry layers must be non-empty and unique')
        self.inner_dim=int(inner_dim)
        camera_layers=tuple(int(layer) for layer in camera_layers)
        if not set(camera_layers).issubset(layers): raise ValueError('camera_layers must be a subset of sightline_layers')
        self.camera_layers=tuple(camera_layers)
        self.layers=nn.ModuleDict({str(layer):SightlineConditioner(inner_dim,**conditioner_kwargs) for layer in layers})
        self.camera_residuals=nn.ModuleDict({str(layer):CameraValueResidual(inner_dim) for layer in camera_layers})
    def for_layer(self, layer):
        key=str(int(layer))
        if key not in self.layers: raise KeyError(f'layer {layer} has no Sightline geometry conditioner')
        return self.layers[key]
    def geometry_parameters(self):
        for layer in self.layers.values():
            yield layer.alpha_q; yield layer.alpha_k
            yield from layer.q_proj.parameters(); yield from layer.k_proj.parameters(); yield from layer.gate.parameters()
            yield from layer.rms_norm_q.parameters(); yield from layer.rms_norm_k.parameters()
    def camera_for_layer(self, layer):
        key=str(int(layer))
        return self.camera_residuals[key] if key in self.camera_residuals else None
    def camera_parameters(self):
        for module in self.camera_residuals.values(): yield from module.parameters()
    def alpha_parameters(self):
        for layer in self.layers.values(): yield layer.alpha_q; yield layer.alpha_k
    def alpha_values(self):
        return ({key:float(layer.alpha_q.detach()) for key,layer in self.layers.items()},
                {key:float(layer.alpha_k.detach()) for key,layer in self.layers.items()})
