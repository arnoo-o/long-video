"""Per-layer Sightline Q/K geometry projections."""
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
        # Opt-in, detached scalars for the standalone numerical diagnostic.
        # Formal training never enables this flag and follows the exact same
        # projection/RMSNorm path as before.
        self.capture_numeric_diagnostics=False
        self.last_pre_norm_rms={'q':None,'k':None}
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
        output_dtype=rays.dtype; parameter_dtype=self.gate.weight.dtype
        s=rays[...,6:7].to(parameter_dtype); s_in=s if scale_delta is None else s+torch.as_tensor(scale_delta,device=s.device,dtype=parameter_dtype)
        # Scale augmentation exclusively affects this gate. E_q/E_k always see
        # true Plücker scale s, never the perturbed value.
        g=torch.sigmoid(self.gate(s_in))
        if kind not in ('q','k'): raise ValueError("kind must be q or k")
        q_in=torch.cat((rays[...,:3].to(parameter_dtype),rays[...,3:6].to(parameter_dtype),s),-1)
        k_in=torch.cat((rays[...,3:6].to(parameter_dtype),rays[...,:3].to(parameter_dtype),s),-1)
        dim=self.q_proj.out_features
        value=self.q_proj(q_in) if kind=='q' else self.k_proj(k_in)
        if self.capture_numeric_diagnostics:
            self.last_pre_norm_rms[kind]=float(value.detach().float().square().mean().sqrt().cpu())
        value=(self.rms_norm_q if kind=='q' else self.rms_norm_k)(value)
        alpha=self.alpha_q if kind=='q' else self.alpha_k
        return (alpha.to(value.dtype)*g*value).to(output_dtype)
    def forward(self, rays_q, rays_k=None, *, training=None, scale_delta=None):
        rays_k = rays_q if rays_k is None else rays_k
        if training is None: training=self.training
        if scale_delta is None: scale_delta=self.sample_scale_delta(rays_q,training)
        return (self.project(rays_q,kind='q',training=training,scale_delta=scale_delta), self.project(rays_k,kind='k',training=training,scale_delta=scale_delta))

class LayeredSightlineConditioner(nn.Module):
    """Independent Q/K geometry projections and gains for every selected layer."""
    def __init__(self, inner_dim: int, layers, **conditioner_kwargs):
        super().__init__(); layers=tuple(int(layer) for layer in layers)
        if not layers or len(set(layers))!=len(layers): raise ValueError('Sightline geometry layers must be non-empty and unique')
        self.inner_dim=int(inner_dim)
        self.layers=nn.ModuleDict({str(layer):SightlineConditioner(inner_dim,**conditioner_kwargs) for layer in layers})
    def for_layer(self, layer):
        key=str(int(layer))
        if key not in self.layers: raise KeyError(f'layer {layer} has no Sightline geometry conditioner')
        return self.layers[key]
    def geometry_parameters(self):
        for layer in self.layers.values():
            yield layer.alpha_q; yield layer.alpha_k
            yield from layer.q_proj.parameters(); yield from layer.k_proj.parameters(); yield from layer.gate.parameters()
            yield from layer.rms_norm_q.parameters(); yield from layer.rms_norm_k.parameters()
    def alpha_parameters(self):
        for layer in self.layers.values(): yield layer.alpha_q; yield layer.alpha_k
    def alpha_values(self):
        return ({key:float(layer.alpha_q.detach()) for key,layer in self.layers.items()},
                {key:float(layer.alpha_k.detach()) for key,layer in self.layers.items()})
