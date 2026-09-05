"""Per-layer SCoPE-style Sightline Q/K geometry projections."""
from __future__ import annotations

import torch
from torch import nn

DEFAULT_TOKEN_TILE = 512


def geometry_sigma_gain(sigma: torch.Tensor | float) -> torch.Tensor:
    """The single training/inference geometry routing rule."""
    value = torch.as_tensor(sigma)
    if not torch.isfinite(value).all():
        raise ValueError("geometry sigma must be finite")
    return ((value - .2) / .6).clamp_(0., 1.)


class SightlineConditioner(nn.Module):
    def __init__(self, inner_dim: int, eps: float = 1e-6,
                 scale_aug_prob: float = .3, scale_aug_range=(-1.2, 1.6)):
        super().__init__()
        self.inner_dim = int(inner_dim)
        self.eps = float(eps)
        self.scale_aug_prob = float(scale_aug_prob)
        self.scale_aug_range = tuple(float(x) for x in scale_aug_range)
        hidden = max(self.inner_dim // 4, 1)
        self.q_proj = nn.Sequential(nn.Linear(7, 64, bias=False), nn.GELU(), nn.Linear(64, self.inner_dim, bias=False))
        self.k_proj = nn.Sequential(nn.Linear(7, 64, bias=False), nn.GELU(), nn.Linear(64, self.inner_dim, bias=False))
        self.gate = nn.Sequential(nn.Linear(1, hidden), nn.SiLU(), nn.Linear(hidden, self.inner_dim), nn.Sigmoid())
        self.rms_norm_q = nn.RMSNorm(self.inner_dim, eps=self.eps)
        self.rms_norm_k = nn.RMSNorm(self.inner_dim, eps=self.eps)
        self.alpha_q = nn.Parameter(torch.ones(())); self.alpha_k = nn.Parameter(torch.ones(()))
        self.capture_numeric_diagnostics = False; self.last_pre_norm_rms = {'q': None, 'k': None}; self.last_gate_stats = {'q': None, 'k': None}
        for projection in (self.q_proj, self.k_proj):
            nn.init.kaiming_uniform_(projection[0].weight, a=5**.5)
            nn.init.zeros_(projection[2].weight)
        nn.init.zeros_(self.gate[0].bias); nn.init.zeros_(self.gate[2].bias)

    def sample_scale_delta(self, rays, training=None):
        if training is None: training = self.training
        if training and torch.rand((), device=rays.device) < self.scale_aug_prob:
            return torch.empty((), device=rays.device, dtype=rays.dtype).uniform_(*self.scale_aug_range)
        return None

    def project(self, rays, *, kind: str, training=None, scale_delta=None, detach_alpha: bool = False):
        if rays.shape[-1] != 7: raise ValueError("rays must contain d, m_hat, log_norm")
        if kind not in ('q', 'k'): raise ValueError("kind must be q or k")
        projection = self.q_proj if kind == 'q' else self.k_proj
        norm = self.rms_norm_q if kind == 'q' else self.rms_norm_k
        alpha = self.alpha_q if kind == 'q' else self.alpha_k
        flat = rays.reshape(-1, 7).to(next(projection.parameters()).dtype)
        values = []; pre_norm_sq = 0.0; gate_samples=[]
        for start in range(0, flat.shape[0], DEFAULT_TOKEN_TILE):
            ray = flat[start:start + DEFAULT_TOKEN_TILE].float(); scale = ray[:, 6:7]
            geometric = torch.cat((ray[:, 3:6], ray[:, :3], scale), -1) if kind == 'k' else ray
            # Scale augmentation is intentionally gate-only; projector sees physical log_norm.
            gate_input = scale if scale_delta is None else scale + scale_delta
            raw = projection.float()(geometric)
            gate = self.gate.float()(gate_input)
            value = (alpha.detach() if detach_alpha else alpha).float() * gate * norm.float()(raw)
            values.append(value.to(rays.dtype))
            if self.capture_numeric_diagnostics:
                pre_norm_sq += float(raw.detach().square().sum())
                gate_samples.append(gate.detach().flatten()[::max(1, gate.numel() // 4096)].float())
        output = torch.cat(values, 0).reshape(*rays.shape[:-1], self.inner_dim)
        if self.capture_numeric_diagnostics:
            self.last_pre_norm_rms[kind] = (pre_norm_sq / max(1, flat.shape[0] * self.inner_dim)) ** .5
            sample=torch.cat(gate_samples) if gate_samples else torch.empty(0,device=rays.device)
            self.last_gate_stats[kind] = {'mean':float(sample.mean().cpu()),'p05':float(torch.quantile(sample,.05).cpu()),'p50':float(torch.quantile(sample,.5).cpu()),'p95':float(torch.quantile(sample,.95).cpu())} if sample.numel() else None
        return output

    def forward(self, rays_q, rays_k=None, *, training=None, scale_delta=None, detach_alpha: bool = False):
        rays_k = rays_q if rays_k is None else rays_k
        if training is None: training = self.training
        if scale_delta is None: scale_delta = self.sample_scale_delta(rays_q, training)
        return (self.project(rays_q, kind='q', training=training, scale_delta=scale_delta, detach_alpha=detach_alpha),
                self.project(rays_k, kind='k', training=training, scale_delta=scale_delta, detach_alpha=detach_alpha))


class LayeredSightlineConditioner(nn.Module):
    def __init__(self, inner_dim: int, layers, **conditioner_kwargs):
        super().__init__(); layers=tuple(int(layer) for layer in layers)
        if not layers or len(set(layers)) != len(layers): raise ValueError('Sightline geometry layers must be non-empty and unique')
        self.inner_dim=int(inner_dim)
        self.layers=nn.ModuleDict({str(layer): SightlineConditioner(self.inner_dim, **conditioner_kwargs) for layer in layers})
    def for_layer(self, layer):
        key=str(int(layer))
        if key not in self.layers: raise KeyError(f'layer {layer} has no Sightline geometry conditioner')
        return self.layers[key]
    def geometry_parameters(self):
        for layer in self.layers.values(): yield from layer.parameters()
    def alpha_parameters(self):
        for layer in self.layers.values(): yield layer.alpha_q; yield layer.alpha_k
    def alpha_values(self):
        return ({key:float(layer.alpha_q.detach()) for key,layer in self.layers.items()}, {key:float(layer.alpha_k.detach()) for key,layer in self.layers.items()})
