"""Per-layer zero-initialized Sightline Geometry projections.

Geometry is intentionally small and explicit.  Helios Q/K tensors are already
normalized and rotary-embedded when this module is called, so the only
trainable Geometry objects are the two ray projectors and bounded rho logits.
"""
from __future__ import annotations

import torch
from torch import nn

from .bounded_ops import DEFAULT_TOKEN_TILE, GEOMETRY_RMS_EPSILON, token_blocked_sightline_geometry_project
MAX_DIAGNOSTIC_QUANTILE_VALUES = 262_144


def _bounded_quantile_sample(values: torch.Tensor) -> torch.Tensor:
    flat = values.reshape(-1)
    if flat.numel() <= MAX_DIAGNOSTIC_QUANTILE_VALUES:
        return flat
    stride = (flat.numel() + MAX_DIAGNOSTIC_QUANTILE_VALUES - 1) // MAX_DIAGNOSTIC_QUANTILE_VALUES
    return flat[::stride]


def soft_rms(value: torch.Tensor) -> torch.Tensor:
    return value / torch.sqrt(value.float().square().mean(dim=-1, keepdim=True) + GEOMETRY_RMS_EPSILON)


class SightlineConditioner(nn.Module):
    def __init__(self, inner_dim: int, scale_aug_prob: float = .3,
                 scale_aug_range=(-1.2, 1.6), rho_init: float = .4):
        super().__init__()
        self.inner_dim = int(inner_dim)
        # Compatibility handles for old probes.  Formal config uses 0 and
        # therefore never consumes an RNG value.
        self.scale_aug_prob = float(scale_aug_prob)
        self.scale_aug_range = tuple(map(float, scale_aug_range))
        self.token_tile = DEFAULT_TOKEN_TILE
        self.q_proj = nn.Linear(7, self.inner_dim, bias=False)
        self.k_proj = nn.Linear(7, self.inner_dim, bias=False)
        if not 0.0 < float(rho_init) < 1.0:
            raise ValueError('rho_init must be strictly inside (0, 1)')
        beta_init = torch.logit(torch.tensor(float(rho_init)))
        self.beta_q = nn.Parameter(beta_init.clone())
        self.beta_k = nn.Parameter(beta_init.clone())
        nn.init.zeros_(self.q_proj.weight)
        nn.init.zeros_(self.k_proj.weight)
        self.capture_numeric_diagnostics = False
        self.last_pre_norm_rms = {'q': None, 'k': None}
        self.last_soft_gain = {'q': None, 'k': None}
        self.last_delta_ratio = {'q': None, 'k': None}

    def sample_scale_delta(self, rays, training=None):
        if training is None:
            training = self.training
        if not training or self.scale_aug_prob <= 0.0:
            return None
        if torch.rand((), device=rays.device) < self.scale_aug_prob:
            return torch.empty((), device=rays.device, dtype=rays.dtype).uniform_(*self.scale_aug_range)
        return None

    def rho_values(self):
        return torch.sigmoid(self.beta_q), torch.sigmoid(self.beta_k)

    @staticmethod
    def native_rms(native: torch.Tensor | None, eps: float = 0.0):
        if native is None:
            return None
        if native.ndim < 2:
            raise ValueError(f'native Q/K tensor must include a channel dimension, got {tuple(native.shape)}')
        return native.float().square().mean(dim=-1, keepdim=True).add(float(eps)).sqrt().detach()

    def project(self, rays, native=None, *, kind: str, training=None,
                scale_delta=None, native_rms=None, detach_rho: bool = False):
        if rays.shape[-1] != 7 or kind not in ('q', 'k'):
            raise ValueError('rays must be 7D and kind must be q or k')
        if native is None:
            native = rays.new_zeros((*rays.shape[:-1], self.inner_dim))
        if native_rms is None:
            native_rms = self.native_rms(native)
        beta = self.beta_q if kind == 'q' else self.beta_k
        # No gate, scale augmentation, or learned Geometry norm participates
        # in the formal formula.  scale_delta remains an inert API parameter.
        output = token_blocked_sightline_geometry_project(
            rays, self.q_proj if kind == 'q' else self.k_proj,
            beta.detach() if detach_rho else beta,
            native_rms=native_rms, token_tile=self.token_tile,
        )
        if self.capture_numeric_diagnostics:
            with torch.no_grad():
                raw = (self.q_proj if kind == 'q' else self.k_proj)(
                    rays.reshape(-1, 7).to(self.q_proj.weight.dtype)).float()
                raw_rms = raw.square().mean(-1).sqrt()
                gain = raw_rms / torch.sqrt(raw_rms.square() + GEOMETRY_RMS_EPSILON)
                sample = _bounded_quantile_sample(gain)
                self.last_pre_norm_rms[kind] = float(raw_rms.mean())
                self.last_soft_gain[kind] = {
                    'mean': float(gain.mean()), 'p05': float(torch.quantile(sample, .05)),
                    'p50': float(torch.quantile(sample, .50)), 'p95': float(torch.quantile(sample, .95)),
                }
                self.last_delta_ratio[kind] = float(
                    output.detach().float().norm() /
                    native.detach().float().norm().clamp_min(1e-30)
                )
        return output

    def forward(self, rays_q, rays_k=None, native_q=None, native_k=None, *,
                training=None, scale_delta=None, detach_rho: bool = False):
        rays_k = rays_q if rays_k is None else rays_k
        if native_q is None:
            native_q = rays_q.new_zeros((*rays_q.shape[:-1], self.inner_dim))
        if native_k is None:
            native_k = rays_k.new_zeros((*rays_k.shape[:-1], self.inner_dim))
        if scale_delta is None:
            scale_delta = self.sample_scale_delta(rays_q, training)
        return (
            self.project(rays_q, native_q, kind='q', training=training,
                         scale_delta=scale_delta, native_rms=self.native_rms(native_q),
                         detach_rho=detach_rho),
            self.project(rays_k, native_k, kind='k', training=training,
                         scale_delta=scale_delta, native_rms=self.native_rms(native_k),
                         detach_rho=detach_rho),
        )


class LayeredSightlineConditioner(nn.Module):
    def __init__(self, inner_dim: int, layers, **kwargs):
        super().__init__()
        layers = tuple(map(int, layers))
        if not layers or len(set(layers)) != len(layers):
            raise ValueError('Sightline geometry layers must be non-empty and unique')
        self.inner_dim = int(inner_dim)
        self.layers = nn.ModuleDict({str(layer): SightlineConditioner(self.inner_dim, **kwargs) for layer in layers})

    def for_layer(self, layer):
        key = str(int(layer))
        if key not in self.layers:
            raise KeyError(f'layer {layer} has no Sightline geometry conditioner')
        return self.layers[key]

    def geometry_parameters(self):
        for layer in self.layers.values():
            yield from layer.q_proj.parameters(); yield from layer.k_proj.parameters()
            yield layer.beta_q; yield layer.beta_k

    def projector_parameters(self):
        for layer in self.layers.values():
            yield from layer.q_proj.parameters(); yield from layer.k_proj.parameters()

    def rho_parameters(self):
        for layer in self.layers.values():
            yield layer.beta_q; yield layer.beta_k

    def alpha_parameters(self):
        yield from self.rho_parameters()

    def alpha_values(self):
        return self.rho_values()

    def rho_values(self):
        return ({key: float(layer.rho_values()[0].detach()) for key, layer in self.layers.items()},
                {key: float(layer.rho_values()[1].detach()) for key, layer in self.layers.items()})
