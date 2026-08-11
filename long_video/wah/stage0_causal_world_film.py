"""Point/image-aligned Stage0 FiLM for the pinned Warp-as-History pipeline."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


TEMPORAL_GROUPS = ((0, 1),) + tuple((1 + 4 * i, 5 + 4 * i) for i in range(8))


class PointEncoder(nn.Module):
    """Per-winning-pixel 8->32->16 encoder without spatial mixing."""
    def __init__(self):
        super().__init__()
        self.input = nn.Linear(8, 32)
        self.output = nn.Linear(32, 16)

    def forward(self, value):
        return self.output(F.silu(self.input(value)))


class PointFiLMHead(nn.Module):
    """Per-position [FW, Z-FW_sigma, V, sigma] -> gamma/beta."""
    def __init__(self):
        super().__init__()
        self.input = nn.Conv3d(34, 32, 1)
        self.output = nn.Conv3d(32, 32, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, latent, point_feature, point_feature_sigma, visibility, sigma):
        expected = (latent.shape[0], 1, *latent.shape[2:])
        if tuple(point_feature.shape) != tuple(latent.shape):
            raise ValueError("Point-FiLM feature and Stage0 latent must match")
        if tuple(visibility.shape) != expected:
            raise ValueError(f"Point-FiLM visibility must be {expected}")
        sigma_map = torch.as_tensor(sigma, device=latent.device, dtype=torch.float32)
        while sigma_map.ndim < 5: sigma_map = sigma_map.unsqueeze(-1)
        sigma_map = sigma_map.expand(expected)
        value = torch.cat([
            point_feature.float(), latent.float() - point_feature_sigma.float(),
            visibility.float(), sigma_map,
        ], dim=1)
        gamma, beta = self.output(F.silu(self.input(value.to(self.input.weight.dtype)))).chunk(2, 1)
        visible = visibility.to(gamma.dtype).clamp(0, 1)
        return (latent.to(gamma.dtype) * (1 + visible * gamma) + visible * beta).to(latent.dtype)


def fixed_source_scale(source_visible_depth):
    depth = np.asarray(source_visible_depth, np.float32)
    valid = np.isfinite(depth) & (depth > 0)
    if not np.any(valid): raise ValueError("source has no positive visible depth for fixed scale")
    return float(np.median(depth[valid]))


def world_xyz_to_fixed_source(xyz_world, source_c2w, source_scale):
    pose = np.asarray(source_c2w, np.float32)
    if pose.shape != (4, 4) or float(source_scale) <= 0: raise ValueError("invalid fixed source frame/scale")
    xyz = np.asarray(xyz_world, np.float32)
    return ((xyz - pose[:3, 3]) @ pose[:3, :3] / float(source_scale)).astype(np.float32)


def aggregate_winning_points(warp, source_c2w, source_scale, encoder, *, device=None, dtype=None):
    """Encode winning pixels, then confidence-average to [B,16,9,12,20]."""
    if warp.point_index is None or warp.winning_xyz_world is None:
        raise ValueError("renderer must provide final winning point_index and XYZ")
    rgb = np.asarray(warp.rgb, np.float32)
    xyz = world_xyz_to_fixed_source(warp.winning_xyz_world, source_c2w, source_scale)
    depth = np.asarray(warp.depth, np.float32) / float(source_scale)
    confidence = np.asarray(warp.confidence, np.float32)
    visibility = np.asarray(warp.visibility, bool)
    point_index = np.asarray(warp.point_index, np.int64)
    if rgb.shape[:3] != (33, 384, 640):
        raise ValueError(f"Point-FiLM expects renderer [33,384,640], got {rgb.shape}")
    valid = visibility & (point_index >= 0) & np.isfinite(depth) & np.isfinite(xyz).all(-1)
    if np.any(point_index[~visibility] >= 0): raise ValueError("invisible pixel has a winning point index")
    dev = device or next(encoder.parameters()).device
    features_np = np.concatenate([rgb, xyz, depth[..., None], confidence[..., None]], -1)
    features = torch.as_tensor(features_np, device=dev, dtype=next(encoder.parameters()).dtype)
    valid_t = torch.as_tensor(valid, device=dev)
    flat = features.reshape(-1, 8); flat_valid = valid_t.reshape(-1)
    encoded_flat = torch.zeros((flat.shape[0], 16), device=dev, dtype=features.dtype)
    indices = torch.nonzero(flat_valid, as_tuple=False).flatten()
    if len(indices): encoded_flat = encoded_flat.index_copy(0, indices, encoder(flat.index_select(0, indices)))
    encoded = encoded_flat.reshape(33, 384, 640, 16).permute(0, 3, 1, 2)
    conf = torch.as_tensor(confidence, device=dev, dtype=features.dtype) * valid_t.to(features.dtype)
    visible = valid_t.to(features.dtype)
    feature_groups, coverage_groups = [], []
    for start, stop in TEMPORAL_GROUPS:
        weighted = (encoded[start:stop] * conf[start:stop, None]).sum(0, keepdim=True)
        weights = conf[start:stop].sum(0, keepdim=True)
        weighted = F.avg_pool2d(weighted, kernel_size=(32, 32), stride=(32, 32))
        weights = F.avg_pool2d(weights[:, None], kernel_size=(32, 32), stride=(32, 32))
        feature_groups.append(weighted / weights.clamp_min(1e-6))
        coverage = F.avg_pool2d(
            visible[start:stop].mean(0, keepdim=True)[:, None],
            kernel_size=(32, 32), stride=(32, 32),
        )
        coverage_groups.append(coverage)
    point_feature = torch.stack(feature_groups, 2)
    coverage = torch.stack(coverage_groups, 2)
    return point_feature.to(dtype=dtype or point_feature.dtype), coverage.to(dtype=dtype or coverage.dtype)


def scheduler_aligned_point_feature(point_feature, start_point, sigma, end_sigma):
    """Use the exact native Stage0 start/noise and sigma coordinate."""
    clean_endpoint = float(end_sigma) * start_point + (1.0 - float(end_sigma)) * point_feature
    sigma_value = torch.as_tensor(sigma, device=start_point.device, dtype=start_point.dtype)
    while sigma_value.ndim < start_point.ndim: sigma_value = sigma_value.unsqueeze(-1)
    return sigma_value * start_point + (1 - sigma_value) * clean_endpoint


class Stage0PointFiLMController(nn.Module):
    def __init__(self):
        super().__init__()
        self.point_encoder = PointEncoder()
        self.film_head = PointFiLMHead()
        self._feature = self._visibility = None
        self._start_point = self._sigma = None
        self._end_sigma = 0.0
        self._scheduler = None
        self._inference_mode = False
        self.applied_calls = 0
        self.relative_modulation = None

    def set_point_context(self, feature, visibility):
        self._feature, self._visibility = feature, visibility.detach()
        self._start_point = self._sigma = None

    def set_training_schedule(self, item, end_sigma):
        if int(item["stage_id"]) != 0: raise ValueError("Point-FiLM only supports native Stage0")
        self._start_point = item["start_point"].detach()
        self._sigma = item["sigmas"].detach()
        self._end_sigma = float(end_sigma)
        self._inference_mode = False

    def bind_inference_scheduler(self, scheduler):
        self._scheduler = scheduler; self._inference_mode = True; self._start_point = self._sigma = None

    def clear_context(self):
        self._feature = self._visibility = self._start_point = self._sigma = None

    def transformer_pre_hook(self, _module, args, kwargs):
        if not self._inference_mode or self._feature is None: return None
        latent = kwargs.get("hidden_states", args[0] if args else None)
        if not torch.is_tensor(latent) or tuple(latent.shape) != tuple(self._feature.shape): return None
        if self._start_point is None: self._start_point = latent.detach().clone()
        timestep = kwargs.get("timestep")
        if timestep is None or self._scheduler is None: raise RuntimeError("inference Point-FiLM needs native timestep")
        times = self._scheduler.timesteps.to(device=timestep.device, dtype=torch.float32)
        index = int(torch.argmin((times - timestep.flatten()[0].float()).abs()).item())
        self._sigma = self._scheduler.sigmas[index].to(device=latent.device, dtype=latent.dtype)
        self._end_sigma = float(self._scheduler.end_sigmas[0])
        return None

    def patch_embedding_pre_hook(self, _module, args):
        if not args or self._feature is None or self._start_point is None or self._sigma is None: return None
        latent = args[0]
        if not torch.is_tensor(latent) or tuple(latent.shape) != tuple(self._feature.shape): return None
        feature = self._feature.to(device=latent.device, dtype=latent.dtype)
        visibility = self._visibility.to(device=latent.device, dtype=latent.dtype)
        aligned = scheduler_aligned_point_feature(feature, self._start_point.to(latent), self._sigma, self._end_sigma)
        output = self.film_head(latent, feature, aligned, visibility, self._sigma)
        self.relative_modulation = ((output.float() - latent.float()).norm() / latent.float().norm().clamp_min(1e-8)).detach()
        self.applied_calls += 1
        return (output, *args[1:])


def install_stage0_causal_world_film(transformer):
    existing = getattr(transformer, "stage0_causal_world_film", None)
    if existing is not None: return existing
    if not hasattr(transformer, "patch_embedding"): raise AttributeError("Helios transformer has no patch_embedding")
    controller = Stage0PointFiLMController()
    transformer.add_module("stage0_causal_world_film", controller)
    controller._transformer_hook = transformer.register_forward_pre_hook(
        controller.transformer_pre_hook, with_kwargs=True,
    )
    controller._patch_hook = transformer.patch_embedding.register_forward_pre_hook(
        controller.patch_embedding_pre_hook,
    )
    return controller


def freeze_for_stage0_film_training(model):
    controller = getattr(model, "stage0_causal_world_film", None)
    if controller is None: raise RuntimeError("install Point-FiLM before freezing")
    for parameter in model.parameters(): parameter.requires_grad_(False)
    names = []
    for prefix, module in (("point_encoder", controller.point_encoder), ("film_head", controller.film_head)):
        for name, parameter in module.named_parameters():
            parameter.requires_grad_(True); names.append(f"stage0_causal_world_film.{prefix}.{name}")
    return names


def posterior_mode_or_mean(encoded: Any):
    posterior = getattr(encoded, "latent_dist", encoded)
    mode = getattr(posterior, "mode", None)
    if callable(mode):
        value = mode()
        if value is not None: return value
    mean = getattr(posterior, "mean", None)
    if mean is None: raise TypeError("VAE posterior must expose mode() or mean")
    return mean


@dataclass(frozen=True)
class CausalTrainingContract:
    conditioning_frame_end: int
    target_frame_start: int
    uses_future_gt: bool = False

    def validate(self):
        if self.uses_future_gt: raise ValueError("future GT is forbidden from causal world construction")
        if self.conditioning_frame_end >= self.target_frame_start:
            raise ValueError("causal conditioning overlaps the supervised target")
