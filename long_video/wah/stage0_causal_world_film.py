"""Stage-0-only causal-world FiLM for the pinned Warp-as-History pipeline."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F


class Stage0CausalWorldFiLM(nn.Module):
    """Per-position 16->32->(gamma16,beta16) modulation."""

    def __init__(self, channels: int = 16, hidden_channels: int = 32):
        super().__init__()
        if channels != 16 or hidden_channels != 32:
            raise ValueError("Stage0 causal-world FiLM is fixed at 16->32->32")
        self.channels = channels
        self.input = nn.Conv3d(channels, hidden_channels, 1)
        self.output = nn.Conv3d(hidden_channels, channels * 2, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, latent, world, visibility):
        if latent.ndim != 5 or latent.shape[1] != self.channels:
            raise ValueError(f"Stage0 latent must be [B,16,T,H,W], got {tuple(latent.shape)}")
        if tuple(world.shape) != tuple(latent.shape):
            raise ValueError("world and Stage0 latent shapes must match")
        expected = (latent.shape[0], 1, *latent.shape[2:])
        if tuple(visibility.shape) != expected:
            raise ValueError(f"Stage0 visibility must be {expected}")
        modulation = self.output(F.silu(self.input(world.to(self.input.weight.dtype))))
        gamma, beta = modulation.chunk(2, dim=1)
        visible = visibility.to(device=latent.device, dtype=gamma.dtype).clamp(0, 1)
        return (latent.to(gamma.dtype) * (1 + visible * gamma) + visible * beta).to(latent.dtype)


class Stage0FiLMController(nn.Module):
    def __init__(self):
        super().__init__()
        self.film = Stage0CausalWorldFiLM()
        self._world = None
        self._visibility = None
        self.applied_calls = 0

    def set_context(self, world, visibility):
        self._world, self._visibility = world.detach(), visibility.detach()

    def clear_context(self):
        self._world = self._visibility = None

    def patch_embedding_pre_hook(self, _module, args):
        if not args or self._world is None:
            return None
        latent = args[0]
        if not torch.is_tensor(latent) or tuple(latent.shape) != tuple(self._world.shape):
            return None
        self.applied_calls += 1
        return (self.film(latent, self._world, self._visibility), *args[1:])


def install_stage0_causal_world_film(transformer):
    existing = getattr(transformer, "stage0_causal_world_film", None)
    if existing is not None:
        return existing
    if not hasattr(transformer, "patch_embedding"):
        raise AttributeError("Helios transformer has no patch_embedding")
    controller = Stage0FiLMController()
    transformer.add_module("stage0_causal_world_film", controller)
    controller._hook_handle = transformer.patch_embedding.register_forward_pre_hook(
        controller.patch_embedding_pre_hook
    )
    return controller


def freeze_for_stage0_film_training(model):
    controller = getattr(model, "stage0_causal_world_film", None)
    if controller is None:
        raise RuntimeError("install Stage0 FiLM before freezing")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    names = []
    for name, parameter in controller.film.named_parameters():
        parameter.requires_grad_(True)
        names.append(f"stage0_causal_world_film.film.{name}")
    return names


def posterior_mode_or_mean(encoded: Any):
    posterior = getattr(encoded, "latent_dist", encoded)
    mode = getattr(posterior, "mode", None)
    if callable(mode):
        value = mode()
        if value is not None:
            return value
    mean = getattr(posterior, "mean", None)
    if mean is None:
        raise TypeError("VAE posterior must expose mode() or mean")
    return mean


def encode_world_video(pipe, video, *, dtype=None):
    """Deterministic 33-frame to 9-slot encoding; never samples the posterior."""
    device = pipe._wah_execution_device()
    vae = pipe.vae
    value = video.to(device=device, dtype=getattr(vae, "dtype", video.dtype))
    latent = posterior_mode_or_mean(vae.encode(value))
    mean, std = pipe._latent_stats(device)
    return ((latent - mean) * std).to(dtype=dtype or latent.dtype)


def renderer_visibility_to_latent(visibility, *, latent_frames, latent_height, latent_width, temporal_scale=4):
    value = torch.as_tensor(visibility, dtype=torch.float32)
    if value.ndim == 3:
        value = value[None, None]
    elif value.ndim == 4:
        value = value[:, None]
    expected_frames = 1 + (latent_frames - 1) * temporal_scale
    if value.ndim != 5 or value.shape[1] != 1 or value.shape[2] != expected_frames:
        raise ValueError(f"visibility must become [B,1,{expected_frames},H,W]")
    groups = [(0, 1)] + [(1 + i * temporal_scale, 1 + (i + 1) * temporal_scale) for i in range(latent_frames - 1)]
    grouped = torch.stack([value[:, :, start:end].mean(2) for start, end in groups], 2)
    flat = grouped.permute(0, 2, 1, 3, 4).flatten(0, 1)
    flat = F.interpolate(flat, (latent_height, latent_width), mode="area")
    return flat.reshape(grouped.shape[0], latent_frames, 1, latent_height, latent_width).permute(0, 2, 1, 3, 4)


def build_stage0_context(pipe, warp_rgb, warp_visibility, *, dtype=None):
    rgb = torch.as_tensor(warp_rgb)
    if rgb.ndim != 4 or rgb.shape[-1] != 3:
        raise ValueError("warp RGB must be [T,H,W,3]")
    device = pipe._wah_execution_device()
    video = pipe._coerce_warp_video_tensor(
        warp_rgb, height=int(rgb.shape[1]), width=int(rgb.shape[2]), device=device,
    )
    world = encode_world_video(pipe, video, dtype=dtype)
    if world.shape[1] != 16:
        raise ValueError("Helios world latent must have 16 channels")
    height, width = world.shape[-2] // 4, world.shape[-1] // 4
    world0 = F.interpolate(world, (world.shape[2], height, width), mode="trilinear", align_corners=False) * 4
    visible0 = renderer_visibility_to_latent(
        warp_visibility, latent_frames=world.shape[2], latent_height=height, latent_width=width
    ).to(device=world0.device, dtype=world0.dtype)
    return world0.detach(), visible0.detach()


@dataclass(frozen=True)
class CausalTrainingContract:
    conditioning_frame_end: int
    target_frame_start: int
    uses_future_gt: bool = False

    def validate(self):
        if self.uses_future_gt:
            raise ValueError("future GT is forbidden from causal world construction")
        if self.conditioning_frame_end >= self.target_frame_start:
            raise ValueError("causal conditioning overlaps the supervised target")
