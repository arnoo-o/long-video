"""Camera/world Q/K-only conditioning for frozen Helios attention."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch
from torch import nn


GEOTOKEN_BLOCKS = (8, 12, 16, 20, 24, 28)
TEMPORAL_GROUPS = ((0,),) + tuple(tuple(range(start, start + 4)) for start in range(1, 33, 4))
CAMERA_CHANNELS = 6
WORLD_BASE_CHANNELS = 9  # XYZ, ray, log-depth, visibility, confidence
WORLD_CHANNELS = WORLD_BASE_CHANNELS + 3 * 2 * 4
STAGE_SCALES = (1.0, 0.7, 0.4)
STAGE_RMS_CAPS = (0.15, 0.10, 0.06)


def progress_from_sigma(sigma) -> float:
    return min(max(1.0 - float(torch.as_tensor(sigma).float().mean()), 0.0), 1.0)


def time_scale_from_progress(progress: float) -> float:
    progress = min(max(float(progress), 0.0), 1.0)
    return 1.0 if progress <= 0.6 else 0.25 + 0.75 * 0.5 * (1 + math.cos(math.pi * (progress - 0.6) / 0.4))


def effective_strengths(geotoken_strength: float, camera_strength: float, world_strength: float):
    return float(geotoken_strength) * float(camera_strength), float(geotoken_strength) * float(world_strength)


def scheduler_progress_from_timestep(scheduler, timestep) -> float:
    """Resolve an inference timestep through the scheduler's actual sigma table."""
    sigmas = torch.as_tensor(getattr(scheduler, "sigmas", ()), dtype=torch.float32).flatten()
    float_timesteps = torch.as_tensor(getattr(scheduler, "timesteps", ()), dtype=torch.float32).flatten()
    count = len(sigmas) - 1
    if count <= 0 or len(float_timesteps) < count:
        raise RuntimeError("Helios scheduler must expose aligned timesteps/sigmas for GeoToken timing")
    timesteps = float_timesteps[:count].to(torch.int64)
    received = torch.as_tensor(timestep).to(torch.int64).flatten()
    if not len(received) or not bool((received == received[0]).all()):
        raise RuntimeError("transformer timestep must contain one unique int64 value")
    value = received[0]
    matches = torch.nonzero(timesteps == value, as_tuple=False).flatten()
    if len(matches) != 1:
        raise RuntimeError(
            f"transformer timestep {int(value)} must match exactly one scheduler int64 timestep; "
            f"matched {len(matches)}"
        )
    index = int(matches[0])
    return progress_from_sigma(sigmas[index])


@dataclass
class GeometryTokenBatch:
    """Parallel CameraRay and WorldGeo token streams aligned to Helios."""

    camera: torch.Tensor
    world: torch.Tensor
    world_support: torch.Tensor

    def __post_init__(self):
        if self.camera.ndim != 3 or self.world.shape != self.camera.shape:
            raise ValueError("camera/world GeoTokens must both be [B,N,256]")
        if self.world_support.shape != self.camera.shape[:2] + (1,):
            raise ValueError("world support must be [B,N,1]")


class _FrameTokenizer(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(channels, 128), nn.SiLU(), nn.Linear(128, 256), nn.SiLU())
        self.score = nn.Linear(256, 1)

    def forward(self, values: torch.Tensor, weights: torch.Tensor):
        # [B,T,H,W,C], [B,T,H,W,1] -> [B,256,9,H,W], [B,1,9,H,W]
        encoded = self.encoder(values.float())
        pooled, supports = [], []
        for group in TEMPORAL_GROUPS:
            index = torch.as_tensor(group, device=values.device)
            feature = encoded.index_select(1, index)
            weight = weights.index_select(1, index).clamp_min(0)
            present = weight.sum(1) > 0
            logits = self.score(feature).masked_fill(weight <= 0, -torch.inf)
            logits = torch.where(present[:, None], logits, torch.zeros_like(logits))
            attention = torch.softmax(logits, 1) * weight
            attention = attention / attention.sum(1, keepdim=True).clamp_min(1e-8)
            item = (feature * attention).sum(1)
            pooled.append(torch.where(present, item, torch.zeros_like(item)))
            supports.append(weight.mean(1))
        return (torch.stack(pooled, 1).permute(0, 4, 1, 2, 3).contiguous(),
                torch.stack(supports, 1).permute(0, 4, 1, 2, 3).contiguous())


class GeometryTokenizer(nn.Module):
    """Separate camera-ray and world-geometry encoders; both retain 33 -> 9 slots."""
    def __init__(self, inner_dim: int):
        super().__init__()
        self.inner_dim = int(inner_dim)
        self.camera = _FrameTokenizer(CAMERA_CHANNELS)
        self.world = _FrameTokenizer(WORLD_CHANNELS)

    @staticmethod
    def _world_features(world: torch.Tensor):
        xyz = world[..., :3]
        features = [world[..., :WORLD_BASE_CHANNELS]]
        for frequency in (1.0, 2.0, 4.0, 8.0):
            features.extend((torch.sin(math.pi * frequency * xyz), torch.cos(math.pi * frequency * xyz)))
        return torch.cat(features, -1)

    def forward(self, camera: torch.Tensor, world: torch.Tensor):
        if camera.ndim != 5 or camera.shape[1:3] != (CAMERA_CHANNELS, 33):
            raise ValueError(f"camera must be [B,6,33,H,W], got {tuple(camera.shape)}")
        if world.ndim != 5 or world.shape[1:3] != (WORLD_BASE_CHANNELS, 33):
            raise ValueError(f"world must be [B,9,33,H,W], got {tuple(world.shape)}")
        camera_values = camera.permute(0, 2, 3, 4, 1)
        world_values = world.permute(0, 2, 3, 4, 1)
        camera_feature, _ = self.camera(camera_values, torch.ones_like(camera_values[..., :1]))
        world_support = (world_values[..., 7:8].clamp(0, 1) * world_values[..., 8:9].clamp(0, 1))
        world_feature, world_support = self.world(self._world_features(world_values), world_support)
        return camera_feature, world_feature, world_support


class _QKAdapter(nn.Module):
    def __init__(self, inner_dim: int):
        super().__init__()
        self.norm = nn.RMSNorm(256)
        self.down = nn.Linear(256, 64, bias=False)
        self.act = nn.SiLU()
        self.up = nn.Linear(64, inner_dim, bias=False)
        nn.init.normal_(self.up.weight, std=1e-3)

    def forward(self, value):
        return self.up(self.act(self.down(self.norm(value))))


class GeoTokenConditioner(nn.Module):
    """Build Q/K deltas only. Values, hidden states and FFNs are untouched."""
    def __init__(self, inner_dim: int, block_indices: Iterable[int] = GEOTOKEN_BLOCKS):
        super().__init__()
        self.tokenizer = GeometryTokenizer(inner_dim)
        self.block_indices = tuple(map(int, block_indices))
        if self.block_indices != GEOTOKEN_BLOCKS:
            raise ValueError(f"GeoToken block indices must be {GEOTOKEN_BLOCKS}")
        self.adapters = nn.ModuleDict({str(i): nn.ModuleDict({
            "camera_q_adapter": _QKAdapter(inner_dim), "camera_k_adapter": _QKAdapter(inner_dim),
            "world_q_adapter": _QKAdapter(inner_dim), "world_k_adapter": _QKAdapter(inner_dim),
        }) for i in self.block_indices})
        self.gates = nn.ParameterDict({f"{i}_{kind}": nn.Parameter(torch.zeros(()))
            for i in self.block_indices for kind in ("camera_q", "camera_k", "world_q", "world_k")})
        self._active: GeometryTokenBatch | None = None
        self._history: GeometryTokenBatch | None = None
        self.camera_strength = 1.0
        self.world_strength = 1.0
        self.stage_index = 0
        self.denoise_progress = 0.0
        self.diagnostics: dict[int, dict] = {}

    def configure_strengths(self, *, geotoken: float = 1.0, camera: float = 1.0, world: float = 1.0):
        camera, world = effective_strengths(geotoken, camera, world)
        if camera < 0 or world < 0:
            raise ValueError("GeoToken strengths must be non-negative")
        self.camera_strength, self.world_strength = float(camera), float(world)

    def set_timing(self, *, stage_index: int = 0, denoise_progress: float = 0.0):
        self.stage_index = int(stage_index)
        self.denoise_progress = float(denoise_progress)

    def set_active(self, current: GeometryTokenBatch, history: GeometryTokenBatch | None = None):
        self._active, self._history = current, history

    def clear_active(self): self._active = self._history = None

    def attach(self, transformer: nn.Module):
        blocks = getattr(transformer, "blocks", None)
        if blocks is None or max(self.block_indices) >= len(blocks):
            raise TypeError("patched Helios transformer blocks are unavailable")
        for index in self.block_indices:
            # The WAH patch invokes this callable after Q/K/V projection and before norm_q/norm_k.
            blocks[index].attn1.geotoken_qk_binding = self._make_qk_binding(index)

    def _joined(self, total_length: int, current_length: int):
        if self._active is None: return None
        if self._active.camera.shape[1] != current_length:
            raise RuntimeError("GeoToken current token count differs from Helios current sequence")
        history_length = total_length - current_length
        if history_length:
            if self._history is None or self._history.camera.shape[1] != history_length:
                raise RuntimeError("GeoToken and WAH history tokens are not one-to-one")
            return GeometryTokenBatch(torch.cat((self._history.camera, self._active.camera), 1),
                torch.cat((self._history.world, self._active.world), 1),
                torch.cat((self._history.world_support, self._active.world_support), 1))
        return self._active

    def _make_qk_binding(self, index):
        def binding(query, key, value, original_context_length=None, **_):
            if self._active is None: return query, key, value
            current_length = int(original_context_length or self._active.camera.shape[1])
            tokens = self._joined(query.shape[1], current_length)
            if tokens is None: return query, key, value
            adapters = self.adapters[str(index)]
            if self.stage_index not in (0, 1, 2):
                raise RuntimeError(f"invalid Helios stage index: {self.stage_index}")
            stage = STAGE_SCALES[self.stage_index]
            progress = min(max(self.denoise_progress, 0.), 1.)
            time = time_scale_from_progress(progress)
            cam_q = adapters["camera_q_adapter"](tokens.camera) * torch.tanh(self.gates[f"{index}_camera_q"])
            cam_k = adapters["camera_k_adapter"](tokens.camera) * torch.tanh(self.gates[f"{index}_camera_k"])
            world_q = adapters["world_q_adapter"](tokens.world) * torch.tanh(self.gates[f"{index}_world_q"])
            world_k = adapters["world_k_adapter"](tokens.world) * torch.tanh(self.gates[f"{index}_world_k"])
            scale = stage * time
            delta_q = (self.camera_strength * cam_q + self.world_strength * tokens.world_support * world_q) * scale
            delta_k = (self.camera_strength * cam_k + self.world_strength * tokens.world_support * world_k) * scale
            # Previous appearance history is a frozen WAH query. Its keys remain world/camera-bound.
            delta_q[:, :-current_length] = 0
            cap = STAGE_RMS_CAPS[self.stage_index]
            def cap_delta(delta, base):
                ratio = delta.float().square().mean(-1, keepdim=True).sqrt() / base.float().square().mean(-1, keepdim=True).sqrt().clamp_min(1e-8)
                return delta * (cap / ratio.clamp_min(cap)).to(delta.dtype), ratio.max()
            delta_q, q_ratio = cap_delta(delta_q, query); delta_k, k_ratio = cap_delta(delta_k, key)
            with torch.no_grad():
                self.diagnostics[index] = {"camera_q_gate": float(torch.tanh(self.gates[f"{index}_camera_q"])), "camera_k_gate": float(torch.tanh(self.gates[f"{index}_camera_k"])), "world_q_gate": float(torch.tanh(self.gates[f"{index}_world_q"])), "world_k_gate": float(torch.tanh(self.gates[f"{index}_world_k"])), "stage_scale": stage, "time_scale": time, "q_delta_ratio": float(q_ratio), "k_delta_ratio": float(k_ratio), "world_support_mean": float(tokens.world_support.mean())}
            return query + delta_q.to(query.dtype), key + delta_k.to(key.dtype), value
        return binding


def install_geotoken(transformer: nn.Module) -> GeoTokenConditioner:
    existing = getattr(transformer, "geotoken", None)
    if existing is not None: return existing
    for parameter in transformer.parameters(): parameter.requires_grad_(False)
    module = GeoTokenConditioner(int(transformer.inner_dim)); transformer.add_module("geotoken", module); module.attach(transformer)
    for parameter in module.parameters(): parameter.requires_grad_(True)
    assert_geotoken_only_trainable(transformer); return module


def assert_geotoken_only_trainable(transformer):
    items = [(n, p) for n, p in transformer.named_parameters() if p.requires_grad]
    if not items or any("geotoken." not in n for n, _ in items): raise RuntimeError("only GeoToken parameters may train")
    return items


def camera_channels_from_cameras(c2w, intrinsics, source_center, scene_scale, height, width, *, device):
    poses = torch.as_tensor(c2w, dtype=torch.float32, device=device); k = torch.as_tensor(intrinsics, dtype=torch.float32, device=device)
    y, x = torch.meshgrid(torch.arange(height, device=device), torch.arange(width, device=device), indexing="ij")
    pix = torch.stack((x.float(), y.float(), torch.ones_like(x, dtype=torch.float32)), -1)
    ray_cam = torch.einsum("tij,hwj->thwi", torch.linalg.inv(k), pix); ray_cam = ray_cam / torch.linalg.vector_norm(ray_cam, dim=-1, keepdim=True).clamp_min(1e-8)
    direction = torch.einsum("tij,thwj->thwi", poses[:, :3, :3], ray_cam)
    origin = (poses[:, :3, 3] - torch.as_tensor(source_center, dtype=torch.float32, device=device)) / float(scene_scale)
    moment = torch.cross(origin[:, None, None].expand_as(direction), direction, dim=-1)
    return torch.cat((direction, moment), -1)


def world_channels_from_cuda_render(xyz, depth, visibility, confidence, c2w, source_center, scene_scale):
    device = xyz.device; poses = torch.as_tensor(c2w, dtype=torch.float32, device=device); origin = poses[:, None, None, :3, 3]
    valid = visibility.bool() & torch.isfinite(xyz).all(-1) & torch.isfinite(depth) & (depth > 0) & torch.isfinite(confidence)
    safe = torch.where(valid[..., None], xyz, origin); direction = safe-origin; direction = direction / torch.linalg.vector_norm(direction, dim=-1, keepdim=True).clamp_min(1e-8)
    result = torch.zeros((*depth.shape, 7), dtype=torch.float32, device=device)
    result[..., :3] = ((safe - torch.as_tensor(source_center, dtype=torch.float32, device=device)) / float(scene_scale)).clamp(-64, 64)
    result[..., 3:6] = direction
    # Compact contract: XYZ, ray, log-depth, visibility, confidence.
    result[..., 6] = torch.log(torch.where(valid, depth / float(scene_scale), torch.ones_like(depth)).clamp_min(1e-6)).clamp(-16,16)
    return torch.cat((result[..., :7], valid[..., None].float(), torch.where(valid, confidence, torch.zeros_like(confidence))[..., None].clamp(0,1)), -1)
