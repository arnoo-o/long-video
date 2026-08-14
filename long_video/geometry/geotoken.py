"""Geometry-only conditioning tokens for frozen Helios transformers."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch
from torch import nn


GEOTOKEN_CHANNELS = 12
GEOTOKEN_BLOCKS = (8, 12, 16, 20, 24, 28)
TEMPORAL_GROUPS = ((0,),) + tuple(tuple(range(start, start + 4)) for start in range(1, 33, 4))


@dataclass
class GeometryTokenBatch:
    """Encoded geometry aligned with an already-patched Helios sequence."""

    tokens: torch.Tensor
    support: torch.Tensor

    def __post_init__(self):
        if self.tokens.ndim != 3:
            raise ValueError("geometry tokens must be [B,N,D]")
        if self.support.shape != self.tokens.shape[:2] + (1,):
            raise ValueError("geometry support must be [B,N,1]")


class GeometryTokenizer(nn.Module):
    """Encode 12-channel pixels and attention-pool 33 frames into 9 slots."""

    def __init__(self, inner_dim: int):
        super().__init__()
        self.inner_dim = int(inner_dim)
        if self.inner_dim <= 0:
            raise ValueError("inner_dim must be positive")
        self.frame_encoder = nn.Sequential(
            nn.Linear(GEOTOKEN_CHANNELS, 128),
            nn.SiLU(),
            nn.Linear(128, 256),
            nn.SiLU(),
        )
        self.temporal_score = nn.Linear(256, 1)
        self.geometry_projection = nn.Linear(256, self.inner_dim)

    def forward(self, geometry: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return `[B,D,9,H,W]` features and `[B,1,9,H,W]` support."""
        if geometry.ndim != 5 or geometry.shape[1] != GEOTOKEN_CHANNELS:
            raise ValueError(f"geometry must be [B,12,33,H,W], got {tuple(geometry.shape)}")
        if geometry.shape[2] != 33:
            raise ValueError("GeoToken requires the Helios 33-frame layout")
        x = geometry.permute(0, 2, 3, 4, 1).float()
        valid = x[..., 10:11].clamp(0, 1)
        confidence = x[..., 11:12].clamp(0, 1) * valid
        encoded = self.frame_encoder(x) * valid
        pooled, supports = [], []
        for group in TEMPORAL_GROUPS:
            index = torch.as_tensor(group, device=x.device)
            group_features = encoded.index_select(1, index)
            group_weights = confidence.index_select(1, index)
            logits = self.temporal_score(group_features).float()
            logits = logits.masked_fill(group_weights <= 0, -torch.inf)
            any_valid = group_weights.sum(dim=1) > 0
            safe_logits = torch.where(any_valid[:, None], logits, torch.zeros_like(logits))
            weights = torch.softmax(safe_logits, dim=1) * group_weights
            weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
            feature = (group_features.float() * weights).sum(dim=1)
            feature = torch.where(any_valid, feature, torch.zeros_like(feature))
            pooled.append(feature)
            supports.append(group_weights.mean(dim=1))
        pooled = torch.stack(pooled, dim=1)
        support = torch.stack(supports, dim=1).clamp(0, 1)
        projected = self.geometry_projection(pooled) * (support > 0)
        projected = projected.permute(0, 4, 1, 2, 3).contiguous()
        support = support.permute(0, 4, 1, 2, 3).contiguous()
        return projected, support


class GeoTokenConditioner(nn.Module):
    """Inject aligned geometry after patch embedding at selected blocks."""

    def __init__(self, inner_dim: int, block_indices: Iterable[int] = GEOTOKEN_BLOCKS):
        super().__init__()
        self.tokenizer = GeometryTokenizer(inner_dim)
        self.block_indices = tuple(int(value) for value in block_indices)
        if self.block_indices != GEOTOKEN_BLOCKS:
            raise ValueError(f"GeoToken block indices must be {GEOTOKEN_BLOCKS}")
        self.injection_gates = nn.ParameterDict({
            str(index): nn.Parameter(torch.zeros((), dtype=torch.float32))
            for index in self.block_indices
        })
        self._active: GeometryTokenBatch | None = None
        self._history: GeometryTokenBatch | None = None
        self._hook_handles = []

    def set_active(
        self, tokens: torch.Tensor, support: torch.Tensor,
        history_tokens: torch.Tensor | None = None, history_support: torch.Tensor | None = None,
    ) -> None:
        self._active = GeometryTokenBatch(tokens=tokens, support=support)
        self._history = (
            None if history_tokens is None
            else GeometryTokenBatch(tokens=history_tokens, support=history_support)
        )

    def clear_active(self) -> None:
        self._active = None
        self._history = None

    def attach(self, transformer: nn.Module) -> None:
        if self._hook_handles:
            return
        blocks = getattr(transformer, "blocks", None)
        if blocks is None:
            raise TypeError("Helios transformer must expose blocks")
        if max(self.block_indices) >= len(blocks):
            raise ValueError("GeoToken injection block is outside transformer depth")
        for index in self.block_indices:
            self._hook_handles.append(blocks[index].register_forward_pre_hook(
                self._make_hook(index), with_kwargs=True,
            ))

    def _make_hook(self, index: int):
        def hook(_module, args, kwargs):
            if self._active is None:
                return args, kwargs
            if not args:
                raise RuntimeError("Helios block did not receive hidden_states positionally")
            hidden = args[0]
            current = self._active
            current_length = int(args[4]) if len(args) > 4 else current.tokens.shape[1]
            if current.tokens.shape[1] != current_length:
                raise RuntimeError(
                    f"GeoToken current length mismatch: {current.tokens.shape[1]} != {current_length}"
                )
            history_length = hidden.shape[1] - current_length
            if history_length < 0:
                raise RuntimeError("Helios original_context_length exceeds sequence length")
            if history_length:
                if self._history is None or self._history.tokens.shape[1] != history_length:
                    raise RuntimeError(
                        "history geometry must align one-to-one with patched WAH history tokens: "
                        f"expected {history_length}, got "
                        f"{None if self._history is None else self._history.tokens.shape[1]}"
                    )
                tokens = torch.cat([self._history.tokens, current.tokens], dim=1)
                support = torch.cat([self._history.support, current.support], dim=1)
            else:
                tokens, support = current.tokens, current.support
            if tokens.shape != hidden.shape:
                raise RuntimeError(f"GeoToken/Helios sequence mismatch: {tokens.shape} != {hidden.shape}")
            update = (
                self.injection_gates[str(index)].to(dtype=hidden.dtype)
                * support.to(device=hidden.device, dtype=hidden.dtype)
                * tokens.to(device=hidden.device, dtype=hidden.dtype)
            )
            return (hidden + update, *args[1:]), kwargs
        return hook


def install_geotoken(transformer: nn.Module) -> GeoTokenConditioner:
    """Register GeoToken on a transformer and freeze every non-GeoToken parameter."""
    existing = getattr(transformer, "geotoken", None)
    if existing is not None:
        return existing
    inner_dim = int(getattr(transformer, "inner_dim"))
    for parameter in transformer.parameters():
        parameter.requires_grad_(False)
    module = GeoTokenConditioner(inner_dim)
    transformer.add_module("geotoken", module)
    module.attach(transformer)
    for parameter in module.parameters():
        parameter.requires_grad_(True)
    assert_geotoken_only_trainable(transformer)
    return module


def assert_geotoken_only_trainable(transformer: nn.Module) -> list[tuple[str, nn.Parameter]]:
    items = [(name, value) for name, value in transformer.named_parameters() if value.requires_grad]
    if not items or any("geotoken." not in name for name, _ in items):
        raise RuntimeError(f"only GeoToken parameters may train: {[name for name, _ in items]}")
    return items


def flatten_geometry_tokens(features: torch.Tensor, support: torch.Tensor) -> GeometryTokenBatch:
    """Flatten `[B,D,T,H,W]` in the same order as Helios patch tokens."""
    tokens = features.flatten(2).transpose(1, 2)
    weights = support.flatten(2).transpose(1, 2)
    return GeometryTokenBatch(tokens=tokens, support=weights)


def geometry_channels_from_render(
    winning_xyz_world, depth, visibility, confidence, c2w, source_center, scene_scale,
) -> np.ndarray:
    """Build the geometry-only 12-channel input from final z-buffer winners."""
    xyz = np.asarray(winning_xyz_world, np.float32)
    depth = np.asarray(depth, np.float32)
    valid = (
        np.asarray(visibility, bool)
        & np.isfinite(depth)
        & (depth > 0)
        & np.isfinite(xyz).all(axis=-1)
    )
    confidence = np.asarray(confidence, np.float32)
    valid &= np.isfinite(confidence)
    poses = np.asarray(c2w, np.float32)
    c0 = np.asarray(source_center, np.float32).reshape(3)
    scale = float(scene_scale)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("trajectory scene_scale must be finite and positive")
    if xyz.shape[:3] != valid.shape or xyz.shape[-1] != 3:
        raise ValueError("winning XYZ and visibility shapes do not align")
    if poses.shape != (xyz.shape[0], 4, 4):
        raise ValueError("one target camera is required for every geometry frame")
    h, w = valid.shape[-2:]
    yy, xx = np.meshgrid(np.arange(h, dtype=np.float32), np.arange(w, dtype=np.float32), indexing="ij")
    output = np.zeros((len(poses), h, w, GEOTOKEN_CHANNELS), np.float32)
    for frame, pose in enumerate(poses):
        origin = pose[:3, 3]
        direction = xyz[frame] - origin
        norm = np.linalg.norm(direction, axis=-1, keepdims=True)
        direction = direction / np.maximum(norm, 1e-8)
        output[frame, ..., 0:3] = (xyz[frame] - c0) / scale
        output[frame, ..., 3:6] = (origin - c0) / scale
        output[frame, ..., 6:9] = direction
        output[frame, ..., 9] = np.log(np.maximum(depth[frame] / scale, 1e-6))
        output[frame, ..., 10] = valid[frame]
        output[frame, ..., 11] = np.clip(confidence[frame], 0, 1) * valid[frame]
    output *= valid[..., None]
    return output


def source_scene_scale(depth, visibility) -> float:
    depth = np.asarray(depth, np.float32)
    valid = np.asarray(visibility, bool) & np.isfinite(depth) & (depth > 0)
    values = depth[valid]
    if not len(values):
        raise ValueError("source view has no valid geometry depth")
    return float(np.median(values))
