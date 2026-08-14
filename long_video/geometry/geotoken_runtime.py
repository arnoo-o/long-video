"""Runtime alignment of point-world geometry with Helios current/history tokens."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

from ..data.camera import resize_intrinsics
from ..types import CameraBatch
from .geotoken import GeometryTokenBatch, geometry_channels_from_render
from .point_renderer import render


def _node(points_xyz, points_confidence):
    count = len(points_xyz)
    return SimpleNamespace(
        points_xyz=np.asarray(points_xyz, np.float32),
        points_rgb=np.zeros((count, 3), np.uint8),
        points_confidence=np.asarray(points_confidence, np.float32),
        points_source=np.zeros(count, np.int8),
        parent_point_count=None,
        quality_metrics={},
    )


def _pad_pool(value: torch.Tensor, kernel: tuple[int, int, int]) -> torch.Tensor:
    pt, ph, pw = kernel
    _, _, t, h, w = value.shape
    padding = (0, (pw - w % pw) % pw, 0, (ph - h % ph) % ph, 0, (pt - t % pt) % pt)
    value = F.pad(value, padding, mode="replicate")
    return F.avg_pool3d(value, kernel, kernel)


class PointWorldGeoTokenProvider:
    """Render one causal point source directly at every requested token grid."""

    def __init__(self, conditioner, *, device, source_center, scene_scale):
        self.conditioner = conditioner
        self.device = torch.device(device)
        self.source_center = np.asarray(source_center, np.float32)
        self.scene_scale = float(scene_scale)
        self.points_xyz = None
        self.points_confidence = None
        self.current_c2w = None
        self.current_k = None
        self.history_camera_chunks: list[tuple[np.ndarray, np.ndarray]] = []
        self._handle = None
        self._cache = {}

    def attach(self, transformer):
        if self._handle is None:
            self._handle = transformer.register_forward_pre_hook(self._pre_forward, with_kwargs=True)

    def configure_world(self, points_xyz, points_confidence):
        self.points_xyz = np.asarray(points_xyz, np.float32)
        self.points_confidence = np.asarray(points_confidence, np.float32)
        self._cache.clear()

    def configure_chunk(self, c2w, intrinsics, history_camera_chunks):
        self.current_c2w = np.asarray(c2w, np.float32)
        self.current_k = np.asarray(intrinsics, np.float32)
        self.history_camera_chunks = [
            (np.asarray(poses, np.float32), np.asarray(k, np.float32))
            for poses, k in history_camera_chunks
        ]
        self._cache.clear()

    def _render_channels(self, c2w, intrinsics, height, width):
        if self.points_xyz is None:
            raise RuntimeError("GeoToken provider has no available scene geometry")
        scaled = resize_intrinsics(intrinsics, (384, 640), (int(height), int(width)))
        cameras = CameraBatch(c2w, scaled, int(height), int(width))
        warp = render(
            _node(self.points_xyz, self.points_confidence), cameras,
            device=str(self.device), point_radius=0,
        )
        channels = geometry_channels_from_render(
            warp.winning_xyz_world, warp.depth, warp.visibility, warp.confidence,
            c2w, self.source_center, self.scene_scale,
        )
        return torch.from_numpy(channels).permute(3, 0, 1, 2).unsqueeze(0).to(self.device)

    def _encode_chunk(self, c2w, intrinsics, height, width):
        key = (id(c2w), int(height), int(width), torch.is_grad_enabled())
        if key not in self._cache:
            channels = self._render_channels(c2w, intrinsics, height, width)
            features, support = self.conditioner.tokenizer(channels)
            self._cache[key] = (features, support)
        return self._cache[key]

    def _history_slots(self, length, height, width):
        chunks = []
        for poses, intrinsics in self.history_camera_chunks:
            chunks.append(self._encode_chunk(poses, intrinsics, height, width))
        if not chunks:
            feature = torch.zeros(
                1, self.conditioner.tokenizer.inner_dim, length, height, width, device=self.device,
            )
            support = torch.zeros(1, 1, length, height, width, device=self.device)
            return feature, support
        feature = torch.cat([item[0] for item in chunks], dim=2)
        support = torch.cat([item[1] for item in chunks], dim=2)
        if feature.shape[2] < length:
            pad = length - feature.shape[2]
            feature = torch.cat([torch.zeros_like(feature[:, :, :1]).expand(-1, -1, pad, -1, -1), feature], 2)
            support = torch.cat([torch.zeros_like(support[:, :, :1]).expand(-1, -1, pad, -1, -1), support], 2)
        return feature[:, :, -length:], support[:, :, -length:]

    def _history_part(self, kwargs, name, kernel):
        latent = kwargs.get(f"latents_history_{name}")
        indices = kwargs.get(f"indices_latents_history_{name}")
        if latent is None or indices is None or latent.shape[2] == 0:
            return None
        kt, kh, kw = kernel
        out_h = (latent.shape[-2] + kh - 1) // kh
        out_w = (latent.shape[-1] + kw - 1) // kw
        feature, support = self._history_slots(int(latent.shape[2]), out_h, out_w)
        if kt > 1:
            numerator = _pad_pool(feature * support, (kt, 1, 1))
            denominator = _pad_pool(support, (kt, 1, 1))
            feature = numerator / denominator.clamp_min(1e-8)
            support = denominator
        tokens = feature.flatten(2).transpose(1, 2)
        weights = support.flatten(2).transpose(1, 2)
        mask = kwargs.get(f"history_visible_mask_{name}")
        attention = kwargs.get("attention_kwargs") or {}
        if mask is not None and str(attention.get("history_visible_token_mode", "drop")) == "drop":
            pooled = _pad_pool(mask.float(), kernel).flatten()
            keep = pooled >= float(attention.get("history_visible_token_threshold", 0.5))
            tokens, weights = tokens[:, keep], weights[:, keep]
        return GeometryTokenBatch(tokens, weights)

    def _build(self, kwargs):
        hidden = kwargs.get("hidden_states")
        if isinstance(hidden, list):
            raise RuntimeError("GeoToken runtime expects one real Helios stage per forward")
        patch = tuple(int(value) for value in getattr(kwargs.pop("_geotoken_patch_size", (1, 2, 2))))
        current_h = int(hidden.shape[-2]) // patch[1]
        current_w = int(hidden.shape[-1]) // patch[2]
        current_feature, current_support = self._encode_chunk(
            self.current_c2w, self.current_k, current_h, current_w,
        )
        current = GeometryTokenBatch(
            current_feature.flatten(2).transpose(1, 2),
            current_support.flatten(2).transpose(1, 2),
        )
        parts = []
        for name, kernel in (("long", (4, 8, 8)), ("mid", (2, 4, 4)), ("short", (1, 2, 2))):
            part = self._history_part(kwargs, name, kernel)
            if part is not None:
                parts.append(part)
        if parts:
            history_tokens = torch.cat([part.tokens for part in parts], 1)
            history_support = torch.cat([part.support for part in parts], 1)
        else:
            history_tokens = history_support = None
        return current, history_tokens, history_support

    def _pre_forward(self, module, args, kwargs):
        if self.current_c2w is None:
            self.conditioner.clear_active()
            return args, kwargs
        local = dict(kwargs)
        local["_geotoken_patch_size"] = tuple(module.config.patch_size)
        current, history, history_support = self._build(local)
        self.conditioner.set_active(
            current.tokens, current.support,
            None if history is None else history,
            history_support,
        )
        return args, kwargs
