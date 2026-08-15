"""Runtime alignment of point-world geometry with Helios current/history tokens."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import torch
import torch.nn.functional as F

from ..data.camera import resize_intrinsics
from ..types import CameraBatch
from .geotoken import GeometryTokenBatch, geometry_channels_from_cuda_render
from .point_renderer import render_geometry_cuda


def _pad_pool(value: torch.Tensor, kernel: tuple[int, int, int]) -> torch.Tensor:
    pt, ph, pw = kernel
    _, _, t, h, w = value.shape
    padding = (0, (pw - w % pw) % pw, 0, (ph - h % ph) % ph, 0, (pt - t % pt) % pt)
    value = F.pad(value, padding, mode="replicate")
    return F.avg_pool3d(value, kernel, kernel)


@dataclass(frozen=True)
class FrozenGeometrySnapshot:
    """Detached geometry rendered by the active world for one generated chunk."""

    chunk_index: int
    frame_start: int
    world_version: object
    c2w: np.ndarray
    source_center: np.ndarray
    scene_scale: float
    grids: dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]


class PointWorldGeoTokenProvider:
    """Render one causal point source directly at every requested token grid."""

    def __init__(self, conditioner, *, device, source_center, scene_scale):
        self.conditioner = conditioner
        self.device = torch.device(device)
        self.source_center = np.asarray(source_center, np.float32)
        self.scene_scale = float(scene_scale)
        self.points_xyz: torch.Tensor | None = None
        self.points_confidence: torch.Tensor | None = None
        self.parent_point_count = None
        self.world_version = None
        self.current_c2w = None
        self.current_k = None
        self.history_snapshots: list[FrozenGeometrySnapshot] = []
        self._handle = None
        self._render_cache = {}
        self._feature_cache = {}
        self._render_events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []

    def attach(self, transformer):
        if self._handle is None:
            self._handle = transformer.register_forward_pre_hook(self._pre_forward, with_kwargs=True)

    def configure_world(self, points_xyz, points_confidence, *, world_version=None, parent_point_count=None):
        """Upload a point-world once; invalidate only on world-version changes."""
        version = world_version if world_version is not None else id(points_xyz)
        if version == self.world_version:
            return
        self.points_xyz = torch.as_tensor(points_xyz, dtype=torch.float32, device=self.device).contiguous()
        self.points_confidence = torch.as_tensor(points_confidence, dtype=torch.float32, device=self.device).contiguous()
        self.parent_point_count = parent_point_count
        self.world_version = version
        self.clear_world_cache()

    def configure_active_node(self, node):
        version = getattr(node, "node_id", id(node))
        self.configure_world(
            node.points_xyz, node.points_confidence, world_version=version,
            parent_point_count=getattr(node, "parent_point_count", None),
        )
        return {"world_version": version}

    def clear_world_cache(self):
        self._render_cache.clear()
        self._feature_cache.clear()

    def clear_feature_cache(self):
        """Release autograd graphs after one stage backward, not raw z-buffers."""
        self._feature_cache.clear()

    def configure_chunk(self, c2w, intrinsics, history_snapshots=()):
        self.current_c2w = np.asarray(c2w, np.float32)
        self.current_k = np.asarray(intrinsics, np.float32)
        self.history_snapshots = list(history_snapshots)
        # Camera changes invalidate only its keyed render entries; retaining
        # matching history/current camera renders avoids rerasterizing a world.

    def _render_channels(self, c2w, intrinsics, height, width):
        if self.points_xyz is None:
            raise RuntimeError("GeoToken provider has no available scene geometry")
        scaled = resize_intrinsics(intrinsics, (384, 640), (int(height), int(width)))
        cameras = CameraBatch(c2w, scaled, int(height), int(width))
        cache_key = (self.world_version, id(c2w), id(intrinsics), int(height), int(width))
        if cache_key not in self._render_cache:
            started = torch.cuda.Event(enable_timing=True)
            finished = torch.cuda.Event(enable_timing=True)
            with torch.cuda.device(self.device):
                started.record()
                self._render_cache[cache_key] = render_geometry_cuda(
                    self.points_xyz, self.points_confidence, cameras,
                    parent_point_count=self.parent_point_count,
                )
                finished.record()
            self._render_events.append((started, finished))
        xyz, depth, visibility, confidence = self._render_cache[cache_key]
        channels = geometry_channels_from_cuda_render(
            xyz, depth, visibility, confidence, c2w, self.source_center, self.scene_scale,
        )
        return channels.permute(3, 0, 1, 2).unsqueeze(0)

    def _encode_chunk(self, c2w, intrinsics, height, width):
        key = (self.world_version, id(c2w), id(intrinsics), int(height), int(width), torch.is_grad_enabled())
        if key not in self._feature_cache:
            channels = self._render_channels(c2w, intrinsics, height, width)
            features, support = self.conditioner.tokenizer(channels)
            self._feature_cache[key] = (features, support)
        return self._feature_cache[key]

    def point_render_seconds(self):
        if not self._render_events:
            return 0.0
        torch.cuda.synchronize(self.device)
        return sum(start.elapsed_time(end) for start, end in self._render_events) / 1000.0

    def freeze_current_snapshot(self, *, chunk_index, frame_start, **_ignored):
        """Freeze actual current-world renders after the formal Helios call."""
        grids = {}
        for key, value in self._render_cache.items():
            version, c2w_id, _k_id, height, width = key
            if version == self.world_version and c2w_id == id(self.current_c2w):
                grids[(int(height), int(width))] = tuple(item.detach().clone() for item in value)
        if not grids:
            raise RuntimeError("GeoToken current chunk produced no geometry render to freeze")
        return FrozenGeometrySnapshot(
            chunk_index=int(chunk_index), frame_start=int(frame_start), world_version=self.world_version,
            c2w=self.current_c2w.copy(),
            source_center=self.source_center.copy(), scene_scale=self.scene_scale, grids=grids,
        )

    def _encode_snapshot(self, snapshot, height, width):
        if snapshot.scene_scale != self.scene_scale or not np.array_equal(snapshot.source_center, self.source_center):
            raise RuntimeError("history snapshot normalization does not match this trajectory")
        grid = snapshot.grids.get((int(height), int(width)))
        if grid is None:
            raise RuntimeError(
                f"frozen geometry snapshot {snapshot.chunk_index} lacks requested token grid {(height, width)}"
            )
        key = ("snapshot", snapshot.chunk_index, id(snapshot), int(height), int(width), torch.is_grad_enabled())
        if key not in self._feature_cache:
            xyz, depth, visibility, confidence = grid
            channels = geometry_channels_from_cuda_render(
                xyz, depth, visibility, confidence, snapshot.c2w,
                snapshot.source_center, snapshot.scene_scale,
            ).permute(3, 0, 1, 2).unsqueeze(0)
            self._feature_cache[key] = self.conditioner.tokenizer(channels)
        return self._feature_cache[key]

    def _history_by_indices(self, indices, height, width):
        """Select frozen 9-slot chunks by the exact official WAH indices.

        WAH short mode uses a source-prefix slot followed by previous
        long/mid/short slots.  ``indices`` is the authoritative layout: this
        routine never derives history order from the current world or a simple
        trailing slice.
        """
        if indices is None:
            raise RuntimeError("WAH history latent was supplied without temporal indices")
        flat = torch.as_tensor(indices).reshape(-1).detach().cpu().tolist()
        slots = {}
        for snapshot in self.history_snapshots:
            feature, support = self._encode_snapshot(snapshot, height, width)
            # The 33-frame VAE layout is [frame0, frames1..4, ..., frames29..32].
            # Chunks advance by 32 RGB frames, i.e. eight latent slots.
            base = int(snapshot.frame_start) // 4
            for local_slot in range(feature.shape[2]):
                slots[base + local_slot] = (feature[:, :, local_slot:local_slot + 1],
                                            support[:, :, local_slot:local_slot + 1])
        feature_parts, support_parts = [], []
        zero_feature = torch.zeros(1, self.conditioner.tokenizer.inner_dim, 1, height, width, device=self.device)
        zero_support = torch.zeros(1, 1, 1, height, width, device=self.device)
        for index in flat:
            feature, support = slots.get(int(index), (zero_feature, zero_support))
            feature_parts.append(feature); support_parts.append(support)
        return torch.cat(feature_parts, 2), torch.cat(support_parts, 2)

    def _history_part(self, kwargs, name, kernel):
        latent = kwargs.get(f"latents_history_{name}")
        indices = kwargs.get(f"indices_latents_history_{name}")
        if latent is None or indices is None or latent.shape[2] == 0:
            return None
        kt, kh, kw = kernel
        out_h = (latent.shape[-2] + kh - 1) // kh
        out_w = (latent.shape[-1] + kw - 1) // kw
        feature, support = self._history_by_indices(indices, out_h, out_w)
        if feature.shape[2] != latent.shape[2]:
            raise RuntimeError("GeoToken temporal history must match official WAH history slots")
        if kt > 1:
            numerator = _pad_pool(feature * support, (kt, 1, 1))
            denominator = _pad_pool(support, (kt, 1, 1))
            feature = numerator / denominator.clamp_min(1e-8)
            support = denominator
        tokens = feature.flatten(2).transpose(1, 2)
        weights = support.flatten(2).transpose(1, 2)
        expected_raw_tokens = int(latent.shape[2] * out_h * out_w)
        if tokens.shape[1] != expected_raw_tokens:
            raise RuntimeError(f"GeoToken raw {name} token count mismatch with WAH: {tokens.shape[1]} != {expected_raw_tokens}")
        mask = kwargs.get(f"history_visible_mask_{name}")
        attention = kwargs.get("attention_kwargs") or {}
        if mask is not None and str(attention.get("history_visible_token_mode", "drop")) == "drop":
            pooled = _pad_pool(mask.float(), kernel).flatten()
            keep = pooled >= float(attention.get("history_visible_token_threshold", 0.5))
            tokens, weights = tokens[:, keep], weights[:, keep]
        # The same visible-token filtering is applied to both appearance and
        # geometry, leaving every retained WAH token with one geometry token.
        if tokens.shape != weights.expand_as(tokens).shape:
            raise RuntimeError("GeoToken history support does not align with history tokens")
        return GeometryTokenBatch(tokens, weights)

    def _build(self, kwargs):
        hidden = kwargs.get("hidden_states")
        if isinstance(hidden, list):
            raise RuntimeError("GeoToken runtime expects one real Helios stage per forward")
        patch = tuple(int(value) for value in kwargs.pop("_geotoken_patch_size", (1, 2, 2)))
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


def source_scene_scale_from_active_node(node, source_c2w, intrinsics, *, device, height=384, width=640):
    """Median visible source-camera depth for causal ReCal3R/world normalization.

    This is intentionally independent of ReCal3R and is shared by Phase C and
    formal GeoToken inference callers.
    """
    cameras = CameraBatch(
        np.asarray(source_c2w, np.float32)[None], np.asarray(intrinsics, np.float32)[None],
        int(height), int(width),
    )
    xyz = torch.as_tensor(node.points_xyz, dtype=torch.float32, device=device)
    confidence = torch.as_tensor(node.points_confidence, dtype=torch.float32, device=device)
    _, depth, visibility, _ = render_geometry_cuda(
        xyz, confidence, cameras,
        parent_point_count=getattr(node, "parent_point_count", None),
    )
    values = depth[visibility & torch.isfinite(depth) & (depth > 0)]
    if not len(values):
        raise RuntimeError("initial ReCal3R active node has no valid source-visible depth")
    scale = float(values.median().item())
    if not np.isfinite(scale) or scale <= 0:
        raise RuntimeError("initial ReCal3R source scene scale is invalid")
    return scale
