"""Runtime alignment of point-world geometry with Helios current/history tokens."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import numpy as np
import torch
import torch.nn.functional as F

from ..data.camera import resize_intrinsics
from ..types import CameraBatch
from .geotoken import (GeometryTokenBatch, camera_channels_from_cameras,
                       world_channels_from_cuda_render)
from .point_renderer import render_geometry_cuda
from .world_identity import point_world_snapshot_identity


STAGE_GRIDS = {(12, 20): 0, (24, 40): 1, (48, 80): 2}


def stage_for_grid(height: int, width: int) -> int:
    grid = (int(height), int(width))
    if grid not in STAGE_GRIDS:
        raise RuntimeError(f"unknown Helios GeoToken grid {grid}; stage inference is forbidden")
    return STAGE_GRIDS[grid]


def stage_for_hidden_states(hidden_states: torch.Tensor) -> int:
    return stage_for_grid(int(hidden_states.shape[-2]), int(hidden_states.shape[-1]))


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
    intrinsics: np.ndarray
    source_center: np.ndarray
    scene_scale: float
    grids: dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]


class PointWorldGeoTokenProvider:
    """Render one causal point source directly at every requested token grid."""

    def __init__(self, conditioner, *, device, source_center, scene_scale, render_height=384, render_width=640):
        self.conditioner = conditioner
        self.device = torch.device(device)
        self.source_center = np.asarray(source_center, np.float32)
        self.scene_scale = float(scene_scale)
        self.render_resolution=(int(render_height),int(render_width))
        self.points_xyz: torch.Tensor | None = None
        self.points_confidence: torch.Tensor | None = None
        self.parent_point_count = None
        self.world_version = None
        self.current_c2w = None
        self.current_k = None
        self.history_snapshots: list[FrozenGeometrySnapshot] = []
        self.history_window = []
        self.source_geometry = None
        self._handle = None
        self._render_cache = {}
        self._feature_cache = {}
        self._render_events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
        self.world_slot_dropout = 0.0
        self.timing_resolver = None
        self.timing_enabled = True

    def set_world_slot_dropout(self, probability: float):
        self.world_slot_dropout = float(probability)

    def set_timing_resolver(self, resolver):
        self.timing_resolver = resolver

    def set_timing_enabled(self, enabled: bool):
        self.timing_enabled = bool(enabled)
        if not self.timing_enabled:
            self._render_events.clear()

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
        version = point_world_snapshot_identity(node)
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

    def configure_chunk(self, c2w, intrinsics, history_snapshots=(), *, history_window=(), source_geometry=None):
        self.current_c2w = np.asarray(c2w, np.float32)
        self.current_k = np.asarray(intrinsics, np.float32)
        self.history_snapshots = list(history_snapshots)
        self.history_window = list(history_window)
        if source_geometry is not None:
            self.source_geometry = source_geometry
        # Camera changes invalidate only its keyed render entries; retaining
        # matching history/current camera renders avoids rerasterizing a world.

    def _render_inputs(self, c2w, intrinsics, height, width):
        if self.points_xyz is None:
            raise RuntimeError("GeoToken provider has no available scene geometry")
        scaled = resize_intrinsics(intrinsics, self.render_resolution, (int(height), int(width)))
        cameras = CameraBatch(c2w, scaled, int(height), int(width))
        camera_id = self._camera_identity(c2w, intrinsics)
        cache_key = (self.world_version, camera_id, int(height), int(width))
        if cache_key not in self._render_cache:
            if self.timing_enabled:
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
            else:
                self._render_cache[cache_key] = render_geometry_cuda(
                    self.points_xyz, self.points_confidence, cameras,
                    parent_point_count=self.parent_point_count,
                )
        xyz, depth, visibility, confidence = self._render_cache[cache_key]
        camera = camera_channels_from_cameras(
            c2w, scaled, self.source_center, self.scene_scale, int(height), int(width), device=self.device,
        )
        world = world_channels_from_cuda_render(
            xyz, depth, visibility, confidence, c2w, self.source_center, self.scene_scale,
        )
        return camera.permute(3, 0, 1, 2).unsqueeze(0), world.permute(3, 0, 1, 2).unsqueeze(0)

    def _encode_chunk(self, c2w, intrinsics, height, width):
        key = (self.world_version, self._camera_identity(c2w, intrinsics), int(height), int(width), torch.is_grad_enabled())
        if key not in self._feature_cache:
            camera, world = self._render_inputs(c2w, intrinsics, height, width)
            self._feature_cache[key] = self.conditioner.tokenizer(camera, world)
        return self._feature_cache[key]

    @staticmethod
    def _camera_identity(c2w, intrinsics):
        digest=hashlib.sha256()
        for value in (np.asarray(c2w,np.float32),np.asarray(intrinsics,np.float32)):
            value=np.ascontiguousarray(value); digest.update(np.asarray(value.shape,np.int64).tobytes()); digest.update(value.tobytes())
        return digest.hexdigest()

    def point_render_seconds(self):
        if not self._render_events:
            return 0.0
        torch.cuda.synchronize(self.device)
        return sum(start.elapsed_time(end) for start, end in self._render_events) / 1000.0

    def freeze_current_snapshot(self, *, chunk_index, frame_start, **_ignored):
        """Freeze actual current-world renders after the formal Helios call."""
        grids = {}
        for key, value in self._render_cache.items():
            version, camera_id, height, width = key
            if version == self.world_version and camera_id == self._camera_identity(self.current_c2w,self.current_k):
                grids[(int(height), int(width))] = tuple(item.detach().clone() for item in value)
        if not grids:
            raise RuntimeError("GeoToken current chunk produced no geometry render to freeze")
        return FrozenGeometrySnapshot(
            chunk_index=int(chunk_index), frame_start=int(frame_start), world_version=self.world_version,
            c2w=self.current_c2w.copy(),
            intrinsics=self.current_k.copy(),
            source_center=self.source_center.copy(), scene_scale=self.scene_scale, grids=grids,
        )

    def ensure_source_geometry(self, source_c2w, source_intrinsics):
        """Freeze W0 at source pose before the first transformer forward."""
        if self.source_geometry is not None:
            return self.source_geometry
        original_c2w, original_k = self.current_c2w, self.current_k
        self.current_c2w = np.repeat(np.asarray(source_c2w, np.float32)[None], 33, 0)
        self.current_k = np.repeat(np.asarray(source_intrinsics, np.float32)[None], 33, 0)
        # All pyramid token grids that may be requested by WAH history.
        for height, width in ((12, 20), (24, 40), (48, 80)):
            self._render_inputs(self.current_c2w, self.current_k, height, width)
        snapshot = self.freeze_current_snapshot(chunk_index=-1, frame_start=0)
        self.source_geometry = (snapshot, 0)
        self.current_c2w, self.current_k = original_c2w, original_k
        return self.source_geometry

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
            camera = camera_channels_from_cameras(snapshot.c2w, resize_intrinsics(snapshot.intrinsics, self.render_resolution, (int(height), int(width))), snapshot.source_center, snapshot.scene_scale, int(height), int(width), device=self.device)
            world = world_channels_from_cuda_render(xyz, depth, visibility, confidence, snapshot.c2w, snapshot.source_center, snapshot.scene_scale)
            self._feature_cache[key] = self.conditioner.tokenizer(camera.permute(3,0,1,2).unsqueeze(0), world.permute(3,0,1,2).unsqueeze(0))
        return self._feature_cache[key]

    def _slot_from_snapshot(self, item, height, width):
        snapshot, local_slot = item
        camera, world, support = self._encode_snapshot(snapshot, height, width)
        return camera[:, :, local_slot:local_slot + 1], world[:, :, local_slot:local_slot + 1], support[:, :, local_slot:local_slot + 1]

    def _history_by_indices(self, indices, height, width, current_feature, current_support):
        """Select frozen 9-slot chunks by the exact official WAH indices.

        WAH short mode uses a source-prefix slot followed by previous
        long/mid/short slots.  ``indices`` is the authoritative layout: this
        routine never derives history order from the current world or a simple
        trailing slice.
        """
        if indices is None:
            raise RuntimeError("WAH history latent was supplied without temporal indices")
        flat = torch.as_tensor(indices).reshape(-1).detach().cpu().tolist()
        camera_parts, world_parts, support_parts = [], [], []
        zero_feature = torch.zeros(1, 256, 1, height, width, device=self.device)
        zero_support = torch.zeros(1, 1, 1, height, width, device=self.device)
        for index in flat:
            index = int(index)
            if index == 0:
                # The source prefix is a persistent geometry slot. Before the
                # first chunk has frozen, its current slot-0 is the same
                # source view and is therefore the exact causal fallback.
                if self.source_geometry is None:
                    raise RuntimeError("source geometry snapshot must exist before first GeoToken forward")
                camera, world, support = self._slot_from_snapshot(self.source_geometry, height, width)
            elif 1 <= index <= 19 and index - 1 < len(self.history_window):
                camera, world, support = self._slot_from_snapshot(self.history_window[index - 1], height, width)
            elif index == 19 and not self.history_window:
                # Official WAH fake-short slot in chunk0 is derived from the
                # source prefix, so it is the same frozen source geometry.
                camera, world, support = self._slot_from_snapshot(self.source_geometry, height, width)
            elif 20 <= index <= 28:
                camera, world, support = (current_feature[0][:, :, index - 20:index - 19], current_feature[1][:, :, index - 20:index - 19], current_feature[2][:, :, index - 20:index - 19])
            else:
                camera, world, support = zero_feature, zero_feature, zero_support
            camera_parts.append(camera); world_parts.append(world); support_parts.append(support)
        return torch.cat(camera_parts, 2), torch.cat(world_parts, 2), torch.cat(support_parts, 2)

    def _history_part(self, kwargs, name, kernel, current_feature, current_support):
        latent = kwargs.get(f"latents_history_{name}")
        indices = kwargs.get(f"indices_latents_history_{name}")
        if latent is None or indices is None or latent.shape[2] == 0:
            return None
        kt, kh, kw = kernel
        out_h = (latent.shape[-2] + kh - 1) // kh
        out_w = (latent.shape[-1] + kw - 1) // kw
        # Resize the current live geometry only through the same stage grid
        # requested by Helios; previous slots stay frozen snapshots.
        if current_feature[0].shape[-2:] != (out_h, out_w):
            current_feature = self._encode_chunk(self.current_c2w, self.current_k, out_h, out_w)
            current_support = current_feature[2]
        camera, feature, support = self._history_by_indices(indices, out_h, out_w, current_feature, current_support)
        if feature.shape[2] != latent.shape[2]:
            raise RuntimeError("GeoToken temporal history must match official WAH history slots")
        if kt > 1:
            numerator = _pad_pool(feature * support, (kt, 1, 1))
            denominator = _pad_pool(support, (kt, 1, 1))
            feature = numerator / denominator.clamp_min(1e-8)
            support = denominator
            camera = _pad_pool(camera, (kt, 1, 1))
        camera_tokens = camera.flatten(2).transpose(1, 2)
        tokens = feature.flatten(2).transpose(1, 2)
        weights = support.flatten(2).transpose(1, 2)
        expected_tokens = int(
            ((int(latent.shape[2]) + kt - 1) // kt)
            * ((int(latent.shape[-2]) + kh - 1) // kh)
            * ((int(latent.shape[-1]) + kw - 1) // kw)
        )
        if tokens.shape[1] != expected_tokens:
            raise RuntimeError(f"GeoToken pooled {name} token count mismatch with WAH: {tokens.shape[1]} != {expected_tokens}")
        mask = kwargs.get(f"history_visible_mask_{name}")
        attention = kwargs.get("attention_kwargs") or {}
        if mask is not None and str(attention.get("history_visible_token_mode", "drop")) == "drop":
            pooled = _pad_pool(mask.float(), kernel).flatten()
            keep = pooled >= float(attention.get("history_visible_token_threshold", 0.5))
            camera_tokens, tokens, weights = camera_tokens[:, keep], tokens[:, keep], weights[:, keep]
        # The same visible-token filtering is applied to both appearance and
        # geometry, leaving every retained WAH token with one geometry token.
        if tokens.shape != weights.expand_as(tokens).shape:
            raise RuntimeError("GeoToken history support does not align with history tokens")
        return GeometryTokenBatch(camera_tokens, tokens, weights)

    def _build(self, kwargs):
        hidden = kwargs.get("hidden_states")
        if isinstance(hidden, list):
            raise RuntimeError("GeoToken runtime expects one real Helios stage per forward")
        # Helios hidden_states already use the latent grid. Transformer patch
        # projection happens inside the model and must not be applied twice.
        current_h = int(hidden.shape[-2])
        current_w = int(hidden.shape[-1])
        stage = stage_for_hidden_states(hidden)
        self.conditioner.set_timing(stage_index=stage, denoise_progress=self.conditioner.denoise_progress)
        current_feature = self._encode_chunk(
            self.current_c2w, self.current_k, current_h, current_w,
        )
        if self.world_slot_dropout and self.conditioner.training:
            drop = torch.rand((current_feature[2].shape[0], 1, current_feature[2].shape[2], 1, 1), device=self.device) < self.world_slot_dropout
            current_feature = (current_feature[0], current_feature[1], current_feature[2].masked_fill(drop, 0))
        current = GeometryTokenBatch(
            current_feature[0].flatten(2).transpose(1, 2), current_feature[1].flatten(2).transpose(1, 2),
            current_feature[2].flatten(2).transpose(1, 2),
        )
        parts = []
        for name, kernel in (("long", (4, 8, 8)), ("mid", (2, 4, 4)), ("short", (1, 2, 2))):
            part = self._history_part(kwargs, name, kernel, current_feature, current_feature[2])
            if part is not None:
                parts.append(part)
        if parts:
            history = GeometryTokenBatch(torch.cat([part.camera for part in parts],1), torch.cat([part.world for part in parts],1), torch.cat([part.world_support for part in parts],1))
        else:
            history = None
        return current, history

    def _pre_forward(self, module, args, kwargs):
        if self.current_c2w is None:
            self.conditioner.clear_active()
            return args, kwargs
        local = dict(kwargs)
        if self.timing_resolver is not None:
            self.conditioner.set_timing(
                stage_index=self.conditioner.stage_index,
                denoise_progress=float(self.timing_resolver(local)),
            )
        current, history = self._build(local)
        self.conditioner.set_active(current, history)
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
