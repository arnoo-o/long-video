"""Causal explicit point-world accumulation from frozen ReCal3R predictions."""
from __future__ import annotations

from dataclasses import replace
import numpy as np

from ..geometry.point_renderer import render
from ..types import CameraBatch
from ..types import ScaleMetadata


class ReCal3RWorldAccumulator:
    """Fuse each newly generated frame once; no candidate/promotion policy."""

    def __init__(self, backend, initial_node, *, trajectory_id, voxel_size=0.02):
        self.backend = backend
        self.initial_node = initial_node
        self.trajectory_id = str(trajectory_id)
        self.voxel_size = float(voxel_size)
        if self.voxel_size != 0.02:
            raise ValueError("ReCal3R world accumulation is fixed to voxel_size=0.02")
        self.reset()

    def reset(self):
        self.backend.reset()
        node = self.initial_node
        self._xyz = np.asarray(node.points_xyz, np.float32).copy()
        self._rgb = np.asarray(node.points_rgb, np.uint8).copy()
        self._observations = np.asarray(node.observation_count, np.int32).clip(1).copy()
        # Stored node confidence is an observation mean, not a total fusion
        # weight.  Preserve that meaning across accumulator reset/publish.
        self._weight = (np.asarray(node.points_confidence, np.float32).clip(1e-6)
                        * self._observations.astype(np.float32))
        self._seen_frame_ids = set()
        self._version = 0
        self._scale_anchor = None
        self._scale_uncertainty = 1.0
        self._node = replace(node)
        self._publish()

    def _publish(self):
        keys = np.floor(self._xyz / self.voxel_size).astype(np.int64)
        # State is already fused, but fusion remains explicit and deterministic.
        _, inverse = np.unique(keys, axis=0, return_inverse=True)
        count = int(inverse.max()) + 1 if len(inverse) else 0
        weight = np.bincount(inverse, weights=self._weight, minlength=count).astype(np.float32)
        xyz = np.zeros((count, 3), np.float32); rgb = np.zeros((count, 3), np.float32)
        np.add.at(xyz, inverse, self._xyz * self._weight[:, None])
        np.add.at(rgb, inverse, self._rgb.astype(np.float32) * self._weight[:, None])
        observations = np.bincount(inverse, weights=self._observations, minlength=count).astype(np.int32)
        xyz /= weight[:, None].clip(1e-8); rgb = np.clip(rgb / weight[:, None].clip(1e-8), 0, 255).astype(np.uint8)
        bmin = xyz.min(0) if len(xyz) else np.zeros(3, np.float32)
        bmax = xyz.max(0) if len(xyz) else np.zeros(3, np.float32)
        scale = ScaleMetadata(
            mode="relative" if self._scale_anchor is None else "metric_anchor",
            meters_per_world_unit=self._scale_anchor,
            uncertainty=float(self._scale_uncertainty),
            anchor_source="causal_camera_depth_overlap" if self._scale_anchor is not None else "causal_overlap_not_yet_available",
        )
        self._node.points_xyz = xyz; self._node.points_rgb = rgb
        self._node.points_confidence = np.clip(weight / np.maximum(observations, 1), 0, 1).astype(np.float32)
        self._node.points_source = np.full(len(xyz), 2, np.int8)
        self._node.observation_count = np.minimum(observations, np.iinfo(np.uint16).max).astype(np.uint16)
        self._node.bbox_min = bmin.astype(np.float32); self._node.bbox_max = bmax.astype(np.float32)
        self._node.coverage_radius = float(np.linalg.norm(bmax - bmin) * 0.5)
        self._node.scale = scale
        self._node.quality_metrics.update({"recal3r_world_version": self._version, "voxel_size": self.voxel_size,
                                           "accumulator_points": int(len(xyz)), "scale_uncertainty": float(self._scale_uncertainty),
                                           "fusion_weight_sum": float(weight.sum())})

    def _maybe_lock_scale_anchor(self, rgb, c2w, intrinsics, global_frame_index):
        """Use only current causal overlap, never a future/offline depth map."""
        if self._scale_anchor is not None or len(self._xyz) == 0:
            return
        height, width = map(int, np.asarray(rgb).shape[:2])
        # Rendering the existing world at the current observed camera yields
        # the causal overlap map at the exact ReCal input resolution.
        camera = CameraBatch(np.asarray(c2w, np.float32)[None], np.asarray(intrinsics, np.float32)[None], height, width)
        old = render(self._node, camera, device="cpu")
        old_depth = np.asarray(old.depth[0], np.float32)
        recal_depth = self.backend.raw_recal_depth(self.trajectory_id, global_frame_index)
        valid = (np.isfinite(old_depth) & (old_depth > 0) & np.isfinite(recal_depth) & (recal_depth > 0))
        if int(valid.sum()) < 256:
            return
        ratios = old_depth[valid] / recal_depth[valid]
        scale = float(np.median(ratios))
        mad = float(np.median(np.abs(ratios - scale)))
        if not np.isfinite(scale) or not 1e-4 < scale < 1e4:
            return
        self._scale_anchor = scale
        self._scale_uncertainty = float(min(1.0, mad / max(abs(scale), 1e-8)))
        self._node.quality_metrics.update({
            "recal3r_scale_anchor_frame": int(global_frame_index),
            "recal3r_scale_anchor_mad": mad,
            "recal3r_scale_anchor_overlap": int(valid.sum()),
        })

    def update_frame(self, rgb, c2w, intrinsics, global_frame_index):
        identity = (self.trajectory_id, int(global_frame_index))
        if identity in self._seen_frame_ids:
            raise RuntimeError(f"ReCal3R frame processed twice: {identity}")
        prediction = self.backend.update_frame(rgb, c2w, intrinsics, trajectory_id=self.trajectory_id,
                                               global_frame_index=int(global_frame_index))
        self._maybe_lock_scale_anchor(rgb, c2w, intrinsics, int(global_frame_index))
        xyz = np.asarray(prediction.point_maps[0], np.float32)
        confidence = np.asarray(prediction.geometry_confidence[0], np.float32)
        valid = np.isfinite(xyz).all(-1) & np.isfinite(confidence) & (confidence > 0)
        points, conf, colors = xyz[valid], confidence[valid], np.asarray(rgb, np.uint8)[valid]
        if len(points):
            self._xyz = np.concatenate([self._xyz, points])
            self._rgb = np.concatenate([self._rgb, colors])
            self._weight = np.concatenate([self._weight, conf])
            self._observations = np.concatenate([self._observations, np.ones(len(points), np.int16)])
        self._seen_frame_ids.add(identity); self._version += 1; self._publish()
        return self._node

    def get_point_world(self):
        return self._node
