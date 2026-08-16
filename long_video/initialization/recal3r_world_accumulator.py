"""Causal explicit point-world accumulation from frozen ReCal3R predictions."""
from __future__ import annotations

from dataclasses import replace
import numpy as np

from ..geometry.voxel_fusion import fuse_voxels
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
        self._weight = (
            np.asarray(node.points_confidence, np.float32).clip(1e-6)
            * self._observations.astype(np.float32)
        )
        anchors = getattr(node, "appearance_anchors", {})
        self._anchor_confidence = np.asarray(
            anchors.get("anchor_confidence", node.points_confidence), np.float32
        ).copy()
        self._anchor_frame = np.asarray(
            anchors.get("anchor_frame", np.zeros(len(node.points_xyz))), np.int32
        ).copy()
        self._source_locked = np.asarray(
            anchors.get("source_locked", np.ones(len(node.points_xyz), bool)), bool
        ).copy()
        self._seen_frame_ids = set()
        self._fused_frame_ids = set()
        self._pending_frame_ids = set()
        self._replay_rgb = [np.asarray(node.view_rgb[0], np.uint8).copy()]
        self._replay_c2w = [np.asarray(node.view_c2w[0], np.float32).copy()]
        self._replay_k = [np.asarray(node.view_intrinsics[0], np.float32).copy()]
        self._version = 0
        self.last_update_metrics = {
            "world_version": self._version,
            "frames_submitted": 0,
            "frames_fused": 0,
            "frames_pending": 0,
            "fused_observation_points": 0,
        }
        self._node = replace(node)
        self._publish()

    def _publish(self):
        means = self._weight / self._observations.clip(1)
        xyz, rgb, confidence, observations, _, anchors = fuse_voxels(
            self._xyz,
            self._rgb,
            means,
            self._observations,
            self.voxel_size,
            anchor_confidence=self._anchor_confidence,
            anchor_frame=self._anchor_frame,
            source_locked=self._source_locked,
            return_anchors=True,
        )
        weight = confidence * observations.astype(np.float32)
        bmin = xyz.min(0) if len(xyz) else np.zeros(3, np.float32)
        bmax = xyz.max(0) if len(xyz) else np.zeros(3, np.float32)
        scale = ScaleMetadata(
            mode="relative",
            meters_per_world_unit=None,
            uncertainty=1.0,
            anchor_source="pi3x_w0_source_geometry_commanded_pose",
        )
        self._node.points_xyz = xyz
        self._node.points_rgb = rgb
        self._node.points_confidence = confidence
        self._node.points_source = np.full(len(xyz), 2, np.int8)
        self._node.observation_count = np.minimum(
            observations, np.iinfo(np.uint16).max
        ).astype(np.uint16)
        self._node.bbox_min = bmin.astype(np.float32)
        self._node.bbox_max = bmax.astype(np.float32)
        self._node.coverage_radius = float(np.linalg.norm(bmax - bmin) * 0.5)
        self._node.scale = scale
        self._node.appearance_anchors = anchors
        self._node.quality_metrics.update(
            {
                "recal3r_world_version": self._version,
                "voxel_size": self.voxel_size,
                "accumulator_points": int(len(xyz)),
                "fusion_weight_sum": float(weight.sum()),
            }
        )
        self._xyz, self._rgb = xyz.copy(), rgb.copy()
        self._observations = observations.astype(np.int32, copy=True)
        self._weight = confidence * self._observations.astype(np.float32)
        self._anchor_confidence = anchors["anchor_confidence"].copy()
        self._anchor_frame = anchors["anchor_frame"].copy()
        self._source_locked = anchors["source_locked"].copy()

    def _render_source_w0_depth(self):
        """Render immutable Pi3X W0 z-depth in the source camera."""
        rgb = np.asarray(self.initial_node.view_rgb[0], np.uint8)
        height, width = rgb.shape[:2]
        c2w = np.asarray(self.initial_node.view_c2w[0], np.float32)
        intrinsics = np.asarray(self.initial_node.view_intrinsics[0], np.float32)
        xyz = np.asarray(self.initial_node.points_xyz, np.float32)

        local = (xyz - c2w[:3, 3]) @ c2w[:3, :3]
        z = local[:, 2]
        uvw = local @ intrinsics.T
        uv = uvw[:, :2] / np.maximum(z[:, None], 1e-8)
        x = np.rint(uv[:, 0]).astype(np.int64)
        y = np.rint(uv[:, 1]).astype(np.int64)
        valid = (
            np.isfinite(local).all(1)
            & (z > 0.05)
            & (z < 100.0)
            & (x >= 0)
            & (x < width)
            & (y >= 0)
            & (y < height)
        )
        depth = np.full(height * width, np.inf, np.float32)
        np.minimum.at(depth, y[valid] * width + x[valid], z[valid])
        depth[~np.isfinite(depth)] = np.nan
        return depth.reshape(height, width)

    def update_frame(self, rgb, c2w, intrinsics, global_frame_index):
        return self.update_chunk(
            np.asarray(rgb)[None],
            np.asarray(c2w)[None],
            np.asarray(intrinsics)[None],
            [global_frame_index],
        )

    def update_chunk(self, rgb, c2w, intrinsics, global_frame_indices):
        """Fuse/publish once after ReCal3R has causally processed one chunk."""
        indices = [int(value) for value in global_frame_indices]
        identities = [(self.trajectory_id, index) for index in indices]
        if any(identity in self._seen_frame_ids for identity in identities):
            raise RuntimeError("ReCal3R frame processed twice")
        if not indices or indices[0] != len(self._replay_rgb):
            raise RuntimeError("ReCal chunk must append exactly after the previous unique global frame")

        self._replay_rgb.extend(np.asarray(image, np.uint8).copy() for image in rgb)
        self._replay_c2w.extend(np.asarray(pose, np.float32).copy() for pose in c2w)
        self._replay_k.extend(np.asarray(k, np.float32).copy() for k in intrinsics)

        before_point_count = int(len(self._node.points_xyz))
        before_observations = int(
            np.asarray(self._node.observation_count, np.int64).sum()
        )

        replay = self.backend.replay_prefix(
            np.stack(self._replay_rgb),
            np.stack(self._replay_c2w),
            np.stack(self._replay_k),
            trajectory_id=self.trajectory_id,
            global_frame_indices=range(len(self._replay_rgb)),
        )
        if self.backend.get_state().get("alignment", {}).get("status") != "locked":
            replay = self.backend.lock_source_geometry_alignment(
                self._render_source_w0_depth()
            )

        candidates = sorted(self._pending_frame_ids | set(indices))
        point_parts = []
        confidence_parts = []
        color_parts = []
        frame_parts = []
        fused_frames = []
        validation_keys = (
            "pixel_count",
            "inside_count",
            "finite_world_count",
            "finite_depth_count",
            "positive_depth_count",
            "finite_confidence_count",
            "confidence_threshold_count",
            "valid_count",
        )
        validation_totals = {key: 0 for key in validation_keys}
        raw_confidence_min = None
        raw_confidence_max = None
        raw_confidence_weighted_sum = 0.0
        raw_confidence_weight = 0

        for index in candidates:
            if index == 0 or index in self._fused_frame_ids:
                continue
            prediction = replay[index]
            image = self._replay_rgb[index]
            validation = self.backend.geometry_validation(self.trajectory_id, index)
            for key in validation_keys:
                validation_totals[key] += int(validation[key])
            if validation["raw_confidence_min"] is not None:
                raw_confidence_min = (
                    validation["raw_confidence_min"]
                    if raw_confidence_min is None
                    else min(raw_confidence_min, validation["raw_confidence_min"])
                )
                raw_confidence_max = (
                    validation["raw_confidence_max"]
                    if raw_confidence_max is None
                    else max(raw_confidence_max, validation["raw_confidence_max"])
                )
                raw_confidence_weighted_sum += (
                    validation["raw_confidence_mean"]
                    * validation["finite_confidence_count"]
                )
                raw_confidence_weight += validation["finite_confidence_count"]

            xyz = np.asarray(prediction.point_maps[0], np.float32)
            confidence = np.asarray(prediction.geometry_confidence[0], np.float32)
            valid = (
                np.isfinite(xyz).all(-1)
                & np.isfinite(confidence)
                & (confidence > 0)
            )
            if not bool(valid.any()):
                self._pending_frame_ids.add(index)
                continue

            point_parts.append(xyz[valid])
            confidence_parts.append(confidence[valid])
            color_parts.append(np.asarray(image, np.uint8)[valid])
            frame_parts.append(np.full(int(valid.sum()), index, np.int32))
            fused_frames.append(index)
            self._pending_frame_ids.discard(index)
            self._fused_frame_ids.add(index)

        points = (
            np.concatenate(point_parts) if point_parts else np.empty((0, 3), np.float32)
        )
        conf = (
            np.concatenate(confidence_parts)
            if confidence_parts
            else np.empty(0, np.float32)
        )
        colors = (
            np.concatenate(color_parts)
            if color_parts
            else np.empty((0, 3), np.uint8)
        )
        frames = (
            np.concatenate(frame_parts)
            if frame_parts
            else np.empty(0, np.int32)
        )

        if len(points):
            self._xyz = np.concatenate([self._xyz, points])
            self._rgb = np.concatenate([self._rgb, colors])
            self._weight = np.concatenate([self._weight, conf])
            self._observations = np.concatenate(
                [self._observations, np.ones(len(points), np.int32)]
            )
            self._anchor_confidence = np.concatenate(
                [self._anchor_confidence, conf]
            )
            self._anchor_frame = np.concatenate([self._anchor_frame, frames])
            self._source_locked = np.concatenate(
                [self._source_locked, np.zeros(len(points), bool)]
            )

        self._seen_frame_ids.update(identities)
        self._version += 1
        self._publish()
        after_observations = int(
            np.asarray(self._node.observation_count, np.int64).sum()
        )
        alignment = self.backend.get_state().get("alignment", {})
        self.last_update_metrics = {
            "world_version": int(self._version),
            "frames_submitted": int(len(indices)),
            "frames_fused": int(len(fused_frames)),
            "fused_frame_indices": fused_frames,
            "frames_pending": int(len(self._pending_frame_ids)),
            "pending_frame_indices": sorted(self._pending_frame_ids),
            "fused_observation_points": int(len(points)),
            "world_point_count_before": before_point_count,
            "world_point_count_after": int(len(self._node.points_xyz)),
            "observation_count_before": before_observations,
            "observation_count_after": after_observations,
            "alignment": alignment,
            "geometry_validation": {
                **validation_totals,
                "raw_confidence_min": raw_confidence_min,
                "raw_confidence_max": raw_confidence_max,
                "raw_confidence_mean": (
                    raw_confidence_weighted_sum / raw_confidence_weight
                    if raw_confidence_weight
                    else None
                ),
            },
        }
        return self._node

    def get_point_world(self):
        return self._node

    def debug_geometry_for_frames(self, global_frame_indices):
        """Expose the current full-prefix ReCal maps without retaining them."""
        return [
            self.backend.raw_recal_debug(self.trajectory_id, int(index))
            for index in global_frame_indices
        ]
