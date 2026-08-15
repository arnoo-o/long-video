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
        # Stored node confidence is an observation mean, not a total fusion
        # weight.  Preserve that meaning across accumulator reset/publish.
        self._weight = (np.asarray(node.points_confidence, np.float32).clip(1e-6)
                        * self._observations.astype(np.float32))
        self._seen_frame_ids = set()
        self._replay_rgb = [np.asarray(node.view_rgb[0], np.uint8).copy()]
        self._replay_c2w = [np.asarray(node.view_c2w[0], np.float32).copy()]
        self._replay_k = [np.asarray(node.view_intrinsics[0], np.float32).copy()]
        self._version = 0
        self._alignment = None
        self._node = replace(node)
        self._publish()

    def _publish(self):
        # Convert internal total weights back to means so Pi3X and ReCal3R
        # invoke the exact same fusion primitive.
        means = self._weight / self._observations.clip(1)
        xyz, rgb, confidence, observations, _ = fuse_voxels(
            self._xyz, self._rgb, means, self._observations, self.voxel_size,
        )
        weight = confidence * observations.astype(np.float32)
        bmin = xyz.min(0) if len(xyz) else np.zeros(3, np.float32)
        bmax = xyz.max(0) if len(xyz) else np.zeros(3, np.float32)
        scale = ScaleMetadata(mode="relative", meters_per_world_unit=None, uncertainty=1.0,
                              anchor_source="pi3x_w0_recal_relative_alignment")
        self._node.points_xyz = xyz; self._node.points_rgb = rgb
        self._node.points_confidence = confidence
        self._node.points_source = np.full(len(xyz), 2, np.int8)
        self._node.observation_count = np.minimum(observations, np.iinfo(np.uint16).max).astype(np.uint16)
        self._node.bbox_min = bmin.astype(np.float32); self._node.bbox_max = bmax.astype(np.float32)
        self._node.coverage_radius = float(np.linalg.norm(bmax - bmin) * 0.5)
        self._node.scale = scale
        self._node.quality_metrics.update({"recal3r_world_version": self._version, "voxel_size": self.voxel_size,
                                           "accumulator_points": int(len(xyz)),
                                           "fusion_weight_sum": float(weight.sum())})


    def update_frame(self, rgb, c2w, intrinsics, global_frame_index):
        return self.update_chunk(np.asarray(rgb)[None], np.asarray(c2w)[None], np.asarray(intrinsics)[None], [global_frame_index])

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
        replay = self.backend.replay_prefix(np.stack(self._replay_rgb), np.stack(self._replay_c2w), np.stack(self._replay_k),
                                            trajectory_id=self.trajectory_id, global_frame_indices=range(len(self._replay_rgb)))
        # source prediction initializes ReCal only; only this newly generated
        # tail may enter W0, and it may do so exactly once globally.
        predictions = replay[-len(indices):]
        point_parts=[]; confidence_parts=[]; color_parts=[]
        for image, pose, k, index, prediction in zip(rgb, c2w, intrinsics, indices, predictions):
            xyz = np.asarray(prediction.point_maps[0], np.float32)
            confidence = np.asarray(prediction.geometry_confidence[0], np.float32)
            valid = np.isfinite(xyz).all(-1) & np.isfinite(confidence) & (confidence > 0)
            point_parts.append(xyz[valid]); confidence_parts.append(confidence[valid]); color_parts.append(np.asarray(image, np.uint8)[valid])
        points = np.concatenate(point_parts) if point_parts else np.empty((0,3),np.float32)
        conf = np.concatenate(confidence_parts) if confidence_parts else np.empty(0,np.float32)
        colors = np.concatenate(color_parts) if color_parts else np.empty((0,3),np.uint8)
        if len(points):
            self._xyz = np.concatenate([self._xyz, points]); self._rgb = np.concatenate([self._rgb, colors])
            self._weight = np.concatenate([self._weight, conf]); self._observations = np.concatenate([self._observations, np.ones(len(points), np.int32)])
        self._seen_frame_ids.update(identities); self._version += 1; self._publish()
        return self._node

    def _legacy_update_frame(self, rgb, c2w, intrinsics, global_frame_index):
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
