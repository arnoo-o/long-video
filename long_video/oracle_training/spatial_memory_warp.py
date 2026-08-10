"""Causal generated-boundary geometry for Phase-B spatial memory attention."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..geometry.backprojection import backproject
from ..geometry.point_renderer import render
from ..types import CameraBatch, SpatialNode, Z_DEPTH


def full_rotation_angle_degrees(left_c2w, right_c2w) -> float:
    left = np.asarray(left_c2w, np.float64)
    right = np.asarray(right_c2w, np.float64)
    relative = left[:3, :3].T @ right[:3, :3]
    cosine = np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


@dataclass
class SpatialMemoryWarpEntry:
    entry_id: int
    pose: np.ndarray
    intrinsics: np.ndarray
    points_xyz: np.ndarray
    points_rgb: np.ndarray
    points_confidence: np.ndarray
    points_source: np.ndarray
    frame_id: int
    chunk_id: int
    provenance: dict

    def as_node(self) -> SpatialNode:
        xyz = np.asarray(self.points_xyz, np.float32)
        if len(xyz):
            bbox_min, bbox_max = xyz.min(axis=0), xyz.max(axis=0)
        else:
            bbox_min = bbox_max = np.zeros(3, np.float32)
        rgb = np.asarray(self.points_rgb, np.uint8)
        confidence = np.asarray(self.points_confidence, np.float32)
        source = np.asarray(self.points_source, np.int8)
        return SpatialNode(
            node_id=f"spatial_memory_{self.entry_id:04d}", status="active",
            parent_id=None, center_c2w=np.asarray(self.pose, np.float32),
            created_frame=int(self.frame_id), coverage_radius=0.0,
            bbox_min=bbox_min, bbox_max=bbox_max,
            view_rgb=rgb[None], view_depth=np.empty((1, 0), np.float32),
            view_c2w=np.asarray(self.pose, np.float32)[None],
            view_intrinsics=np.asarray(self.intrinsics, np.float32)[None],
            points_xyz=xyz, points_rgb=rgb, points_confidence=confidence,
            points_source=source, observation_count=np.ones(len(xyz), np.int32),
            depth_convention=Z_DEPTH,
        )


class SpatialMemoryWarpBank:
    """Session-local point memories built only from generated RGB and causal M0 depth."""

    def __init__(self, translation_threshold=3.0, rotation_threshold_degrees=30.0):
        self.translation_threshold = float(translation_threshold)
        self.rotation_threshold_degrees = float(rotation_threshold_degrees)
        self.entries: list[SpatialMemoryWarpEntry] = []

    def query(self, pose):
        pose = np.asarray(pose, np.float32)
        candidates = []
        for entry in self.entries:
            if not len(entry.points_xyz):
                continue
            translation = float(np.linalg.norm(entry.pose[:3, 3] - pose[:3, 3]))
            rotation = full_rotation_angle_degrees(entry.pose, pose)
            if translation <= self.translation_threshold and rotation <= self.rotation_threshold_degrees:
                candidates.append((translation + rotation / 30.0, translation, rotation, entry))
        if not candidates:
            return None, {
                "memory_hit": False, "memory_entry_id": None,
                "memory_translation_distance": None, "memory_rotation_distance_degrees": None,
            }
        _, translation, rotation, entry = min(candidates, key=lambda item: item[0])
        return entry, {
            "memory_hit": True, "memory_entry_id": int(entry.entry_id),
            "memory_translation_distance": translation,
            "memory_rotation_distance_degrees": rotation,
        }

    def add_generated_boundary(
        self, *, rgb, depth, visibility, confidence, pose, intrinsics,
        frame_id, chunk_id, provenance,
    ):
        """Backproject generated RGB with same-time causal-M0 geometry only."""
        rgb = np.asarray(rgb)
        if rgb.dtype != np.uint8:
            rgb = np.rint(np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)
        depth = np.asarray(depth, np.float32).copy()
        valid = np.asarray(visibility, bool) & np.isfinite(depth) & (depth > 0)
        depth[~valid] = np.nan
        causal_confidence = np.asarray(confidence, np.float32) * valid.astype(np.float32)
        source = np.zeros(depth.shape, np.int8)
        xyz, colors, point_confidence, point_source = backproject(
            depth, rgb, pose, intrinsics, causal_confidence, source,
            depth_convention=Z_DEPTH,
        )
        entry = SpatialMemoryWarpEntry(
            entry_id=len(self.entries), pose=np.asarray(pose, np.float32).copy(),
            intrinsics=np.asarray(intrinsics, np.float32).copy(), points_xyz=xyz,
            points_rgb=np.asarray(colors, np.uint8),
            points_confidence=np.asarray(point_confidence, np.float32),
            points_source=np.asarray(point_source, np.int8), frame_id=int(frame_id),
            chunk_id=int(chunk_id), provenance={**dict(provenance or {}),
                "rgb_origin": "model_generated", "geometry_origin": "causal_M0_renderer",
                "uses_future_gt": False},
        )
        self.entries.append(entry)
        return entry

    def render_query(self, *, poses, intrinsics, height, width, device="cpu", **renderer_kwargs):
        entry, report = self.query(np.asarray(poses)[0])
        shape = (len(poses), int(height), int(width))
        if entry is None or not len(entry.points_xyz):
            return {
                "rgb": np.zeros((*shape, 3), np.uint8),
                "visibility": np.zeros(shape, bool),
                "confidence": np.zeros(shape, np.float32),
                "report": {**report, "uses_future_gt": False},
            }
        cameras = CameraBatch(
            np.asarray(poses, np.float32), np.asarray(intrinsics, np.float32),
            int(height), int(width),
        )
        warp = render(entry.as_node(), cameras, device=device, **renderer_kwargs)
        return {
            "rgb": np.rint(np.clip(warp.rgb, 0.0, 1.0) * 255.0).astype(np.uint8),
            "visibility": np.asarray(warp.visibility, bool),
            "confidence": np.asarray(warp.confidence, np.float32),
            "report": {**report, "uses_future_gt": False,
                "valid_pixel_count": int(np.asarray(warp.visibility).sum())},
        }

    def summary(self):
        return [{
            "entry_id": int(entry.entry_id), "frame_id": int(entry.frame_id),
            "chunk_id": int(entry.chunk_id), "point_count": int(len(entry.points_xyz)),
            "pose": entry.pose.tolist(), "intrinsics": entry.intrinsics.tolist(),
            "provenance": dict(entry.provenance),
        } for entry in self.entries]


__all__ = [
    "SpatialMemoryWarpBank", "SpatialMemoryWarpEntry", "full_rotation_angle_degrees",
]
