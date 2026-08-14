"""Shared geometry backend contracts independent of a specific model."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ..geometry.backprojection import backproject_ray_distance, backproject_z_depth
from ..types import RAY_DISTANCE, Z_DEPTH


@dataclass
class GeometryPrediction:
    depth: np.ndarray
    depth_confidence: np.ndarray
    point_maps: Optional[np.ndarray] = None
    predicted_c2w: Optional[np.ndarray] = None
    geometry_confidence: Optional[np.ndarray] = None
    scale_info: dict = field(default_factory=dict)
    diagnostics: dict = field(default_factory=dict)
    depth_convention: str = Z_DEPTH


class MultiViewGeometryBackend:
    def predict(self, view_rgb, view_c2w, intrinsics, known_depth=None, known_mask=None,
                known_depth_convention=None, known_scale=None):
        raise NotImplementedError


class GroundTruthGeometryBackend(MultiViewGeometryBackend):
    def predict(self, view_rgb, view_c2w, intrinsics, known_depth=None, known_mask=None,
                known_depth_convention=None, known_scale=None):
        if known_depth is None:
            raise ValueError("GroundTruthGeometryBackend requires known_depth")
        if known_depth_convention not in (RAY_DISTANCE, Z_DEPTH):
            raise ValueError("known_depth_convention must be RAY_DISTANCE or Z_DEPTH")
        depth = np.asarray(known_depth, np.float32).copy()
        valid = np.isfinite(depth) & (depth > 0)
        if known_mask is not None:
            valid &= np.asarray(known_mask, bool)
        depth[~valid] = np.nan
        point_maps = []
        for index in range(len(depth)):
            local = (backproject_z_depth if known_depth_convention == Z_DEPTH else backproject_ray_distance)(
                depth[index], intrinsics[index]
            )
            pose = np.asarray(view_c2w[index], np.float32)
            point_maps.append(local @ pose[:3, :3].T + pose[:3, 3])
        confidence = valid.astype(np.float32)
        return GeometryPrediction(
            depth=depth, depth_confidence=confidence, point_maps=np.stack(point_maps).astype(np.float32),
            geometry_confidence=confidence,
            scale_info={"mode": "dataset_calibrated", "meters_per_world_unit": 1.0,
                        "uncertainty": 0.0, "anchor_source": "ground_truth_depth"},
            diagnostics={"backend": "ground_truth", "valid_ratio": float(valid.mean())},
            depth_convention=known_depth_convention,
        )
