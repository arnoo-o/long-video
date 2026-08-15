"""Construct a source-only Pi3X W0 with canonical PointWorld voxel fusion."""
from __future__ import annotations

import numpy as np
from ..geometry.voxel_fusion import fuse_voxels
from ..types import ScaleMetadata, SpatialNode


def build_pi3x_source_world(rgb, c2w, intrinsics, backend, *, node_id="node_000", voxel_size=0.02):
    prediction = backend.predict_source(rgb, c2w, intrinsics)
    colors = prediction.diagnostics.pop("source_rgb_resized")
    xyz, colors, confidence, observations, _ = fuse_voxels(
        prediction.point_maps[0].reshape(-1, 3), colors.reshape(-1, 3),
        prediction.geometry_confidence[0].reshape(-1), voxel_size=voxel_size,
    )
    if not len(xyz):
        raise RuntimeError("Pi3X source reconstruction contains no valid points")
    depth = prediction.depth.astype(np.float32)
    node = SpatialNode(node_id=node_id, status="active", parent_id=None, center_c2w=np.asarray(c2w, np.float32),
        created_frame=0, coverage_radius=float(np.linalg.norm(xyz.max(0)-xyz.min(0))*0.5),
        bbox_min=xyz.min(0), bbox_max=xyz.max(0), view_rgb=np.asarray(rgb, np.uint8)[None], view_depth=depth,
        view_c2w=np.asarray(c2w, np.float32)[None], view_intrinsics=np.asarray(intrinsics, np.float32)[None],
        points_xyz=xyz, points_rgb=colors, points_confidence=confidence, points_source=np.zeros(len(xyz), np.int8),
        observation_count=observations, depth_convention=prediction.depth_convention, schema_version=5,
        quality_metrics={"initialization_mode": "pi3x_source_only", "voxel_size": 0.02,
                         "pi3x_diagnostics": prediction.diagnostics, "uses_future_or_past": False},
        scale=ScaleMetadata(mode="relative", meters_per_world_unit=None, uncertainty=1.0,
                            anchor_source="pi3x_source_only"))
    return node
