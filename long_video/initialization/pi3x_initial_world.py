"""Construct a source-only Pi3X W0 with canonical PointWorld voxel fusion."""
from __future__ import annotations

import numpy as np
from ..geometry.voxel_fusion import fuse_voxels
from ..types import ScaleMetadata, SpatialNode


def build_pi3x_source_world(rgb, c2w, intrinsics, backend, *, node_id="node_000", voxel_size=0.02):
    prediction = backend.predict_source(rgb, c2w, intrinsics)
    colors = prediction.diagnostics.pop("source_rgb_resized")
    # Preserve the exact v3 W0 numeric semantics first: source RGB is a
    # confidence-weighted voxel average, just like XYZ.
    xyz, colors, confidence, observations, _ = fuse_voxels(
        prediction.point_maps[0].reshape(-1, 3), colors.reshape(-1, 3),
        prediction.geometry_confidence[0].reshape(-1), voxel_size=voxel_size,
        rgb_mode="weighted",
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
    # Lock only the already-fused voxel colors. ReCal observations can never
    # replace these source anchors, while old Pi3X cache values remain exact.
    node.appearance_anchors = {
        "anchor_rgb": colors.copy(),
        "anchor_confidence": confidence.copy(),
        "anchor_frame": np.zeros(len(xyz), np.int32),
        "source_locked": np.ones(len(xyz), bool),
    }
    return node
