"""Metric Oracle M0 built directly from one complete Holo360D ERP RGB-D frame."""

from __future__ import annotations

import numpy as np

from ..data.camera import rgb_to_uint8
from ..data.erp_geometry import backproject_erp_ray_distance
from ..types import RAY_DISTANCE, ScaleMetadata, SpatialNode


def _voxel_average(points: np.ndarray, colors: np.ndarray, voxel_size: float):
    if voxel_size <= 0:
        return points.astype(np.float32), rgb_to_uint8(colors)
    keys = np.floor(points / float(voxel_size)).astype(np.int64)
    _, inverse = np.unique(keys, axis=0, return_inverse=True)
    count = np.bincount(inverse).astype(np.float32)
    xyz = np.zeros((len(count), 3), np.float32)
    rgb = np.zeros((len(count), 3), np.float32)
    np.add.at(xyz, inverse, points)
    np.add.at(rgb, inverse, rgb_to_uint8(colors).astype(np.float32))
    return xyz / count[:, None], np.rint(rgb / count[:, None]).clip(0, 255).astype(np.uint8)


def build_oracle_erp_node(
    source_erp_rgb: np.ndarray,
    source_erp_depth_ray_distance: np.ndarray,
    source_erp_mask: np.ndarray,
    *,
    source_c2w_local: np.ndarray | None = None,
    node_id: str = "node_000",
    created_frame: int = 0,
    voxel_size: float = 0.01,
    pixel_center: float = 0.5,
    model_versions: dict | None = None,
) -> SpatialNode:
    """Build M0 from the full 2:1 ERP without canonical-view recropping."""
    rgb = rgb_to_uint8(source_erp_rgb)
    depth = np.asarray(source_erp_depth_ray_distance, np.float32)
    mask = np.asarray(source_erp_mask, bool)
    if rgb.shape[:2] != depth.shape or depth.shape != mask.shape:
        raise ValueError("source ERP RGB/depth/mask shapes do not match")
    if rgb.shape[1] != 2 * rgb.shape[0]:
        raise ValueError(f"Oracle source ERP must remain 2:1, got {rgb.shape[:2]}")
    c2w = np.eye(4, dtype=np.float32) if source_c2w_local is None else np.asarray(source_c2w_local, np.float32)
    np.testing.assert_allclose(c2w, np.eye(4), atol=1e-5, err_msg="Oracle M0 must use source-relative identity c2w")
    points, colors = backproject_erp_ray_distance(
        rgb, depth, mask, c2w, pixel_center=pixel_center
    )
    points, colors = _voxel_average(points, colors, float(voxel_size))
    if not len(points):
        raise ValueError("Oracle ERP contains no valid geometry")
    bbox_min, bbox_max = points.min(0), points.max(0)
    radius = float(np.linalg.norm(bbox_max - bbox_min) * 0.5)
    point_count = len(points)
    pixel_shape = (1,) + depth.shape
    return SpatialNode(
        node_id=node_id,
        status="active",
        parent_id=None,
        center_c2w=c2w,
        created_frame=int(created_frame),
        coverage_radius=radius,
        bbox_min=bbox_min.astype(np.float32),
        bbox_max=bbox_max.astype(np.float32),
        view_rgb=rgb[None],
        view_depth=depth[None],
        view_c2w=c2w[None],
        view_intrinsics=np.full((1, 3, 3), np.nan, np.float32),
        points_xyz=points.astype(np.float32),
        points_rgb=colors,
        points_confidence=np.ones(point_count, np.float32),
        points_source=np.zeros(point_count, np.int8),
        observation_count=np.ones(point_count, np.int16),
        depth_convention=RAY_DISTANCE,
        schema_version=4,
        quality_metrics={
            "builder": "full_erp_point_cloud",
            "input_valid_pixels": int((mask & np.isfinite(depth) & (depth > 0)).sum()),
            "fused_points": int(point_count),
            "future_geometry_used": False,
        },
        view_source=np.zeros(pixel_shape, np.int8),
        view_image_confidence=np.ones(pixel_shape, np.float32),
        view_depth_confidence=(mask & np.isfinite(depth) & (depth > 0))[None].astype(np.float32),
        point_view_mask=np.ones(point_count, np.uint64),
        scale=ScaleMetadata(
            mode="dataset_calibrated",
            meters_per_world_unit=1.0,
            uncertainty=0.0,
            anchor_source="Holo360D_mesh_depth",
        ),
        model_versions=dict(model_versions or {}),
        points_rgb_content_origin=np.full(point_count, "oracle_source", dtype="U24"),
        points_depth_content_origin=np.full(point_count, "oracle_source", dtype="U24"),
        points_evidence_role=np.full(point_count, "direct_source", dtype="U24"),
        view_rgb_content_origin=np.full(pixel_shape, "oracle_source", dtype="U24"),
        view_depth_content_origin=np.full(pixel_shape, "oracle_source", dtype="U24"),
        view_evidence_role=np.full(pixel_shape, "direct_source", dtype="U24"),
        points_rgb_evidence_role=np.full(point_count, "direct_source", dtype="U24"),
        points_depth_evidence_role=np.full(point_count, "direct_source", dtype="U24"),
        view_rgb_evidence_role=np.full(pixel_shape, "direct_source", dtype="U24"),
        view_depth_evidence_role=np.full(pixel_shape, "direct_source", dtype="U24"),
    )
