"""Non-destructive point-cloud views of persistent spatial nodes."""
from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np


_POINT_ALIGNED_FIELDS = (
    "points_xyz",
    "points_rgb",
    "points_confidence",
    "points_source",
    "observation_count",
    "points_normal",
    "point_view_mask",
    "points_rgb_content_origin",
    "points_depth_content_origin",
    "points_evidence_role",
    "points_rgb_evidence_role",
    "points_depth_evidence_role",
)


def filter_node_to_observed_erp(node: Any, observed_erp_mask: Any):
    """Return a copied node containing only rays covered by the source image.

    The input node and its completed panorama remain untouched.  The mask is
    sampled by projecting each world point back to the panorama center stored
    in ``center_c2w`` using the repository's OpenCV/ERP convention.
    """
    mask = np.asarray(observed_erp_mask, bool)
    if mask.ndim != 2 or not mask.any():
        raise ValueError("observed ERP mask must be a non-empty [H,W] array")
    points = np.asarray(node.points_xyz, np.float32)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0:
        raise ValueError("node point cloud must be non-empty [N,3]")
    center = np.asarray(node.center_c2w, np.float32)
    local = (points - center[:3, 3]) @ center[:3, :3]
    norm = np.linalg.norm(local, axis=1)
    direction = local / np.maximum(norm[:, None], 1e-8)
    longitude = np.arctan2(direction[:, 0], direction[:, 2])
    latitude = -np.arcsin(np.clip(direction[:, 1], -1.0, 1.0))
    height, width = mask.shape
    u = (longitude / (2.0 * np.pi) + 0.5) * width - 0.5
    v = (0.5 - latitude / np.pi) * height - 0.5
    x = np.mod(np.rint(u).astype(np.int64), width)
    y = np.clip(np.rint(v).astype(np.int64), 0, height - 1)
    keep = (norm > 1e-8) & mask[y, x]
    kept_count = int(keep.sum())
    if kept_count == 0:
        raise RuntimeError("observed ERP mask removed every source-world point")

    updates = {}
    for name in _POINT_ALIGNED_FIELDS:
        value = getattr(node, name, None)
        if value is None:
            continue
        array = np.asarray(value)
        if len(array) != len(points):
            raise ValueError(f"point-aligned field {name} has {len(array)} rows, expected {len(points)}")
        updates[name] = array[keep].copy()
    filtered_points = updates["points_xyz"]
    bbox_min = filtered_points.min(axis=0).astype(np.float32)
    bbox_max = filtered_points.max(axis=0).astype(np.float32)
    report = {
        "mode": "original_perspective_points_only",
        "mask_height": int(height),
        "mask_width": int(width),
        "observed_erp_ratio": float(mask.mean()),
        "original_point_count": int(len(points)),
        "kept_point_count": kept_count,
        "removed_completion_point_count": int(len(points) - kept_count),
        "kept_point_ratio": float(keep.mean()),
        "input_node_unchanged": True,
    }
    quality_metrics = dict(node.quality_metrics)
    quality_metrics["inference_point_filter"] = report
    filtered = replace(
        node,
        **updates,
        bbox_min=bbox_min,
        bbox_max=bbox_max,
        coverage_radius=float(np.linalg.norm(bbox_max - bbox_min) * 0.5),
        quality_metrics=quality_metrics,
    )
    return filtered, report
