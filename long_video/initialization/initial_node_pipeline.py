"""Build the initial world node from eight causal RGB views using Pi3."""
from __future__ import annotations

from dataclasses import replace
import numpy as np

from ..memory.node_builder import build_from_views
from ..types import ScaleMetadata, ViewSet


def initialize_spatial_node(views: ViewSet, geometry_backend, config: dict):
    """Predict geometry for pre-existing causal views; no completion model is used."""
    if not isinstance(views, ViewSet) or len(views.rgb) != 8:
        raise ValueError("initial causal world requires exactly eight RGB views for Pi3")
    view_frames = config.get("view_frame_indices")
    target_start = config.get("target_frame_start")
    if target_start is not None:
        if view_frames is None or len(view_frames) != 8:
            raise ValueError("training initialization requires eight causal view_frame_indices")
        if any(int(frame) >= int(target_start) for frame in view_frames):
            raise ValueError("future/current GT views cannot initialize the causal world")
    has_depth = bool(np.isfinite(views.depth).any())
    prediction = geometry_backend.predict(
        views.rgb, views.c2w, views.intrinsics,
        known_depth=views.depth if has_depth else None,
        known_mask=np.isfinite(views.depth) if has_depth else None,
        known_depth_convention=views.depth_convention if has_depth else None,
    )
    completed = replace(
        views, depth=np.asarray(prediction.depth, np.float32),
        depth_confidence=np.asarray(prediction.depth_confidence, np.float32),
        depth_convention=prediction.depth_convention,
    )
    node = build_from_views(
        completed, node_id=str(config.get("node_id", "node_000")),
        center_c2w=np.asarray(config.get("center_c2w", completed.c2w[0]), np.float32),
        created_frame=int(config.get("created_frame", 0)),
        voxel_size=float(config.get("voxel_size", 0.01)), status="active",
    )
    node.quality_metrics.update(
        initialization_mode="causal_pi3_views",
        geometry_diagnostics=prediction.diagnostics,
        scale_info=prediction.scale_info,
    )
    node.scale = ScaleMetadata(
        mode=prediction.scale_info.get("mode", "relative"),
        meters_per_world_unit=prediction.scale_info.get("meters_per_world_unit"),
        uncertainty=float(prediction.scale_info.get("uncertainty", 1.0)),
        anchor_source=prediction.scale_info.get("anchor_source", "unspecified"),
    )
    if config.get("node_store") is not None:
        config["node_store"].save(node)
    return node
