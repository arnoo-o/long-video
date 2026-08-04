"""Unified construction of the initial spatial-memory node."""
from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np

from ..memory.node_builder import build_from_views


def _completion_result(completion_backend, observed_images, camera_specs, prompt, config):
    mode = str(config.get("mode", "mvdiffusion_pi3"))
    if mode == "holo_oracle":
        if not isinstance(observed_images, dict):
            raise TypeError("holo_oracle expects panorama/depth/mask/c2w input fields")
        return completion_backend.complete(
            observed_images["panorama"],
            observed_images["depth"],
            panorama_c2w=observed_images.get("c2w"),
            mask=observed_images.get("mask"),
            observed_indices=tuple(config.get("observed_indices", (0,))),
        )
    if mode == "precomputed":
        return completion_backend.complete()
    if mode != "mvdiffusion_pi3":
        raise ValueError(f"Unsupported initialization mode: {mode}")
    return completion_backend.complete(
        observed_images,
        camera_specs,
        prompt,
        output_dir=config.get("completion_output_dir"),
        height=int(config.get("height", 512)),
        width=int(config.get("width", 512)),
        target_fov_degrees=float(config.get("fov_degrees", 90.0)),
    )


def initialize_spatial_node(
    observed_images: Any,
    camera_specs,
    prompt: str,
    completion_backend,
    geometry_backend,
    config: dict,
):
    """Complete RGB views, predict geometry, and build the unique active M0."""
    views = _completion_result(
        completion_backend, observed_images, camera_specs, prompt, config
    )
    has_known_depth = bool(np.isfinite(views.depth).any())
    prediction = geometry_backend.predict(
        views.rgb,
        views.c2w,
        views.intrinsics,
        known_depth=views.depth if has_known_depth else None,
        known_mask=np.isfinite(views.depth) if has_known_depth else None,
    )
    completed = replace(
        views,
        depth=np.asarray(prediction.depth, np.float32),
        depth_confidence=np.asarray(prediction.depth_confidence, np.float32),
        depth_convention=prediction.depth_convention,
    )
    node = build_from_views(
        completed,
        node_id=str(config.get("node_id", "node_000")),
        center_c2w=np.asarray(config.get("center_c2w", completed.c2w[0]), np.float32),
        created_frame=int(config.get("created_frame", 0)),
        voxel_size=float(config.get("voxel_size", 0.02)),
        status="active",
    )
    node.quality_metrics.update(
        initialization_mode=str(config.get("mode", "mvdiffusion_pi3")),
        geometry_diagnostics=prediction.diagnostics,
        scale_info=prediction.scale_info,
        prompt=str(prompt),
    )
    store = config.get("node_store")
    if store is not None:
        store.save(node)
    return node
