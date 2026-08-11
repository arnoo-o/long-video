"""Information-flow guards shared by causal-world training and inference."""
from typing import Any
import numpy as np

_GT_KEYS = {
    "target_rgb_for_loss", "target_z_depth_for_eval",
    "target_ray_distance_for_reference", "ground_truth_supervision_only",
}


def assert_no_supervision_content(payload: Any, destination: str):
    if isinstance(payload, dict):
        overlap = _GT_KEYS.intersection(payload)
        if overlap:
            raise ValueError(f"{destination} cannot receive supervision fields: {sorted(overlap)}")
        for value in payload.values():
            assert_no_supervision_content(value, destination)
    elif isinstance(payload, (list, tuple)):
        for value in payload:
            assert_no_supervision_content(value, destination)


def validate_content_labels(rgb_origin, depth_origin, rgb_evidence_role, depth_evidence_role=None):
    rgb, depth, rgb_role = map(np.asarray, (rgb_origin, depth_origin, rgb_evidence_role))
    depth_role = rgb_role if depth_evidence_role is None else np.asarray(depth_evidence_role)
    if not (rgb.shape == depth.shape == rgb_role.shape == depth_role.shape):
        raise ValueError("content-origin and evidence-role arrays must have identical shapes")
    if np.any(rgb == "ground_truth_supervision_only") or np.any(depth == "ground_truth_supervision_only"):
        raise ValueError("ground-truth supervision cannot enter a SpatialNode")
    if np.any((rgb_role == "current_generation") & (rgb != "model_generated")):
        raise ValueError("current_generation RGB must be model-generated")
    if np.any((depth_role == "geometry_prediction") & (depth != "pi3_prediction")):
        raise ValueError("geometry_prediction depth must come from Pi3")
