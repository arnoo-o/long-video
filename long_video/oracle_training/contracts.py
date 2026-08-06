"""Strict information-flow contracts preventing supervision leakage."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


_GT_KEYS = {
    "target_rgb_for_loss", "target_z_depth_for_eval",
    "target_ray_distance_for_reference", "ground_truth_supervision_only",
}


def assert_no_supervision_content(payload: Any, destination: str) -> None:
    """Fail if recursively supplied memory/history payload contains GT fields."""
    if isinstance(payload, dict):
        overlap = _GT_KEYS.intersection(payload)
        if overlap:
            raise ValueError(f"{destination} cannot receive supervision fields: {sorted(overlap)}")
        for value in payload.values():
            assert_no_supervision_content(value, destination)
    elif isinstance(payload, (list, tuple)):
        for value in payload:
            assert_no_supervision_content(value, destination)


@dataclass(frozen=True)
class SupervisionBatch:
    target_rgb_for_loss: np.ndarray
    target_z_depth_for_eval: np.ndarray
    target_valid_mask: np.ndarray

    def as_memory_content(self):
        raise RuntimeError("ground-truth supervision can never be converted to memory content")


@dataclass(frozen=True)
class GeneratedMemoryBatch:
    generated_rgb_for_memory: np.ndarray

    def __post_init__(self):
        rgb = np.asarray(self.generated_rgb_for_memory)
        if rgb.ndim != 4 or rgb.shape[-1] != 3:
            raise ValueError("generated_rgb_for_memory must be [T,H,W,3]")


def assert_history_frames_are_generated(history_frames: Any, target_rgb_for_loss: Any) -> None:
    if history_frames is target_rgb_for_loss:
        raise ValueError("WAH history must not alias target_rgb_for_loss")


def validate_content_labels(
    rgb_origin, depth_origin, rgb_evidence_role, depth_evidence_role=None
) -> None:
    rgb = np.asarray(rgb_origin)
    depth = np.asarray(depth_origin)
    rgb_role = np.asarray(rgb_evidence_role)
    depth_role = rgb_role if depth_evidence_role is None else np.asarray(depth_evidence_role)
    if not (rgb.shape == depth.shape == rgb_role.shape == depth_role.shape):
        raise ValueError("content-origin and modality evidence-role arrays must have identical shapes")
    if np.any(rgb == "ground_truth_supervision_only") or np.any(depth == "ground_truth_supervision_only"):
        raise ValueError("ground_truth_supervision_only cannot enter a SpatialNode")
    new_rgb = rgb_role == "current_generation"
    if np.any(new_rgb & (rgb != "model_generated")):
        raise ValueError("current_generation RGB must have model_generated content origin")
    generated_depth = depth_role == "geometry_prediction"
    if np.any(generated_depth & (depth != "pi3_prediction")):
        raise ValueError("geometry_prediction depth must have pi3_prediction content origin")
