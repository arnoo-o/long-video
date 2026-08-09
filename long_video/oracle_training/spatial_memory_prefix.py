"""Session-local Spatial Memory Prefix semantics.

This is the small, dependency-light part of the formal rollout that decides
which clean first-frame latent and support mask should occupy the history
prefix.  It intentionally does not render or mutate model parameters.  Both
training and inference can use the same policy and hidden tests can exercise
the geometry/priority rules with NumPy arrays.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np


def rotation_delta_degrees(left: np.ndarray, right: np.ndarray) -> float:
    """Full SO(3) relative-angle distance in degrees."""

    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    relative = left[:3, :3].T @ right[:3, :3]
    cosine = np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def binary_support_confidence(visibility: Any):
    """Return hard 0/1 confidence on renderer-supported pixels/tokens."""

    # Torch tensors are accepted without importing torch at module import time.
    if hasattr(visibility, "to") and hasattr(visibility, "dtype"):
        # ``float`` is available on torch tensors and keeps confidence masks
        # compatible with the patched WAH history path even for bool input.
        if hasattr(visibility, "float"):
            return (visibility > 0).float()
        return (visibility > 0).to(dtype=getattr(visibility, "dtype", None))
    return (np.asarray(visibility) > 0).astype(np.float32)


@dataclass
class SpatialMemoryEntry:
    canonical_c2w: np.ndarray
    intrinsics: np.ndarray
    clean_image_latent: Any
    visibility: Any
    confidence: Any
    frame_id: int
    chunk_id: int
    source_type: str
    entry_id: int = -1

    def clone(self) -> "SpatialMemoryEntry":
        def clone_value(value):
            if hasattr(value, "detach"):
                return value.detach().clone()
            if hasattr(value, "copy"):
                return value.copy()
            return deepcopy(value)

        return SpatialMemoryEntry(
            canonical_c2w=np.asarray(self.canonical_c2w, np.float64).copy(),
            intrinsics=np.asarray(self.intrinsics, np.float64).copy(),
            clean_image_latent=clone_value(self.clean_image_latent),
            visibility=clone_value(self.visibility),
            confidence=clone_value(self.confidence),
            frame_id=int(self.frame_id), chunk_id=int(self.chunk_id),
            source_type=str(self.source_type), entry_id=int(self.entry_id),
        )


class SpatialMemoryPrefixBank:
    """Deterministic nearest-prefix store with M0 priority.

    A hit requires *both* translation and full rotation to be inside the
    configured bounds.  Generated entries may coexist with M0 entries, but
    never replace an M0 entry.  If an M0 observation arrives for a generated
    slot, it upgrades the generated slot.
    """

    def __init__(self, translation_threshold: float = 3.0, rotation_threshold: float = 30.0):
        if float(translation_threshold) < 0 or float(rotation_threshold) < 0:
            raise ValueError("Spatial Memory thresholds must be non-negative")
        self.translation_threshold = float(translation_threshold)
        self.rotation_threshold = float(rotation_threshold)
        self.entries: list[SpatialMemoryEntry] = []

    def _distances(self, pose: np.ndarray, entry: SpatialMemoryEntry) -> tuple[float, float]:
        pose = np.asarray(pose, np.float64)
        translation = float(np.linalg.norm(pose[:3, 3] - entry.canonical_c2w[:3, 3]))
        rotation = rotation_delta_degrees(entry.canonical_c2w, pose)
        return translation, rotation

    def _rank(self, score: float, entry: SpatialMemoryEntry) -> tuple[int, float, int]:
        # Explicitly prefer M0 over generated for the same valid query.  The
        # entry id makes ties deterministic across Python versions.
        return (0 if str(entry.source_type).upper() == "M0" else 1, float(score), int(entry.entry_id))

    def query(self, pose: np.ndarray):
        """Return ``(selected, nearest)`` where each item includes distances."""

        candidates = []
        nearest = None
        for entry in self.entries:
            translation, rotation = self._distances(pose, entry)
            score = translation / max(self.translation_threshold, 1e-12) + rotation / max(
                self.rotation_threshold, 1e-12
            )
            item = (score, translation, rotation, entry)
            if nearest is None or (score, int(entry.entry_id)) < (nearest[0], int(nearest[3].entry_id)):
                nearest = item
            if translation <= self.translation_threshold and rotation <= self.rotation_threshold:
                candidates.append(item)
        selected = min(candidates, key=lambda item: self._rank(item[0], item[3])) if candidates else None
        return selected, nearest

    def add_if_novel(
        self,
        *,
        pose: np.ndarray,
        intrinsics: np.ndarray,
        latent: Any,
        visibility: Any,
        confidence: Any,
        frame_id: int,
        chunk_id: int,
        source_type: str,
    ) -> dict:
        source_type = str(source_type)
        selected, nearest = self.query(pose)
        if selected is not None:
            _, translation, rotation, existing = selected
            # A real M0 observation can replace a generated approximation.  A
            # generated observation can never overwrite M0.
            upgraded = source_type.upper() == "M0" and str(existing.source_type).upper() != "M0"
            if upgraded:
                existing.canonical_c2w = np.asarray(pose, np.float64).copy()
                existing.intrinsics = np.asarray(intrinsics, np.float64).copy()
                existing.clean_image_latent = _clone(latent)
                existing.visibility = _clone(visibility)
                existing.confidence = _clone(confidence)
                existing.frame_id = int(frame_id)
                existing.chunk_id = int(chunk_id)
                existing.source_type = "M0"
            return {
                "created": False,
                "upgraded_generated_to_M0": bool(upgraded),
                "matched_entry_id": int(existing.entry_id),
                "source_type": str(existing.source_type),
                "translation": float(translation),
                "rotation_degrees": float(rotation),
            }
        entry = SpatialMemoryEntry(
            canonical_c2w=np.asarray(pose, np.float64).copy(),
            intrinsics=np.asarray(intrinsics, np.float64).copy(),
            clean_image_latent=_clone(latent), visibility=_clone(visibility),
            confidence=_clone(confidence), frame_id=int(frame_id), chunk_id=int(chunk_id),
            source_type=source_type, entry_id=len(self.entries),
        )
        self.entries.append(entry)
        return {
            "created": True, "entry_id": int(entry.entry_id),
            "source_type": source_type,
            "nearest_translation": None if nearest is None else float(nearest[1]),
            "nearest_rotation_degrees": None if nearest is None else float(nearest[2]),
        }

    def summary(self) -> list[dict]:
        return [
            {
                "entry_id": int(entry.entry_id), "frame_id": int(entry.frame_id),
                "chunk_id": int(entry.chunk_id), "source_type": str(entry.source_type),
                "canonical_c2w": entry.canonical_c2w.tolist(),
                "intrinsics": entry.intrinsics.tolist(),
                "canonical_translation": entry.canonical_c2w[:3, 3].tolist(),
            }
            for entry in self.entries
        ]


def choose_prefix(
    bank: SpatialMemoryPrefixBank,
    *,
    pose: np.ndarray,
    m0_latent: Any,
    m0_visibility: Any,
    m0_confidence: Any | None = None,
    m0_has_support: bool | None = None,
    zero_latent_factory=None,
) -> tuple[Any, Any, Any, dict]:
    """Choose ``(latent, visibility, confidence, report)`` for one chunk.

    ``m0_latent`` is the clean current-boundary M0 latent.  On a miss it is
    used only when the current M0 renderer has support; otherwise an all-zero
    invalid prefix is returned.  The caller can provide ``zero_latent_factory``
    for tensors; NumPy arrays are zeroed directly by default.
    """

    selected, nearest = bank.query(pose)
    if selected is not None:
        _, translation, rotation, entry = selected
        return (
            _clone(entry.clean_image_latent), _clone(entry.visibility), _clone(entry.confidence),
            {
                "hit": True, "entry_id": int(entry.entry_id),
                "source_type": str(entry.source_type), "translation": float(translation),
                "rotation_degrees": float(rotation), "prefix_source": "spatial_memory",
            },
        )

    if m0_has_support is None:
        if hasattr(m0_visibility, "any"):
            value = m0_visibility.any()
            support = bool(value.item()) if hasattr(value, "item") else bool(value)
        else:
            support = bool(np.asarray(m0_visibility).any())
    else:
        support = bool(m0_has_support)
    if support:
        if m0_confidence is None:
            m0_confidence = binary_support_confidence(m0_visibility)
        return (
            _clone(m0_latent), _clone(m0_visibility), _clone(m0_confidence),
            {
                "hit": False, "entry_id": None, "source_type": "M0",
                "translation": None if nearest is None else float(nearest[1]),
                "rotation_degrees": None if nearest is None else float(nearest[2]),
                "prefix_source": "current_M0_boundary",
            },
        )

    zero = zero_latent_factory(m0_latent) if zero_latent_factory else _zeros_like(m0_latent)
    return (
        zero, _zeros_like(m0_visibility), _zeros_like(m0_visibility),
        {
            "hit": False, "entry_id": None, "source_type": None,
            "translation": None if nearest is None else float(nearest[1]),
            "rotation_degrees": None if nearest is None else float(nearest[2]),
            "prefix_source": "masked_invalid",
        },
    )


def _clone(value):
    if hasattr(value, "detach"):
        return value.detach().clone()
    if hasattr(value, "copy"):
        return value.copy()
    return deepcopy(value)


def _zeros_like(value):
    if hasattr(value, "detach"):
        return value.detach().clone().zero_()
    return np.zeros_like(value)


__all__ = [
    "SpatialMemoryEntry",
    "SpatialMemoryBank",
    "SpatialMemoryPrefixBank",
    "binary_support_confidence",
    "choose_prefix",
    "rotation_delta_degrees",
]

# Reuse the inference-facing spelling in lightweight training callers.
SpatialMemoryBank = SpatialMemoryPrefixBank
