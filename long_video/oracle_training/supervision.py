"""Validation helpers for chunk-local Phase B supervision masks."""
from __future__ import annotations

import numpy as np


def validate_current_chunk_supervision(weights, target_temporal_length: int) -> list[int]:
    """Validate the shared-boundary contract and return effective indices.

    The first latent in each chunk is the shared autoregressive boundary and
    must never carry gradient.  At least one non-boundary target latent must
    retain positive weight, and the temporal mask must match the encoded GT
    target shape exactly.
    """

    values = np.asarray(weights)
    expected = int(target_temporal_length)
    if values.ndim != 1 or len(values) != expected:
        raise ValueError(
            f"Phase B loss weights must be 1-D with target T={expected}; got {values.shape}"
        )
    if not np.isfinite(values).all() or bool((values < 0).any()):
        raise ValueError("Phase B loss weights must be finite and non-negative")
    if float(values[0]) != 0.0:
        raise ValueError("shared boundary supervision must remain weights[0]=0")
    indices = np.flatnonzero(values > 0).astype(int).tolist()
    if not indices:
        raise ValueError("Phase B current chunk has no effective non-boundary target latent")
    if indices[0] == 0:
        raise ValueError("shared boundary index 0 cannot participate in Phase B loss")
    return indices


__all__ = ["validate_current_chunk_supervision"]
