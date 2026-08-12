"""Training utilities retained for dataset-agnostic causal rollout."""
from .causal_rollout import (
    AllChunkRoundRobin, BoundaryCacheKey, build_boundary_states_once,
    current_chunk_loss_weights, validate_boundary_cache,
)
__all__ = [
    "AllChunkRoundRobin", "BoundaryCacheKey", "build_boundary_states_once",
    "current_chunk_loss_weights", "validate_boundary_cache",
]
