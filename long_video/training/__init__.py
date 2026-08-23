"""Training utilities retained for dataset-agnostic causal rollout.

Imports are lazy so manifest validation and preprocessing do not require the
heavy torch training runtime.
"""
_ROLLOUT_EXPORTS = {
    "AllChunkRoundRobin", "BoundaryCacheKey", "build_boundary_states_once",
    "current_chunk_loss_weights", "validate_boundary_cache",
}
__all__ = [
    "AllChunkRoundRobin", "BoundaryCacheKey", "build_boundary_states_once",
    "current_chunk_loss_weights", "validate_boundary_cache",
]


def __getattr__(name):
    if name in _ROLLOUT_EXPORTS:
        from . import causal_rollout
        return getattr(causal_rollout, name)
    raise AttributeError(name)
