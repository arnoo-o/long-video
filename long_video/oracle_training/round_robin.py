"""Deterministic current-chunk scheduling for Phase B training.

The Phase B objective is deliberately local to one chunk at a time.  This
module keeps the selection policy independent of the model/runtime so it can
be tested without loading CUDA weights.  Chunk zero is the shared source
boundary and is therefore never eligible for supervision.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Iterable, Mapping


def eligible_current_chunks(chunk_count: int) -> tuple[int, ...]:
    """Return all trainable chunks for an ``N`` chunk trajectory.

    ``0`` is the shared first-frame boundary and chunks are one based from the
    training perspective.  A trajectory with fewer than two chunks has no
    trainable chunk and is rejected explicitly rather than silently selecting
    the boundary.
    """

    count = int(chunk_count)
    if count < 2:
        raise ValueError(f"Phase B trajectory must contain at least 2 chunks, got {count}")
    return tuple(range(1, count))


@dataclass
class RoundRobinChunkScheduler:
    """Per-trajectory deterministic round-robin cursor and coverage counts."""

    cursors: Dict[str, int] = field(default_factory=dict)
    supervision_counts: Dict[str, Dict[str, int]] = field(default_factory=dict)

    def _trajectory_key(self, trajectory_id: object) -> str:
        return str(trajectory_id)

    def next_chunk(self, trajectory_id: object, chunk_count: int) -> int:
        """Select the next chunk and advance only that trajectory's cursor."""

        key = self._trajectory_key(trajectory_id)
        eligible = eligible_current_chunks(chunk_count)
        cursor = int(self.cursors.get(key, 0))
        selected = int(eligible[cursor % len(eligible)])
        self.cursors[key] = (cursor + 1) % len(eligible)
        counts = self.supervision_counts.setdefault(key, {})
        selected_key = str(selected)
        counts[selected_key] = int(counts.get(selected_key, 0)) + 1
        return selected

    def select(self, trajectory_id: object, chunk_count: int, *, step: int | None = None) -> int:
        """Compatibility sampler spelling.

        Without ``step`` this advances the per-trajectory cursor.  Supplying a
        step gives a pure deterministic selection, useful for diagnostics
        without mutating a resumed training cursor.
        """

        if step is None:
            return self.next_chunk(trajectory_id, chunk_count)
        return round_robin_current_chunk(trajectory_id, chunk_count, cursor=int(step))

    def record(self, trajectory_id: object, current_chunk_index: int, chunk_count: int) -> None:
        """Record an externally selected chunk without moving the cursor.

        This is useful when restoring a checkpoint or when gradient
        accumulation has already made the sample choice in a caller.
        """

        selected = int(current_chunk_index)
        if selected not in eligible_current_chunks(chunk_count):
            raise ValueError(
                f"current_chunk_index must be in 1..N-1; got {selected} for N={chunk_count}"
            )
        key = self._trajectory_key(trajectory_id)
        counts = self.supervision_counts.setdefault(key, {})
        selected_key = str(selected)
        counts[selected_key] = int(counts.get(selected_key, 0)) + 1

    def snapshot(self) -> dict:
        """JSON/checkpoint-safe scheduler state."""

        return {
            "cursors": {str(key): int(value) for key, value in self.cursors.items()},
            "supervision_counts": {
                str(trajectory): {str(chunk): int(value) for chunk, value in counts.items()}
                for trajectory, counts in self.supervision_counts.items()
            },
        }

    def restore(self, state: Mapping | None) -> None:
        """Restore state from metadata, tolerating the old step-600 schema."""

        if not state:
            return
        cursors = state.get("cursors", {})
        counts = state.get("supervision_counts", {})
        if not isinstance(cursors, Mapping) or not isinstance(counts, Mapping):
            raise ValueError("round-robin state must contain mapping cursors and supervision_counts")
        self.cursors = {str(key): int(value) for key, value in cursors.items()}
        self.supervision_counts = {
            str(trajectory): {str(chunk): int(value) for chunk, value in values.items()}
            for trajectory, values in counts.items()
            if isinstance(values, Mapping)
        }

    def counts_for(self, trajectory_id: object, chunk_count: int | None = None) -> dict[str, int]:
        """Return counts including zeroes for every eligible chunk."""

        key = self._trajectory_key(trajectory_id)
        current = dict(self.supervision_counts.get(key, {}))
        if chunk_count is not None:
            current = {
                str(chunk): int(current.get(str(chunk), 0))
                for chunk in eligible_current_chunks(chunk_count)
            }
        return current


def round_robin_current_chunk(
    trajectory_id: object,
    chunk_count: int,
    *,
    cursor: int = 0,
    step: int | None = None,
) -> int:
    """Pure helper used by tests and callers that manage cursors themselves."""

    eligible = eligible_current_chunks(chunk_count)
    if step is not None:
        cursor = int(step)
    return int(eligible[int(cursor) % len(eligible)])


__all__ = [
    "RoundRobinChunkScheduler",
    "eligible_current_chunks",
    "eligible_chunks",
    "round_robin_current_chunk",
]

# Small naming alias used by diagnostics/tests that call the policy directly.
eligible_chunks = eligible_current_chunks
