"""Immutable accepted shadows activate exactly two chunks later."""
from collections import deque
from dataclasses import dataclass


@dataclass(frozen=True)
class ScheduledNodeActivation:
    node: object
    created_after_chunk: int
    activate_at_chunk: int

    @property
    def node_id(self):
        return str(self.node.node_id)


class DelayedNodeActivationQueue:
    def __init__(self, delay_chunks=2, max_pending=1):
        if delay_chunks != 2:
            raise ValueError("causal world activation delay is fixed at two chunks")
        if max_pending != 1:
            raise ValueError("only one immutable shadow may be pending")
        self._queue = deque()

    @property
    def pending(self):
        return self._queue[0] if self._queue else None

    def __len__(self):
        return len(self._queue)

    def schedule(self, node, *, created_after_chunk):
        if self._queue:
            raise RuntimeError("a pending shadow cannot be replaced")
        entry = ScheduledNodeActivation(node, int(created_after_chunk), int(created_after_chunk) + 2)
        self._queue.append(entry)
        return entry

    def activate_due(self, chunk_index):
        if not self._queue:
            return None
        entry = self._queue[0]
        if chunk_index < entry.activate_at_chunk:
            return None
        if chunk_index > entry.activate_at_chunk:
            raise RuntimeError(f"missed activation of {entry.node_id}")
        return self._queue.popleft()
