"""All-chunk round-robin and one-pass causal boundary-state caching."""
from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json

import torch


@dataclass(frozen=True)
class BoundaryCacheKey:
    trajectory_id: str
    current_chunk_index: int
    model_fingerprint: str
    schema_version: int = 1

    def digest(self):
        return hashlib.sha256(json.dumps(self.__dict__, sort_keys=True).encode()).hexdigest()


class AllChunkRoundRobin:
    """Deterministically supervise 0..N-1 with per-trajectory imbalance <= 1."""
    def __init__(self):
        self.cursors = defaultdict(int)
        self.counts = defaultdict(lambda: defaultdict(int))

    def next(self, trajectory_id, chunk_count):
        if int(chunk_count) not in (8, 12): raise ValueError("DL3DV trajectories must have 8 or 12 chunks")
        key = str(trajectory_id); chunk = self.cursors[key] % int(chunk_count)
        self.cursors[key] = (chunk + 1) % int(chunk_count); self.counts[key][chunk] += 1
        return chunk

    def state_dict(self):
        return {"cursors": dict(self.cursors),
                "counts": {k: {str(i): int(v) for i, v in values.items()} for k, values in self.counts.items()}}

    def load_state_dict(self, state):
        self.cursors = defaultdict(int, {str(k): int(v) for k, v in state.get("cursors", {}).items()})
        self.counts = defaultdict(lambda: defaultdict(int))
        for key, values in state.get("counts", {}).items():
            self.counts[str(key)].update({int(i): int(v) for i, v in values.items()})

    def coverage_report(self, manifest_records):
        report = {}
        for record in manifest_records:
            key, count = str(record["trajectory_id"]), int(record["chunk_count"])
            report[key] = [int(self.counts[key].get(i, 0)) for i in range(count)]
        return report


def current_chunk_loss_weights(chunk_count, current_chunk_index, *, latent_frames=9):
    if not 0 <= int(current_chunk_index) < int(chunk_count): raise IndexError("current chunk out of range")
    weights = torch.ones(int(latent_frames), dtype=torch.float32)
    weights[0] = 0.0  # source for chunk0; generated shared boundary thereafter
    return weights


def _snapshot(value):
    if torch.is_tensor(value): return value.detach().cpu().clone()
    if isinstance(value, dict): return {k: _snapshot(v) for k, v in value.items()}
    if isinstance(value, list): return [_snapshot(v) for v in value]
    if isinstance(value, tuple): return tuple(_snapshot(v) for v in value)
    return deepcopy(value)


def build_boundary_states_once(record, source_state, rollout_no_grad, *, model_fingerprint):
    """One no-grad AR pass, caching the state before every trainable chunk.

    ``rollout_no_grad(chunk_index, state)`` receives no target GT.  It must use
    normal inference generation and causal Pi3 world updates, then return the
    next boundary state.  The current/future RGB arrays are intentionally not
    arguments to this API.
    """
    if record.get("uses_future_gt") is not False: raise ValueError("future GT is forbidden")
    count, trajectory = int(record["chunk_count"]), str(record["trajectory_id"])
    cache, state = {}, _snapshot(source_state)
    for current in range(count):
        key = BoundaryCacheKey(trajectory, current, str(model_fingerprint))
        cache[key.digest()] = {"key": key.__dict__, "state": _snapshot(state),
                               "uses_future_gt": False, "history_chunks": list(range(current))}
        if current + 1 < count:
            with torch.no_grad(): state = rollout_no_grad(current, state)
    return cache


def validate_boundary_cache(cache, record):
    count = int(record["chunk_count"])
    if len(cache) != count: raise ValueError("one boundary state is required for every current chunk")
    seen = set()
    for item in cache.values():
        if item.get("uses_future_gt") is not False: raise ValueError("boundary cache leaks future GT")
        current = int(item["key"]["current_chunk_index"]); seen.add(current)
        if item["history_chunks"] != list(range(current)): raise ValueError("non-causal history prefix")
    if seen != set(range(count)): raise ValueError("cache does not cover all chunks 0..N-1")
    return True
