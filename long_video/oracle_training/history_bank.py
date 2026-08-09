"""Versioned generated-history cache keys and leakage validation."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json


@dataclass(frozen=True)
class HistoryBankKey:
    checkpoint_sha: str
    global_step: int
    scene_id: str
    source_id: str
    trajectory_id: str
    history_chunk_index: int
    generation_config: tuple
    prompt: str
    seed: int
    # Version 2 includes the trajectory/current-chunk Spatial Memory Prefix
    # semantics and therefore must not collide with old generated-history
    # entries.
    history_schema_version: int = 2

    @property
    def current_chunk_index(self) -> int:
        """Explicit alias used by Phase B cache maps and diagnostics."""

        return int(self.history_chunk_index)

    @property
    def trajectory_chunk(self) -> tuple[str, int]:
        """Stable map key preventing history reuse across chunks."""

        return str(self.trajectory_id), int(self.history_chunk_index)

    def digest(self):
        payload = asdict(self)
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def history_bank_cache_key(trajectory_id: object, current_chunk_index: int, *parts) -> tuple:
    """Build a map key that always carries trajectory and current chunk.

    ``parts`` may contain scene/sample/checkpoint identifiers.  The first two
    fields are intentionally fixed and make accidental cross-chunk reuse
    visible in both tests and serialized indexes.
    """

    chunk = int(current_chunk_index)
    if chunk < 1:
        raise ValueError("history bank current_chunk_index must be >= 1")
    return (str(trajectory_id), chunk, *parts)


def validate_history_bank_entry(entry):
    required = {"TEMP_LONG", "TEMP_MID", "TEMP_SHORT", "key", "metadata"}
    missing = required - set(entry)
    if missing:
        raise ValueError(f"history bank entry is incomplete: {sorted(missing)}")
    forbidden = {"target_rgb", "target_depth", "future_rgb", "future_depth", "gt_future"}
    found = forbidden & set(entry)
    if found:
        raise ValueError(f"history bank entry contains future GT fields: {sorted(found)}")
    metadata = entry["metadata"]
    if metadata.get("uses_gt_future") is not False:
        raise ValueError("history bank metadata must explicitly declare uses_gt_future=false")
    provenance = metadata.get("warp_provenance")
    if not isinstance(provenance, list) or not provenance:
        raise ValueError("history bank metadata must include causal warp provenance")
    for item in provenance:
        if not isinstance(item, dict) or item.get("causal") is not True:
            raise ValueError("history bank warp provenance must be causal")
        if item.get("uses_future_gt") is not False:
            raise ValueError("history bank warp provenance must declare uses_future_gt=false")
    return True
