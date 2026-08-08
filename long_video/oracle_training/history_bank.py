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
    history_schema_version: int = 1

    def digest(self):
        payload = asdict(self)
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


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
    return True
