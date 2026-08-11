#!/usr/bin/env python3
"""Validate a generic RGB-video run and initialize Stage0-FiLM-only training.

The actual flow batches are produced by the pinned WAH ``train_exact`` runner;
this entrypoint intentionally rejects manifests in which a supervised current
frame is also present in the causal-world history.
"""
import argparse
import json
from pathlib import Path

from long_video.training.stage0_causal_world import (
    validate_dl3dv_film_manifest, validate_generic_rgb_manifest,
)
from long_video.training.causal_rollout import AllChunkRoundRobin


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.manifest.read_text(encoding="utf-8"))
    is_dl3dv = isinstance(payload, dict) and payload.get("dataset") == "DL3DV-10K official 480P images+poses"
    records = validate_dl3dv_film_manifest(args.manifest) if is_dl3dv else validate_generic_rgb_manifest(args.manifest)
    round_robin = AllChunkRoundRobin()
    report = {
        "architecture": "original_wah+pi3_causal_world+stage0_film+native_helios",
        "record_count": len(records),
        "uses_future_gt": False,
        "trainable": ["stage0_causal_world_film.film.*"],
        "flow_matching_stage_id": 0,
        "all_chunk_round_robin": bool(is_dl3dv),
        "trainable_chunk_indices": ({x["trajectory_id"]: list(range(int(x["chunk_count"]))) for x in records}
                                    if is_dl3dv else None),
        "round_robin_state": round_robin.state_dict(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
