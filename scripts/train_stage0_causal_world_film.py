#!/usr/bin/env python3
"""Validate a generic RGB-video run and initialize Stage0-FiLM-only training.

The actual flow batches are produced by the pinned WAH ``train_exact`` runner;
this entrypoint intentionally rejects manifests in which a supervised current
frame is also present in the causal-world history.
"""
import argparse
import json
from pathlib import Path

from long_video.training.stage0_causal_world import validate_generic_rgb_manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records = validate_generic_rgb_manifest(args.manifest)
    report = {
        "architecture": "original_wah+pi3_causal_world+stage0_film+native_helios",
        "record_count": len(records),
        "uses_future_gt": False,
        "trainable": ["stage0_causal_world_film.film.*"],
        "flow_matching_stage_id": 0,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
