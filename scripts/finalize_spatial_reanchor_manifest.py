#!/usr/bin/env python3
"""Finalize Phase B choices with actual M0 renderer overlap on built windows."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args()


def _renderer_overlap(record, candidate):
    import numpy as np
    from long_video.geometry.point_renderer import render
    from long_video.memory.node_store import NodeStore
    from long_video.types import CameraBatch

    root = Path(record["path"])
    node = NodeStore(root / "session").load("node_000")
    poses = np.load(root / "target" / "target_c2w_local.npy")[::8]
    intrinsics = np.load(root / "target" / "intrinsics.npy")[::8]
    offsets = [int(candidate["earlier_anchor_offset"]), int(candidate["later_anchor_offset"])]
    cameras = CameraBatch(poses[offsets], intrinsics[offsets], 192, 320)
    rendered = render(
        node, cameras, device="cpu", near=0.05, far=100.0,
        point_radius=1, chunk_points=1000000,
    )
    left, right = np.asarray(rendered.visibility[0], bool), np.asarray(rendered.visibility[1], bool)
    union = left | right
    return float((left & right).sum() / max(int(union.sum()), 1))


def main():
    args = _args()
    from long_video.oracle_training.revisit import (
        add_renderer_overlap, choose_independent_final_candidates,
        score_large_motion_window, score_revisit_window,
    )

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    phase_b = [item for item in manifest["sequences"] if item["phase"] == "B"]
    candidates = {"revisit": [], "large_motion": []}
    for record in phase_b:
        root = Path(record["path"])
        poses = __import__("numpy").load(root / "target" / "target_c2w_local.npy")[::8]
        anchors = int(record["chunk_count"]) * 4 + 1
        if len(poses) < anchors:
            raise ValueError(f"record anchor trajectory is truncated: {record['sequence_id']}")
        if record["sample_type"] == "revisit":
            raw_candidates = [score_revisit_window(poses[:anchors], 0, anchors)]
        else:
            raw_candidates = []
            for chunk_index in range(int(record["chunk_count"])):
                local = score_large_motion_window(poses[4 * chunk_index:4 * chunk_index + 5], 0, 5)
                local["training_chunk_index"] = chunk_index
                local["earlier_anchor_offset"] += 4 * chunk_index
                local["later_anchor_offset"] += 4 * chunk_index
                local["anchor_count"] = anchors
                local["chunks"] = int(record["chunk_count"])
                local["dense_frames"] = 1 + 32 * int(record["chunk_count"])
                raw_candidates.append(local)
        for candidate in raw_candidates:
            candidate["chunks"] = int(record["chunk_count"])
            candidate["dense_frames"] = 1 + 32 * int(record["chunk_count"])
            candidate = add_renderer_overlap(
                candidate, _renderer_overlap(record, candidate), sample_type=record["sample_type"],
            )
            candidate["sequence_id"] = record["sequence_id"]
            candidates[record["sample_type"]].append(candidate)

    revisit, motion = choose_independent_final_candidates(
        candidates["revisit"], candidates["large_motion"],
    )
    selected_ids = {item["sequence_id"] for item in revisit + motion}
    for record in phase_b:
        if record["sequence_id"] not in selected_ids:
            record["split"] = "excluded_phase_b_candidate"
            continue
        candidate = next(item for item in revisit + motion if item["sequence_id"] == record["sequence_id"])
        record["training_chunk_index"] = int(candidate["training_chunk_index"])
        record.setdefault("metadata", {})["phase_b_selection"] = {
            key: value for key, value in candidate.items() if key != "sequence_id"
        }
        record["metadata"]["node_mode"] = "M0-only"
        record["metadata"]["uses_gt_future"] = False

    manifest["schema_version"] = max(3, int(manifest.get("schema_version", 1)))
    manifest["phase_b_selection"] = {
        "selection_stage": "renderer_final",
        "node_mode": "M0-only",
        "uses_future_gt": False,
        "revisit": revisit,
        "large_motion": motion,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    temporary.replace(args.output)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(manifest["phase_b_selection"], indent=2), encoding="utf-8")
    print(json.dumps(manifest["phase_b_selection"], indent=2))


if __name__ == "__main__":
    main()
