#!/usr/bin/env python3
"""Validate built Spatial Re-Anchored WAH data without loading model weights."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _frame_count(path: Path):
    return len(list(path.glob("*.png")))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    scan = json.loads(Path(manifest["scan_report"]).read_text(encoding="utf-8"))
    scan_scenes = {item["scene_id"]: item for item in scan["scenes"]}
    counts = {}
    validated = []
    for record in manifest["sequences"]:
        root = Path(record["path"])
        chunks = int(record["chunk_count"])
        anchors = int(record["anchor_count"])
        expected_frames = 1 + 32 * chunks
        expected_anchors = 1 + 4 * chunks
        if anchors != expected_anchors:
            raise ValueError(f"{record['sequence_id']}: expected {expected_anchors} anchors, got {anchors}")
        if _frame_count(root / "target" / "target_rgb_for_loss") != expected_frames:
            raise ValueError(f"{record['sequence_id']}: target PNG count violates multi-chunk contract")
        poses = np.load(root / "target" / "target_c2w_local.npy")
        intrinsics = np.load(root / "target" / "intrinsics.npy")
        rgb_weights = np.load(root / "primary_loss_weight_rgb.npy")
        latent_weights = np.load(root / "primary_loss_weight_latent.npy")
        if poses.shape != (expected_frames, 4, 4) or intrinsics.shape != (expected_frames, 3, 3):
            raise ValueError(f"{record['sequence_id']}: camera array shape mismatch")
        np.testing.assert_allclose(poses[0], np.eye(4), atol=1e-5, rtol=0)
        if rgb_weights.shape != (expected_frames,) or latent_weights.shape != (1 + 8 * chunks,):
            raise ValueError(f"{record['sequence_id']}: supervision weight shape mismatch")
        anchor_indices = np.arange(0, expected_frames, 8)
        rife_indices = np.setdiff1d(np.arange(expected_frames), anchor_indices)
        if rgb_weights[0] != 0 or not np.all(rgb_weights[anchor_indices[1:]] == 1.0):
            raise ValueError(f"{record['sequence_id']}: real-anchor supervision weights changed")
        if not np.all(rgb_weights[rife_indices] == 0.25):
            raise ValueError(f"{record['sequence_id']}: RIFE-only supervision weights changed")
        scene = scan_scenes[record["scene_id"]]
        start = int(record["anchor_start"])
        if not any(
            int(run["start"]) <= start and start + anchors <= int(run["end"])
            for run in scene["continuous_runs"]
        ):
            raise ValueError(f"{record['sequence_id']}: selected window crosses an acquisition gap")
        if record["phase"] == "B" and record["split"] == "train":
            training_chunk = int(record.get("training_chunk_index", -1))
            if not 1 <= training_chunk < chunks:
                raise ValueError(
                    f"{record['sequence_id']}: Phase B training chunk must be in 1..N-1, "
                    f"got {training_chunk} for N={chunks}"
                )
            selection = record.get("metadata", {}).get("phase_b_selection", {})
            if selection.get("renderer_overlap_metric") != "bidirectional_depth_reprojection_overlap":
                raise ValueError(
                    f"{record['sequence_id']}: renderer overlap metric is not depth reprojection"
                )
            if record.get("metadata", {}).get("uses_gt_future") is not False:
                raise ValueError(f"{record['sequence_id']}: uses_gt_future must be false")
        counts[record["scene_id"]] = counts.get(record["scene_id"], 0) + 1
        validated.append({
            "sequence_id": record["sequence_id"], "scene_id": record["scene_id"],
            "phase": record["phase"], "chunks": chunks, "frames": expected_frames,
            "anchors": anchors,
        })
    phase_a_train = [item for item in manifest["sequences"] if item["phase"] == "A" and item["split"] == "train"]
    phase_a_diag = [
        item for item in manifest["sequences"]
        if item["phase"] == "A" and item["split"] == "diagnostic"
    ]
    phase_b_train = [
        item for item in manifest["sequences"]
        if item["phase"] == "B" and item["split"] == "train"
    ]
    phase_a_scenes = sorted({item["scene_id"] for item in phase_a_train})
    phase_b_scenes = sorted({item["scene_id"] for item in phase_b_train})
    phase_a_by_scene = {
        scene: sum(item["scene_id"] == scene for item in phase_a_train)
        for scene in phase_a_scenes
    }
    phase_a_diag_by_scene = {
        scene: sum(item["scene_id"] == scene for item in phase_a_diag)
        for scene in phase_a_scenes
    }
    if (
        len(phase_a_scenes) < 2
        or len(set(phase_a_by_scene.values())) != 1
        or min(phase_a_by_scene.values()) < 4
        or any(phase_a_diag_by_scene.get(scene, 0) < 1 for scene in phase_a_scenes)
        or not phase_b_scenes
        or set(phase_a_scenes) != set(phase_b_scenes)
    ):
        raise ValueError(f"Phase A scene balance changed: {phase_a_by_scene}")
    payload = {
        "status": "passed", "manifest": str(args.manifest),
        "sequence_count": len(validated), "scene_counts": counts,
        "phase_a_train_by_scene": phase_a_by_scene,
        "phase_a_diagnostic_by_scene": phase_a_diag_by_scene,
        "phase_b_scenes": phase_b_scenes,
        "sequences": validated,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
