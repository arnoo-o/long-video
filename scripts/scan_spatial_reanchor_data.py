#!/usr/bin/env python3
"""Scan Holo360D archives and select gap-safe real-pose revisit windows."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gap-factor", type=float, default=2.5)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    from long_video.oracle_training.revisit import (
        scan_holo360d_zip, select_large_motion_windows, select_revisit_windows,
    )

    scenes = []
    for archive in args.archive:
        report, frame_ids, poses, runs = scan_holo360d_zip(archive, gap_factor=args.gap_factor)
        revisit_candidates = select_revisit_windows(poses, runs)
        motion_candidates = select_large_motion_windows(poses, runs)
        report["phase_b_revisit_pose_candidates"] = revisit_candidates
        report["phase_b_large_motion_pose_candidates"] = motion_candidates
        report["available_phase_b_chunk_counts"] = sorted({
            int(item["chunks"]) for item in revisit_candidates
        })
        report["usable_anchor_count"] = int(sum(item["anchor_count"] for item in report["continuous_runs"]))
        scenes.append(report)
    payload = {"schema_version": 2, "gap_factor": args.gap_factor, "scenes": scenes}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
