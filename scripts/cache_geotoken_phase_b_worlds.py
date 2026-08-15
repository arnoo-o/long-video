#!/usr/bin/env python3
"""Precompute compact causal ReCal3R voxel worlds for GeoToken Phase B."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


FRAME_LIMITS = (0, 32, 64, 96, 128, 160)
GEOMETRY_CACHE_VERSION = "recal-causal-teacher-world-v2"


def fuse(root: Path, frame_limit: int, voxel_size: float):
    xyz = np.load(root / "xyz_world.npy", mmap_mode="r")[: frame_limit + 1]
    valid = np.load(root / "valid.npy", mmap_mode="r")[: frame_limit + 1].astype(bool)
    confidence = np.load(root / "confidence.npy", mmap_mode="r")[: frame_limit + 1]
    points, weights = np.asarray(xyz[valid], np.float32), np.asarray(confidence[valid], np.float32)
    finite = np.isfinite(points).all(1) & np.isfinite(weights) & (weights > 0)
    points, weights = points[finite], weights[finite]
    if not len(points):
        raise RuntimeError(f"no geometry at causal frame limit {frame_limit}")
    voxel_keys = np.floor(points / voxel_size).astype(np.int64)
    unique, inverse = np.unique(voxel_keys, axis=0, return_inverse=True)
    sums = np.zeros((len(unique), 3), np.float64)
    totals = np.zeros(len(unique), np.float64)
    counts = np.zeros(len(unique), np.int32)
    np.add.at(sums, inverse, points * weights[:, None])
    np.add.at(totals, inverse, weights)
    np.add.at(counts, inverse, 1)
    return unique, (sums / totals[:, None].clip(1e-8)).astype(np.float32), (
        totals / counts.clip(1)
    ).clip(0, 1).astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--recal3r-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--trajectory-ids-json", type=Path, required=True)
    parser.add_argument("--voxel-size", type=float, default=0.02)
    args = parser.parse_args()
    trajectory_ids = json.loads(args.trajectory_ids_json.read_text())
    for trajectory_id in trajectory_ids:
        source = args.recal3r_root / trajectory_id
        metadata = json.loads((source / "metadata.json").read_text())
        if not metadata.get("valid", False):
            continue
        target = args.output_root / trajectory_id
        target.mkdir(parents=True, exist_ok=True)
        (target / "cache_metadata.json").write_text(json.dumps({
            "schema_version": 3, "geometry_implementation_version": GEOMETRY_CACHE_VERSION,
            "source_recal_metadata": metadata, "voxel_size": float(args.voxel_size),
            "alignment_version": "recal_to_pi3x_w0_v1",
        }, indent=2))
        for frame_limit in FRAME_LIMITS:
            keys, points, confidence = fuse(source, frame_limit, args.voxel_size)
            np.savez_compressed(
                target / f"frame_{frame_limit:03d}.npz", voxel_keys=keys,
                points_xyz=points, points_confidence=confidence,
            )


if __name__ == "__main__":
    main()
