#!/usr/bin/env python3
"""Precompute compact causal ReCal3R voxel worlds for GeoToken Phase B."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from long_video.geometry.voxel_fusion import fuse_voxels


FRAME_LIMITS = (0, 32, 64, 96, 128, 160)
GEOMETRY_CACHE_VERSION = "recal-causal-teacher-world-v3"


def fuse(root: Path, dataset_root: Path, frame_limit: int, voxel_size: float):
    xyz = np.load(root / "xyz_world.npy", mmap_mode="r")[: frame_limit + 1]
    valid = np.load(root / "valid.npy", mmap_mode="r")[: frame_limit + 1].astype(bool)
    confidence = np.load(root / "confidence.npy", mmap_mode="r")[: frame_limit + 1]
    metadata = json.loads((root / "metadata.json").read_text())
    rgb_dir = dataset_root / metadata["rgb_dir"]
    frames = [np.asarray(Image.open(path).convert("RGB"), np.uint8) for path in sorted(rgb_dir.glob("*"))[:frame_limit + 1]]
    rgb = np.stack(frames)
    points, weights, colors = np.asarray(xyz[valid], np.float32), np.asarray(confidence[valid], np.float32), np.asarray(rgb[valid], np.uint8)
    frame_ids = np.broadcast_to(np.arange(frame_limit + 1, dtype=np.int32)[:, None, None], valid.shape)[valid]
    finite = np.isfinite(points).all(1) & np.isfinite(weights) & (weights > 0)
    points, weights = points[finite], weights[finite]
    if not len(points):
        raise RuntimeError(f"no geometry at causal frame limit {frame_limit}")
    keys = np.floor(points / voxel_size).astype(np.int64)
    unique, inverse = np.unique(keys, axis=0, return_inverse=True)
    max_frame = np.full(len(unique), -1, np.int32); np.maximum.at(max_frame, inverse, frame_ids)
    xyz, rgb, conf, obs, fused_keys = fuse_voxels(points, colors, weights, voxel_size=voxel_size)
    return fused_keys, xyz, rgb, conf, obs, max_frame


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--recal3r-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
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
            keys, points, rgb, confidence, observation_count, max_frame = fuse(source, args.dataset_root, frame_limit, args.voxel_size)
            if int(max_frame.max()) > frame_limit: raise RuntimeError("future observation leaked into causal cache")
            np.savez_compressed(
                target / f"frame_{frame_limit:03d}.npz", voxel_keys=keys,
                points_xyz=points, points_rgb=rgb, points_confidence=confidence,
                observation_count=observation_count, max_observation_frame=max_frame,
            )


if __name__ == "__main__":
    main()
