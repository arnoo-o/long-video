#!/usr/bin/env python3
"""Precompute compact causal ReCal3R voxel worlds for GeoToken Phase B."""
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np
from PIL import Image
from long_video.geometry.voxel_fusion import fuse_voxels


FRAME_LIMITS = (0, 32, 64, 96, 128, 160)
GEOMETRY_CACHE_VERSION = "recal-causal-teacher-world-v7-p40-surface-ownership"


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
    points, weights, colors, frame_ids = points[finite], weights[finite], colors[finite], frame_ids[finite]
    if not len(points):
        raise RuntimeError(f"no geometry at causal frame limit {frame_limit}")
    keys = np.floor(points / voxel_size).astype(np.int64)
    unique, inverse = np.unique(keys, axis=0, return_inverse=True)
    max_frame = np.full(len(unique), -1, np.int32); np.maximum.at(max_frame, inverse, frame_ids)
    # Offline causal teacher equivalent of persistent ownership: frame-0
    # surfaces are source-owned immediately; all other voxels need three
    # distinct frame supports. Their XYZ is the coordinate-wise median of the
    # first three supporting frames and is then kept immutable while confidence,
    # observation count and appearance anchors continue to aggregate.
    pairs = np.stack([inverse, frame_ids.astype(np.int64)], axis=1)
    unique_pairs, first_observation = np.unique(pairs, axis=0, return_index=True)
    support_counts = np.bincount(unique_pairs[:, 0], minlength=len(unique))
    support_starts = np.r_[0, np.cumsum(support_counts)]
    support_rank = np.arange(len(unique_pairs)) - np.repeat(support_starts[:-1], support_counts)
    first_three = support_rank < 3
    support_grid = np.full((len(unique), 3, 3), np.nan, np.float32)
    support_grid[unique_pairs[first_three, 0], support_rank[first_three]] = points[first_observation[first_three]]
    canonical = np.nanmedian(support_grid, axis=1).astype(np.float32)
    source_pairs = unique_pairs[:, 1] == 0
    source_voxels = unique_pairs[source_pairs, 0]
    canonical[source_voxels] = points[first_observation[source_pairs]]
    keep = (support_counts >= 3)
    keep[source_voxels] = True
    observation_keep = keep[inverse]
    points = canonical[inverse[observation_keep]]
    colors, weights, frame_ids = colors[observation_keep], weights[observation_keep], frame_ids[observation_keep]
    xyz, rgb, conf, obs, fused_keys, anchors = fuse_voxels(points, colors, weights, voxel_size=voxel_size,
        anchor_frame=frame_ids, return_anchors=True)
    max_by_key = {tuple(key): max_frame[index] for index, key in enumerate(unique) if keep[index]}
    max_frame = np.asarray([max_by_key[tuple(key)] for key in fused_keys], np.int32)
    return fused_keys, xyz, rgb, conf, obs, max_frame, anchors


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
        if (not metadata.get("valid", False) or metadata.get("schema_version") != 3
                or metadata.get("geometry_implementation_version") != "recal-full-teacher-world-v7-p40-confidence-rgb-anchor"
                or float(metadata.get("voxel_size", -1)) != 0.02):
            raise RuntimeError(f"stale Phase A ReCal cache: {source}")
        target = args.output_root / trajectory_id
        temporary = args.output_root / f".{trajectory_id}.tmp-{os.getpid()}"
        if temporary.exists(): shutil.rmtree(temporary)
        temporary.mkdir(parents=True, exist_ok=True)
        built=[]
        (temporary / "cache_metadata.json").write_text(json.dumps({
            "schema_version": 3, "geometry_implementation_version": GEOMETRY_CACHE_VERSION,
            "source_recal_metadata": metadata, "voxel_size": float(args.voxel_size),
            "alignment_version": "offline-full-trajectory-recal-to-dataset-v3",
            "surface_ownership_version": "immutable-xyz-three-frame-confirmation-v1",
        }, indent=2))
        for frame_limit in FRAME_LIMITS:
            keys, points, rgb, confidence, observation_count, max_frame, anchors = fuse(source, args.dataset_root, frame_limit, args.voxel_size)
            if int(max_frame.max()) > frame_limit: raise RuntimeError("future observation leaked into causal cache")
            np.savez_compressed(
            temporary / f"frame_{frame_limit:03d}.npz", voxel_keys=keys,
                points_xyz=points, points_rgb=rgb, points_confidence=confidence,
                observation_count=observation_count, max_observation_frame=max_frame,
                **anchors,
            ); built.append(frame_limit)
        if tuple(built) != FRAME_LIMITS: raise RuntimeError("incomplete Phase B cache build")
        backup = args.output_root / f".{trajectory_id}.old-{os.getpid()}"
        if backup.exists(): shutil.rmtree(backup)
        if target.exists(): target.replace(backup)
        temporary.replace(target)
        if backup.exists(): shutil.rmtree(backup)


if __name__ == "__main__":
    main()
