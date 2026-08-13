#!/usr/bin/env python3
"""Cache immutable source-only Pi3 node_000 worlds for DL3DV trajectories."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--pi3-repo", type=Path, required=True)
    parser.add_argument("--pi3-checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--split", default="train")
    parser.add_argument("--record-count", type=int, default=100)
    return parser.parse_args()


def digest_inputs(paths, *arrays):
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    for array in arrays:
        value = np.ascontiguousarray(array)
        digest.update(str(value.dtype).encode())
        digest.update(np.asarray(value.shape, np.int64).tobytes())
        digest.update(value.tobytes())
    return digest.hexdigest()


def valid_cache(path, record, source_digest, checkpoint):
    try:
        metadata = json.loads((path / "cache_metadata.json").read_text())
        return (
            metadata["trajectory_id"] == record["trajectory_id"]
            and metadata["source_views_sha256"] == source_digest
            and metadata["pi3_checkpoint"] == str(checkpoint.resolve())
            and metadata["uses_future_gt"] is False
            and (path / "nodes/node_000/metadata.json").is_file()
        )
    except (FileNotFoundError, KeyError, json.JSONDecodeError):
        return False


def main():
    args = parse_args()
    from long_video.initialization.geometry_backend import Pi3GeometryBackend
    from long_video.initialization.initial_node_pipeline import initialize_spatial_node
    from long_video.memory.node_store import NodeStore
    from long_video.training.stage2_cleanup import select_balanced_training_records
    from long_video.types import ViewSet

    manifest = json.loads((args.dataset_root / "dl3dv_24fps_manifest.json").read_text())
    if args.split != "train":
        raise ValueError("initial world training cache only supports the train split")
    records = select_balanced_training_records(manifest["records"], args.record_count)
    backend = Pi3GeometryBackend(args.pi3_checkpoint, args.pi3_repo, args.device)
    args.cache_root.mkdir(parents=True, exist_ok=True)
    built = cached = skipped = 0
    for ordinal, record in enumerate(records, 1):
        rgb_paths = sorted((args.dataset_root / record["pi3_initial_rgb_dir"]).glob("*"))
        if not 1 <= len(rgb_paths) <= 8:
            raise RuntimeError(f"invalid causal Pi3 view count: {record['trajectory_id']}")
        real_indices = np.load(args.dataset_root / record["pi3_initial_real_frame_indices"])
        if len(real_indices) != len(rgb_paths) or int(real_indices.max()) > int(record["source_global_frame"]):
            raise RuntimeError(f"future view in Pi3 initialization: {record['trajectory_id']}")
        c2w = np.load(args.dataset_root / record["pi3_initial_c2w_local"]).astype(np.float32)
        intrinsics = np.load(args.dataset_root / record["pi3_initial_intrinsics"]).astype(np.float32)
        source_digest = digest_inputs(rgb_paths, c2w, intrinsics, real_indices)
        target = args.cache_root / record["trajectory_id"]
        if valid_cache(target, record, source_digest, args.pi3_checkpoint):
            cached += 1
            print(json.dumps({"index": ordinal, "trajectory_id": record["trajectory_id"],
                              "status": "cached"}), flush=True)
            continue
        rgb = [np.asarray(Image.open(path).convert("RGB"), np.uint8) for path in rgb_paths]
        padding = 8 - len(rgb)
        if padding:
            rgb = [rgb[0]] * padding + rgb
            c2w = np.concatenate([np.repeat(c2w[:1], padding, axis=0), c2w])
            intrinsics = np.concatenate([np.repeat(intrinsics[:1], padding, axis=0), intrinsics])
            padded_indices = [int(real_indices[0])] * padding + real_indices.astype(int).tolist()
        else:
            padded_indices = real_indices.astype(int).tolist()
        rgb = np.stack(rgb)
        shape = rgb.shape[:3]
        views = ViewSet(rgb=rgb, depth=np.full(shape, np.nan, np.float32),
                        depth_confidence=np.zeros(shape, np.float32), c2w=c2w,
                        intrinsics=intrinsics, source=np.zeros(shape, np.int8),
                        image_confidence=np.ones(shape, np.float32))
        store = NodeStore(target)
        initialize_spatial_node(
            views, backend,
            {"voxel_size": 0.02, "node_store": store,
             "view_frame_indices": padded_indices,
             "target_frame_start": int(record["source_global_frame"]) + 1},
        )
        metadata = {
            "schema_version": 1, "trajectory_id": record["trajectory_id"],
            "scene_hash": record["scene_hash"], "source_views_sha256": source_digest,
            "source_global_frame": int(record["source_global_frame"]),
            "pi3_initial_real_frame_indices": real_indices.tolist(),
            "pi3_effective_view_frame_indices": padded_indices,
            "causal_view_padding_count": padding,
            "pi3_checkpoint": str(args.pi3_checkpoint.resolve()),
            "voxel_size": 0.02, "uses_future_gt": False,
        }
        (target / "cache_metadata.json").write_text(json.dumps(metadata, indent=2))
        built += 1
        print(json.dumps({"index": ordinal, "total": len(records),
                          "trajectory_id": record["trajectory_id"], "status": "built"}), flush=True)
    summary = {"records": len(records), "built": built, "cached": cached, "skipped": skipped,
               "complete": built + cached + skipped == len(records), "uses_future_gt": False}
    (args.cache_root / "cache_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
