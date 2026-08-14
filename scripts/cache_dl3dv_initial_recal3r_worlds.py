#!/usr/bin/env python3
"""Build causal ReCal3R node_000 caches; replacement for Pi3 initial worlds."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from PIL import Image


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-root", type=Path, required=True)
    p.add_argument("--cache-root", type=Path, required=True)
    p.add_argument("--recal3r-repo", type=Path, required=True)
    p.add_argument("--recal3r-checkpoint", type=Path, required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--record-count", type=int, default=100)
    a = p.parse_args()
    from long_video.initialization.initial_node_pipeline import initialize_spatial_node
    from long_video.initialization.recal3r_geometry_backend import ReCal3RGeometryBackend
    from long_video.memory.node_store import NodeStore
    from long_video.training.wpf_adaptation import select_balanced_training_records
    from long_video.types import ViewSet
    records = json.loads((a.dataset_root / "dl3dv_24fps_manifest.json").read_text())["records"]
    records = select_balanced_training_records(records, a.record_count)
    backend = ReCal3RGeometryBackend(a.recal3r_checkpoint, a.recal3r_repo, a.device)
    a.cache_root.mkdir(parents=True, exist_ok=True)
    for ordinal, record in enumerate(records, 1):
        target = a.cache_root / record["trajectory_id"]
        if (target / "cache_metadata.json").is_file():
            continue
        paths = sorted((a.dataset_root / record["pi3_initial_rgb_dir"]).glob("*"))
        indices = np.load(a.dataset_root / record["pi3_initial_real_frame_indices"])
        if not 1 <= len(paths) <= 8 or int(indices.max()) > int(record["source_global_frame"]):
            raise RuntimeError(f"invalid causal source views: {record['trajectory_id']}")
        c2w = np.load(a.dataset_root / record["pi3_initial_c2w_local"]).astype(np.float32)
        intrinsics = np.load(a.dataset_root / record["pi3_initial_intrinsics"]).astype(np.float32)
        rgb = [np.asarray(Image.open(path).convert("RGB"), np.uint8) for path in paths]
        pad = 8 - len(rgb)
        if pad:
            rgb = [rgb[0]] * pad + rgb
            c2w = np.concatenate([np.repeat(c2w[:1], pad, 0), c2w])
            intrinsics = np.concatenate([np.repeat(intrinsics[:1], pad, 0), intrinsics])
            indices = np.r_[np.repeat(indices[:1], pad), indices]
        rgb = np.stack(rgb); shape = rgb.shape[:3]
        backend.reset()
        views = ViewSet(rgb=rgb, depth=np.full(shape, np.nan, np.float32),
                        depth_confidence=np.zeros(shape, np.float32), c2w=c2w, intrinsics=intrinsics,
                        source=np.zeros(shape, np.int8), image_confidence=np.ones(shape, np.float32))
        initialize_spatial_node(views, backend, {"voxel_size": .02, "node_store": NodeStore(target),
            "view_frame_indices": indices.astype(int).tolist(), "target_frame_start": int(record["source_global_frame"]) + 1})
        (target / "cache_metadata.json").write_text(json.dumps({"trajectory_id": record["trajectory_id"],
            "geometry_backend": "recal3r", "recal3r_checkpoint": str(a.recal3r_checkpoint.resolve()),
            "uses_future_gt": False}, indent=2))
        print(json.dumps({"index": ordinal, "trajectory_id": record["trajectory_id"]}), flush=True)


if __name__ == "__main__":
    main()
