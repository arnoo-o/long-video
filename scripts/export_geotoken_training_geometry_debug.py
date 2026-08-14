#!/usr/bin/env python3
"""Export GT RGB beside its ReCal3R full-scene point-cloud render."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import imageio.v2 as imageio
import numpy as np
from PIL import Image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--recal3r-root", type=Path, required=True)
    parser.add_argument("--trajectory-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--frames", type=int, default=33)
    args = parser.parse_args()
    from long_video.geometry.point_renderer import render
    from long_video.types import CameraBatch

    manifest = json.loads((args.dataset_root / "dl3dv_24fps_manifest.json").read_text())
    record = next(item for item in manifest["records"] if item["trajectory_id"] == args.trajectory_id)
    paths = sorted(path for path in (args.dataset_root / record["rgb_dir"]).iterdir() if path.is_file())
    count = min(int(args.frames), len(paths))
    gt = np.stack([np.asarray(Image.open(path).convert("RGB"), np.uint8) for path in paths[:count]])
    c2w = np.load(args.dataset_root / record["target_c2w_local"]).astype(np.float32)[:count]
    intrinsics = np.load(args.dataset_root / record["intrinsics"]).astype(np.float32)[:count]
    scene = np.load(args.recal3r_root / args.trajectory_id / "scene_points.npz")
    points_xyz, points_rgb, confidence = scene["points_xyz"], scene["points_rgb"], scene["points_confidence"]
    node = SimpleNamespace(
        points_xyz=points_xyz, points_rgb=points_rgb, points_confidence=confidence,
        points_source=np.zeros(len(points_xyz), np.int8), parent_point_count=None, quality_metrics={},
    )
    warp = render(node, CameraBatch(c2w, intrinsics, 384, 640), device=args.device, point_radius=0)
    rendered = np.rint(np.clip(warp.rgb, 0, 1) * 255).astype(np.uint8)
    rendered[~warp.visibility] = 0
    panel = np.concatenate([gt, rendered], axis=2)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(args.output, panel, fps=24, macro_block_size=1)
    print(json.dumps({
        "output": str(args.output), "trajectory_id": args.trajectory_id,
        "frames": count, "point_count": int(len(points_xyz)),
        "mean_visibility": float(warp.visibility.mean()), "left": "gt_rgb", "right": "recal3r_full_scene_render",
    }))


if __name__ == "__main__":
    main()
