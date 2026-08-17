#!/usr/bin/env python3
"""Create the source-only session consumed by causal Pi3X inference."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-image", type=Path, required=True)
    parser.add_argument("--output-session", type=Path, required=True)
    parser.add_argument("--height", type=int, default=384)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--fov-degrees", type=float, default=90.0)
    return parser.parse_args()


def main():
    args = parse_args()
    from long_video.memory.node_store import NodeStore
    from long_video.types import ScaleMetadata, SpatialNode, Z_DEPTH

    if args.height <= 0 or args.width <= 0 or not 1 < args.fov_degrees < 179:
        raise ValueError("invalid image dimensions or field of view")
    image = ImageOps.fit(
        Image.open(args.source_image).convert("RGB"), (args.width, args.height),
        method=Image.Resampling.LANCZOS, centering=(0.5, 0.5),
    )
    rgb = np.asarray(image, np.uint8)[None]
    c2w = np.eye(4, dtype=np.float32)[None]
    focal = 0.5 * args.width / np.tan(np.deg2rad(args.fov_degrees) * 0.5)
    intrinsics = np.array([
        [focal, 0.0, (args.width - 1) * 0.5],
        [0.0, focal, (args.height - 1) * 0.5],
        [0.0, 0.0, 1.0],
    ], dtype=np.float32)[None]
    empty_xyz = np.empty((0, 3), np.float32)
    node = SpatialNode(
        node_id="node_000", status="active", parent_id=None,
        center_c2w=c2w[0].copy(), created_frame=0, coverage_radius=0.0,
        bbox_min=np.zeros(3, np.float32), bbox_max=np.zeros(3, np.float32),
        view_rgb=rgb, view_depth=np.full((1, args.height, args.width), np.nan, np.float32),
        view_c2w=c2w, view_intrinsics=intrinsics,
        points_xyz=empty_xyz, points_rgb=np.empty((0, 3), np.uint8),
        points_confidence=np.empty(0, np.float32), points_source=np.empty(0, np.int8),
        observation_count=np.empty(0, np.int32), depth_convention=Z_DEPTH,
        view_source=np.zeros((1, args.height, args.width), np.int8),
        view_image_confidence=np.ones((1, args.height, args.width), np.float32),
        view_depth_confidence=np.zeros((1, args.height, args.width), np.float32),
        scale=ScaleMetadata(),
        quality_metrics={"source_only": True, "canonical_source_frame": 0},
        model_versions={"initializer": "source-only-session-v1"},
    )
    NodeStore(args.output_session).save(node)
    source_path = args.output_session / "source.png"
    image.save(source_path)
    summary = {
        "source_image": str(args.source_image), "prepared_source_image": str(source_path),
        "resolution": [args.height, args.width], "fov_degrees": args.fov_degrees,
        "source_only": True, "point_count": 0, "uses_future_gt": False,
    }
    (args.output_session / "initialization.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8",
    )
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()

