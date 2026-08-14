#!/usr/bin/env python3
"""Initialize a causal ReCal3R node_000 from one source image."""
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
    parser.add_argument("--recal3r-repo", type=Path, required=True)
    parser.add_argument("--recal3r-checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--height", type=int, default=384)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--fov-degrees", type=float, default=90.0)
    parser.add_argument("--voxel-size", type=float, default=0.008)
    return parser.parse_args()


def main():
    args = parse_args()
    from long_video.initialization.initial_node_pipeline import initialize_spatial_node
    from long_video.initialization.recal3r_geometry_backend import ReCal3RGeometryBackend
    from long_video.memory.node_store import NodeStore
    from long_video.types import ViewSet, Z_DEPTH

    image = ImageOps.fit(Image.open(args.source_image).convert("RGB"), (args.width, args.height),
                         method=Image.Resampling.LANCZOS, centering=(0.5, 0.5))
    rgb = np.repeat(np.asarray(image, np.uint8)[None], 8, axis=0)
    poses = np.repeat(np.eye(4, dtype=np.float32)[None], 8, axis=0)
    focal = 0.5 * args.width / np.tan(np.deg2rad(args.fov_degrees) * 0.5)
    intrinsics = np.repeat(np.array([[focal, 0.0, (args.width - 1) * 0.5],
                                    [0.0, focal, (args.height - 1) * 0.5],
                                    [0.0, 0.0, 1.0]], dtype=np.float32)[None], 8, axis=0)
    shape = rgb.shape[:3]
    views = ViewSet(rgb=rgb, depth=np.full(shape, np.nan, np.float32),
                    depth_confidence=np.zeros(shape, np.float32), c2w=poses,
                    intrinsics=intrinsics, source=np.zeros(shape, np.int8),
                    image_confidence=np.ones(shape, np.float32), depth_convention=Z_DEPTH)
    node = initialize_spatial_node(
        views, ReCal3RGeometryBackend(args.recal3r_checkpoint, args.recal3r_repo, args.device),
        {"node_store": NodeStore(args.output_session), "node_id": "node_000",
         "center_c2w": np.eye(4, dtype=np.float32), "created_frame": 0,
         "voxel_size": args.voxel_size},
    )
    source_path = args.output_session / "source.png"
    image.save(source_path)
    summary = {"source_image": str(args.source_image), "prepared_source_image": str(source_path),
               "resolution": [args.height, args.width], "fov_degrees": args.fov_degrees,
               "geometry_backend": "recal3r", "voxel_size": args.voxel_size,
               "point_count": int(len(node.points_xyz)), "uses_future_gt": False}
    (args.output_session / "initialization.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
