#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from long_video.data.holo360d import Holo360DReader
from long_video.initialization.geometry_backend import Pi3GeometryBackend
from long_video.initialization.view_completion import HoloOracleCompletion


def extract_first(zip_path, destination):
    with ZipFile(zip_path) as archive:
        rgb = sorted(name for name in archive.namelist() if "/rgb/" in name and name.endswith(".jpg"))
        if not rgb:
            raise RuntimeError(f"No RGB frames in {zip_path}")
        root = rgb[0].split("/")[0]
        stem = Path(rgb[0]).stem
        for name in (
            f"{root}/rgb/{stem}.jpg",
            f"{root}/depth/mesh_depth/{stem}.exr",
            f"{root}/mask/{stem}.jpg",
            f"{root}/poses/{stem}.txt",
        ):
            archive.extract(name, destination)
    return Path(destination) / root


def preview(depth):
    valid = np.isfinite(depth) & (depth > 0)
    image = np.zeros(depth.shape, np.uint8)
    if valid.any():
        low, high = np.percentile(depth[valid], (2, 98))
        image[valid] = np.clip((depth[valid] - low) / max(high - low, 1e-6) * 255, 0, 255)
    return image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--height", type=int, default=518)
    parser.add_argument("--width", type=int, default=518)
    parser.add_argument("--fov", type=float, default=90.0)
    args = parser.parse_args()

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    scene = extract_first(args.zip, output / "extracted")
    frame = Holo360DReader(scene).read(0)
    views = HoloOracleCompletion(args.fov, args.height, args.width).complete(
        frame.rgb, frame.depth, frame.c2w, frame.mask, observed_indices=(0,)
    )
    prediction = Pi3GeometryBackend(args.checkpoint, args.repo).predict(
        views.rgb, views.c2w, views.intrinsics, views.depth, np.isfinite(views.depth)
    )
    for index in range(8):
        Image.fromarray(views.rgb[index]).save(output / f"rgb_{index:02d}.png")
        Image.fromarray(preview(prediction.depth[index])).save(output / f"depth_pred_{index:02d}.png")
        Image.fromarray((prediction.depth_confidence[index] * 255).astype(np.uint8)).save(
            output / f"confidence_{index:02d}.png"
        )
    metrics = {
        **prediction.diagnostics,
        **prediction.scale_info,
        "depth_shape": list(prediction.depth.shape),
        "point_maps_shape": list(prediction.point_maps.shape),
        "predicted_c2w_shape": list(prediction.predicted_c2w.shape),
    }
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
