#!/usr/bin/env python3
"""Extract one matched Holo360D frame and export the canonical 8-view sequence."""
import argparse
import tempfile
from pathlib import Path
from zipfile import ZipFile

from long_video.data.holo360d import Holo360DReader
from long_video.data.sequence import write_sequence
from long_video.initialization.view_completion import HoloOracleCompletion


def extract_frame(archive, output, frame_index=0):
    with ZipFile(archive) as handle:
        rgb = sorted(name for name in handle.namelist() if "/rgb/" in name and name.endswith(".jpg"))
        if not rgb:
            raise RuntimeError(f"No RGB panorama found in {archive}")
        selected = rgb[int(frame_index)]
        root, stem = selected.split("/")[0], Path(selected).stem
        for suffix in (
            f"rgb/{stem}.jpg", f"depth/mesh_depth/{stem}.exr",
            f"mask/{stem}.jpg", f"poses/{stem}.txt",
        ):
            handle.extract(f"{root}/{suffix}", output)
    return Path(output) / root


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--frame-index", type=int, default=0)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--fov", type=float, default=90)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="holo360d_") as temporary:
        scene = extract_frame(args.zip, temporary, args.frame_index)
        frame = Holo360DReader(scene).read(0)
        views = HoloOracleCompletion(args.fov, args.height, args.width).complete(
            frame.rgb, frame.depth, frame.c2w, frame.mask
        )
        write_sequence(
            args.output, views.rgb, views.depth, views.depth_confidence > 0,
            views.c2w, views.intrinsics, prompt="indoor panorama",
            metadata={"source": "Holo360D", "frame_id": frame.frame_id,
                      "depth_convention": views.depth_convention},
        )
    print(args.output)


if __name__ == "__main__":
    main()
