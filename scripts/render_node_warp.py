#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np

from long_video.geometry.point_renderer import render
from long_video.memory.node_store import NodeStore
from long_video.types import CameraBatch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", required=True)
    parser.add_argument("--node-id", default="node_000")
    parser.add_argument("--poses", required=True)
    parser.add_argument("--intrinsics", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    node = NodeStore(args.session).load(args.node_id)
    poses = np.load(args.poses).astype(np.float32)
    intrinsics = np.load(args.intrinsics).astype(np.float32)
    if intrinsics.ndim == 2:
        intrinsics = np.repeat(intrinsics[None], len(poses), axis=0)
    warp = render(
        node, CameraBatch(poses, intrinsics, args.height, args.width), device=args.device
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output, rgb=warp.rgb, depth=warp.depth, visibility=warp.visibility,
        confidence=warp.confidence, source=warp.source,
        coverage_per_frame=warp.coverage_per_frame,
    )
    print({"output": str(output), "coverage": warp.coverage_per_frame.tolist()})


if __name__ == "__main__":
    main()
