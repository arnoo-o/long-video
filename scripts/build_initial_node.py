#!/usr/bin/env python3
import argparse

from long_video.initialization.view_completion import PrecomputedCompletion
from long_video.memory.node_builder import build_from_views
from long_video.memory.node_store import NodeStore


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--views", required=True)
    parser.add_argument("--session", required=True)
    parser.add_argument("--node-id", default="node_000")
    parser.add_argument("--voxel-size", type=float, default=0.02)
    args = parser.parse_args()
    views = PrecomputedCompletion(args.views).complete()
    node = build_from_views(
        views, node_id=args.node_id, center_c2w=views.c2w[0],
        voxel_size=args.voxel_size, status="active",
    )
    NodeStore(args.session).save(node)
    print({"node_id": node.node_id, "points": len(node.points_xyz), "session": args.session})


if __name__ == "__main__":
    main()
