#!/usr/bin/env python3
"""Download only requested TartanGround pinhole RGB-D trajectories officially."""
from __future__ import annotations

import argparse
from pathlib import Path

import tartanair as ta


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--env", action="append", required=True)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    ta.init(str(args.root))
    ta.download_ground(
        env=args.env,
        version=["diff"],
        traj=[],
        modality=["image", "depth", "meta"],
        camera_name=["lcam_front"],
        unzip=True,
        delete_zip=True,
        num_workers=args.workers,
        data_source="airlab",
    )


if __name__ == "__main__":
    main()
