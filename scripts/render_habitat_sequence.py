#!/usr/bin/env python3
import argparse
from pathlib import Path

from long_video.habitat.renderer import HabitatSequenceRenderer
from long_video.habitat.trajectory_generator import generate_validation_trajectory


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-dataset-config", required=True)
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--hfov", type=float, default=90)
    parser.add_argument("--move-speed", type=float, default=0.5)
    args = parser.parse_args()
    renderer = HabitatSequenceRenderer(
        args.scene_dataset_config, args.scene_id, args.height, args.width, args.hfov
    )
    try:
        poses, controls = generate_validation_trajectory(
            renderer.initial_c2w(), move_speed=args.move_speed
        )
        poses=renderer.constrain_to_navmesh(poses)
        renderer.render(poses, args.output, controls)
    finally:
        renderer.close()
    print(args.output)


if __name__ == "__main__":
    main()
