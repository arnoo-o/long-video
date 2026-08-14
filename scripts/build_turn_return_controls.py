#!/usr/bin/env python3
"""Build chunked, ease-in/ease-out turn-and-return camera controls."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chunks-each", type=int, default=6)
    parser.add_argument("--turn-degrees", type=float, default=180.0)
    parser.add_argument("--forward-distance", type=float, default=0.5)
    parser.add_argument("--fps", type=float, default=24.0)
    return parser.parse_args()


def eased_increments(total, count):
    cumulative = [
        total * (0.5 - 0.5 * math.cos(math.pi * index / count))
        for index in range(count + 1)
    ]
    return [right - left for left, right in zip(cumulative, cumulative[1:])]


def main():
    args = parse_args()
    if args.chunks_each <= 0 or args.fps <= 0:
        raise ValueError("chunks-each and fps must be positive")
    step_count = args.chunks_each * 32
    yaw_out = eased_increments(math.radians(args.turn_degrees), step_count)
    move_out = eased_increments(args.forward_distance, step_count)
    yaw_back = eased_increments(-math.radians(args.turn_degrees), step_count)
    yaw = yaw_out + yaw_back
    distance = move_out + [0.0] * step_count
    dt = 1.0 / args.fps
    chunks = []
    for chunk_index in range(args.chunks_each * 2):
        controls = [{
            "delta_time": dt, "yaw_delta": 0.0, "forward": 0.0,
            "backward": 0.0, "strafe_left": 0.0, "strafe_right": 0.0,
        }]
        start = chunk_index * 32
        for yaw_delta, move_delta in zip(
            yaw[start:start + 32], distance[start:start + 32], strict=True,
        ):
            controls.append({
                "delta_time": dt,
                "yaw_delta": float(yaw_delta),
                "forward": float(move_delta / dt),
                "backward": 0.0,
                "strafe_left": 0.0,
                "strafe_right": 0.0,
            })
        chunks.append(controls)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(chunks, indent=2), encoding="utf-8")
    print(json.dumps({
        "chunks": len(chunks),
        "frames_per_chunk": 33,
        "outbound_turn_degrees": math.degrees(sum(yaw_out)),
        "outbound_forward_distance": sum(move_out),
        "return_turn_degrees": math.degrees(sum(yaw_back)),
    }, indent=2))


if __name__ == "__main__":
    main()
