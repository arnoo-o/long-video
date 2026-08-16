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
    parser.add_argument("--forward-distance", type=float, default=1.0)
    parser.add_argument("--return-distance", type=float, default=1.0)
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
    # The return leg is left-front in the camera basis at that point.  The
    # independent eased components keep velocity zero at the phase boundary.
    return_forward = eased_increments(args.return_distance / math.sqrt(2.0), step_count)
    return_left = eased_increments(args.return_distance / math.sqrt(2.0), step_count)
    dt = 1.0 / args.fps
    chunks = []
    for chunk_index in range(args.chunks_each * 2):
        start = chunk_index * 32
        controls = []
        for local_index, yaw_delta in enumerate(yaw[start:start + 32]):
            global_index = start + local_index
            forward_delta = move_out[global_index] if global_index < step_count else return_forward[global_index - step_count]
            left_delta = 0.0 if global_index < step_count else return_left[global_index - step_count]
            controls.append({
                "delta_time": dt,
                "yaw_delta": float(yaw_delta),
                "forward": float(forward_delta / dt),
                "backward": 0.0,
                "strafe_left": float(left_delta / dt),
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
        "return_left_front_distance": args.return_distance,
    }, indent=2))


if __name__ == "__main__":
    main()
