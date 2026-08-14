#!/usr/bin/env python3
"""Read-only status query for a GeoToken training run."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    metrics = sorted((args.run_dir / "metrics").glob("step_*.json"))
    checkpoints = sorted((args.run_dir / "checkpoints").glob("*.pt"))
    if not metrics:
        raise SystemExit("no GeoToken metrics have been written")
    latest = json.loads(metrics[-1].read_text())
    recent = [json.loads(path.read_text()) for path in metrics[-20:]]
    def step_seconds(item):
        # Current runs report complete wall-clock time; retain compatibility
        # with earlier GeoToken metrics that only had optimizer_step_seconds.
        value = item.get("total_step_seconds")
        if value is None:
            value = item["optimizer_step_seconds"]
        return float(value)

    average = sum(step_seconds(item) for item in recent) / len(recent)
    remaining = max(0, 2000 - int(latest["global_step"])) * average
    print(json.dumps({
        "global_step": f"{latest['global_step']}/2000",
        "phase": latest["phase"],
        "latest_metrics": str(metrics[-1]),
        "flow_loss": latest["flow_loss"],
        "stage0_flow_mse": latest["stage0_flow_mse"],
        "stage1_flow_mse": latest["stage1_flow_mse"],
        "stage2_flow_mse": latest["stage2_flow_mse"],
        "rollout_length": latest["rollout_length"],
        "learning_rate": latest["learning_rate"],
        "grad_norm": latest["grad_norm"],
        "latest_checkpoint": str(checkpoints[-1]) if checkpoints else None,
        "elapsed_seconds": latest["elapsed_seconds"],
        "recent_average_step_seconds": average,
        "estimated_remaining_seconds": remaining,
    }, indent=2))


if __name__ == "__main__":
    main()
