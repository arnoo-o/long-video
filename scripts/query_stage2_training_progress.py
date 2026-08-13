#!/usr/bin/env python3
"""Read Stage2 cleanup training logs without touching the training process."""
from __future__ import annotations
import argparse, json, statistics
from datetime import timedelta
from pathlib import Path


def progress_summary(run_dir: Path):
    metrics = sorted((run_dir / "metrics").glob("step_*.json"))
    checkpoints = sorted((run_dir / "checkpoints").glob("checkpoint_step_*.pt"))
    status_path = run_dir / "status.json"
    status = json.loads(status_path.read_text()) if status_path.exists() else {}
    if not metrics:
        status = run_dir / "status.json"
        return json.loads(status.read_text()) if status.exists() else {"global_step": 0, "total_steps": 1500}
    latest = json.loads(metrics[-1].read_text())
    recent = [json.loads(path.read_text()) for path in metrics[-20:]]
    durations = [float(value) for value in status.get("recent_step_times_sec", [])]
    if not durations:
        durations = [float(item["optimizer_step_time_sec"]) for item in recent if "optimizer_step_time_sec" in item]
    average = statistics.fmean(durations) if durations else None
    remaining = max(0, 1500 - int(latest["global_step"]))
    return {
        "global_step": int(latest["global_step"]), "total_steps": 1500,
        "latest_metrics": str(metrics[-1]), "total_loss": latest.get("total_loss"),
        "selected_stage2_step": latest.get("selected_stage2_step"),
        "stage2_losses": [latest.get(f"stage2_step{i}_loss") for i in range(4)],
        "learning_rate": latest.get("lr"), "grad_norm": latest.get("grad_norm"),
        "trajectory_length": latest.get("trajectory_length"),
        "world_point_count": latest.get("world_point_count"),
        "latest_checkpoint": str(checkpoints[-1]) if checkpoints else None,
        "elapsed_training_time_sec": latest.get("elapsed_training_time_sec"),
        "recent_20_average_step_sec": average,
        "estimated_remaining": str(timedelta(seconds=round(remaining * average))) if average else None,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(progress_summary(args.run_dir), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
