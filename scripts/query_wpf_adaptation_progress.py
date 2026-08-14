#!/usr/bin/env python3
"""Read WPF adaptation training logs without touching the training process."""
from __future__ import annotations

import argparse
import json
import statistics
from datetime import timedelta
from pathlib import Path


def progress_summary(run_dir: Path):
    metrics = sorted((run_dir / "metrics").glob("step_*.json"))
    checkpoints = sorted((run_dir / "checkpoints").glob("checkpoint_step_*.pt"))
    status_path = run_dir / "status.json"
    status = json.loads(status_path.read_text()) if status_path.exists() else {}
    if not metrics:
        return status or {"global_step": 0, "total_steps": 1400}
    latest = json.loads(metrics[-1].read_text())
    durations = [float(value) for value in status.get("recent_step_times_sec", [])]
    if not durations:
        recent = [json.loads(path.read_text()) for path in metrics[-20:]]
        durations = [float(item["optimizer_step_time_sec"]) for item in recent]
    average = statistics.fmean(durations) if durations else None
    total_steps = int(status.get("total_steps", 1400))
    remaining = max(0, total_steps - int(latest["global_step"]))
    return {
        "global_step": int(latest["global_step"]), "total_steps": total_steps,
        "latest_metrics": str(metrics[-1]), "L_total": latest.get("L_total"),
        "L_fill": latest.get("L_fill"), "L_keep": latest.get("L_keep"),
        "selected_stage": latest.get("selected_stage"),
        "selected_step": latest.get("selected_step"), "sigma_t": latest.get("sigma_t"),
        "world_mask_mean": latest.get("world_mask_mean"),
        "learning_rate": latest.get("lr"), "grad_norm": latest.get("grad_norm"),
        "position_counts": {
            key: latest.get(key) for key in (
                "stage1_step0", "stage1_step1", "stage2_step0", "stage2_step1",
            )
        },
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
