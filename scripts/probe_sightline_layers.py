"""Aggregate real Helios Sightline probe measurements; never fabricate metrics."""
from __future__ import annotations
import argparse, json
from pathlib import Path

REQUIRED = {
    "layer", "sigma", "correspondence_mrr", "top1", "top5",
    "positive_attention_mass", "memory_attention_mass", "wrong_ray_delta",
    "memory_zero_delta", "memory_shuffle_delta", "fm_loss", "corr_loss",
    "alpha", "alpha_grad", "vram_gb", "step_time_sec", "ablation_time_sec",
    "ranking_source", "raw_qk_mrr", "raw_qk_top1", "raw_qk_top5",
}

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input-jsonl", required=True, help="JSONL emitted by the real Helios probe adapter")
    p.add_argument("--out", required=True)
    p.add_argument("--plot-dir")
    a = p.parse_args()
    rows = []
    for line in Path(a.input_jsonl).read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        missing = REQUIRED - row.keys()
        if missing:
            raise SystemExit(f"probe row is missing real measurements: {sorted(missing)}")
        rows.append(row)
    if not rows:
        raise SystemExit("real Helios probe produced no rows; refusing to create output")
    rows.sort(key=lambda row: (int(row["layer"]), float(row["sigma"])))
    Path(a.out).write_text(json.dumps({"source": "real_helios_probe", "rows": rows}, indent=2) + "\n")
    if a.plot_dir:
        import matplotlib.pyplot as plt
        out = Path(a.plot_dir); out.mkdir(parents=True, exist_ok=True)
        x = list(range(len(rows)))
        charts = {
            "loss_curve": (("fm_loss", "corr_loss"), "loss"),
            "sightline_signal": (("alpha", "alpha_grad", "wrong_ray_delta"), "signal"),
            "correspondence": (("raw_qk_mrr", "raw_qk_top1", "raw_qk_top5"), "score"),
            "memory_signal": (("memory_attention_mass", "memory_zero_delta", "memory_shuffle_delta"), "signal"),
            "efficiency": (("vram_gb", "step_time_sec", "ablation_time_sec"), "value"),
        }
        for name, (keys, ylabel) in charts.items():
            plt.figure()
            for key in keys:
                values=[float('nan') if row[key] is None else float(row[key]) for row in rows]
                plt.plot(x, values, marker="o", label=key)
            plt.title(name); plt.xlabel("measured probe row"); plt.ylabel(ylabel); plt.legend(); plt.tight_layout()
            plt.savefig(out / f"{name}.png"); plt.close()

if __name__ == "__main__":
    main()
