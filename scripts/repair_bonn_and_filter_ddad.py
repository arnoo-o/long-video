#!/usr/bin/env python3
"""Repair Bonn pose scale and filter weak DDAD supervision records.

This utility is intentionally explicit and idempotent.  It changes only
canonical Bonn pose/correspondence artifacts and training manifests; RGB,
depth, and latent caches are never rewritten.
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from long_video.data.rgbd_memory import build_causal_correspondence_cache, localize_c2w


def atomic_npy(path: Path, value: np.ndarray) -> None:
    temporary = path.with_name(path.stem + ".repair.tmp.npy")
    np.save(temporary, value)
    os.replace(temporary, path)


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def rigid_rotation(rotation: np.ndarray) -> np.ndarray:
    """Remove the uniform scale accidentally retained in Bonn's marker frame."""
    u, _, vt = np.linalg.svd(np.asarray(rotation, dtype=np.float64))
    corrected = u @ vt
    if np.linalg.det(corrected) < 0:
        u[:, -1] *= -1
        corrected = u @ vt
    if not np.allclose(corrected.T @ corrected, np.eye(3), atol=1e-8):
        raise RuntimeError("Bonn rotation repair did not produce an SO(3) matrix")
    return corrected


def repair_bonn_row(row: dict, *, apply: bool) -> dict:
    pose_path = Path(row["c2w_abs"])
    local_path = Path(row["c2w_local"])
    cache_path = Path(row["correspondence_cache"])
    c2w = np.load(pose_path)
    repaired = np.array(c2w, dtype=np.float64, copy=True)
    before = np.linalg.det(repaired[:, :3, :3])
    repaired[:, :3, :3] = np.stack([rigid_rotation(value) for value in repaired[:, :3, :3]])
    after = np.linalg.det(repaired[:, :3, :3])
    if not np.allclose(after, 1.0, atol=1e-8):
        raise RuntimeError(f"{row['record_id']}: repaired c2w is not rigid")
    if apply:
        atomic_npy(pose_path, repaired)
        atomic_npy(local_path, localize_c2w(repaired))
        metadata = json.loads(Path(row["metadata"]).read_text(encoding="utf-8"))
        stride = int((metadata.get("correspondence") or {}).get("pixel_stride", 4))
        depth_dir = Path(row["depth_dir"])
        depth = sorted(depth_dir.glob("*.png"), key=lambda path: path.name)
        intrinsics = np.load(row["intrinsics"])
        temporary_cache = cache_path.with_name(cache_path.stem + ".repair.tmp.npz")
        if temporary_cache.exists():
            temporary_cache.unlink()
        stats = build_causal_correspondence_cache(depth, repaired, intrinsics, temporary_cache, chunk_count=int(row["chunk_count"]), pixel_stride=stride)
        os.replace(temporary_cache, cache_path)
        metadata["pose_repair"] = {
            "method": "nearest_SO3_per_pose",
            "reason": "remove uniform scale from Bonn marker transform",
            "determinant_before_min": float(before.min()),
            "determinant_before_max": float(before.max()),
            "determinant_after_min": float(after.min()),
            "determinant_after_max": float(after.max()),
        }
        metadata["correspondence"] = stats
        atomic_json(Path(row["metadata"]), metadata)
    return {"record_id": row["record_id"], "det_before": [float(before.min()), float(before.max())], "det_after": [float(after.min()), float(after.max())]}


def repair_bonn_entry(value: tuple[dict, bool]) -> dict:
    return repair_bonn_row(value[0], apply=value[1])


def depth_p10(row: dict) -> float:
    frames = sorted(Path(row["depth_dir"]).glob("*.png"), key=lambda path: path.name)
    values = []
    for path in frames[: int(row["frame_count"])]:
        depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if depth is None:
            raise RuntimeError(f"unreadable depth: {path}")
        values.append(float(np.count_nonzero(depth) / depth.size))
    return float(np.quantile(values, 0.1))


def correspondence_rows(row: dict) -> int:
    with np.load(row["correspondence_cache"], allow_pickle=False) as cache:
        return int(len(cache["query_frame"]))


def ddad_quality(row: dict) -> dict:
    return {"depth_valid_p10": depth_p10(row), "correspondence_rows": correspondence_rows(row)}


def filter_manifest(path: Path, rejected: set[str], qualities: dict[str, dict], *, apply: bool) -> tuple[int, int]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    original = payload["records"]
    kept = []
    for row in original:
        if row.get("dataset") == "ddad" and row.get("record_id") in rejected:
            continue
        if row.get("dataset") == "ddad" and row.get("record_id") in qualities:
            row = dict(row)
            row["quality_filter"] = qualities[row["record_id"]]
        kept.append(row)
    payload["records"] = kept
    if apply:
        atomic_json(path, payload)
    return len(original), len(kept)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--unified-root", type=Path, required=True)
    parser.add_argument("--min-depth-p10", type=float, default=0.005)
    parser.add_argument("--min-correspondence-rows", type=int, default=1000)
    parser.add_argument("--bonn-workers", type=int, default=1)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    unified = args.unified_root
    manifest_all = unified / "manifest_all.json"
    rows = json.loads(manifest_all.read_text(encoding="utf-8"))["records"]
    bonn = [row for row in rows if row.get("dataset") == "bonn"]
    ddad = [row for row in rows if row.get("dataset") == "ddad"]
    if args.bonn_workers < 1:
        raise ValueError("--bonn-workers must be positive")
    jobs = [(row, args.apply) for row in bonn]
    if args.bonn_workers == 1:
        repaired = [repair_bonn_entry(job) for job in jobs]
    else:
        with ProcessPoolExecutor(max_workers=args.bonn_workers) as pool:
            repaired = list(pool.map(repair_bonn_entry, jobs))
    qualities = {row["record_id"]: ddad_quality(row) for row in ddad}
    rejected = {record_id for record_id, quality in qualities.items() if quality["depth_valid_p10"] < args.min_depth_p10 or quality["correspondence_rows"] < args.min_correspondence_rows}
    manifests = [unified / name for name in ("manifest_train.json", "manifest_train_p3.json", "manifest_train_units_3chunk.json", "manifest_all.json")]
    manifest_counts = {path.name: filter_manifest(path, rejected, qualities, apply=args.apply) for path in manifests if path.is_file()}
    report = {
        "mode": "apply" if args.apply else "dry_run",
        "bonn_records": len(bonn),
        "bonn_repaired": len(repaired),
        "ddad_thresholds": {"depth_valid_p10": args.min_depth_p10, "correspondence_rows": args.min_correspondence_rows},
        "ddad_total": len(ddad), "ddad_kept": len(ddad) - len(rejected), "ddad_rejected": len(rejected),
        "rejected_ddad": [{"record_id": key, **qualities[key]} for key in sorted(rejected)],
        "manifest_counts": {key: {"before": value[0], "after": value[1]} for key, value in manifest_counts.items()},
    }
    if args.apply:
        atomic_json(unified / "ddad_quality_filter_report.json", report)
    print(json.dumps({key: report[key] for key in ("mode", "bonn_records", "ddad_thresholds", "ddad_total", "ddad_kept", "ddad_rejected", "manifest_counts")}, indent=2))


if __name__ == "__main__":
    main()
