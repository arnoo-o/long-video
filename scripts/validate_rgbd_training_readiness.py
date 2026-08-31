#!/usr/bin/env python3
"""Strict end-to-end readiness audit for the formal RGB-D training manifests."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from long_video.training.rgbd_memory_data import RGBDMemoryRecord
from long_video.training.sightline_data import validate_rgbd_record_latent, validate_rgbd_unit_latent


def identity(record) -> tuple[str, str]:
    if record.raw["dataset"] == "scannet":
        return record.raw["dataset"], record.raw["scene_id"]
    if record.raw["dataset"] == "arkitscenes":
        return record.raw["dataset"], str(record.raw["sequence_id"]).rsplit("/", 1)[0]
    return record.raw["dataset"], record.raw["sequence_id"]


def validate_full_record(record) -> int:
    if float(record.raw.get("fps", -1)) != 24.0:
        raise ValueError(f"{record.record_id}: formal data is not marked 24 FPS")
    target = record.load_timestamps()
    expected = target[0] + np.arange(record.frame_count, dtype=np.float64) / 24.0
    if not np.allclose(target, expected, rtol=0.0, atol=1e-8):
        raise ValueError(f"{record.record_id}: target clock is not exact 24 FPS")
    source_times, source_indices = record.load_source_identity()
    if (len(source_times) != record.frame_count or len(np.unique(source_indices)) != record.frame_count
            or np.any(np.diff(source_indices) <= 0)):
        raise ValueError(f"{record.record_id}: source frames are not real unique increasing observations")
    c2w, K = record.load_cameras(local=False); rotations = c2w[:, :3, :3]
    if (not np.allclose(rotations.transpose(0, 2, 1) @ rotations, np.eye(3), atol=2e-4)
            or not np.allclose(np.linalg.det(rotations), 1.0, atol=2e-4)
            or np.any(K[:, 0, 0] <= 0) or np.any(K[:, 1, 1] <= 0)):
        raise ValueError(f"{record.record_id}: invalid camera geometry")
    if "pointcloud" not in record.raw:
        raise ValueError(f"{record.record_id}: no pointcloud")
    with np.load(record.path("pointcloud"), allow_pickle=False) as cloud:
        offsets = cloud["offsets"]
        if (offsets.shape != (record.frame_count + 1,) or offsets[0] != 0
                or offsets[-1] != len(cloud["xyz_world"]) or np.any(np.diff(offsets) <= 0)
                or not np.array_equal(cloud["source_frame_indices"].astype(np.int64), source_indices)
                or not np.array_equal(cloud["timestamps"].astype(np.float64), target)):
            raise ValueError(f"{record.record_id}: pointcloud/source identity mismatch")
    cache = record.load_correspondences(); count = len(cache.get("query_frame", ()))
    if (count == 0 or np.any(cache["key_frame"] >= cache["query_frame"])
            or np.any(cache["key_chunk"] >= cache["query_chunk"])):
        raise ValueError(f"{record.record_id}: correspondence is empty or non-causal")
    return count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--unified-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    def raw(name: str) -> list[dict]:
        return json.loads((args.unified_root / name).read_text(encoding="utf-8"))["records"]
    all_rows = raw("manifest_all.json")
    train_rows, val_rows = raw("manifest_train.json"), raw("manifest_val.json")
    p3_rows, unit_rows = raw("manifest_train_p3.json"), raw("manifest_train_units_3chunk.json")
    train = [RGBDMemoryRecord(row, args.unified_root) for row in train_rows]
    val = [RGBDMemoryRecord(row, args.unified_root) for row in val_rows]
    p3 = [RGBDMemoryRecord(row, args.unified_root) for row in p3_rows]
    units = [RGBDMemoryRecord(row, args.unified_root) for row in unit_rows]
    all_ids = {str(row["record_id"]) for row in all_rows}
    if ({record.record_id for record in train} | {record.record_id for record in val}) != all_ids:
        raise ValueError("train/val manifests do not exactly partition manifest_all")
    if {record.record_id for record in p3} != {record.record_id for record in train}:
        raise ValueError("P3 manifest differs from formal train split")
    overlap = {identity(record) for record in train} & {identity(record) for record in val}
    if overlap:
        raise ValueError(f"train/val sequence leakage: {sorted(overlap)[:5]}")
    counts = Counter(); correspondences = Counter()
    # Validate sequentially so large NPZ correspondence arrays are released
    # after each record instead of retaining the whole corpus in RAM.
    for row in all_rows:
        record = RGBDMemoryRecord(row, args.unified_root)
        record.validate()
        counts[record.raw["dataset"]] += 1
        correspondences[record.raw["dataset"]] += validate_full_record(record)
    for record in p3:
        path = record.path("latent_cache")
        validate_rgbd_record_latent(record, path)
    for record in units:
        path = record.path("gt_latent_cache")
        validate_rgbd_unit_latent(record, path)
    report = {
        "ready": True, "records": len(all_rows), "train": len(train), "val": len(val),
        "p3_latents_valid": len(p3), "p1_p2_units_valid": len(units),
        "dataset_records": dict(counts), "correspondences": dict(correspondences),
        "all_exact_24fps": True, "all_real_unique_source_frames": True,
        "all_camera_geometry_valid": True, "all_pointclouds_valid": True,
        "all_correspondence_causal": True, "sequence_leakage": False,
    }
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
