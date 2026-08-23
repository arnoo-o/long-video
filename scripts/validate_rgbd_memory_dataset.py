#!/usr/bin/env python3
"""Validate manifests, summarize corpus statistics, and render 8 correspondence QAs."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import random

import cv2
import numpy as np

from long_video.training.rgbd_memory_data import load_rgbd_memory_manifest


def _size(path: Path) -> int:
    return sum(file.stat().st_size for file in path.rglob("*") if file.is_file())


def _render(record, row, output: Path):
    query = cv2.imread(str(record.rgb_paths()[row["query_frame"]]))
    key = cv2.imread(str(record.rgb_paths()[row["key_frame"]]))
    token_h, token_w = 480 / 30, 832 / 52
    q = (int((row["query_x"] + 0.5) * token_w), int((row["query_y"] + 0.5) * token_h))
    k = (int((row["key_x"] + 0.5) * token_w), int((row["key_y"] + 0.5) * token_h))
    panel = np.concatenate((key, query), axis=1)
    q_panel = (q[0] + 832, q[1])
    cv2.circle(panel, k, 8, (0, 255, 0), 2); cv2.circle(panel, q_panel, 8, (0, 0, 255), 2)
    cv2.line(panel, k, q_panel, (255, 255, 0), 2)
    label = f"{record.record_id} key={row['key_frame']} query={row['query_frame']} w={row['weight']:.3f}"
    cv2.putText(panel, label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, .65, (255, 255, 255), 2, cv2.LINE_AA)
    output.parent.mkdir(parents=True, exist_ok=True); cv2.imwrite(str(output), panel)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--qa-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path)
    args = parser.parse_args()
    train = load_rgbd_memory_manifest(args.dataset_root / "manifest_train.json")
    val = load_rgbd_memory_manifest(args.dataset_root / "manifest_val.json")
    all_records = load_rgbd_memory_manifest(args.dataset_root / "manifest_all.json")
    train_sequences = {(row.raw["dataset"], row.raw["sequence_id"]) for row in train}
    val_sequences = {(row.raw["dataset"], row.raw["sequence_id"]) for row in val}
    if train_sequences & val_sequences:
        raise RuntimeError("sequence leakage between train and val")
    summary = defaultdict(lambda: Counter(sequences=set()))
    for record in all_records:
        bucket = summary[record.raw["dataset"]]
        bucket["clips"] += 1; bucket["frames"] += 97
        bucket[record.raw["split"]] += 1
        cache = record.load_correspondences()
        bucket["correspondences"] += len(cache.get("query_frame", ()))
        bucket["camera_only"] += int(not record.memory_eligible)
        bucket["memory_eligible"] += int(record.memory_eligible)
        bucket["sequences"].add(record.raw["sequence_id"])
    candidates = []
    for record in all_records:
        cache = record.load_correspondences()
        if cache and len(cache["query_frame"]):
            index = int(np.argmax(cache["weight"]))
            row = {key: int(cache[key][index]) for key in ("query_frame", "key_frame", "query_chunk", "key_chunk", "query_y", "query_x", "key_y", "key_x")}
            row["weight"] = float(cache["weight"][index])
            candidates.append((record, row))
    random.Random(20260824).shuffle(candidates)
    qa = []
    for index, (record, row) in enumerate(candidates[:8]):
        output = args.qa_dir / f"{index:02d}_{record.record_id}.jpg"
        _render(record, row, output); qa.append(str(output))
    datasets = {}
    for dataset, counts in summary.items():
        report_path = args.dataset_root / "reports" / f"{dataset}.json"
        build_report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.is_file() else {}
        filtered = Counter(build_report.get("filters", {}))
        for sequence in build_report.get("sequences", {}).values():
            association = sequence.get("association", {})
            for key, value in association.items():
                if (key.startswith("dropped_") or key == "invalid_pose") and isinstance(value, (int, float)):
                    filtered[key] += value
        datasets[dataset] = {
            **{key: value for key, value in counts.items() if key != "sequences"},
            "sequences": len(counts["sequences"]),
            "original_sequences": int(build_report.get("sequence_count", len(counts["sequences"]))),
            "raw_bytes": _size(args.raw_root / dataset) if args.raw_root and (args.raw_root / dataset).is_dir() else None,
            "processed_bytes": _size(args.dataset_root / "records" / dataset),
            "filters": dict(filtered),
        }
    payload = {
        "schema_version": "rgbd-memory-validation-v1", "record_count": len(all_records),
        "train_count": len(train), "val_count": len(val), "sequence_leakage": False,
        "processed_bytes": _size(args.dataset_root / "records"), "qa": qa,
        "datasets": datasets,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
