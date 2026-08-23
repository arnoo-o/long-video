#!/usr/bin/env python3
"""Create an exact-size, sequence-disjoint formal RGB-D train/val split."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path


def _atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def _choose_sequences(groups: list[tuple[str, list[dict]]], target: int, seed: str) -> set[str]:
    ordered = sorted(groups, key=lambda item: hashlib.sha256(f"{seed}/{item[0]}".encode()).hexdigest())
    choices: dict[int, tuple[str, ...]] = {0: ()}
    for sequence_id, rows in ordered:
        size = len(rows)
        for total, chosen in list(choices.items())[::-1]:
            candidate = total + size
            if candidate <= target and candidate not in choices:
                choices[candidate] = chosen + (sequence_id,)
    if target not in choices:
        available = sorted(choices)
        raise ValueError(f"cannot select {target} clips from whole sequences; reachable totals include {available[-10:]}")
    return set(choices[target])


def split_records(records: list[dict], train_count: int, seed: str = "sightline-rgbd-400-65") -> tuple[list[dict], list[dict], list[dict]]:
    if not 0 < train_count < len(records):
        raise ValueError("train_count must be inside the corpus")
    by_dataset: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for record in records:
        by_dataset[record["dataset"]][record["sequence_id"]].append(record)
    val_count = len(records) - train_count
    totals = {dataset: sum(len(rows) for rows in sequences.values()) for dataset, sequences in by_dataset.items()}
    raw_targets = {dataset: val_count * total / len(records) for dataset, total in totals.items()}
    targets = {dataset: int(value) for dataset, value in raw_targets.items()}
    for dataset, _ in sorted(raw_targets.items(), key=lambda item: (-(item[1] - int(item[1])), item[0]))[:val_count - sum(targets.values())]:
        targets[dataset] += 1
    val_sequences = {(dataset, sequence_id) for dataset, sequences in by_dataset.items() for sequence_id in _choose_sequences(list(sequences.items()), targets[dataset], f"{seed}/{dataset}")}
    all_rows = []
    for record in records:
        row = dict(record)
        row["split"] = "val" if (row["dataset"], row["sequence_id"]) in val_sequences else "train"
        all_rows.append(row)
    train = [row for row in all_rows if row["split"] == "train"]
    val = [row for row in all_rows if row["split"] == "val"]
    if len(train) != train_count or len(val) != val_count:
        raise RuntimeError("split cardinality mismatch")
    train_sequences = {(row["dataset"], row["sequence_id"]) for row in train}
    val_sequences = {(row["dataset"], row["sequence_id"]) for row in val}
    if train_sequences & val_sequences:
        raise RuntimeError("sequence leakage")
    return all_rows, train, val


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--train-count", type=int, default=400)
    parser.add_argument("--seed", default="sightline-rgbd-400-65")
    args = parser.parse_args()
    all_path = args.dataset_root / "manifest_all.json"
    payload = json.loads(all_path.read_text(encoding="utf-8"))
    all_rows, train, val = split_records(payload["records"], args.train_count, args.seed)
    header = {key: value for key, value in payload.items() if key != "records"}
    _atomic_json(all_path, {**header, "records": all_rows})
    _atomic_json(args.dataset_root / "manifest_train.json", {**header, "records": train})
    _atomic_json(args.dataset_root / "manifest_val.json", {**header, "records": val})
    print(json.dumps({"train": len(train), "val": len(val), "seed": args.seed}, indent=2))


if __name__ == "__main__":
    main()
