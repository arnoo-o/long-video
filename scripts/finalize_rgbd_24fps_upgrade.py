#!/usr/bin/env python3
"""Publish rebuilt Bonn/TUM records and ScanNet full latents atomically."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def read_records(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload["records"] if isinstance(payload, dict) else payload


def attach_latent(row: dict, cache: Path) -> dict:
    if not cache.is_file():
        raise FileNotFoundError(f"missing latent for {row['record_id']}: {cache}")
    result = dict(row)
    result["latent_cache"] = str(cache)
    result["latent_schema"] = f"continuous_{1 + (int(row['frame_count']) - 1) // 4}"
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--unified-root", type=Path, required=True)
    parser.add_argument("--rebuilt-root", type=Path, required=True)
    parser.add_argument("--unit-latent-root", type=Path, required=True)
    parser.add_argument("--full-latent-root", type=Path, required=True)
    args = parser.parse_args()

    rebuilt_all = read_records(args.rebuilt_root / "manifest_all.json")
    rebuilt_train = [row for row in rebuilt_all if row["split"] == "train"]
    rebuilt_val = [row for row in rebuilt_all if row["split"] == "val"]
    rebuilt = {
        "manifest_all.json": rebuilt_all,
        "manifest_train.json": rebuilt_train,
        "manifest_train_p3.json": rebuilt_train,
        "manifest_val.json": rebuilt_val,
    }
    expected_ids = {row["record_id"] for row in rebuilt_all}
    if len(expected_ids) != len(rebuilt_all):
        raise ValueError("duplicate rebuilt record id")

    for name, additions in rebuilt.items():
        path = args.unified_root / name
        payload = json.loads(path.read_text(encoding="utf-8"))
        kept = [row for row in payload["records"] if row.get("dataset") not in {"bonn", "tum"}]
        published: list[dict] = []
        for row in kept:
            if row.get("dataset") == "scannet":
                cache = args.full_latent_root / row["record_id"].replace(":", "_") / "continuous_49.pt"
                row = attach_latent(row, cache)
            published.append(row)
        for row in additions:
            cache = args.unit_latent_root / row["record_id"].replace(":", "_") / "continuous_25.pt"
            published.append(attach_latent(row, cache))
        if len({row["record_id"] for row in published}) != len(published):
            raise ValueError(f"duplicate ids in {name}")
        atomic_json(path, {**{key: value for key, value in payload.items() if key != "records"}, "records": published})
        print(json.dumps({"manifest": name, "records": len(published),
                          "bonn_tum": len(additions),
                          "scannet_latents": sum(row.get("dataset") == "scannet" for row in published)}))


if __name__ == "__main__":
    main()
