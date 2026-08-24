#!/usr/bin/env python3
"""Parallel wrapper around Apple's official ARKitScenes raw downloader."""
from __future__ import annotations

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import subprocess
import sys


ASSETS = ("lowres_depth", "lowres_wide.traj", "lowres_wide", "lowres_wide_intrinsics")


def complete(root: Path, split: str, video_id: str) -> bool:
    path = root / "raw" / split / video_id
    return ((path / "lowres_wide.traj").is_file()
            and (path / "lowres_wide").is_dir()
            and (path / "lowres_depth").is_dir()
            and (path / "lowres_wide_intrinsics").is_dir())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--official-root", type=Path, required=True)
    parser.add_argument("--download-root", type=Path, required=True)
    parser.add_argument("--train-additional", type=int, default=120)
    parser.add_argument("--val-additional", type=int, default=10)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    split_csv = args.official_root / "raw" / "raw_train_val_splits.csv"
    with split_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    targets = []
    for split, count in (("Training", args.train_additional), ("Validation", args.val_additional)):
        candidates = [row["video_id"] for row in rows if row["fold"] == split and not complete(args.download_root, split, row["video_id"])]
        if len(candidates) < count:
            raise ValueError(f"official {split} list has only {len(candidates)} remaining videos")
        targets.extend((split, video_id) for video_id in candidates[:count])

    script = args.official_root / "download_data.py"
    def download(item: tuple[str, str]) -> dict:
        split, video_id = item
        command = [sys.executable, str(script), "raw", "--split", split, "--video_id", video_id,
                   "--download_dir", str(args.download_root), "--raw_dataset_assets", *ASSETS]
        result = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        return {"split": split, "video_id": video_id, "returncode": result.returncode,
                "complete": complete(args.download_root, split, video_id), "tail": result.stdout[-1000:]}

    failures = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(download, item) for item in targets]
        for future in as_completed(futures):
            result = future.result()
            print(json.dumps(result), flush=True)
            if result["returncode"] or not result["complete"]:
                failures.append(result)
    if failures:
        raise RuntimeError(f"{len(failures)} ARKitScenes video downloads are incomplete; rerun to resume completed-video selection")


if __name__ == "__main__":
    main()
