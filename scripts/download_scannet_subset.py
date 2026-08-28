#!/usr/bin/env python3
"""Resumable, stateful ScanNet shard downloader.

The requested repository is intentionally treated as a shard source: no
snapshot download is used, and a shard is never fetched twice once its SHA256
has been recorded in the state file.
"""
from __future__ import annotations
import argparse, hashlib, json, os, shutil, time
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

REPO = "kairunwen/scannet_temp"


def atomic_json(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--start-part", type=int, default=0)
    parser.add_argument("--max-parts", type=int, default=1)
    parser.add_argument("--min-free-gib", type=float, default=200)
    parser.add_argument("--retries", type=int, default=5)
    args = parser.parse_args()
    root = args.output_root; shards = root / "shards"; shards.mkdir(parents=True, exist_ok=True)
    state_path = root / "download_state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {"repo": REPO, "completed": {}, "cleaned": {}, "failed": {}}
    state.setdefault("cleaned", {})
    api = HfApi()
    tree = list(api.list_repo_tree(REPO, repo_type="dataset", recursive=False, expand=True))
    entries = {item.path: item for item in tree if item.path.startswith("scannet_scans_part_") and item.path.endswith(".tar.gz")}
    files = sorted(entries)
    selected = files[args.start_part:args.start_part + args.max_parts]
    if not selected:
        raise ValueError("no requested ScanNet shard exists")
    for name in selected:
        target = shards / Path(name).name
        if name in state["cleaned"]:
            print(json.dumps({"status": "already_processed_and_cleaned", "shard": name})); continue
        remote_size = int(entries[name].size)
        available = shutil.disk_usage(root).free
        required = remote_size + int(args.min_free_gib * 2**30)
        if available < required:
            raise RuntimeError(f"{name}: only {available / 2**30:.1f} GiB free; need shard plus {args.min_free_gib:.1f} GiB reserve")
        if name in state["completed"] and target.is_file() and sha256(target) == state["completed"][name]["sha256"]:
            print(json.dumps({"status": "already_complete", "shard": name})); continue
        for attempt in range(1, args.retries + 1):
            try:
                cached = Path(hf_hub_download(REPO, name, repo_type="dataset", local_dir=shards, resume_download=True))
                if cached.resolve() != target.resolve():
                    cached.replace(target)
                if target.stat().st_size != remote_size:
                    raise IOError(f"{name}: expected {remote_size} bytes, got {target.stat().st_size}")
                state["completed"][name] = {"sha256": sha256(target), "bytes": target.stat().st_size, "completed_at": time.time()}
                state["failed"].pop(name, None); atomic_json(state_path, state)
                print(json.dumps({"status": "complete", "shard": name, **state["completed"][name]})); break
            except Exception as exc:
                state["failed"][name] = {"attempt": attempt, "error": repr(exc), "time": time.time()}; atomic_json(state_path, state)
                if attempt == args.retries: raise
                time.sleep(min(60, 2**attempt))


if __name__ == "__main__":
    main()
