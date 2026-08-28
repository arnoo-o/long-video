#!/usr/bin/env python3
"""Download lightweight ARKitScenes RRD metadata and selected video layers."""
from __future__ import annotations
import argparse, json, os, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from huggingface_hub import HfApi, hf_hub_download

REPO = "rerun/arkitscenes-rrd"
LAYERS = ("base", "calibration", "video_wide", "depth")

def atomic(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)

def video_ids() -> list[str]:
    return sorted(Path(x.path).stem for x in HfApi().list_repo_tree(REPO, repo_type="dataset", recursive=True) if x.path.startswith("base/") and x.path.endswith(".rrd"))

def _fetch(root: Path, key: str, retries: int) -> tuple[str, str, int]:
    for attempt in range(retries):
        try:
            path = Path(hf_hub_download(REPO, key, repo_type="dataset", local_dir=root))
            return key, str(path), path.stat().st_size
        except Exception:
            if attempt + 1 == retries: raise
            time.sleep(2 ** attempt)
    raise RuntimeError(key)

def download(root: Path, ids: list[str], layers: tuple[str, ...], retries: int = 4, workers: int = 8) -> dict:
    state_path = root / "download_state.json"
    state = json.loads(state_path.read_text()) if state_path.is_file() else {"repo": REPO, "completed": {}, "failed": {}}
    pending = [f"{layer}/{video_id}.rrd" for video_id in ids for layer in layers if not (key := f"{layer}/{video_id}.rrd") in state["completed"] or not (root / key).is_file()]
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = {pool.submit(_fetch, root, key, retries): key for key in pending}
        for future in as_completed(futures):
            key = futures[future]
            try:
                key, path, size = future.result()
                state["completed"][key] = {"path": path, "bytes": size, "completed_at": time.time()}; state["failed"].pop(key, None)
            except Exception as exc:
                state["failed"][key] = {"error": repr(exc)}
            atomic(state_path, state)
    return state

def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--ids", nargs="*"); p.add_argument("--start", type=int, default=0); p.add_argument("--count", type=int)
    p.add_argument("--ids-file", type=Path, help="newline-separated video ids")
    p.add_argument("--layers", nargs="+", choices=LAYERS, default=list(LAYERS)); p.add_argument("--workers", type=int, default=8); a = p.parse_args()
    if a.ids_file:
        ids = [x.strip() for x in a.ids_file.read_text().splitlines() if x.strip()]
    else:
        ids = a.ids or video_ids()[a.start : a.start + a.count if a.count else None]
    print(json.dumps({"repo": REPO, "video_count": len(ids), "video_ids": ids, "layers": a.layers}, indent=2))
    download(a.output_root, ids, tuple(a.layers), workers=a.workers)

if __name__ == "__main__": main()
