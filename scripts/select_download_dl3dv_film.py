#!/usr/bin/env python3
"""Metadata-first, per-hash DL3DV-10K 480P downloader for Stage0 FiLM."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import urllib.request
import zipfile

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from long_video.data.dl3dv import (OFFICIAL_REPO, load_dl3dv_scene, ranked_candidates,
                                   read_official_metadata, select_revisit_trajectories)

CSV_URL = "https://raw.githubusercontent.com/DL3DV-10K/Dataset/main/cache/DL3DV-valid.csv"
HTML_URL = "https://raw.githubusercontent.com/DL3DV-10K/Dataset/main/visualize/index.html"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--target-scenes", type=int, default=256)
    parser.add_argument("--indoor-target", type=int, default=128)
    parser.add_argument("--source-fps", type=float, default=30.0)
    parser.add_argument("--metadata-root", type=Path)
    parser.add_argument("--keep-rejected", action="store_true")
    return parser.parse_args()


def official_metadata(root):
    root.mkdir(parents=True, exist_ok=True)
    paths = (root / "DL3DV-valid.csv", root / "index.html")
    for path, url in zip(paths, (CSV_URL, HTML_URL)):
        if not path.exists(): urllib.request.urlretrieve(url, path)
    return paths


def verify_official_access():
    try:
        from huggingface_hub import HfFileSystem
    except ImportError as error:
        raise RuntimeError("install huggingface_hub before DL3DV access validation") from error
    try:
        entries = HfFileSystem().ls(f"datasets/{OFFICIAL_REPO}", detail=False)
    except Exception as error:
        raise PermissionError(
            f"No authorized access to official {OFFICIAL_REPO}; accept its Hugging Face terms "
            "and log in on this machine. No alternate source will be used."
        ) from error
    if not entries: raise PermissionError(f"official {OFFICIAL_REPO} returned no accessible files")


def download_hash(record, raw_root):
    from huggingface_hub import hf_hub_download
    batch, scene_hash = record["batch"], record["scene_hash"]
    destination = raw_root / batch / scene_hash
    if destination.exists(): return destination
    raw_root.mkdir(parents=True, exist_ok=True)
    cache_root = raw_root / ".hf_download_cache"
    archive = Path(hf_hub_download(
        repo_id=OFFICIAL_REPO, repo_type="dataset", filename=f"{batch}/{scene_hash}.zip",
        local_dir=raw_root, cache_dir=cache_root,
    ))
    batch_root = raw_root / batch
    before = {x.resolve() for x in batch_root.iterdir()} if batch_root.exists() else set()
    with zipfile.ZipFile(archive) as handle:
        handle.extractall(batch_root)
    archive.unlink()
    if cache_root.exists(): shutil.rmtree(cache_root)
    if not destination.exists():
        after = [x for x in batch_root.iterdir() if x.resolve() not in before and x.is_dir()]
        if len(after) == 1: after[0].rename(destination)
    if not destination.exists(): raise FileNotFoundError(f"archive did not create {destination}")
    return destination


def image_quality(scene):
    from PIL import Image
    sample = np.linspace(0, len(scene.image_paths) - 1, min(12, len(scene.image_paths))).round().astype(int)
    brightness, sharpness = [], []
    for index in sample:
        with Image.open(scene.image_paths[index]) as image:
            gray = np.asarray(image.convert("L").resize((160, 96)), np.float32) / 255
        brightness.append(float(gray.mean()))
        gx, gy = np.diff(gray, axis=1), np.diff(gray, axis=0)
        sharpness.append(float(gx.var() + gy.var()))
    return {"brightness_mean": float(np.mean(brightness)), "sharpness_mean": float(np.mean(sharpness))}


def qualifies(path, source_fps, duration=None):
    scene = load_dl3dv_scene(path, source_fps=source_fps, duration=duration)
    quality = image_quality(scene)
    trajectories = select_revisit_trajectories(scene)
    reasons = []
    if not 0.08 <= quality["brightness_mean"] <= 0.92: reasons.append("extreme_brightness")
    if quality["sharpness_mean"] < 0.0005: reasons.append("severe_blur")
    if not trajectories: reasons.append("no_8_or_12_chunk_revisit")
    return not reasons, {**quality, "trajectory_count": len(trajectories), "reasons": reasons}


def main():
    args = parse_args()
    if args.target_scenes != 256:
        raise ValueError("formal DL3DV FiLM corpus is fixed to exactly 256 qualified scenes")
    if args.indoor_target * 2 != args.target_scenes:
        raise ValueError("formal corpus requires an approximately equal indoor/outdoor split")
    verify_official_access()
    metadata_root = args.metadata_root or args.raw_root / "official_metadata"
    csv_path, html_path = official_metadata(metadata_root)
    candidates = ranked_candidates(read_official_metadata(csv_path, html_path))
    state = {"schema_version": 1, "official_repo": OFFICIAL_REPO, "attempted": [], "qualified": []}
    if args.state.exists(): state = json.loads(args.state.read_text(encoding="utf-8"))
    attempted = {x["scene_hash"] for x in state["attempted"]}
    counts = {env: sum(x["environment"] == env for x in state["qualified"])
              for env in ("indoor", "outdoor")}
    for record in candidates:
        if len(state["qualified"]) >= args.target_scenes: break
        env = record["environment"]
        if counts[env] >= args.indoor_target or record["scene_hash"] in attempted: continue
        path = download_hash(record, args.raw_root)
        try:
            accepted, report = qualifies(path, args.source_fps, record.get("duration"))
        except Exception as error:
            accepted, report = False, {"reasons": [f"inspection_error:{type(error).__name__}:{error}"]}
        attempt = {**record, "raw_path": str(path), "accepted": accepted, "inspection": report}
        state["attempted"].append(attempt); attempted.add(record["scene_hash"])
        if accepted:
            state["qualified"].append(attempt); counts[env] += 1
        elif not args.keep_rejected:
            resolved = path.resolve()
            if args.raw_root.resolve() not in resolved.parents: raise RuntimeError("cleanup escaped raw root")
            shutil.rmtree(resolved)
        args.state.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.state.with_suffix(".tmp")
        temporary.write_text(json.dumps(state, indent=2), encoding="utf-8"); temporary.replace(args.state)
        print(json.dumps({"qualified": len(state["qualified"]), "indoor": counts["indoor"],
                          "outdoor": counts["outdoor"], "last": attempt}, ensure_ascii=False))
    if len(state["qualified"]) != args.target_scenes:
        raise RuntimeError(f"exhausted official metadata with {len(state['qualified'])}/256 qualified scenes")


if __name__ == "__main__":
    main()
