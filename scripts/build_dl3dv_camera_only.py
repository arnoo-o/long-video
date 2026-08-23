#!/usr/bin/env python3
"""Build independent 65-frame/2-chunk DL3DV RGB + camera samples.

This path intentionally reads only official ``images+poses`` scene archives. It
does not inspect or create depth, point clouds, ReCal3R, correspondence, or
Memory artifacts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time
import urllib.request
import zipfile

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from long_video.data.dl3dv import (
    OFFICIAL_REPO, TARGET_FPS, TARGET_HW, build_interpolated_timeline,
    center_crop_resize_geometry, interpolate_timeline_c2w, load_dl3dv_scene,
    ranked_candidates, read_official_metadata,
)

CSV_URL = "https://raw.githubusercontent.com/DL3DV-10K/Dataset/main/cache/DL3DV-valid.csv"
HTML_URL = "https://raw.githubusercontent.com/DL3DV-10K/Dataset/main/visualize/index.html"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--raw-root", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--state", type=Path, required=True)
    p.add_argument("--metadata-root", type=Path)
    p.add_argument("--target-clips", type=int, default=2000)
    p.add_argument("--scene-batch", type=int, default=32)
    p.add_argument("--max-clips-per-scene", type=int, default=3)
    p.add_argument("--source-fps", type=float, default=30.0)
    p.add_argument("--rife-root", type=Path, required=True)
    p.add_argument("--rife-checkpoint", type=Path, required=True)
    p.add_argument("--device", default="cuda:1")
    p.add_argument("--retries", type=int, default=3)
    p.add_argument("--seed", type=int, default=20260823)
    p.add_argument("--candidate-shard-index", type=int, default=0)
    p.add_argument("--candidate-shard-count", type=int, default=1)
    p.add_argument("--manifest", type=Path)
    return p.parse_args()


def official_metadata(root: Path):
    root.mkdir(parents=True, exist_ok=True)
    paths = (root / "DL3DV-valid.csv", root / "index.html")
    for path, url in zip(paths, (CSV_URL, HTML_URL)):
        if not path.exists():
            urllib.request.urlretrieve(url, path)
    return paths


def download_scene(record: dict, raw_root: Path, retries: int = 3) -> Path:
    destination = raw_root / str(record["batch"]) / str(record["scene_hash"])
    if destination.is_dir():
        return destination
    from huggingface_hub import hf_hub_download
    batch_root = destination.parent
    batch_root.mkdir(parents=True, exist_ok=True)
    last = None
    for attempt in range(max(1, retries)):
        try:
            archive = Path(hf_hub_download(
                repo_id=OFFICIAL_REPO, repo_type="dataset",
                filename=f"{record['batch']}/{record['scene_hash']}.zip",
                local_dir=raw_root, cache_dir=raw_root / ".hf_download_cache",
                resume_download=True,
            ))
            with zipfile.ZipFile(archive) as handle:
                handle.extractall(batch_root)
            archive.unlink(missing_ok=True)
            cache = raw_root / ".hf_download_cache"
            if cache.exists():
                shutil.rmtree(cache)
            if destination.is_dir():
                return destination
            # Archives occasionally contain a single directory with a different name.
            candidates = [x for x in batch_root.iterdir() if x.is_dir() and x.name != destination.name]
            if len(candidates) == 1:
                candidates[0].replace(destination)
                return destination
            raise FileNotFoundError(destination)
        except Exception as exc:  # retry transient HF/network failures
            last = exc
            time.sleep(2 ** attempt)
    raise RuntimeError(f"failed to download {record['scene_hash']}") from last


def _rotation_components(local: np.ndarray):
    r = local[-1, :3, :3]
    # OpenCV camera convention: yaw around Y, pitch around X, roll around Z.
    yaw = math.degrees(math.atan2(float(r[0, 2]), float(r[2, 2])))
    pitch = math.degrees(math.atan2(float(-r[1, 2]), math.hypot(float(r[0, 2]), float(r[2, 2]))))
    roll = math.degrees(math.atan2(float(r[1, 0]), float(r[1, 1])))
    total = float(np.degrees(np.arccos(np.clip((np.trace(r) - 1) / 2, -1, 1))))
    return yaw, pitch, roll, total


def _motion_type(yaw, pitch, translation, forward, lateral, vertical):
    # Rotation is in degrees and translation is in scene units, so never
    # compare their raw magnitudes.  The thresholds form the stable motion
    # buckets used for corpus balancing and metadata.
    if abs(yaw) < 0.5 and abs(pitch) < 0.5 and translation < 0.03:
        return "static"
    rotating = abs(yaw) >= 12.0 or abs(pitch) >= 10.0
    translating = translation >= 0.7
    if rotating and translating:
        return "rotation_translation"
    if abs(yaw) >= max(abs(pitch), 8.0):
        return "left_yaw" if yaw < 0 else "right_yaw"
    if abs(pitch) >= 5.0:
        return "pitch_up" if pitch < 0 else "pitch_down"
    if abs(forward) >= max(abs(lateral), abs(vertical), 0.3):
        return "forward" if forward > 0 else "backward"
    if abs(lateral) >= max(abs(vertical), 0.08):
        return "left" if lateral < 0 else "right"
    if abs(vertical) >= 0.1:
        return "vertical"
    return "other"


def _quality(rgb_paths):
    from PIL import Image, ImageFilter
    brightness, sharpness, transitions = [], [], []
    previous = None
    for path in rgb_paths[:: max(1, len(rgb_paths) // 8)]:
        with Image.open(path) as image:
            gray = np.asarray(image.convert("L").resize((160, 96)), np.float32) / 255.0
        brightness.append(float(gray.mean()))
        sharpness.append(float(np.diff(gray, axis=1).var() + np.diff(gray, axis=0).var()))
        if previous is not None:
            transitions.append(float(np.mean(np.abs(gray - previous))))
        previous = gray
    return float(np.mean(brightness)), float(np.mean(sharpness)), float(max(transitions, default=0.0))


class RifeInterpolator:
    """Pinned Practical-RIFE adapter, loaded once for the whole corpus run."""
    def __init__(self, root: Path, checkpoint: Path, device: str):
        import torch
        if not (checkpoint / "flownet.pkl").is_file():
            raise FileNotFoundError(f"missing Practical-RIFE checkpoint: {checkpoint / 'flownet.pkl'}")
        self.torch, self.device = torch, torch.device("cuda")
        sys.path.insert(0, str(root.resolve()))
        from train_log.RIFE_HDv3 import Model
        self.model = Model(); self.model.load_model(str(checkpoint), -1); self.model.eval()
        self.cache = {}

    def _image(self, scene, index: int, crop):
        key = (str(scene.root), int(index), tuple(crop))
        if key not in self.cache:
            from PIL import Image
            with Image.open(scene.image_paths[int(index)]) as image:
                self.cache[key] = np.asarray(image.convert("RGB").crop(crop).resize(
                    (TARGET_HW[1], TARGET_HW[0]), Image.Resampling.LANCZOS), np.uint8)
        return self.cache[key]

    def render(self, scene, timeline, crop):
        from PIL import Image
        frames, sources = [], []
        with self.torch.inference_mode():
            for output_index, (left, right, alpha, real, real_index) in enumerate(zip(
                    timeline["left_real_indices"], timeline["right_real_indices"], timeline["alpha"],
                    timeline["is_real"], timeline["rgb_real_indices"])):
                if real:
                    frame, kind = self._image(scene, real_index, crop), "real"
                else:
                    def tensor(value):
                        return self.torch.from_numpy(value.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(self.device)
                    predicted = self.model.inference(tensor(self._image(scene, left, crop)),
                                                     tensor(self._image(scene, right, crop)),
                                                     timestep=float(alpha))
                    frame = np.rint(predicted[0].clamp(0, 1).permute(1, 2, 0).float().cpu().numpy() * 255).astype(np.uint8)
                    kind = "rife"
                frames.append(frame)
                sources.append({"output_index": output_index, "timestamp": float(timeline["timestamps"][output_index]),
                                "source": kind, "left_real_index": int(left), "right_real_index": int(right),
                                "alpha": float(alpha)})
        return frames, sources


def process_window(scene, start: int, output: Path, record: dict, *, interpolator: RifeInterpolator):
    timeline = build_interpolated_timeline(scene.frame_times, start, 65, TARGET_FPS)
    indices = np.asarray(timeline["rgb_real_indices"], dtype=np.int64)
    if not np.all(np.diff(timeline["timestamps"]) > 0):
        raise ValueError("non-monotonic timestamps")
    crop, k_dense = center_crop_resize_geometry(scene.source_hw, scene.intrinsics)
    c2w = interpolate_timeline_c2w(scene.c2w_opencv, timeline)
    k = np.stack([(1 - a) * k_dense[l] + a * k_dense[r] for l, r, a in zip(
        timeline["left_real_indices"], timeline["right_real_indices"], timeline["alpha"])])
    if not np.isfinite(c2w).all() or not np.isfinite(k).all():
        raise ValueError("non-finite pose/intrinsics")
    brightness, sharpness, max_transition = _quality([scene.image_paths[i] for i in indices])
    if not 0.05 <= brightness <= 0.95 or sharpness < 0.0002 or max_transition > 0.45:
        raise ValueError("quality filter rejected window")
    rotations = c2w[:, :3, :3]
    if not np.allclose(np.linalg.det(rotations), 1.0, atol=2e-2):
        raise ValueError("invalid rotation matrix")
    local_raw = c2w.astype(np.float32)
    # The first output pose defines the local frame.  Force the exact identity
    # after SLERP/float conversion so downstream loaders do not inherit tiny
    # scene-dependent numerical drift.
    local_raw[0] = np.eye(4, dtype=np.float32)
    translation_distance = float(np.linalg.norm(local_raw[-1, :3, 3]))
    translation_scale = max(float(np.max(np.linalg.norm(local_raw[:, :3, 3], axis=1))), 1e-6)
    local = local_raw.copy(); local[:, :3, 3] /= translation_scale
    yaw, pitch, roll, total = _rotation_components(local_raw)
    # ``local_raw`` is OpenCV c2w in the source-camera coordinate system:
    # X points right, Y points down, and Z points forward.  Keep these named
    # motion components in that convention instead of treating XYZ as F/L/V.
    lateral, vertical, forward = map(float, local_raw[-1, :3, 3])
    motion = _motion_type(yaw, pitch, translation_distance, forward, lateral, vertical)
    if motion == "static" or (total < 1.0 and translation_distance < 0.01):
        raise ValueError("nearly static window")
    output.mkdir(parents=True, exist_ok=False)
    rgb = output / "rgb"; rgb.mkdir()
    from PIL import Image
    frames, sources = interpolator.render(scene, timeline, crop)
    if len(frames) != 65:
        raise RuntimeError("RIFE timeline did not yield exactly 65 frames")
    for out_index, frame in enumerate(frames):
        Image.fromarray(frame).save(rgb / f"{out_index:06d}.jpg", quality=95, subsampling=0)
    np.save(output / "target_c2w_local.npy", local)
    np.save(output / "target_c2w_local_raw.npy", local_raw)
    np.save(output / "intrinsics.npy", np.asarray(k, np.float32))
    metadata = {
        "trajectory_id": f"{record['scene_hash']}_camera2_{start:06d}",
        "scene_hash": record["scene_hash"], "source_frame_range": [int(indices[0]), int(indices[-1])],
        "source_indices": indices.tolist(), "frame_sources": sources, "fps": 24, "height": 384, "width": 640,
        "chunk_count": 2, "translation_scale": translation_scale,
        "total_rotation_deg": total, "yaw_deg": yaw, "pitch_deg": pitch,
        "roll_deg": roll, "translation_distance": translation_distance,
        "forward_translation": forward, "lateral_translation": lateral,
        "vertical_translation": vertical, "motion_type": motion,
        "quality": {"brightness_mean": brightness, "sharpness_mean": sharpness,
                    "max_sample_transition": max_transition},
        "rgb_unique": True, "rife_interpolated": any(item["source"] == "rife" for item in sources),
        "uses_depth": False, "uses_pointcloud": False,
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def _atomic_json(path: Path, payload):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(path)


def main():
    args = parse_args()
    if not 0 <= args.candidate_shard_index < args.candidate_shard_count:
        raise ValueError("candidate shard index must be in [0, candidate shard count)")
    metadata_root = args.metadata_root or args.raw_root / "official_metadata"
    csv_path, html_path = official_metadata(metadata_root)
    candidates = ranked_candidates(read_official_metadata(csv_path, html_path))
    state = {"schema_version": 1, "official_repo": OFFICIAL_REPO, "processed_scenes": [],
             "failed_scenes": {}, "records": []}
    if args.state.exists():
        state = json.loads(args.state.read_text(encoding="utf-8"))
    done_scenes = set(state.get("processed_scenes", [])); args.output_root.mkdir(parents=True, exist_ok=True)
    if not args.device.startswith("cuda:"):
        raise ValueError("camera-only RIFE preprocessing requires an explicit cuda:N device")
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device.split(":", 1)[1]
    interpolator = RifeInterpolator(args.rife_root, args.rife_checkpoint, args.device)
    batch_counter = 0
    for candidate_index, record in enumerate(candidates):
        if candidate_index % args.candidate_shard_count != args.candidate_shard_index:
            continue
        if len(state["records"]) >= args.target_clips: break
        scene_hash = record["scene_hash"]
        if scene_hash in done_scenes: continue
        try:
            scene_root = download_scene(record, args.raw_root, args.retries)
            scene = load_dl3dv_scene(scene_root, source_fps=args.source_fps, duration=record.get("duration"))
            # Widely spaced starts maximize diversity while allowing up to 3 clips/scene.
            max_start = len(scene.frame_times) - 2
            starts = np.linspace(0, max(0, max_start), args.max_clips_per_scene, dtype=np.int64)
            made = 0
            for start in np.unique(starts):
                if len(state["records"]) >= args.target_clips or made >= args.max_clips_per_scene: break
                out = args.output_root / f"{scene_hash}_camera2_{int(start):06d}"
                if out.exists(): continue
                try:
                    item = process_window(scene, int(start), out, record, interpolator=interpolator)
                    item["rgb_dir"] = str(out.relative_to(args.output_root).joinpath("rgb"))
                    item["target_c2w_local"] = str(out.relative_to(args.output_root).joinpath("target_c2w_local.npy"))
                    item["target_c2w_local_raw"] = str(out.relative_to(args.output_root).joinpath("target_c2w_local_raw.npy"))
                    item["intrinsics"] = str(out.relative_to(args.output_root).joinpath("intrinsics.npy"))
                    state["records"].append(item); made += 1
                except (ValueError, OSError):
                    if out.exists(): shutil.rmtree(out)
            done_scenes.add(scene_hash); state["processed_scenes"] = sorted(done_scenes)
            batch_counter += 1
            state["last_batch_size"] = min(batch_counter, args.scene_batch)
            _atomic_json(args.state, state)
            # Raw scene is disposable after all derived clips are safely recorded.
            if scene_root.exists() and args.raw_root.resolve() in scene_root.resolve().parents:
                shutil.rmtree(scene_root)
            print(json.dumps({"scene_hash": scene_hash, "clips": made, "total": len(state["records"])}))
            if batch_counter >= args.scene_batch:
                batch_counter = 0
        except Exception as exc:
            failures = state.setdefault("failed_scenes", {})
            previous = failures.get(scene_hash, {})
            failures[scene_hash] = {"attempts": int(previous.get("attempts", 0)) + 1,
                                    "last_error": f"{type(exc).__name__}: {exc}"}
            _atomic_json(args.state, state)
            print(json.dumps({"scene_hash": scene_hash, "error": f"{type(exc).__name__}: {exc}"}))
    manifest = {"schema_version": "camera-only-v1", "fps": 24, "height": 384, "width": 640,
                "chunk_count": 2, "records": state["records"]}
    _atomic_json(args.manifest or args.output_root / "camera_only_manifest.json", manifest)
    if len(state["records"]) < args.target_clips:
        raise RuntimeError(f"only built {len(state['records'])}/{args.target_clips} camera-only clips")


if __name__ == "__main__":
    main()
