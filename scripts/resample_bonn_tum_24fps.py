#!/usr/bin/env python3
"""Rebuild processed Bonn/TUM records on a real-frame 24 FPS timeline.

The source is the already synchronized, geometrically normalized RGB-D record
set.  Records from the same original sequence are first joined by their saved
source identity.  Target timestamps are then matched to the nearest unique,
strictly increasing real observation.  No RGB, depth, or pose interpolation is
performed.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path
import shutil
import sys
import uuid

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from long_video.data.rgbd_memory import build_causal_correspondence_cache, localize_c2w


FRAME_COUNT = 97
CHUNK_COUNT = 3
TARGET_FPS = 24.0
HEIGHT = 480
WIDTH = 832


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def resolve(base: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else base / path


def nearest_real_indices(source_times: np.ndarray, targets: np.ndarray) -> np.ndarray:
    right = np.searchsorted(source_times, targets, side="left")
    right = np.clip(right, 0, len(source_times) - 1)
    left = np.clip(right - 1, 0, len(source_times) - 1)
    choose_left = np.abs(source_times[left] - targets) <= np.abs(source_times[right] - targets)
    return np.where(choose_left, left, right).astype(np.int64)


def collect_sequences(manifest: Path, datasets: set[str]) -> tuple[dict, dict]:
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    rows = payload["records"] if isinstance(payload, dict) else payload
    base = manifest.resolve().parent
    sequences: dict[tuple[str, str], dict[int, dict]] = defaultdict(dict)
    split_by_sequence: dict[tuple[str, str], str] = {}
    for row in rows:
        dataset = str(row.get("dataset", "")).lower()
        if dataset not in datasets:
            continue
        sequence_id = str(row["sequence_id"])
        key = (dataset, sequence_id)
        split = str(row.get("split", "train"))
        previous_split = split_by_sequence.setdefault(key, split)
        if previous_split != split:
            raise ValueError(f"split leakage in source sequence {key}: {previous_split} vs {split}")
        root = resolve(base, row["metadata"]).parent
        metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
        source_ids = metadata.get("source_indices")
        if not isinstance(source_ids, list) or len(source_ids) != FRAME_COUNT:
            raise ValueError(f"{row['record_id']}: invalid source_indices")
        rgb = sorted((root / "rgb").glob("*.png"))
        depth = sorted((root / "depth").glob("*.png"))
        c2w = np.load(root / "c2w_abs.npy")
        K = np.load(root / "intrinsics.npy")
        timestamps = np.load(root / "timestamps.npy")
        if not (len(rgb) == len(depth) == len(c2w) == len(K) == len(timestamps) == FRAME_COUNT):
            raise ValueError(f"{row['record_id']}: inconsistent source record")
        for i, source_id in enumerate(source_ids):
            observation = {
                "source_index": int(source_id), "source_timestamp": float(timestamps[i]),
                "rgb": str(rgb[i]), "depth": str(depth[i]),
                "c2w": np.asarray(c2w[i], np.float64), "K": np.asarray(K[i], np.float64),
                "source_record_id": row["record_id"],
            }
            existing = sequences[key].get(int(source_id))
            if existing is not None and abs(existing["source_timestamp"] - observation["source_timestamp"]) > 1e-7:
                raise ValueError(f"conflicting source identity in {key} frame {source_id}")
            sequences[key][int(source_id)] = observation
    return sequences, split_by_sequence


def make_jobs(sequences: dict, split_by_sequence: dict) -> tuple[list[dict], Counter]:
    jobs: list[dict] = []
    stats = Counter()
    for (dataset, sequence_id), indexed in sorted(sequences.items()):
        observations = [indexed[index] for index in sorted(indexed)]
        segments: list[list[dict]] = []
        for observation in observations:
            if not segments:
                segments.append([observation])
                continue
            previous = segments[-1][-1]
            dt = observation["source_timestamp"] - previous["source_timestamp"]
            if observation["source_index"] != previous["source_index"] + 1 or not (0.0 < dt <= 0.1):
                segments.append([])
                stats["continuity_boundaries"] += 1
            segments[-1].append(observation)
        ordinal = 0
        for segment_index, segment in enumerate(segments):
            if len(segment) < FRAME_COUNT:
                stats["short_source_segments"] += 1
                continue
            source_times = np.asarray([item["source_timestamp"] for item in segment], np.float64)
            target_count = int(np.floor((source_times[-1] - source_times[0]) * TARGET_FPS + 1e-8)) + 1
            targets = source_times[0] + np.arange(target_count, dtype=np.float64) / TARGET_FPS
            selected = nearest_real_indices(source_times, targets)
            valid = np.ones(len(selected), dtype=bool)
            if len(selected) > 1:
                valid[1:] = np.diff(selected) > 0
            stats["target_duplicate_rejections"] += int((~valid).sum())
            # At roughly 30 Hz -> 24 Hz every target should map uniquely.  Split
            # rather than silently deleting inside a training record if not.
            runs: list[tuple[int, int]] = []
            start = 0
            for i in range(1, len(selected)):
                if selected[i] <= selected[i - 1]:
                    runs.append((start, i)); start = i
            runs.append((start, len(selected)))
            for run_start, run_stop in runs:
                run_length = run_stop - run_start
                windows = run_length // FRAME_COUNT
                stats["unused_target_tail"] += run_length - windows * FRAME_COUNT
                for window in range(windows):
                    lo = run_start + window * FRAME_COUNT
                    positions = selected[lo:lo + FRAME_COUNT]
                    target_window = targets[lo:lo + FRAME_COUNT]
                    chosen = [segment[int(position)] for position in positions]
                    source_window = np.asarray([item["source_timestamp"] for item in chosen], np.float64)
                    error = np.abs(source_window - target_window)
                    if len({item["source_index"] for item in chosen}) != FRAME_COUNT:
                        stats["non_unique_window"] += 1; continue
                    if float(error.max()) > 0.020:
                        stats["timing_error_over_20ms"] += 1; continue
                    jobs.append({
                        "dataset": dataset, "sequence_id": sequence_id,
                        "scene_id": sequence_id, "split": split_by_sequence[(dataset, sequence_id)],
                        "segment_index": segment_index, "ordinal": ordinal,
                        "observations": chosen, "timestamps": target_window.tolist(),
                        "source_timestamps": source_window.tolist(),
                    })
                    ordinal += 1
    stats["jobs"] = len(jobs)
    return jobs, stats


def link_or_copy(source: Path, destination: Path) -> None:
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def build_pointcloud(depth_paths: list[Path], K: np.ndarray, c2w: np.ndarray,
                     source_indices: np.ndarray, timestamps: np.ndarray,
                     output: Path, stride: int = 4) -> dict:
    yy, xx = np.mgrid[0:HEIGHT:stride, 0:WIDTH:stride]
    points: list[np.ndarray] = []
    offsets = [0]
    for i, path in enumerate(depth_paths):
        depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if depth is None:
            raise ValueError(f"cannot decode depth {path}")
        depth = depth.astype(np.float32)[::stride, ::stride] / 1000.0
        valid = np.isfinite(depth) & (depth > 0)
        z, x, y = depth[valid], xx[valid], yy[valid]
        camera = (np.linalg.inv(K[i]) @ np.stack((x * z, y * z, z))).T
        world = (c2w[i][:3] @ np.c_[camera, np.ones(len(camera))].T).T.astype(np.float32)
        points.append(world); offsets.append(offsets[-1] + len(world))
    xyz = np.concatenate(points).astype(np.float32, copy=False)
    np.savez_compressed(
        output, xyz_world=xyz, offsets=np.asarray(offsets, np.int64),
        source_frame_indices=source_indices.astype(np.int32), timestamps=timestamps.astype(np.float64),
    )
    return {"pixel_stride": stride, "points": int(len(xyz)),
            "points_per_frame_min": int(np.min(np.diff(offsets))),
            "points_per_frame_median": float(np.median(np.diff(offsets)))}


def build_job(job: dict, output_root: str, correspondence_stride: int) -> dict:
    dataset = job["dataset"]
    safe_sequence = job["sequence_id"].replace("/", "_")
    record_id = f"{dataset}__{safe_sequence}__24fps__{job['ordinal']:06d}"
    destination = Path(output_root) / "records" / dataset / record_id
    if (destination / "metadata.json").is_file():
        return {"status": "existing", "record_id": record_id, "metadata": str(destination / "metadata.json")}
    temporary = destination.with_name(destination.name + f".tmp-{uuid.uuid4().hex}")
    try:
        (temporary / "rgb").mkdir(parents=True)
        (temporary / "depth").mkdir(parents=True)
        observations = job["observations"]
        for i, observation in enumerate(observations):
            link_or_copy(Path(observation["rgb"]), temporary / "rgb" / f"{i:06d}.png")
            link_or_copy(Path(observation["depth"]), temporary / "depth" / f"{i:06d}.png")
        c2w_abs = np.stack([item["c2w"] for item in observations]).astype(np.float64)
        K = np.stack([item["K"] for item in observations]).astype(np.float64)
        absolute_targets = np.asarray(job["timestamps"], np.float64)
        # Match ScanNet/ARKitScenes: the model timeline is record-local and
        # exactly representable as k/24; absolute sensor time is kept separately.
        timestamps = np.arange(FRAME_COUNT, dtype=np.float64) / TARGET_FPS
        source_timestamps = np.asarray(job["source_timestamps"], np.float64)
        source_indices = np.asarray([item["source_index"] for item in observations], np.int32)
        if not np.all(np.diff(source_indices) > 0) or len(np.unique(source_indices)) != FRAME_COUNT:
            raise ValueError("source frames are not unique and strictly increasing")
        if not np.allclose(np.diff(timestamps), 1.0 / TARGET_FPS, atol=1e-8, rtol=0):
            raise ValueError("nominal timeline is not exactly 24 FPS")
        rotations = c2w_abs[:, :3, :3]
        if float(np.max(np.abs(rotations @ np.transpose(rotations, (0, 2, 1)) - np.eye(3)))) > 2e-4:
            raise ValueError("non-rigid c2w")
        np.save(temporary / "c2w_abs.npy", c2w_abs)
        np.save(temporary / "c2w_local.npy", localize_c2w(c2w_abs))
        np.save(temporary / "intrinsics.npy", K)
        np.save(temporary / "timestamps.npy", timestamps)
        np.save(temporary / "source_timestamps.npy", source_timestamps)
        np.save(temporary / "source_frame_indices.npy", source_indices)
        depths = sorted((temporary / "depth").glob("*.png"))
        cloud = build_pointcloud(depths, K, c2w_abs, source_indices, timestamps, temporary / "pointcloud.npz")
        correspondence = build_causal_correspondence_cache(
            depths, c2w_abs, K, temporary / "correspondence_cache.npz",
            chunk_count=CHUNK_COUNT, pixel_stride=correspondence_stride,
        )
        if int(correspondence["row_count"]) <= 0:
            raise ValueError("empty correspondence")
        metadata = {
            "schema_version": "rgbd-memory-record-v3", "record_id": record_id,
            "dataset": dataset, "scene_id": job["scene_id"], "sequence_id": job["sequence_id"],
            "split": job["split"],
            "frame_count": FRAME_COUNT, "chunk_count": CHUNK_COUNT, "chunk_stride": 32,
            "fps": TARGET_FPS, "duration_seconds": 4.0,
            "timestamps": "nominal 24 FPS target timeline", "source_timestamps": "nearest synchronized real RGB-D observation",
            "source_frame_indices": source_indices.tolist(), "interpolation": "none",
            "source_selection": "nearest real frame; strictly increasing and unique",
            "target_time_origin_seconds": float(absolute_targets[0]),
            "max_target_source_error_seconds": float(np.max(np.abs(source_timestamps - absolute_targets))),
            "pose_convention": "OpenCV camera-to-world (+x right, +y down, +z forward)",
            "depth_unit": "millimeters uint16; 0 invalid", "height": HEIGHT, "width": WIDTH,
            "pointcloud": cloud,
            "correspondence": {"row_count": int(correspondence["row_count"]),
                               "raw_match_count": int(correspondence.get("raw_match_count", correspondence["row_count"])),
                               "pixel_stride": correspondence_stride},
            "source_records": sorted({item["source_record_id"] for item in observations}),
        }
        atomic_json(temporary / "metadata.json", metadata)
        destination.parent.mkdir(parents=True, exist_ok=True)
        os.replace(temporary, destination)
        return {"status": "built", "record_id": record_id, "metadata": str(destination / "metadata.json")}
    except Exception as exc:
        shutil.rmtree(temporary, ignore_errors=True)
        return {"status": "rejected", "record_id": record_id, "reason": str(exc)}


def row_from_metadata(path: Path) -> dict:
    metadata = json.loads(path.read_text(encoding="utf-8")); root = path.parent
    return {
        "schema_version": "rgbd-memory-record-v3", "record_id": metadata["record_id"],
        "dataset": metadata["dataset"], "scene_id": metadata["scene_id"], "sequence_id": metadata["sequence_id"],
        "rgb_dir": str(root / "rgb"), "depth_dir": str(root / "depth"),
        "c2w_abs": str(root / "c2w_abs.npy"), "c2w_local": str(root / "c2w_local.npy"),
        "intrinsics": str(root / "intrinsics.npy"), "timestamps": str(root / "timestamps.npy"),
        "source_timestamps": str(root / "source_timestamps.npy"),
        "source_frame_indices": str(root / "source_frame_indices.npy"),
        "pointcloud": str(root / "pointcloud.npz"),
        "correspondence_cache": str(root / "correspondence_cache.npz"), "metadata": str(path),
        "frame_count": FRAME_COUNT, "chunk_count": CHUNK_COUNT, "chunk_stride": 32,
        "fps": TARGET_FPS, "height": HEIGHT, "width": WIDTH, "split": metadata["split"],
        "memory_eligible": True, "training_scope": "rgbd_memory",
        "pose_convention": "OpenCV camera-to-world",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--dataset", action="append", choices=("bonn", "tum"), default=[])
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--correspondence-pixel-stride", type=int, default=8)
    args = parser.parse_args()
    datasets = set(args.dataset or ("bonn", "tum"))
    sequences, split_by_sequence = collect_sequences(args.source_manifest, datasets)
    jobs, selection_stats = make_jobs(sequences, split_by_sequence)
    args.output_root.mkdir(parents=True, exist_ok=True)
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for result in executor.map(build_job, jobs, [str(args.output_root)] * len(jobs),
                                   [args.correspondence_pixel_stride] * len(jobs)):
            results.append(result)
            print(json.dumps(result), flush=True)
    rows = [row_from_metadata(path) for path in sorted((args.output_root / "records").glob("*/*/metadata.json"))
            if path.parent.parent.name in datasets]
    split_lookup = split_by_sequence
    for row in rows:
        row["split"] = split_lookup[(row["dataset"], row["sequence_id"])]
        metadata_path = Path(row["metadata"])
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["split"] = row["split"]
        atomic_json(metadata_path, metadata)
    header = {"schema_version": "rgbd-memory-manifest-v3", "height": HEIGHT, "width": WIDTH}
    atomic_json(args.output_root / "manifest_all.json", {**header, "records": rows})
    atomic_json(args.output_root / "manifest_train.json", {**header, "records": [row for row in rows if row["split"] == "train"]})
    atomic_json(args.output_root / "manifest_val.json", {**header, "records": [row for row in rows if row["split"] == "val"]})
    status = Counter(result["status"] for result in results)
    report = {"datasets": sorted(datasets), "source_sequences": len(sequences), "records": len(rows),
              "train": sum(row["split"] == "train" for row in rows),
              "val": sum(row["split"] == "val" for row in rows),
              "selection": dict(selection_stats), "build_status": dict(status)}
    atomic_json(args.output_root / "quality_report.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
