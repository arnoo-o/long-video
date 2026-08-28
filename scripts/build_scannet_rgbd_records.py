#!/usr/bin/env python3
"""Build strict 8-second / 24 FPS ScanNet RGB-D memory records.

The input is one extracted ``scannet_scans_part_NNN`` shard.  Each output
record owns copied RGB-D observations and all derived geometry, so a completed
shard can be deleted without invalidating the training corpus.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import struct

import cv2
import numpy as np

from long_video.data.rgbd_memory import (
    HEIGHT,
    WIDTH,
    build_causal_correspondence_cache,
    center_crop_resize_geometry,
    localize_c2w,
)

FRAME_COUNT = 193
CHUNK_COUNT = 6
TARGET_FPS = 24.0
SOURCE_FPS = 30.0
SOURCE_SPAN = 240
QA_FRAMES = (0, 32, 64, 96, 128, 160, 192)
UP_WORLD = np.asarray((0.0, 0.0, 1.0), dtype=np.float64)


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def read_scene_info(path: Path) -> dict[str, str]:
    result = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            result[key.strip()] = value.strip()
    return result


def read_sens_timestamps(path: Path, nominal_fps: float = SOURCE_FPS) -> tuple[np.ndarray, dict]:
    """Read official SensorData frame timestamps without decoding payloads."""
    color, depth = [], []
    with path.open("rb") as handle:
        version = struct.unpack("I", handle.read(4))[0]
        if version != 4:
            raise ValueError(f"{path}: unsupported .sens version {version}")
        name_length = struct.unpack("Q", handle.read(8))[0]
        handle.seek(name_length + 4 * 16 * 4 + 2 * 4 + 4 * 4 + 4, 1)
        count = struct.unpack("Q", handle.read(8))[0]
        for _ in range(count):
            handle.seek(16 * 4, 1)
            timestamp_color, timestamp_depth, color_bytes, depth_bytes = struct.unpack("QQQQ", handle.read(32))
            color.append(timestamp_color); depth.append(timestamp_depth)
            handle.seek(color_bytes + depth_bytes, 1)
    color_raw, depth_raw = np.asarray(color, np.float64), np.asarray(depth, np.float64)
    if len(color_raw) == 0:
        raise ValueError(f"{path}: no sensor frames")
    if np.all(color_raw == 0):
        timestamps = np.arange(len(color_raw), dtype=np.float64) / float(nominal_fps)
        mode = "nominal_30fps_from_frame_index_zero_sens_timestamps"
    else:
        positive = np.diff(color_raw); positive = positive[positive > 0]
        if not len(positive):
            raise ValueError(f"{path}: non-increasing color timestamps")
        scale = min((1.0, 1e-3, 1e-6, 1e-9), key=lambda value: abs(np.median(positive) * value - 1.0 / nominal_fps))
        timestamps = (color_raw - color_raw[0]) * scale
        mode = f"sens_color_timestamp_scale_{scale:g}"
    return timestamps, {
        "mode": mode,
        "frame_count": int(len(timestamps)),
        "raw_color_depth_timestamp_equal_fraction": float(np.mean(color_raw == depth_raw)),
    }


def nearest_source_indices(source_times: np.ndarray, start: int) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    target = source_times[start] + np.arange(FRAME_COUNT, dtype=np.float64) / TARGET_FPS
    position = np.searchsorted(source_times, target)
    right = np.clip(position, 0, len(source_times) - 1)
    left = np.clip(position - 1, 0, len(source_times) - 1)
    choose_right = np.abs(source_times[right] - target) < np.abs(source_times[left] - target)
    selected = np.where(choose_right, right, left).astype(np.int32)
    if selected[0] != start or selected[-1] > start + SOURCE_SPAN:
        return None
    if np.any(np.diff(selected) <= 0) or len(np.unique(selected)) != FRAME_COUNT:
        return None
    selected_times = source_times[selected]
    if np.max(np.abs(selected_times - target)) > 1.0 / 60.0 + 1e-6:
        return None
    if np.max(np.diff(selected_times)) > 2.0 / SOURCE_FPS + 1e-6:
        return None
    return selected, target, selected_times


def contiguous_runs(indices: list[int]) -> list[tuple[int, int]]:
    if not indices:
        return []
    runs, start, previous = [], indices[0], indices[0]
    for value in indices[1:]:
        if value != previous + 1:
            runs.append((start, previous)); start = value
        previous = value
    runs.append((start, previous))
    return runs


def record_windows(scene: Path, *, gap_frames: int) -> tuple[list[dict], Counter]:
    color_dir = scene / ("color_resized" if (scene / "color_resized").is_dir() else "color")
    streams = []
    for directory, suffix in ((color_dir, ".jpg"), (scene / "depth", ".png"), (scene / "pose", ".txt")):
        streams.append({int(path.stem) for path in directory.glob(f"*{suffix}") if path.stem.isdigit()})
    available = sorted(set.intersection(*streams))
    timestamps, timestamp_info = read_sens_timestamps(scene / f"{scene.name}.sens")
    available = [index for index in available if index < len(timestamps)]
    jobs, filters = [], Counter()
    ordinal = 0
    for first, last in contiguous_runs(available):
        start = first
        while start + SOURCE_SPAN <= last:
            selection = nearest_source_indices(timestamps, start)
            if selection is None:
                filters["invalid_24fps_nearest_unique_mapping"] += 1
            else:
                indices, target, source = selection
                jobs.append({
                    "scene": str(scene), "record_ordinal": ordinal,
                    "source_indices": indices.tolist(), "timestamps": target.tolist(),
                    "source_timestamps": source.tolist(), "timestamp_info": timestamp_info,
                })
                ordinal += 1
            start += SOURCE_SPAN + 1 + int(gap_frames)
    if not jobs:
        filters["no_continuous_8_second_window"] += 1
    return jobs, filters


def rotation_from_poses(c2w: np.ndarray) -> tuple[int, dict]:
    right = c2w[:, :3, 0] @ UP_WORLD
    down = c2w[:, :3, 1] @ UP_WORLD
    strength = np.hypot(right, down)
    usable = strength >= 0.35
    if not np.any(usable):
        return 0, {"passed": False, "reason": "gravity_projection_ambiguous"}
    angles = np.arctan2(right[usable], -down[usable])
    vector = np.mean(np.exp(1j * angles))
    degrees = float(np.rad2deg(np.angle(vector)))
    options = np.asarray((0, 90, 180, 270), dtype=np.int32)
    clockwise = int(options[np.argmin(np.abs(((options - degrees + 180) % 360) - 180))])
    residual = np.abs(((np.rad2deg(angles) - clockwise + 180) % 360) - 180)
    passed = bool(np.median(residual) <= 25.0 and np.mean(residual > 45.0) <= 0.1)
    return clockwise, {
        "passed": passed, "rotation_cw_degrees": clockwise,
        "roll_degrees_before_quantization": degrees,
        "median_residual_degrees": float(np.median(residual)),
        "outlier_fraction": float(np.mean(residual > 45.0)),
    }


def camera_rotation(clockwise: int) -> np.ndarray:
    rotations = {
        0: np.eye(3),
        90: np.asarray(((0, 1, 0), (-1, 0, 0), (0, 0, 1)), np.float64),
        180: np.asarray(((-1, 0, 0), (0, -1, 0), (0, 0, 1)), np.float64),
        270: np.asarray(((0, -1, 0), (1, 0, 0), (0, 0, 1)), np.float64),
    }
    return rotations[clockwise]


def rotate_image_geometry(image: np.ndarray, K: np.ndarray, clockwise: int) -> tuple[np.ndarray, np.ndarray]:
    height, width = image.shape[:2]
    if clockwise == 0:
        return image, K.copy()
    if clockwise == 90:
        output = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        transformed = np.asarray(((K[1, 1], 0, height - 1 - K[1, 2]), (0, K[0, 0], K[0, 2]), (0, 0, 1)), np.float64)
    elif clockwise == 180:
        output = cv2.rotate(image, cv2.ROTATE_180)
        transformed = np.asarray(((K[0, 0], 0, width - 1 - K[0, 2]), (0, K[1, 1], height - 1 - K[1, 2]), (0, 0, 1)), np.float64)
    elif clockwise == 270:
        output = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        transformed = np.asarray(((K[1, 1], 0, K[1, 2]), (0, K[0, 0], width - 1 - K[0, 2]), (0, 0, 1)), np.float64)
    else:
        raise ValueError("rotation must be 0/90/180/270")
    return output, transformed


def align_depth_to_color(depth_mm: np.ndarray, K_depth: np.ndarray, K_color: np.ndarray, depth_to_color: np.ndarray, output_hw: tuple[int, int]) -> np.ndarray:
    height, width = depth_mm.shape
    yy, xx = np.mgrid[:height, :width]
    z = depth_mm.astype(np.float32).ravel() / 1000.0
    valid = np.isfinite(z) & (z > 0)
    x, y, z = xx.ravel()[valid], yy.ravel()[valid], z[valid]
    camera_depth = (np.linalg.inv(K_depth) @ np.stack((x * z, y * z, z))).T
    camera_color = (depth_to_color[:3] @ np.c_[camera_depth, np.ones(len(camera_depth))].T).T
    positive = camera_color[:, 2] > 0
    camera_color = camera_color[positive]
    projected = (K_color @ camera_color.T).T
    uv = np.rint(projected[:, :2] / projected[:, 2:3]).astype(np.int32)
    output_height, output_width = output_hw
    inside = (uv[:, 0] >= 0) & (uv[:, 0] < output_width) & (uv[:, 1] >= 0) & (uv[:, 1] < output_height)
    uv, zc = uv[inside], camera_color[inside, 2]
    flat = np.full(output_height * output_width, np.inf, np.float32)
    np.minimum.at(flat, uv[:, 1] * output_width + uv[:, 0], zc.astype(np.float32))
    aligned = flat.reshape(output_height, output_width)
    return np.where(np.isfinite(aligned) & (aligned < 65.535), np.rint(aligned * 1000), 0).astype(np.uint16)


def pose_metrics(c2w: np.ndarray) -> dict:
    translation = np.linalg.norm(np.diff(c2w[:, :3, 3], axis=0), axis=1)
    relative = c2w[:-1, :3, :3].transpose(0, 2, 1) @ c2w[1:, :3, :3]
    angles = np.rad2deg(np.arccos(np.clip((np.trace(relative, axis1=1, axis2=2) - 1) / 2, -1, 1)))
    return {
        "max_translation_m": float(np.max(translation)), "p99_translation_m": float(np.quantile(translation, .99)),
        "max_rotation_degrees": float(np.max(angles)), "p99_rotation_degrees": float(np.quantile(angles, .99)),
    }


def image_metrics(paths: list[Path]) -> dict:
    previous, changes, means, stds = None, [], [], []
    for path in paths:
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError("image_decode_failure")
        small = cv2.resize(image, (160, 120), interpolation=cv2.INTER_AREA)
        means.append(float(small.mean())); stds.append(float(small.std()))
        if previous is not None:
            changes.append(float(np.mean(np.abs(small.astype(np.float32) - previous.astype(np.float32)))))
        previous = small
    return {
        "min_mean": min(means), "min_std": min(stds), "max_adjacent_mad": max(changes),
        "median_adjacent_mad": float(np.median(changes)),
    }


def pointcloud(depth_paths: list[Path], K: np.ndarray, c2w: np.ndarray, source_indices: np.ndarray, timestamps: np.ndarray, output: Path, stride: int = 4) -> dict:
    points, offsets = [], [0]
    yy, xx = np.mgrid[0:HEIGHT:stride, 0:WIDTH:stride]
    for index, path in enumerate(depth_paths):
        depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED).astype(np.float32)[::stride, ::stride] / 1000.0
        valid = np.isfinite(depth) & (depth > 0)
        z, x, y = depth[valid], xx[valid], yy[valid]
        camera = (np.linalg.inv(K[index]) @ np.stack((x * z, y * z, z))).T
        world = (c2w[index][:3] @ np.c_[camera, np.ones(len(camera))].T).T.astype(np.float32)
        points.append(world); offsets.append(offsets[-1] + len(world))
    xyz = np.concatenate(points).astype(np.float32, copy=False)
    np.savez_compressed(output, xyz_world=xyz, offsets=np.asarray(offsets, np.int64), source_frame_indices=source_indices.astype(np.int32), timestamps=timestamps.astype(np.float64))
    return {"pixel_stride": stride, "points": int(len(xyz)), "points_per_frame_min": int(np.min(np.diff(offsets))), "points_per_frame_median": float(np.median(np.diff(offsets)))}


def render_qa(paths: list[Path], output: Path) -> None:
    panels = []
    for frame in QA_FRAMES:
        image = cv2.imread(str(paths[frame]), cv2.IMREAD_COLOR)
        panel = cv2.resize(image, (416, 240), interpolation=cv2.INTER_AREA)
        cv2.putText(panel, f"frame {frame}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, .7, (255, 255, 255), 2, cv2.LINE_AA)
        panels.append(panel)
    cv2.imwrite(str(output), np.concatenate(panels, axis=1))


def build_record(job: dict, output_root: str) -> dict:
    scene = Path(job["scene"]); indices = np.asarray(job["source_indices"], np.int32)
    target_times = np.asarray(job["timestamps"], np.float64); source_times = np.asarray(job["source_timestamps"], np.float64)
    record_id = f"scannet__{scene.name}__{int(job['record_ordinal']):06d}"
    destination = Path(output_root) / "records" / "scannet" / record_id
    if (destination / "metadata.json").is_file():
        return {"status": "existing", "record_id": record_id, "metadata": str(destination / "metadata.json")}
    color_dir = scene / ("color_resized" if (scene / "color_resized").is_dir() else "color")
    color_paths = [color_dir / f"{index}.jpg" for index in indices]
    depth_paths = [scene / "depth" / f"{index}.png" for index in indices]
    pose_paths = [scene / "pose" / f"{index}.txt" for index in indices]
    try:
        c2w_depth = np.stack([np.loadtxt(path, dtype=np.float64) for path in pose_paths])
        if c2w_depth.shape != (FRAME_COUNT, 4, 4) or not np.isfinite(c2w_depth).all():
            raise ValueError("invalid_pose")
        rotations = c2w_depth[:, :3, :3]
        if not np.allclose(rotations.transpose(0, 2, 1) @ rotations, np.eye(3), atol=1e-4) or not np.allclose(np.linalg.det(rotations), 1, atol=1e-4):
            raise ValueError("nonrigid_pose")
        poses = pose_metrics(c2w_depth)
        if poses["max_translation_m"] > .50 or poses["max_rotation_degrees"] > 45.0:
            raise ValueError("pose_jump")
        images = image_metrics(color_paths)
        if images["min_mean"] < 5 or images["min_std"] < 3:
            raise ValueError("black_or_blank_frame")
        if images["max_adjacent_mad"] > 80:
            raise ValueError("rgb_scene_cut")
    except Exception as exc:
        return {"status": "rejected", "record_id": record_id, "reason": str(exc)}
    info = read_scene_info(scene / f"{scene.name}.txt")
    K_color_raw = np.loadtxt(scene / "intrinsic" / "intrinsic_color.txt", dtype=np.float64)[:3, :3]
    K_depth = np.loadtxt(scene / "intrinsic" / "intrinsic_depth.txt", dtype=np.float64)[:3, :3]
    first_color = cv2.imread(str(color_paths[0]), cv2.IMREAD_COLOR)
    first_depth = cv2.imread(str(depth_paths[0]), cv2.IMREAD_UNCHANGED)
    if first_color is None or first_depth is None:
        return {"status": "rejected", "record_id": record_id, "reason": "decode_failure"}
    raw_color_hw = (int(info.get("colorHeight", first_color.shape[0])), int(info.get("colorWidth", first_color.shape[1])))
    K_color = K_color_raw.copy()
    K_color[0] *= first_color.shape[1] / raw_color_hw[1]; K_color[1] *= first_color.shape[0] / raw_color_hw[0]
    color_to_depth = np.asarray([float(value) for value in info.get("colorToDepthExtrinsics", "").split()], np.float64)
    color_to_depth = color_to_depth.reshape(4, 4) if color_to_depth.size == 16 else np.eye(4)
    depth_to_color = np.linalg.inv(color_to_depth)
    # ScanNet's exported camera_to_world is the depth/sensor-camera pose.
    # colorToDepthExtrinsics maps color-camera coordinates into that sensor
    # frame, so aligned color-view depth must use this color-camera c2w.
    c2w_color = c2w_depth @ color_to_depth
    clockwise, orientation = rotation_from_poses(c2w_color)
    if not orientation.get("passed"):
        return {"status": "rejected", "record_id": record_id, "reason": "ambiguous_orientation"}
    Q = np.eye(4); Q[:3, :3] = camera_rotation(clockwise)
    c2w_out = c2w_color @ Q
    temporary = destination.with_name(destination.name + ".tmp")
    if temporary.exists(): shutil.rmtree(temporary)
    (temporary / "rgb").mkdir(parents=True); (temporary / "depth").mkdir()
    intrinsics = np.empty((FRAME_COUNT, 3, 3), np.float64)
    depth_valid = []
    transform_info = None
    try:
        for output_index, (rgb_path, depth_path) in enumerate(zip(color_paths, depth_paths)):
            rgb = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR); depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
            if rgb is None or depth is None or depth.ndim != 2:
                raise ValueError("decode_failure")
            aligned = align_depth_to_color(depth, K_depth, K_color, depth_to_color, rgb.shape[:2])
            rgb_rotated, K_rotated = rotate_image_geometry(rgb, K_color, clockwise)
            depth_rotated, K_depth_rotated = rotate_image_geometry(aligned, K_color, clockwise)
            if not np.allclose(K_rotated, K_depth_rotated):
                raise RuntimeError("RGB/depth intrinsic transform mismatch")
            crop, K_out = center_crop_resize_geometry(*rgb_rotated.shape[:2], K_rotated)
            left, top, right, bottom = crop
            rgb_out = cv2.resize(rgb_rotated[top:bottom, left:right], (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
            depth_out = cv2.resize(depth_rotated[top:bottom, left:right], (WIDTH, HEIGHT), interpolation=cv2.INTER_NEAREST)
            intrinsics[output_index] = K_out
            transform_info = {"rotation_cw_degrees": clockwise, "crop_xyxy": list(crop), "source_hw": list(rgb.shape[:2]), "rotated_hw": list(rgb_rotated.shape[:2]), "target_hw": [HEIGHT, WIDTH]}
            depth_valid.append(float(np.mean(depth_out > 0)))
            if not cv2.imwrite(str(temporary / "rgb" / f"{output_index:06d}.png"), rgb_out): raise OSError("RGB write failed")
            if not cv2.imwrite(str(temporary / "depth" / f"{output_index:06d}.png"), depth_out): raise OSError("depth write failed")
        if min(depth_valid) < .02:
            raise ValueError("insufficient_valid_depth")
        np.save(temporary / "c2w_abs.npy", c2w_out)
        np.save(temporary / "c2w_local.npy", localize_c2w(c2w_out))
        np.save(temporary / "intrinsics.npy", intrinsics)
        np.save(temporary / "timestamps.npy", target_times)
        np.save(temporary / "source_timestamps.npy", source_times)
        np.save(temporary / "source_frame_indices.npy", indices)
        processed_depth = sorted((temporary / "depth").glob("*.png"))
        cloud = pointcloud(processed_depth, intrinsics, c2w_out, indices, target_times, temporary / "pointcloud.npz")
        correspondence = build_causal_correspondence_cache(processed_depth, c2w_out, intrinsics, temporary / "correspondence_cache.npz", chunk_count=CHUNK_COUNT, pixel_stride=8)
        if correspondence["row_count"] == 0:
            raise ValueError("empty_correspondence")
        render_qa(sorted((temporary / "rgb").glob("*.png")), temporary / "qa_7frames.jpg")
        continuity = {
            "passed": True, "source_frame_indices_strictly_increasing": True,
            "source_frame_indices_unique": True, "target_duration_seconds": 8.0,
            "max_target_source_error_seconds": float(np.max(np.abs(target_times - source_times))),
            "max_source_dt_seconds": float(np.max(np.diff(source_times))),
            "pose": poses, "rgb": images,
        }
        metadata = {
            "schema_version": "scannet-rgbd-memory-v1", "record_id": record_id,
            "dataset": "scannet", "scene_id": scene.name.split("_")[0], "sequence_id": scene.name,
            "frame_count": FRAME_COUNT, "chunk_count": CHUNK_COUNT, "chunk_stride": 32,
            "fps": TARGET_FPS, "duration_seconds": 8.0,
            "timestamp_source": job["timestamp_info"], "source_frame_indices": indices.tolist(),
            "orientation_validation": orientation, "continuity_validation": continuity,
            "image_geometry": transform_info, "depth_valid_fraction_min": min(depth_valid),
            "depth_valid_fraction_median": float(np.median(depth_valid)), "pointcloud": cloud,
            "correspondence": {"row_count": correspondence["row_count"], "raw_match_count": correspondence["raw_match_count"], "pixel_stride": correspondence["pixel_stride"]},
            "pose_convention": "OpenCV color-camera-to-world; ScanNet depth camera_to_world @ colorToDepthExtrinsics @ image rotation",
            "depth_alignment": "depth camera unprojection -> colorToDepth inverse -> color z-buffer",
        }
        atomic_json(temporary / "metadata.json", metadata)
        destination.parent.mkdir(parents=True, exist_ok=True)
        os.replace(temporary, destination)
    except Exception as exc:
        if temporary.exists(): shutil.rmtree(temporary)
        return {"status": "rejected", "record_id": record_id, "reason": str(exc)}
    return {"status": "built", "record_id": record_id, "metadata": str(destination / "metadata.json")}


def row_from_metadata(path: Path) -> dict:
    metadata = json.loads(path.read_text(encoding="utf-8")); root = path.parent
    return {
        "schema_version": "rgbd-memory-record-v3", "record_id": metadata["record_id"],
        "dataset": "scannet", "scene_id": metadata["scene_id"], "sequence_id": metadata["sequence_id"],
        "rgb_dir": str(root / "rgb"), "depth_dir": str(root / "depth"),
        "c2w_abs": str(root / "c2w_abs.npy"), "c2w_local": str(root / "c2w_local.npy"),
        "intrinsics": str(root / "intrinsics.npy"), "timestamps": str(root / "timestamps.npy"),
        "source_timestamps": str(root / "source_timestamps.npy"), "source_frame_indices": str(root / "source_frame_indices.npy"),
        "pointcloud": str(root / "pointcloud.npz"), "correspondence_cache": str(root / "correspondence_cache.npz"),
        "metadata": str(path), "qa_preview": str(root / "qa_7frames.jpg"),
        "frame_count": FRAME_COUNT, "chunk_count": CHUNK_COUNT, "chunk_stride": 32,
        "fps": TARGET_FPS, "height": HEIGHT, "width": WIDTH,
        "memory_eligible": True, "training_scope": "rgbd_memory", "pose_convention": "OpenCV camera-to-world",
    }


def round_robin(records: list[dict], count: int) -> list[dict]:
    groups = defaultdict(list)
    for row in records: groups[row["sequence_id"]].append(row)
    for values in groups.values(): values.sort(key=lambda row: row["record_id"])
    selected = []
    while len(selected) < count and groups:
        for key in sorted(list(groups)):
            selected.append(groups[key].pop(0))
            if not groups[key]: del groups[key]
            if len(selected) == count: break
    return selected


def select_split(records: list[dict], train_count: int, val_count: int) -> tuple[list[dict], list[dict]]:
    scenes = defaultdict(list)
    for row in records: scenes[row["scene_id"]].append(row)
    ordered = sorted(scenes, key=lambda key: (hashlib.sha256(key.encode()).hexdigest(), key))
    val_scenes, capacity = [], 0
    total = len(records)
    for scene in ordered:
        if total - capacity - len(scenes[scene]) < train_count: continue
        val_scenes.append(scene); capacity += len(scenes[scene])
        if capacity >= val_count: break
    if capacity < val_count:
        raise RuntimeError("ScanNet scene-isolated capacity cannot realize validation count")
    validation_pool = [row for scene in val_scenes for row in scenes[scene]]
    training_pool = [row for scene, values in scenes.items() if scene not in val_scenes for row in values]
    if len(training_pool) < train_count:
        raise RuntimeError("ScanNet scene-isolated training capacity is insufficient")
    train = round_robin(training_pool, train_count); val = round_robin(validation_pool, val_count)
    for row in train: row["split"] = "train"
    for row in val: row["split"] = "val"
    if {row["scene_id"] for row in train} & {row["scene_id"] for row in val}:
        raise RuntimeError("ScanNet scene leakage")
    return train, val


def finalize(output_root: Path, unified: Path, train_count: int, val_count: int) -> dict:
    candidates = [row_from_metadata(path) for path in sorted((output_root / "records" / "scannet").glob("*/metadata.json"))]
    fallback_all_train = len(candidates) < train_count + val_count
    if fallback_all_train:
        # The mirror may contain fewer usable windows than the requested target.
        # Preserve every strictly validated record rather than inventing windows
        # or failing the build solely because a validation quota is unavailable.
        train = list(candidates)
        val = []
        for row in train:
            row["split"] = "train"
    else:
        train, val = select_split(candidates, train_count, val_count)
    selected = train + val
    from long_video.training.rgbd_memory_data import RGBDMemoryRecord
    filters = Counter()
    for report_path in sorted((output_root / "reports").glob("part_*.json")):
        part_report = json.loads(report_path.read_text(encoding="utf-8"))
        filters.update(part_report.get("summary", {}).get("filters", {}))
    orientations, source_fps, source_dt_max, pose_translation, pose_rotation = Counter(), [], [], [], []
    depth_min, depth_median, cloud_points, corr_rows, corr_raw = [], [], [], [], []
    processed_bytes, validated = 0, 0
    for row in selected:
        record = RGBDMemoryRecord(row, unified)
        record.validate(); validated += 1
        metadata = json.loads(record.path("metadata").read_text(encoding="utf-8"))
        source_times, _ = record.load_source_identity()
        source_fps.append(1.0 / float(np.median(np.diff(source_times))))
        source_dt_max.append(float(np.max(np.diff(source_times))))
        orientations[str(metadata["orientation_validation"]["rotation_cw_degrees"])] += 1
        pose_translation.append(float(metadata["continuity_validation"]["pose"]["max_translation_m"]))
        pose_rotation.append(float(metadata["continuity_validation"]["pose"]["max_rotation_degrees"]))
        depth_min.append(float(metadata["depth_valid_fraction_min"])); depth_median.append(float(metadata["depth_valid_fraction_median"]))
        cloud_points.append(int(metadata["pointcloud"]["points"])); corr_rows.append(int(metadata["correspondence"]["row_count"])); corr_raw.append(int(metadata["correspondence"]["raw_match_count"]))
        processed_bytes += sum(path.stat().st_size for path in record.path("metadata").parent.rglob("*") if path.is_file())
    qa = [row["qa_preview"] for row in sorted(selected, key=lambda value: value["record_id"])[:20]]
    report = {
        "schema_version": "scannet-quality-report-v1", "records": len(selected), "train": len(train), "val": len(val),
        "split_mode": "all_train_fallback" if fallback_all_train else "scene_isolated_540_60",
        "requested_records": train_count + val_count, "requested_train": train_count, "requested_val": val_count,
        "scenes": len({row["scene_id"] for row in selected}), "sequences": len({row["sequence_id"] for row in selected}),
        "train_scenes": len({row["scene_id"] for row in train}), "val_scenes": len({row["scene_id"] for row in val}),
        "scene_leakage": False, "validator_passed": validated, "validator_failed": 0,
        "filters": dict(filters), "target_fps": 24.0, "target_duration_seconds": 8.0,
        "source_fps_median": float(np.median(source_fps)), "source_fps_range": [float(np.min(source_fps)), float(np.max(source_fps))],
        "source_dt_max_seconds": float(np.max(source_dt_max)), "time_discontinuity_records": int(sum(value > 2.0 / 30.0 + 1e-6 for value in source_dt_max)),
        "orientation_counts": dict(orientations),
        "pose_max_translation_m": float(np.max(pose_translation)), "pose_max_rotation_degrees": float(np.max(pose_rotation)),
        "depth_valid_fraction_min": float(np.min(depth_min)), "depth_valid_fraction_median": float(np.median(depth_median)),
        "pointcloud_points_total": int(sum(cloud_points)), "pointcloud_points_per_record_median": float(np.median(cloud_points)),
        "correspondence_rows_total": int(sum(corr_rows)), "correspondence_rows_per_record_median": float(np.median(corr_rows)),
        "correspondence_raw_matches_total": int(sum(corr_raw)), "correspondence_tokenization_ratio": float(sum(corr_rows) / max(sum(corr_raw), 1)),
        "processed_bytes": int(processed_bytes), "qa_previews": qa,
    }
    # Do not publish manifests until all selected records have passed the strict
    # validator.  A scene-isolated 540/60 split can require more than 600
    # candidates because surplus windows from validation scenes cannot leak into
    # training.  Keep exactly the selected 600 after validation.
    for name in ("manifest_all.json", "manifest_train.json", "manifest_train_p3.json", "manifest_val.json"):
        path = unified / name; payload = json.loads(path.read_text(encoding="utf-8")); base = [row for row in payload["records"] if row.get("dataset") != "scannet"]
        additions = selected if name == "manifest_all.json" else train if name in ("manifest_train.json", "manifest_train_p3.json") else val
        atomic_json(path, {**{key: value for key, value in payload.items() if key != "records"}, "records": base + additions})
    selected_ids = {row["record_id"] for row in selected}
    discarded_candidates = 0
    discarded_bytes = 0
    for row in candidates:
        if row["record_id"] in selected_ids:
            continue
        record_root = Path(row["metadata"]).parent
        discarded_bytes += sum(path.stat().st_size for path in record_root.rglob("*") if path.is_file())
        shutil.rmtree(record_root)
        discarded_candidates += 1
    report["discarded_surplus_candidates"] = discarded_candidates
    report["discarded_surplus_bytes"] = discarded_bytes
    atomic_json(unified / "scannet_quality_report.json", report)
    return report


def cleanup_part(output_root: Path, part: int) -> dict:
    scannet_root = output_root.parent
    extracted = scannet_root / "extracted" / f"part_{part:03d}"
    archive_name = f"scannet_scans_part_{part:03d}.tar.gz"
    archive = scannet_root / "shards" / archive_name
    removed = 0
    for path in (extracted, archive):
        if path.is_dir():
            removed += sum(file.stat().st_size for file in path.rglob("*") if file.is_file()); shutil.rmtree(path)
        elif path.is_file():
            removed += path.stat().st_size; path.unlink()
    state_path = scannet_root / "download_state.json"
    state = json.loads(state_path.read_text()) if state_path.is_file() else {"repo": "kairunwen/scannet_temp", "completed": {}, "failed": {}}
    state.setdefault("cleaned", {})[archive_name] = {"bytes_removed": removed}
    atomic_json(state_path, state)
    return {"removed_bytes": removed, "archive": archive_name}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--unified-root", type=Path, required=True)
    parser.add_argument("--part-index", type=int)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--gap-frames", type=int, default=60)
    parser.add_argument("--max-total-records", type=int, default=600)
    parser.add_argument("--finalize", action="store_true")
    parser.add_argument("--cleanup-part", action="store_true")
    args = parser.parse_args()
    if args.finalize:
        print(json.dumps(finalize(args.output_root, args.unified_root, 540, 60), indent=2)); return
    if args.input_root is None or args.part_index is None:
        raise ValueError("processing requires --input-root and --part-index")
    existing = list((args.output_root / "records" / "scannet").glob("*/metadata.json"))
    remaining = max(0, args.max_total_records - len(existing))
    filters, jobs = Counter(), []
    for scene in sorted((args.input_root / "scans").glob("scene*")):
        scene_jobs, scene_filters = record_windows(scene, gap_frames=args.gap_frames)
        filters.update(scene_filters); jobs.extend(scene_jobs)
    jobs = jobs[:remaining]
    results = []
    if args.workers == 1:
        results = [build_record(job, str(args.output_root)) for job in jobs]
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            results = list(pool.map(build_record, jobs, [str(args.output_root)] * len(jobs)))
    filters.update(result.get("reason", "") for result in results if result["status"] == "rejected")
    summary = {
        "part": args.part_index, "scenes": len(list((args.input_root / "scans").glob("scene*"))),
        "candidate_windows": len(jobs), "built": sum(result["status"] == "built" for result in results),
        "existing": sum(result["status"] == "existing" for result in results),
        "rejected": sum(result["status"] == "rejected" for result in results), "filters": dict(filters),
    }
    atomic_json(args.output_root / "reports" / f"part_{args.part_index:03d}.json", {"summary": summary, "results": results})
    if args.cleanup_part:
        summary["cleanup"] = cleanup_part(args.output_root, args.part_index)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
