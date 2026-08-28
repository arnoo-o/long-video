#!/usr/bin/env python3
"""Canonicalize ARKitScenes records to world-gravity-up image orientation.

Each record receives one fixed clockwise 90-degree-multiple rotation.  RGB,
depth, intrinsics, OpenCV c2w, point cloud, and causal correspondence are
rebuilt together.  Existing artifacts are retained as ``.pre_gravity``
backups until an explicit cleanup operation is requested.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path
import shutil

import cv2
import numpy as np

from long_video.data.rgbd_memory import (
    build_causal_correspondence_cache,
    center_crop_resize_geometry,
    localize_c2w,
)

HEIGHT, WIDTH = 480, 832
GRAVITY = np.asarray((0.0, 1.0, 0.0), dtype=np.float64)


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def _rotation_from_c2w(c2w: np.ndarray) -> tuple[int, dict]:
    right, down = c2w[:, :3, 0] @ GRAVITY, c2w[:, :3, 1] @ GRAVITY
    strength = np.hypot(right, down)
    usable = strength >= 0.35
    angles = np.arctan2(right, -down)
    vector = np.mean(np.exp(1j * angles[usable])) if np.any(usable) else 0j
    degrees = float(np.rad2deg(np.angle(vector))) if vector else 0.0
    options = np.asarray((0, 90, 180, 270), dtype=np.int32)
    clockwise = int(options[np.argmin(np.abs(((options - degrees + 180) % 360) - 180))])
    return clockwise, {
        "method": "record_fixed_nearest_90deg_world_gravity_up",
        "clockwise_degrees": clockwise,
        "gravity_strength_median": float(np.median(strength)),
        "gravity_strength_min": float(np.min(strength)),
        "ambiguous_frame_fraction": float(np.mean(~usable)),
        "roll_degrees_before_quantization": degrees,
    }


def _camera_rotation(clockwise: int) -> np.ndarray:
    rotations = {
        0: np.eye(3),
        90: np.asarray(((0, 1, 0), (-1, 0, 0), (0, 0, 1)), dtype=np.float64),
        180: np.asarray(((-1, 0, 0), (0, -1, 0), (0, 0, 1)), dtype=np.float64),
        270: np.asarray(((0, -1, 0), (1, 0, 0), (0, 0, 1)), dtype=np.float64),
    }
    return rotations[clockwise]


def _rotate_pair(rgb: np.ndarray, depth: np.ndarray, K: np.ndarray, clockwise: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if clockwise == 0:
        return rgb, depth, np.asarray(K, dtype=np.float64)
    if clockwise == 90:
        rgb, depth = cv2.rotate(rgb, cv2.ROTATE_90_CLOCKWISE), cv2.rotate(depth, cv2.ROTATE_90_CLOCKWISE)
        K_rot = np.asarray(((K[1, 1], 0, HEIGHT - 1 - K[1, 2]), (0, K[0, 0], K[0, 2]), (0, 0, 1)), dtype=np.float64)
    elif clockwise == 180:
        rgb, depth = cv2.rotate(rgb, cv2.ROTATE_180), cv2.rotate(depth, cv2.ROTATE_180)
        K_rot = np.asarray(((K[0, 0], 0, WIDTH - 1 - K[0, 2]), (0, K[1, 1], HEIGHT - 1 - K[1, 2]), (0, 0, 1)), dtype=np.float64)
    elif clockwise == 270:
        rgb, depth = cv2.rotate(rgb, cv2.ROTATE_90_COUNTERCLOCKWISE), cv2.rotate(depth, cv2.ROTATE_90_COUNTERCLOCKWISE)
        K_rot = np.asarray(((K[1, 1], 0, K[1, 2]), (0, K[0, 0], WIDTH - 1 - K[0, 2]), (0, 0, 1)), dtype=np.float64)
    else:
        raise ValueError("clockwise must be one of 0, 90, 180, 270")
    crop, K_out = center_crop_resize_geometry(*rgb.shape[:2], K_rot)
    left, top, right, bottom = crop
    rgb_out = cv2.resize(rgb[top:bottom, left:right], (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
    depth_out = cv2.resize(depth[top:bottom, left:right], (WIDTH, HEIGHT), interpolation=cv2.INTER_NEAREST)
    return rgb_out, depth_out, K_out


def _pointcloud(depth_paths: list[Path], K: np.ndarray, c2w: np.ndarray, output: Path) -> None:
    points, offsets = [], [0]
    yy, xx = np.mgrid[0:HEIGHT:4, 0:WIDTH:4]
    for index, path in enumerate(depth_paths):
        depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0
        z = depth[::4, ::4]; valid = np.isfinite(z) & (z > 0)
        z, x, y = z[valid], xx[valid], yy[valid]
        camera = (np.linalg.inv(K[index]) @ np.stack((x * z, y * z, z))).T
        world = (c2w[index][:3] @ np.c_[camera, np.ones(len(camera))].T).T.astype(np.float32)
        points.append(world); offsets.append(offsets[-1] + len(world))
    np.savez_compressed(output, xyz_world=np.concatenate(points), offsets=np.asarray(offsets, dtype=np.int64))


def _replace_path(source: Path, backup: Path, replacement: Path) -> None:
    if backup.exists():
        if source.is_dir(): shutil.rmtree(source)
        elif source.exists(): source.unlink()
    else:
        os.replace(source, backup)
    os.replace(replacement, source)


def _record_root(row: dict) -> Path:
    return Path(row["rgb_dir"]).parent


def normalize_record(row: dict, apply: bool) -> dict:
    root = _record_root(row)
    metadata_path = Path(row["metadata"])
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if "gravity_orientation_normalization" in metadata:
        return {"record_id": row["record_id"], "status": "already_normalized", "rotation": metadata["gravity_orientation_normalization"]["clockwise_degrees"]}
    c2w_path, K_path = Path(row["c2w_abs"]), Path(row["intrinsics"])
    c2w, K = np.load(c2w_path), np.load(K_path)
    clockwise, info = _rotation_from_c2w(c2w)
    if clockwise == 0:
        return {"record_id": row["record_id"], "status": "kept", "rotation": 0, **info}
    if not apply:
        return {"record_id": row["record_id"], "status": "would_normalize", "rotation": clockwise, **info}
    source_rgb = root / "rgb.pre_gravity" if (root / "rgb.pre_gravity").is_dir() else root / "rgb"
    source_depth = root / "depth.pre_gravity" if (root / "depth.pre_gravity").is_dir() else root / "depth"
    rgb_paths = sorted(source_rgb.glob("*.png"), key=lambda p: p.name)
    depth_paths = sorted(source_depth.glob("*.png"), key=lambda p: p.name)
    count = int(row["frame_count"])
    if len(rgb_paths) != count or len(depth_paths) != count:
        raise RuntimeError(f"{row['record_id']}: expected {count} synchronized frames")
    temporary = root / ".gravity_normalize_tmp"
    if temporary.exists(): shutil.rmtree(temporary)
    (temporary / "rgb").mkdir(parents=True); (temporary / "depth").mkdir()
    K_new = np.empty_like(K, dtype=np.float64)
    for index, (rgb_path, depth_path) in enumerate(zip(rgb_paths, depth_paths)):
        rgb, depth = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR), cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        if rgb is None or depth is None or rgb.shape[:2] != (HEIGHT, WIDTH) or depth.shape != (HEIGHT, WIDTH):
            raise RuntimeError(f"{row['record_id']}: invalid synchronized input frame {index}")
        rgb, depth, K_new[index] = _rotate_pair(rgb, depth, K[index], clockwise)
        if not cv2.imwrite(str(temporary / "rgb" / rgb_path.name), rgb): raise OSError(rgb_path)
        if not cv2.imwrite(str(temporary / "depth" / depth_path.name), depth): raise OSError(depth_path)
    Q = np.eye(4); Q[:3, :3] = _camera_rotation(clockwise)
    c2w_new = c2w @ Q
    np.save(temporary / "c2w_abs.npy", c2w_new)
    np.save(temporary / "c2w_local.npy", localize_c2w(c2w_new))
    np.save(temporary / "intrinsics.npy", K_new)
    output_depth = sorted((temporary / "depth").glob("*.png"), key=lambda p: p.name)
    _pointcloud(output_depth, K_new, c2w_new, temporary / "pointcloud.npz")
    stride = int((metadata.get("correspondence") or {}).get("pixel_stride", 8))
    correspondence = build_causal_correspondence_cache(output_depth, c2w_new, K_new, temporary / "correspondence_cache.npz", chunk_count=int(row["chunk_count"]), pixel_stride=stride)
    for name in ("rgb", "depth", "c2w_abs.npy", "c2w_local.npy", "intrinsics.npy", "pointcloud.npz", "correspondence_cache.npz"):
        source, replacement = root / name, temporary / name
        suffix = ".pre_gravity" if source.suffix == "" else ".pre_gravity" + source.suffix
        _replace_path(source, root / (source.stem + suffix), replacement)
    metadata["gravity_orientation_normalization"] = info
    metadata["correspondence"] = correspondence
    atomic_json(metadata_path, metadata)
    temporary.rmdir()
    return {"record_id": row["record_id"], "status": "normalized", "rotation": clockwise, "correspondence_rows": correspondence["row_count"], **info}


def normalize_record_entry(value: tuple[dict, bool]) -> dict:
    return normalize_record(value[0], value[1])


def _unit_cache_from_parent(unit: dict, parent: dict) -> None:
    offset, count = int(unit.get("source_frame_start", 0)), int(unit["frame_count"])
    chunk_offset = offset // 32
    with np.load(parent["correspondence_cache"], allow_pickle=False) as value:
        arrays = {key: np.ascontiguousarray(value[key]) for key in value.files}
    keep = ((arrays["query_frame"] >= offset) & (arrays["query_frame"] < offset + count) & (arrays["key_frame"] >= offset) & (arrays["key_frame"] < offset + count) & (arrays["query_chunk"] >= chunk_offset) & (arrays["query_chunk"] < chunk_offset + 3) & (arrays["key_chunk"] >= chunk_offset) & (arrays["key_chunk"] < chunk_offset + 3))
    arrays = {key: value[keep].copy() for key, value in arrays.items()}
    arrays["query_frame"] -= offset; arrays["key_frame"] -= offset
    arrays["query_chunk"] -= chunk_offset; arrays["key_chunk"] -= chunk_offset
    output = Path(unit["correspondence_cache"]); temporary = output.with_name(output.stem + ".gravity.tmp.npz")
    np.savez_compressed(temporary, **arrays); os.replace(temporary, output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--unified-root", type=Path, required=True)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if args.workers < 1: raise ValueError("workers must be positive")
    unified = args.unified_root
    all_payload = json.loads((unified / "manifest_all.json").read_text(encoding="utf-8"))
    rows = [row for row in all_payload["records"] if row.get("dataset") == "arkitscenes"]
    jobs = [(row, args.apply) for row in rows]
    if args.workers == 1: result = [normalize_record_entry(job) for job in jobs]
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool: result = list(pool.map(normalize_record_entry, jobs))
    changed = {item["record_id"] for item in result if item["status"] in {"normalized", "would_normalize"}}
    if args.apply:
        units = json.loads((unified / "manifest_train_units_3chunk.json").read_text(encoding="utf-8"))["records"]
        parents = {row["record_id"]: row for row in rows}
        for unit in units:
            parent_id = unit.get("parent_record_id", unit["record_id"])
            if unit.get("dataset") == "arkitscenes" and parent_id in changed:
                _unit_cache_from_parent(unit, parents[parent_id])
        atomic_json(unified / "arkit_gravity_normalization_report.json", {"records": result, "normalized": len(changed), "kept": len(rows) - len(changed)})
    print(json.dumps({"mode": "apply" if args.apply else "dry_run", "records": len(rows), "normalized": len(changed), "kept": len(rows) - len(changed), "rotations": {str(k): sum(x.get("rotation") == k for x in result) for k in (0,90,180,270)}}, indent=2))


if __name__ == "__main__": main()
