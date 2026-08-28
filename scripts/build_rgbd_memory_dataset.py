#!/usr/bin/env python3
"""Build canonical 97-frame RGB-D records from official dataset archives.

Supported inputs:
* TUM RGB-D dynamic sequences (timestamp association, OpenCV c2w)
* Bonn RGB-D Dynamic (TUM format plus the official marker/sensor transform)
* Neural RGB-D Surface Reconstruction data (frame-id alignment, OpenGL->OpenCV)
* 7Scenes camera-only records (frame-id alignment, official camera-to-world)

7Scenes is excluded from memory/correspondence supervision because its official
RGB and depth streams are uncalibrated and no RGB-depth extrinsic is published.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import os
from pathlib import Path

import cv2
import numpy as np

from long_video.data.rgbd_memory import (
    CHUNK_COUNT, FRAME_COUNT, HEIGHT, WIDTH, associate_timestamp_streams,
    build_causal_correspondence_cache, center_crop_resize_geometry, localize_c2w,
    read_timestamp_file, sequence_split, transform_rgb_depth,
)

# Official RGB calibration for the Freiburg 3 camera used by all eight dynamic
# sequences selected here.  The published depth maps are registered to RGB.
TUM_K = np.asarray(((535.4, 0.0, 320.1), (0.0, 539.2, 247.6), (0.0, 0.0, 1.0)))
BONN_K = np.asarray(((542.822841, 0.0, 315.593520), (0.0, 542.576870, 237.756098), (0.0, 0.0, 1.0)))
BONN_DIST = np.asarray((0.039903, -0.099343, -0.000730, -0.000144, 0.0))
BONN_T_ROS = np.asarray(((-1, 0, 0, 0), (0, 0, 1, 0), (0, 1, 0, 0), (0, 0, 0, 1)), dtype=np.float64)
BONN_T_MARKER = np.asarray(((1.0157, 0.1828, -0.2389, 0.0113), (0.0009, -0.8431, -0.6413, -0.0098), (-0.3009, 0.6147, -0.8085, 0.0111), (0, 0, 0, 1)), dtype=np.float64)
OPENGL_TO_OPENCV = np.diag((1.0, -1.0, -1.0, 1.0))
SEVEN_SCENES_K = np.asarray(((585.0, 0.0, 320.0), (0.0, 585.0, 240.0), (0.0, 0.0, 1.0)))


def _nearest_so3(matrix: np.ndarray) -> np.ndarray:
    """Project a numerically scaled calibration rotation back onto SO(3)."""
    u, _, vt = np.linalg.svd(np.asarray(matrix, dtype=np.float64))
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u @ vt
    return rotation


# The published marker coefficients carry a uniform scale (~1.059).  Keeping
# it in c2w violates the OpenCV rigid-camera contract and corrupts projection.
BONN_T_MARKER[:3, :3] = _nearest_so3(BONN_T_MARKER[:3, :3])


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def _manifest_row(metadata_path: Path, output_root: Path) -> dict | None:
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    record_root = metadata_path.parent
    relative = lambda path: str(path.relative_to(output_root))
    memory_eligible = bool(metadata.get("memory_eligible", metadata.get("correspondence") is not None))
    record = {
            "record_id": metadata["record_id"], "dataset": metadata["dataset"],
            "scene_id": metadata["scene_id"], "sequence_id": metadata["sequence_id"],
            "rgb_dir": relative(record_root / "rgb"), "depth_dir": relative(record_root / "depth"),
            "c2w_abs": relative(record_root / "c2w_abs.npy"), "c2w_local": relative(record_root / "c2w_local.npy"),
            "intrinsics": relative(record_root / "intrinsics.npy"), "timestamps": relative(record_root / "timestamps.npy"),
            "metadata": relative(metadata_path), "frame_count": FRAME_COUNT, "chunk_count": CHUNK_COUNT,
            "height": HEIGHT, "width": WIDTH, "training_scope": metadata.get("training_scope", "rgbd_memory"),
            "memory_eligible": memory_eligible, "intrinsics_quality": metadata.get("intrinsics_quality", "calibrated"),
            "split": sequence_split(metadata["dataset"], metadata["sequence_id"]),
    }
    correspondence = record_root / "correspondence_cache.npz"
    if memory_eligible:
        if not correspondence.is_file():
            return None
        record["correspondence_cache"] = relative(correspondence)
    return record


def _discover_completed_records(output_root: Path) -> list[dict]:
    records = []
    for metadata_path in sorted((output_root / "records").glob("*/*/metadata.json")):
        record = _manifest_row(metadata_path, output_root)
        if record is not None:
            records.append(record)
    return records


def _write_manifests(output_root: Path) -> list[dict]:
    all_records = _discover_completed_records(output_root)
    header = {"schema_version": "rgbd-memory-manifest-v1", "frame_count": 97, "chunk_count": 3, "height": 480, "width": 832}
    _atomic_json(output_root / "manifest_all.json", {**header, "records": all_records})
    _atomic_json(output_root / "manifest_train.json", {**header, "records": [row for row in all_records if row["split"] == "train"]})
    _atomic_json(output_root / "manifest_val.json", {**header, "records": [row for row in all_records if row["split"] == "val"]})
    return all_records


def _continuous_segments(observations: list[dict]) -> tuple[list[list[dict]], int]:
    """Return the reliable synchronized observation stream.

    Frames that cannot be synchronized have already been removed.  Stride=1
    therefore means consecutive entries in this retained stream; splitting it
    again on ordinary sensor timestamp jitter would incorrectly discard most
    of the official TUM/Bonn sequences.
    """
    if not observations:
        return [], 0
    segments = [[observations[0]]]
    boundaries = 0
    for observation in observations[1:]:
        previous = segments[-1][-1]
        if "source_index" in previous and "source_index" in observation and int(observation["source_index"]) != int(previous["source_index"]) + 1:
            segments.append([])
            boundaries += 1
        segments[-1].append(observation)
    return segments, boundaries


def _undistort_pair(rgb: np.ndarray, depth_m: np.ndarray, K: np.ndarray, distortion: np.ndarray | None):
    if distortion is None or not np.any(distortion):
        return rgb, depth_m
    height, width = depth_m.shape
    map_x, map_y = cv2.initUndistortRectifyMap(K, distortion, None, K, (width, height), cv2.CV_32FC1)
    return (
        cv2.remap(rgb, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT),
        cv2.remap(depth_m, map_x, map_y, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT),
    )


def _write_record(dataset: str, scene_id: str, sequence_id: str, window_index: int, observations: list[dict], output_root: Path, K_source: np.ndarray, *, depth_factor: float, distortion: np.ndarray | None, source_metadata: dict, corr_stride: int, build_correspondence: bool = True, training_scope: str = "rgbd_memory", intrinsics_quality: str = "calibrated") -> tuple[dict, dict]:
    if len(observations) != FRAME_COUNT:
        raise ValueError("record window must contain exactly 97 observations")
    record_id = f"{dataset}__{sequence_id.replace('/', '_')}__{window_index:06d}"
    record_root = output_root / "records" / dataset / record_id
    metadata_path = record_root / "metadata.json"
    if metadata_path.is_file():
        completed = _manifest_row(metadata_path, output_root)
        if completed is not None:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            return completed, metadata.get("correspondence") or {"row_count": 0}
    rgb_dir, depth_dir = record_root / "rgb", record_root / "depth"
    rgb_dir.mkdir(parents=True, exist_ok=True); depth_dir.mkdir(parents=True, exist_ok=True)
    intrinsics, transforms = [], []
    for index, observation in enumerate(observations):
        destination_rgb = rgb_dir / f"{index:06d}.png"
        destination_depth = depth_dir / f"{index:06d}.png"
        if destination_rgb.is_file() and destination_depth.is_file():
            source = cv2.imread(str(observation["rgb"]), cv2.IMREAD_COLOR)
            _, K_new = center_crop_resize_geometry(*source.shape[:2], K_source)
            intrinsics.append(K_new)
            continue
        rgb_bgr = cv2.imread(str(observation["rgb"]), cv2.IMREAD_COLOR)
        raw_depth = cv2.imread(str(observation["depth"]), cv2.IMREAD_UNCHANGED)
        if rgb_bgr is None or raw_depth is None:
            raise ValueError("failed to read an RGB/depth observation")
        depth_m = raw_depth.astype(np.float32) / float(depth_factor)
        rgb_bgr, depth_m = _undistort_pair(rgb_bgr, depth_m, K_source, distortion)
        rgb_out, depth_mm, K_new, transform = transform_rgb_depth(rgb_bgr, depth_m, K_source)
        if not cv2.imwrite(str(destination_rgb), rgb_out, (cv2.IMWRITE_PNG_COMPRESSION, 3)):
            raise OSError(destination_rgb)
        if not cv2.imwrite(str(destination_depth), depth_mm, (cv2.IMWRITE_PNG_COMPRESSION, 3)):
            raise OSError(destination_depth)
        intrinsics.append(K_new); transforms.append(transform)
    c2w_abs = np.stack([row["c2w"] for row in observations]).astype(np.float64)
    c2w_local = localize_c2w(c2w_abs)
    K_all = np.stack(intrinsics).astype(np.float64)
    timestamps = np.asarray([row["timestamp"] for row in observations], dtype=np.float64)
    np.save(record_root / "c2w_abs.npy", c2w_abs)
    np.save(record_root / "c2w_local.npy", c2w_local)
    np.save(record_root / "intrinsics.npy", K_all)
    np.save(record_root / "timestamps.npy", timestamps)
    corr_stats = {"row_count": 0}
    if build_correspondence:
        corr_stats = build_causal_correspondence_cache(
            sorted(depth_dir.glob("*.png")), c2w_abs, K_all,
            record_root / "correspondence_cache.npz", pixel_stride=corr_stride,
        )
    metadata = {
        "schema_version": "rgbd-memory-record-v1", "record_id": record_id,
        "dataset": dataset, "scene_id": scene_id, "sequence_id": sequence_id,
        "source": source_metadata, "source_indices": [row.get("source_index") for row in observations],
        "timestamp_range": [float(timestamps[0]), float(timestamps[-1])],
        "pose_convention": "OpenCV camera-to-world (+x right, +y down, +z forward)",
        "depth_unit": "millimeters uint16; 0 invalid", "frame_count": FRAME_COUNT,
        "chunk_count": CHUNK_COUNT, "chunks": [[0, 32], [32, 64], [64, 96]],
        "height": HEIGHT, "width": WIDTH, "stride": 1, "interpolation": "none",
        "image_transform": transforms[0] if transforms else "already complete",
        "intrinsics_transform": "K' = diag(sx,sy,1) @ translate(-crop_left,-crop_top) @ K",
        "training_scope": training_scope, "memory_eligible": build_correspondence,
        "intrinsics_quality": intrinsics_quality,
        "correspondence": corr_stats if build_correspondence else None,
    }
    _atomic_json(record_root / "metadata.json", metadata)
    relative = lambda path: str(path.relative_to(output_root))
    record = {
        "record_id": record_id, "dataset": dataset, "scene_id": scene_id, "sequence_id": sequence_id,
        "rgb_dir": relative(rgb_dir), "depth_dir": relative(depth_dir),
        "c2w_abs": relative(record_root / "c2w_abs.npy"), "c2w_local": relative(record_root / "c2w_local.npy"),
        "intrinsics": relative(record_root / "intrinsics.npy"), "timestamps": relative(record_root / "timestamps.npy"),
        "metadata": relative(record_root / "metadata.json"), "frame_count": FRAME_COUNT,
        "chunk_count": CHUNK_COUNT, "height": HEIGHT, "width": WIDTH,
        "training_scope": training_scope, "memory_eligible": build_correspondence,
        "intrinsics_quality": intrinsics_quality,
        "split": sequence_split(dataset, sequence_id),
    }
    if build_correspondence:
        record["correspondence_cache"] = relative(record_root / "correspondence_cache.npz")
    return record, corr_stats


def _tum_or_bonn_sequences(dataset: str, source_root: Path):
    for sequence_root in sorted({path.parent for path in source_root.rglob("rgb.txt") if (path.parent / "depth.txt").is_file() and (path.parent / "groundtruth.txt").is_file()}):
        rgb = [(row[0], sequence_root / row[1]) for row in read_timestamp_file(sequence_root / "rgb.txt", 2)]
        depth = [(row[0], sequence_root / row[1]) for row in read_timestamp_file(sequence_root / "depth.txt", 2)]
        poses = read_timestamp_file(sequence_root / "groundtruth.txt", 8)
        observations, association = associate_timestamp_streams(rgb, depth, poses)
        if dataset == "bonn":
            for row in observations:
                row["c2w"] = BONN_T_ROS @ row["c2w"] @ BONN_T_ROS @ BONN_T_MARKER
        for source_index, row in enumerate(observations):
            row["source_index"] = source_index
        missing_rgb = sum(not Path(row["rgb"]).is_file() for row in observations)
        missing_depth = sum(not Path(row["depth"]).is_file() for row in observations)
        observations = [row for row in observations if Path(row["rgb"]).is_file() and Path(row["depth"]).is_file()]
        association.update(missing_rgb_files=missing_rgb, missing_depth_files=missing_depth, usable=len(observations))
        yield sequence_root.name, observations, association


def _nrgbd_sequences(source_root: Path):
    for scene_root in sorted(path for path in source_root.rglob("*") if path.is_dir() and (path / "poses.txt").is_file() and (path / "images").is_dir()):
        rgb = sorted((scene_root / "images").glob("*.png"), key=lambda p: int("".join(filter(str.isdigit, p.stem)) or 0))
        depth_root = scene_root / "depth"
        depth = sorted(depth_root.glob("*.png"), key=lambda p: int("".join(filter(str.isdigit, p.stem)) or 0))
        pose_rows = np.loadtxt(scene_root / "poses.txt").reshape(-1, 4, 4)
        focal = float((scene_root / "focal.txt").read_text().split()[0])
        count = min(len(rgb), len(depth), len(pose_rows))
        observations = []
        for index in range(count):
            rgb_id = int("".join(filter(str.isdigit, rgb[index].stem)) or index)
            depth_id = int("".join(filter(str.isdigit, depth[index].stem)) or index)
            if rgb_id != depth_id:
                continue
            observations.append({"timestamp": float(rgb_id), "rgb": str(rgb[index]), "depth": str(depth[index]), "c2w": pose_rows[index] @ OPENGL_TO_OPENCV, "source_index": rgb_id})
        yield scene_root.name, observations, {"rgb_input": len(rgb), "depth_input": len(depth), "pose_input": len(pose_rows), "associated": len(observations), "dropped_rgb": len(rgb) - len(observations), "focal": focal}


def _seven_scenes_sequences(source_root: Path):
    sequence_roots = sorted({path.parent for path in source_root.rglob("frame-*.pose.txt")})
    for sequence_root in sequence_roots:
        relative_parts = sequence_root.relative_to(source_root).parts
        scene_id = relative_parts[0]
        sequence_id = f"{scene_id}/{sequence_root.name}"
        pose_files = {path.name.removesuffix(".pose.txt"): path for path in sequence_root.glob("frame-*.pose.txt")}
        rgb_files = {path.name.removesuffix(".color.png"): path for path in sequence_root.glob("frame-*.color.png")}
        depth_files = {path.name.removesuffix(".depth.png"): path for path in sequence_root.glob("frame-*.depth.png")}
        frame_ids = sorted(set(pose_files) & set(rgb_files) & set(depth_files), key=lambda name: int(name.split("-")[-1]))
        observations = []
        invalid_poses = 0
        for frame_id in frame_ids:
            c2w = np.loadtxt(pose_files[frame_id], dtype=np.float64)
            if c2w.shape != (4, 4) or not np.isfinite(c2w).all():
                invalid_poses += 1
                continue
            source_index = int(frame_id.split("-")[-1])
            observations.append({"timestamp": float(source_index), "rgb": str(rgb_files[frame_id]), "depth": str(depth_files[frame_id]), "c2w": c2w, "source_index": source_index})
        association = {
            "rgb_input": len(rgb_files), "depth_input": len(depth_files), "pose_input": len(pose_files),
            "associated": len(observations), "invalid_pose": invalid_poses,
        }
        yield scene_id, sequence_id, observations, association


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("tum", "bonn", "nrgbd", "7scenes"), required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--correspondence-pixel-stride", type=int, default=4)
    parser.add_argument("--rebuild-manifests-only", action="store_true")
    parser.add_argument("--sequence-prefix", action="append", default=[])
    parser.add_argument("--defer-index", action="store_true", help="Process a disjoint shard without writing shared manifests/reports")
    args = parser.parse_args()
    if args.rebuild_manifests_only:
        records = _write_manifests(args.output_root)
        print(json.dumps({"records": len(records), "mode": "rebuild-manifests-only"}, indent=2))
        return
    if args.dataset in ("tum", "bonn"):
        sequences = _tum_or_bonn_sequences(args.dataset, args.source_root)
        K_source = TUM_K if args.dataset == "tum" else BONN_K
        distortion = None if args.dataset == "tum" else BONN_DIST
        depth_factor = 5000.0
        source_url = "https://cvg.cit.tum.de/data/datasets/rgbd-dataset/download" if args.dataset == "tum" else "https://www.ipb.uni-bonn.de/data/rgbd-dynamic-dataset/"
        pose_conversion = "groundtruth quaternion/translation is the OpenCV optical-center c2w published by TUM" if args.dataset == "tum" else "official Bonn conversion: c2w_sensor = T_ROS @ T_groundtruth @ T_ROS @ T_marker"
    elif args.dataset == "nrgbd":
        sequences = _nrgbd_sequences(args.source_root)
        K_source = distortion = depth_factor = source_url = None
    else:
        sequences = _seven_scenes_sequences(args.source_root)
        K_source, distortion, depth_factor = SEVEN_SCENES_K, None, 1000.0
        source_url = "https://www.microsoft.com/en-us/research/project/rgb-d-dataset-7-scenes/"
    records, report, filters = [], {}, Counter()
    for sequence_row in sequences:
        if args.dataset == "7scenes":
            scene_id, sequence_id, observations, association = sequence_row
        else:
            sequence_id, observations, association = sequence_row
            scene_id = sequence_id
        if args.sequence_prefix and not any(sequence_id.startswith(prefix) for prefix in args.sequence_prefix):
            continue
        segments, gap_count = _continuous_segments(observations)
        filters["timestamp_gap_boundaries"] += gap_count
        filters["missing_rgb_files"] += int(association.get("missing_rgb_files", 0))
        filters["missing_depth_files"] += int(association.get("missing_depth_files", 0))
        sequence_records = []
        record_index = 0
        for segment in segments:
            windows = len(segment) // FRAME_COUNT
            filters["tail_frames_shorter_than_97"] += len(segment) - windows * FRAME_COUNT
            for window_index in range(windows):
                window = segment[window_index * FRAME_COUNT:(window_index + 1) * FRAME_COUNT]
                if args.dataset == "nrgbd":
                    sample = cv2.imread(str(window[0]["rgb"]), cv2.IMREAD_COLOR)
                    focal = float(association["focal"])
                    local_K = np.asarray(((focal, 0, (sample.shape[1] - 1) / 2), (0, focal, (sample.shape[0] - 1) / 2), (0, 0, 1.0)))
                    local_factor = 1000.0
                else:
                    local_K, local_factor = K_source, depth_factor
                record, corr = _write_record(
                    args.dataset, scene_id, sequence_id, record_index, window, args.output_root,
                    local_K, depth_factor=local_factor, distortion=distortion,
                    source_metadata={"official_url": source_url or "https://github.com/dazinovic/neural-rgbd-surface-reconstruction", "association": association, "pose_conversion": pose_conversion if args.dataset in ("tum", "bonn") else ("official OpenGL c2w @ diag(1,-1,-1,1) -> OpenCV c2w" if args.dataset == "nrgbd" else "official 7Scenes camera-to-world matrix; no axis guess or conversion")},
                    corr_stride=args.correspondence_pixel_stride,
                    build_correspondence=args.dataset != "7scenes",
                    training_scope="camera_only" if args.dataset == "7scenes" else "rgbd_memory",
                    intrinsics_quality="official default Kinect depth-camera intrinsics; RGB-depth uncalibrated" if args.dataset == "7scenes" else "calibrated",
                )
                records.append(record); sequence_records.append(record); filters["correspondence_rows"] += corr["row_count"]
                record_index += 1
        report[sequence_id] = {"associated_frames": len(observations), "segments": len(segments), "clips": len(sequence_records), "association": association}
    if not args.defer_index:
        _write_manifests(args.output_root)
        _atomic_json(args.output_root / "reports" / f"{args.dataset}.json", {"dataset": args.dataset, "sequence_count": len(report), "record_count": len(records), "sequences": report, "filters": dict(filters)})
    print(json.dumps({"dataset": args.dataset, "sequences": len(report), "records": len(records), "filters": filters}, indent=2))


if __name__ == "__main__":
    main()
