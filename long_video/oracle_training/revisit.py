"""Gap-safe Holo360D qualification and real-pose revisit selection."""
from __future__ import annotations

from dataclasses import dataclass
import io
from pathlib import Path
from zipfile import ZipFile

import numpy as np

from .dense24 import continuous_runs


def bidirectional_depth_reprojection_overlap(
    depth, visibility, c2w, intrinsics, *, depth_tolerance_m=0.05,
    relative_depth_tolerance=0.03,
):
    """Measure mutual visible geometry using only renderer Z-depth outputs."""
    depth = np.asarray(depth, np.float32)
    visibility = np.asarray(visibility, bool)
    c2w = np.asarray(c2w, np.float32)
    intrinsics = np.asarray(intrinsics, np.float32)
    if depth.shape != visibility.shape or depth.ndim != 3 or depth.shape[0] != 2:
        raise ValueError("depth and visibility must describe exactly two [H,W] renderings")
    if c2w.shape != (2, 4, 4) or intrinsics.shape != (2, 3, 3):
        raise ValueError("bidirectional overlap requires two c2w and intrinsic matrices")
    if not (np.isfinite(c2w).all() and np.isfinite(intrinsics).all()):
        raise ValueError("camera inputs must be finite")
    if depth_tolerance_m < 0 or relative_depth_tolerance < 0:
        raise ValueError("depth tolerances must be non-negative")

    def project(source, target):
        source_depth = depth[source]
        valid = visibility[source] & np.isfinite(source_depth) & (source_depth > 0)
        denominator = int(valid.sum())
        if denominator == 0:
            return 0.0
        y, x = np.nonzero(valid)
        z = source_depth[y, x]
        source_k = intrinsics[source]
        camera_points = np.stack((
            (x - source_k[0, 2]) * z / source_k[0, 0],
            (y - source_k[1, 2]) * z / source_k[1, 1],
            z,
        ), axis=1)
        world = camera_points @ c2w[source, :3, :3].T + c2w[source, :3, 3]
        target_w2c = np.linalg.inv(c2w[target])
        target_camera = world @ target_w2c[:3, :3].T + target_w2c[:3, 3]
        target_z = target_camera[:, 2]
        target_k = intrinsics[target]
        u = np.rint(target_k[0, 0] * target_camera[:, 0] / np.maximum(target_z, 1e-8) + target_k[0, 2]).astype(np.int64)
        v = np.rint(target_k[1, 1] * target_camera[:, 1] / np.maximum(target_z, 1e-8) + target_k[1, 2]).astype(np.int64)
        target_depth = depth[target]
        inside = (target_z > 0) & (u >= 0) & (u < target_depth.shape[1]) & (v >= 0) & (v < target_depth.shape[0])
        matches = np.zeros(denominator, dtype=bool)
        indices = np.flatnonzero(inside)
        if len(indices):
            sampled_depth = target_depth[v[indices], u[indices]]
            sampled_visible = visibility[target, v[indices], u[indices]]
            tolerance = np.maximum(float(depth_tolerance_m), float(relative_depth_tolerance) * sampled_depth)
            matches[indices] = (
                sampled_visible & np.isfinite(sampled_depth) & (sampled_depth > 0)
                & (np.abs(target_z[indices] - sampled_depth) <= tolerance)
            )
        return float(matches.mean())

    overlap = 0.5 * (project(0, 1) + project(1, 0))
    return float(np.clip(overlap if np.isfinite(overlap) else 0.0, 0.0, 1.0))


@dataclass(frozen=True)
class MultiChunkContract:
    chunks: int
    chunk_frames: int = 33
    chunk_stride: int = 32
    anchor_stride: int = 8

    @property
    def dense_frames(self):
        return 1 + self.chunk_stride * int(self.chunks)

    @property
    def anchors(self):
        return 1 + self.chunk_stride * int(self.chunks) // self.anchor_stride

    def validate(self):
        if self.chunk_frames != self.chunk_stride + 1:
            raise ValueError("adjacent chunks must share exactly one boundary frame")
        if self.chunk_stride % self.anchor_stride:
            raise ValueError("chunk boundary must land on a real anchor")
        if self.dense_frames != 1 + 32 * self.chunks or self.anchors != 4 * self.chunks + 1:
            raise ValueError("multi-chunk frame/anchor contract changed")
        return self


def _rotation_angle_degrees(left, right):
    relative = np.swapaxes(left, -1, -2) @ right
    cosine = np.clip((np.trace(relative, axis1=-2, axis2=-1) - 1) * 0.5, -1, 1)
    return np.rad2deg(np.arccos(cosine))


def _read_pose(handle: ZipFile, member: str):
    values = np.loadtxt(io.BytesIO(handle.read(member)), dtype=np.float64).reshape(-1)
    return _pose_from_values(values, member)


def _pose_from_values(values, source: str):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size != 12 or not np.isfinite(values).all():
        raise ValueError(f"invalid pose payload: {source}")
    pose = np.eye(4, dtype=np.float64)
    pose[:3, 3] = values[:3]
    pose[:3, :3] = values[3:].reshape(3, 3)
    error = np.max(np.abs(pose[:3, :3].T @ pose[:3, :3] - np.eye(3)))
    if error > 1e-3 or abs(np.linalg.det(pose[:3, :3]) - 1) > 1e-3:
        raise ValueError(f"pose rotation is not orthonormal: {source}")
    return pose


def _read_consolidated_poses(handle: ZipFile, member: str):
    lines = handle.read(member).decode("utf-8").splitlines()
    if not lines or lines[0].split() != [
        "image", "x", "y", "z", "r0", "r1", "r2", "r3", "r4", "r5", "r6", "r7", "r8",
    ]:
        raise ValueError(f"unexpected consolidated pose header: {member}")
    poses = {}
    for line_number, line in enumerate(lines[1:], start=2):
        fields = line.split()
        if len(fields) != 13:
            raise ValueError(f"invalid consolidated pose row {member}:{line_number}")
        stem = Path(fields[0]).stem
        if stem in poses:
            raise ValueError(f"duplicate consolidated pose timestamp: {stem}")
        poses[stem] = _pose_from_values(fields[1:], f"{member}:{line_number}")
    return poses


def scan_holo360d_zip(path, *, gap_factor=2.5):
    """Verify RGB/depth/mask/pose correspondence and return pose-only statistics."""
    archive = Path(path)
    with ZipFile(archive) as handle:
        names = [item.filename for item in handle.infolist() if not item.is_dir()]
        roots = {name.split("/", 1)[0] for name in names if "/" in name}
        if len(roots) != 1:
            raise ValueError(f"archive must contain exactly one scene root, got {sorted(roots)}")
        root = next(iter(roots))
        suffixes = {
            "rgb": (f"{root}/rgb/", ".jpg"),
            "depth": (f"{root}/depth/mesh_depth/", ".exr"),
            "mask": (f"{root}/mask/", ".jpg"),
        }
        groups = {}
        members = {}
        for key, (prefix, suffix) in suffixes.items():
            selected = [name for name in names if name.startswith(prefix) and name.endswith(suffix)]
            groups[key] = {Path(name).stem for name in selected}
            members[key] = {Path(name).stem: name for name in selected}
        consolidated_member = f"{root}/poses/pose.txt"
        if consolidated_member in names:
            pose_values = _read_consolidated_poses(handle, consolidated_member)
            pose_format = "consolidated"
        else:
            pose_members = {
                Path(name).stem: name
                for name in names
                if name.startswith(f"{root}/poses/") and name.endswith(".txt")
            }
            pose_values = {stem: _read_pose(handle, member) for stem, member in pose_members.items()}
            pose_format = "per_frame"
        groups["pose"] = set(pose_values)
        union = set.union(*groups.values())
        intersection = set.intersection(*groups.values())
        missing = {key: sorted(union - values, key=float) for key, values in groups.items()}
        if any(missing.values()) or not intersection:
            raise ValueError(f"incomplete RGB/depth/mask/pose correspondence: {missing}")
        frame_ids = sorted(intersection, key=float)
        timestamps = np.asarray([float(item) for item in frame_ids], np.float64)
        runs, median = continuous_runs(timestamps, float(gap_factor))
        poses = np.stack([pose_values[item] for item in frame_ids])

    positions = poses[:, :3, 3]
    from_first = np.linalg.norm(positions - positions[:1], axis=1)
    rotation_from_first = _rotation_angle_degrees(poses[:1, :3, :3], poses[:, :3, :3])
    report = {
        "archive": str(archive),
        "scene_id": root,
        "archive_size_bytes": archive.stat().st_size,
        "frame_count": len(frame_ids),
        "matched_counts": {key: len(value) for key, value in groups.items()},
        "pose_format": pose_format,
        "correspondence_complete": True,
        "median_timestamp_delta": float(median),
        "gap_threshold": float(gap_factor * median),
        "continuous_runs": [
            {"start": int(start), "end": int(end), "anchor_count": int(end - start),
             "first_frame_id": frame_ids[start], "last_frame_id": frame_ids[end - 1]}
            for start, end in runs
        ],
        "max_translation": float(from_first.max(initial=0)),
        "max_rotation_degrees": float(rotation_from_first.max(initial=0)),
    }
    return report, frame_ids, poses, runs


def _training_chunk_for_anchor(anchor_offset: int, chunks: int):
    return min(max(int(anchor_offset) // 4, 0), int(chunks) - 1)


def _chunk_anchor_bounds(chunk_index: int):
    start = 4 * int(chunk_index)
    return start, start + 4


def score_revisit_window(poses, start: int, anchors: int):
    """Pose-only revisit prefilter; renderer overlap is added in finalization."""
    window = np.asarray(poses[start:start + anchors], np.float64)
    if len(window) != anchors:
        raise ValueError("revisit window is truncated")
    positions = window[:, :3, 3]
    rotations = window[:, :3, :3]
    max_translation = float(np.linalg.norm(positions - positions[:1], axis=1).max(initial=0))
    max_rotation = float(_rotation_angle_degrees(rotations[:1], rotations).max(initial=0))
    split = max(1, anchors // 2)
    best = None
    for earlier in range(split):
        for later in range(split, anchors):
            separation = later - earlier
            translation = float(np.linalg.norm(positions[later] - positions[earlier]))
            rotation = float(_rotation_angle_degrees(rotations[earlier], rotations[later]))
            orientation_overlap = max(0.0, (np.cos(np.deg2rad(rotation)) + 1.0) * 0.5)
            distance_scale = max(max_translation, 0.25)
            pose_overlap = float(np.exp(-translation / distance_scale) * orientation_overlap)
            score = pose_overlap * np.log1p(separation) + 0.05 * max_translation + 0.002 * max_rotation
            candidate = (score, separation, pose_overlap, translation, rotation, earlier, later)
            if best is None or candidate > best:
                best = candidate
    score, separation, overlap, translation, rotation, earlier, later = best
    chunks = (int(anchors) - 1) // 4
    training_chunk_index = _training_chunk_for_anchor(later, chunks)
    chunk_start, chunk_end = _chunk_anchor_bounds(training_chunk_index)
    if not chunk_start <= later <= chunk_end:
        raise RuntimeError("revisit later anchor is outside its selected training chunk")
    return {
        "start": int(start),
        "anchor_count": int(anchors),
        "max_translation": max_translation,
        "max_rotation_degrees": max_rotation,
        "revisit_translation": float(translation),
        "revisit_rotation_degrees": float(rotation),
        "temporal_separation_anchors": int(separation),
        "earlier_anchor_offset": int(earlier),
        "later_anchor_offset": int(later),
        "pose_overlap_proxy": float(overlap),
        "pose_prefilter_score": float(score),
        "revisit_score": None,
        "renderer_overlap": None,
        "renderer_overlap_metric": "projected_visibility_iou",
        "training_chunk_index": int(training_chunk_index),
        "selection_translation": float(translation),
        "selection_rotation_degrees": float(rotation),
        "selection_temporal_gap_anchors": int(separation),
    }


def score_large_motion_window(poses, start: int, anchors: int):
    """Choose the current chunk where the window's strongest local motion occurs."""
    window = np.asarray(poses[start:start + anchors], np.float64)
    if len(window) != anchors:
        raise ValueError("large-motion window is truncated")
    chunks = (int(anchors) - 1) // 4
    best = None
    for chunk_index in range(chunks):
        anchor_start, anchor_end = _chunk_anchor_bounds(chunk_index)
        local = window[anchor_start:anchor_end + 1]
        positions = local[:, :3, 3]
        rotations = local[:, :3, :3]
        for earlier in range(len(local) - 1):
            for later in range(earlier + 1, len(local)):
                translation = float(np.linalg.norm(positions[later] - positions[earlier]))
                rotation = float(_rotation_angle_degrees(rotations[earlier], rotations[later]))
                temporal_gap = later - earlier
                score = translation + 0.01 * rotation
                candidate = (score, translation, rotation, temporal_gap, chunk_index, earlier, later)
                if best is None or candidate > best:
                    best = candidate
    score, translation, rotation, temporal_gap, chunk_index, earlier, later = best
    earlier_offset = 4 * chunk_index + earlier
    later_offset = 4 * chunk_index + later
    max_translation = float(
        np.linalg.norm(window[:, :3, 3] - window[:1, :3, 3], axis=1).max(initial=0)
    )
    max_rotation = float(
        _rotation_angle_degrees(window[:1, :3, :3], window[:, :3, :3]).max(initial=0)
    )
    distance_scale = max(max_translation, 0.25)
    orientation_overlap = max(0.0, (np.cos(np.deg2rad(rotation)) + 1.0) * 0.5)
    pose_proxy = float(np.exp(-translation / distance_scale) * orientation_overlap)
    return {
        "start": int(start), "anchor_count": int(anchors),
        "max_translation": max_translation, "max_rotation_degrees": max_rotation,
        "earlier_anchor_offset": int(earlier_offset),
        "later_anchor_offset": int(later_offset),
        "pose_overlap_proxy": pose_proxy,
        "pose_prefilter_score": float(score),
        "large_motion_score": None,
        "renderer_overlap": None,
        "renderer_overlap_metric": "projected_visibility_iou",
        "training_chunk_index": int(chunk_index),
        "selection_translation": float(translation),
        "selection_rotation_degrees": float(rotation),
        "selection_temporal_gap_anchors": int(temporal_gap),
    }


def _candidate_windows(poses, runs, chunks, candidate_stride, scorer):
    contract = MultiChunkContract(int(chunks)).validate()
    candidates = []
    for run_start, run_end in runs:
        last_start = int(run_end) - contract.anchors
        for start in range(int(run_start), last_start + 1, int(candidate_stride)):
            item = scorer(poses, start, contract.anchors)
            item.update({"chunks": int(chunks), "dense_frames": contract.dense_frames})
            candidates.append(item)
    return candidates


def select_revisit_windows(
    poses, runs, *, chunk_counts=(8, 12, 16), candidate_stride=4, prefilter_count=8,
):
    """Return pose-prefiltered candidates; these are not final until renderer scoring."""
    selected = []
    for chunks in chunk_counts:
        candidates = _candidate_windows(poses, runs, chunks, candidate_stride, score_revisit_window)
        selected.extend(sorted(
            candidates, key=lambda item: item["pose_prefilter_score"], reverse=True,
        )[:int(prefilter_count)])
    return selected


def select_large_motion_windows(
    poses, runs, *, chunk_counts=(8, 12, 16), candidate_stride=4, prefilter_count=8,
):
    selected = []
    for chunks in chunk_counts:
        candidates = _candidate_windows(poses, runs, chunks, candidate_stride, score_large_motion_window)
        selected.extend(sorted(
            candidates, key=lambda item: item["pose_prefilter_score"], reverse=True,
        )[:int(prefilter_count)])
    return selected


def add_renderer_overlap(candidate, overlap: float, *, sample_type: str):
    """Finalize one pose-prefiltered candidate with measured projected overlap."""
    item = dict(candidate)
    overlap = float(overlap)
    if not np.isfinite(overlap) or not 0.0 <= overlap <= 1.0:
        raise ValueError(f"renderer overlap must be finite in [0,1], got {overlap}")
    item["renderer_overlap"] = overlap
    if sample_type == "revisit":
        item["revisit_score"] = (
            overlap * np.log1p(item["selection_temporal_gap_anchors"])
            + 0.05 * item["max_translation"]
            + 0.002 * item["max_rotation_degrees"]
        )
    elif sample_type == "large_motion":
        item["large_motion_score"] = (
            item["selection_translation"]
            + 0.01 * item["selection_rotation_degrees"]
            + 0.1 * (1.0 - overlap)
        )
    else:
        raise ValueError(f"unsupported Phase B sample type: {sample_type}")
    item["selection_stage"] = "renderer_final"
    return item


def windows_highly_overlap(left, right, threshold=0.8):
    left_start, left_end = int(left["start"]), int(left["start"] + left["anchor_count"])
    right_start, right_end = int(right["start"]), int(right["start"] + right["anchor_count"])
    intersection = max(0, min(left_end, right_end) - max(left_start, right_start))
    union = max(left_end, right_end) - min(left_start, right_start)
    return union > 0 and intersection / union >= float(threshold)


def choose_independent_final_candidates(revisit_candidates, motion_candidates):
    """Choose one revisit/motion pair per scene and curriculum length."""
    final_revisit, final_motion = [], []
    groups = sorted({
        (str(item.get("scene_id", "")), int(item["chunks"]))
        for item in revisit_candidates + motion_candidates
    })
    for scene_id, chunks in groups:
        revisit = sorted(
            [
                item for item in revisit_candidates
                if str(item.get("scene_id", "")) == scene_id
                and int(item["chunks"]) == chunks
            ],
            key=lambda item: item["revisit_score"], reverse=True,
        )
        motion = sorted(
            [
                item for item in motion_candidates
                if str(item.get("scene_id", "")) == scene_id
                and int(item["chunks"]) == chunks
            ],
            key=lambda item: item["large_motion_score"], reverse=True,
        )
        if not revisit or not motion:
            continue
        chosen_revisit = revisit[0]
        independent = [item for item in motion if not windows_highly_overlap(chosen_revisit, item)]
        chosen_motion = independent[0] if independent else motion[0]
        chosen_motion = dict(chosen_motion)
        chosen_motion["independent_from_revisit"] = bool(independent)
        final_revisit.append(chosen_revisit)
        final_motion.append(chosen_motion)
    return final_revisit, final_motion
