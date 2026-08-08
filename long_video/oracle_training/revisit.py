"""Gap-safe Holo360D qualification and real-pose revisit selection."""
from __future__ import annotations

from dataclasses import dataclass
import io
from pathlib import Path
from zipfile import ZipFile

import numpy as np

from .dense24 import continuous_runs


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


def score_revisit_window(poses, start: int, anchors: int):
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
        "revisit_score": float(score),
    }


def select_revisit_windows(poses, runs, *, chunk_counts=(8, 12, 16), candidate_stride=4):
    selected = []
    for chunks in chunk_counts:
        contract = MultiChunkContract(int(chunks)).validate()
        candidates = []
        for run_start, run_end in runs:
            last_start = int(run_end) - contract.anchors
            for start in range(int(run_start), last_start + 1, int(candidate_stride)):
                item = score_revisit_window(poses, start, contract.anchors)
                item.update({"chunks": int(chunks), "dense_frames": contract.dense_frames})
                candidates.append(item)
        if candidates:
            selected.append(max(candidates, key=lambda item: item["revisit_score"]))
    return selected


def select_large_motion_windows(poses, runs, *, chunk_counts=(8, 12, 16), candidate_stride=4):
    selected = []
    for chunks in chunk_counts:
        contract = MultiChunkContract(int(chunks)).validate()
        candidates = []
        for run_start, run_end in runs:
            last_start = int(run_end) - contract.anchors
            for start in range(int(run_start), last_start + 1, int(candidate_stride)):
                item = score_revisit_window(poses, start, contract.anchors)
                item.update({"chunks": int(chunks), "dense_frames": contract.dense_frames})
                item["large_motion_score"] = (
                    item["max_translation"] + 0.01 * item["max_rotation_degrees"]
                    + 0.25 * (1.0 - item["pose_overlap_proxy"])
                )
                candidates.append(item)
        if candidates:
            selected.append(max(candidates, key=lambda item: item["large_motion_score"]))
    return selected
