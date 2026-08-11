"""Official DL3DV-10K 480P selection and causal trajectory preparation.

This module never consumes COLMAP point clouds.  Camera poses are read only to
select trajectories and to condition the model; persistent geometry is built
at runtime from the source image and causally generated history through Pi3.
"""
from __future__ import annotations

from dataclasses import dataclass
from html.parser import HTMLParser
import csv
import json
import math
from pathlib import Path
import re
import numpy as np


OFFICIAL_REPO = "DL3DV/DL3DV-ALL-480P"
CHUNK_FRAMES = 33
CHUNK_STRIDE = 32
TARGET_FPS = 24.0
TARGET_HW = (384, 640)
OPENGL_TO_OPENCV = np.diag([1.0, -1.0, -1.0, 1.0]).astype(np.float32)


class _PreviewParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.rows, self.row, self.text, self.tag = [], None, "", None

    def handle_starttag(self, tag, attrs):
        if tag == "tr":
            self.row = []
        if self.row is not None and tag in {"figcaption", "td"}:
            self.tag, self.text = tag, ""

    def handle_data(self, data):
        if self.tag:
            self.text += data

    def handle_endtag(self, tag):
        if self.row is not None and tag == self.tag:
            value = " ".join(self.text.split())
            if value:
                self.row.append(value)
            self.tag, self.text = None, ""
        if tag == "tr" and self.row is not None:
            if self.row:
                self.rows.append(self.row)
            self.row = None


def read_official_metadata(csv_path, preview_html_path):
    """Join the official valid-scene CSV with official preview labels."""
    valid = {}
    with Path(csv_path).open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            valid[row["hash"]] = {
                "scene_hash": row["hash"], "batch": row["batch"],
                "duration": float(row.get("duration") or 0.0),
                "sensibility_label": row.get("sensibility label", ""),
            }
    parser = _PreviewParser()
    parser.feed(Path(preview_html_path).read_text(encoding="utf-8"))
    fields = ("scene_hash", "batch", "boundedness", "reflection", "transparency",
              "lighting", "poi", "category", "device")
    labels = {row[0]: dict(zip(fields, row[: len(fields)])) for row in parser.rows
              if len(row) >= 8 and re.fullmatch(r"[0-9a-f]{64}", row[0])}
    records = []
    for scene_hash, row in valid.items():
        if scene_hash in labels:
            row.update(labels[scene_hash])
            records.append(row)
    return records


def scene_environment(record):
    text = " ".join(str(record.get(k, "")) for k in ("poi", "category")).lower()
    indoor = ("indoor", "room", "corridor", "hall", "store", "shop", "cafe", "restaurant",
              "museum", "library", "office", "hotel", "mall", "classroom", "kitchen", "bank")
    outdoor = ("outdoor", "park", "campus", "courtyard", "plaza", "square", "building", "street",
               "garden", "recreation", "architecture")
    if any(x in text for x in indoor): return "indoor"
    if any(x in text for x in outdoor): return "outdoor"
    return "other"


def candidate_quality(record):
    """Metadata-only rank before any bytes from a scene are downloaded."""
    text = " ".join(str(v) for v in record.values()).lower()
    excluded = ("drone", "water", "lake", "river", "ocean", "beach", "crowd", "traffic",
                "vehicle", "nightclub", "mirror", "aquarium")
    if any(x in text for x in excluded):
        return -math.inf
    environment = scene_environment(record)
    if environment == "other":
        return -math.inf
    score = 10.0
    score += 2.0 if record.get("boundedness") == "bd" else 0.0
    reflection = str(record.get("reflection", "")).lower()
    transparency = str(record.get("transparency", "")).lower()
    lighting = str(record.get("lighting", "")).lower()
    score += 1.0 if "nonreflection" in reflection else -1.0
    score += 1.0 if "nontransparent" in transparency else -1.0
    score += 1.0 if lighting == "nlight" else -0.5
    score += min(float(record.get("duration", 0.0)), 120.0) / 120.0
    return score


def ranked_candidates(records, *, seed=20260812):
    rng = np.random.default_rng(seed)
    buckets = {"indoor": [], "outdoor": []}
    for item in records:
        score = candidate_quality(item)
        env = scene_environment(item)
        if np.isfinite(score) and env in buckets:
            enriched = dict(item, environment=env, metadata_score=float(score), tie_break=float(rng.random()))
            buckets[env].append(enriched)
    for values in buckets.values():
        values.sort(key=lambda x: (-x["metadata_score"], x["tie_break"], x["scene_hash"]))
    result = []
    while buckets["indoor"] or buckets["outdoor"]:
        for env in ("indoor", "outdoor"):
            if buckets[env]: result.append(buckets[env].pop(0))
    return result


def resample_real_frame_indices(frame_times, target_fps=TARGET_FPS):
    times = np.asarray(frame_times, np.float64)
    if times.ndim != 1 or len(times) < 2 or np.any(np.diff(times) <= 0):
        raise ValueError("frame times must be a strictly increasing vector")
    target = np.arange(times[0], times[-1] + 1e-9, 1.0 / float(target_fps))
    right = np.searchsorted(times, target, side="left").clip(0, len(times) - 1)
    left = (right - 1).clip(0, len(times) - 1)
    choose_left = np.abs(target - times[left]) <= np.abs(times[right] - target)
    return np.where(choose_left, left, right).astype(np.int64)


def _frame_time(frame, ordinal, source_fps):
    for key in ("time", "timestamp", "time_sec"):
        if key in frame: return float(frame[key])
    return float(ordinal) / float(source_fps)


def _intrinsics(meta, frame, width, height):
    def get(name, default=None):
        value = frame.get(name, meta.get(name, default))
        return None if value is None else float(value)
    fx, fy = get("fl_x"), get("fl_y")
    if fx is None and get("camera_angle_x") is not None:
        fx = 0.5 * width / math.tan(0.5 * get("camera_angle_x"))
    if fy is None: fy = fx
    if fx is None: raise ValueError("DL3DV transforms must provide fl_x or camera_angle_x")
    cx, cy = get("cx", (width - 1) / 2), get("cy", (height - 1) / 2)
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], np.float32)


@dataclass
class DL3DVScene:
    root: Path
    image_paths: list[Path]
    frame_times: np.ndarray
    c2w_opencv: np.ndarray
    intrinsics: np.ndarray
    source_hw: tuple[int, int]


def load_dl3dv_scene(scene_root, *, source_fps=30.0):
    """Read official Nerfstudio images+poses, never COLMAP point geometry."""
    root = Path(scene_root)
    matches = sorted(root.rglob("transforms.json"))
    if len(matches) != 1:
        raise ValueError(f"expected one transforms.json under {root}, found {len(matches)}")
    transform_path = matches[0]
    meta = json.loads(transform_path.read_text(encoding="utf-8"))
    frames = meta.get("frames") or []
    if len(frames) < 2: raise ValueError("DL3DV scene has fewer than two registered frames")
    from PIL import Image
    paths, times, poses, ks = [], [], [], []
    first_hw = None
    for ordinal, frame in enumerate(frames):
        rel = str(frame["file_path"])
        path = transform_path.parent / rel
        if not path.suffix:
            for suffix in (".png", ".jpg", ".jpeg"):
                if path.with_suffix(suffix).is_file(): path = path.with_suffix(suffix); break
        if not path.is_file(): raise FileNotFoundError(path)
        with Image.open(path) as im: width, height = im.size
        if first_hw is None: first_hw = (height, width)
        if first_hw != (height, width): raise ValueError("mixed image sizes are unsupported")
        pose = np.asarray(frame["transform_matrix"], np.float32)
        if pose.shape != (4, 4): raise ValueError("transform_matrix must be 4x4")
        paths.append(path); times.append(_frame_time(frame, ordinal, source_fps))
        poses.append(pose @ OPENGL_TO_OPENCV)
        ks.append(_intrinsics(meta, frame, width, height))
    order = np.argsort(np.asarray(times), kind="stable")
    return DL3DVScene(root, [paths[i] for i in order], np.asarray(times)[order],
                      np.asarray(poses)[order], np.asarray(ks)[order], first_hw)


def center_crop_resize_geometry(source_hw, intrinsics, target_hw=TARGET_HW):
    source_h, source_w = map(int, source_hw); target_h, target_w = map(int, target_hw)
    target_aspect = target_w / target_h; source_aspect = source_w / source_h
    if source_aspect >= target_aspect:
        crop_h, crop_w = source_h, int(round(source_h * target_aspect))
    else:
        crop_w, crop_h = source_w, int(round(source_w / target_aspect))
    left, top = (source_w - crop_w) // 2, (source_h - crop_h) // 2
    k = np.asarray(intrinsics, np.float32).copy()
    k[..., 0, 2] -= left; k[..., 1, 2] -= top
    sx, sy = target_w / crop_w, target_h / crop_h
    k[..., 0, 0] *= sx; k[..., 1, 1] *= sy
    k[..., 0, 2] = (k[..., 0, 2] + .5) * sx - .5
    k[..., 1, 2] = (k[..., 1, 2] + .5) * sy - .5
    return (left, top, left + crop_w, top + crop_h), k


def rotation_angle_degrees(a, b):
    relative = np.asarray(a)[..., :3, :3].swapaxes(-1, -2) @ np.asarray(b)[..., :3, :3]
    value = ((np.trace(relative, axis1=-2, axis2=-1) - 1) / 2).clip(-1, 1)
    return np.degrees(np.arccos(value))


def source_relative_opencv_c2w(c2w, source_index):
    poses = np.asarray(c2w, np.float32)
    local = np.linalg.inv(poses[int(source_index)]) @ poses
    local[:, 3] = np.array([0, 0, 0, 1], np.float32)
    return local.astype(np.float32)


def _revisit_score(poses, earlier, later, middle):
    pos = poses[:, :3, 3]
    distance = float(np.linalg.norm(pos[later] - pos[earlier]))
    angle = float(rotation_angle_degrees(poses[earlier], poses[later]))
    excursion_t = float(np.linalg.norm(pos[middle] - pos[earlier]))
    excursion_r = float(rotation_angle_degrees(poses[earlier], poses[middle]))
    gap = later - earlier
    return (2.0 * math.tanh(excursion_t) + excursion_r / 90.0 + math.log1p(gap) / 5.0
            - 1.5 * math.tanh(distance) - angle / 60.0)


def select_revisit_trajectories(scene: DL3DVScene, *, max_trajectories=2):
    sampled = resample_real_frame_indices(scene.frame_times)
    poses = scene.c2w_opencv[sampled]
    candidates = []
    for chunks in (12, 8):
        length = chunks * CHUNK_STRIDE + 1
        if len(sampled) < length: continue
        step = max(1, CHUNK_STRIDE // 2)
        for start in range(step, len(sampled) - length + 1, step):
            end = start + length - 1
            for sample_type in ("source_revisit", "world_revisit"):
                earlier = start if sample_type == "source_revisit" else start + 2 * CHUNK_STRIDE
                min_later = earlier + 3 * CHUNK_STRIDE
                possible = range(min_later, end + 1, max(1, CHUNK_STRIDE // 2))
                best = None
                for later in possible:
                    middle = earlier + int(np.argmax(
                        np.linalg.norm(poses[earlier:later + 1, :3, 3] - poses[earlier, :3, 3], axis=1)
                    ))
                    score = _revisit_score(poses, earlier, later, middle)
                    if best is None or score > best[0]: best = (score, earlier, later, middle)
                if best:
                    candidates.append({"score": best[0], "sample_type": sample_type,
                        "chunk_count": chunks, "start": start, "end": end,
                        "earlier": best[1], "later": best[2], "middle": best[3]})
    candidates.sort(key=lambda x: (-x["score"], -x["chunk_count"], x["start"]))
    selected = []
    for candidate in candidates:
        if candidate["sample_type"] in {x["sample_type"] for x in selected}: continue
        if any(max(candidate["start"], x["start"]) <= min(candidate["end"], x["end"])
               and candidate["sample_type"] == x["sample_type"] for x in selected): continue
        selected.append(candidate)
        if len(selected) >= max_trajectories: break
    for item in selected:
        indices = sampled[item["start"]:item["end"] + 1]
        item["real_frame_indices"] = indices.tolist()
        item["source_global_frame"] = int(indices[0])
        item["revisit_earlier_output_frame"] = int(item["earlier"] - item["start"])
        item["revisit_later_output_frame"] = int(item["later"] - item["start"])
        item["revisit_earlier_chunk"] = item["revisit_earlier_output_frame"] // CHUNK_STRIDE
        item["revisit_later_chunk"] = item["revisit_later_output_frame"] // CHUNK_STRIDE
    return selected


def chunk_real_indices(indices, chunk_count):
    values = list(map(int, indices))
    if len(values) != chunk_count * CHUNK_STRIDE + 1: raise ValueError("trajectory length mismatch")
    return [values[k * CHUNK_STRIDE:k * CHUNK_STRIDE + CHUNK_FRAMES] for k in range(chunk_count)]


def validate_trajectory_record(record, root=None):
    base = Path(root or ".")
    count = int(record["chunk_count"])
    if count not in (8, 12): raise ValueError("chunk_count must be 8 or 12")
    chunks = record["chunk_real_frame_indices"]
    if len(chunks) != count or any(len(x) != CHUNK_FRAMES for x in chunks):
        raise ValueError("every chunk must contain exactly 33 real frame indices")
    for a, b in zip(chunks, chunks[1:]):
        if a[-1] != b[0]: raise ValueError("adjacent chunks must share exactly their boundary frame")
    if record.get("uses_future_gt") is not False: raise ValueError("uses_future_gt must be false")
    if int(record["source_global_frame"]) == 0: raise ValueError("source must not default to frame zero")
    earlier, later = int(record["revisit_earlier_output_frame"]), int(record["revisit_later_output_frame"])
    if not 0 <= earlier < later < count * CHUNK_STRIDE + 1: raise ValueError("invalid revisit pair")
    if set(record["trainable_chunk_indices"]) != set(range(count)):
        raise ValueError("all chunks 0..N-1 must be trainable")
    if root is not None:
        poses = np.load(base / record["target_c2w_local"])
        ks = np.load(base / record["intrinsics"])
        if len(poses) != count * CHUNK_STRIDE + 1 or len(ks) != len(poses):
            raise ValueError("RGB/pose/intrinsics count mismatch")
        if not np.allclose(poses[0], np.eye(4), atol=1e-5): raise ValueError("source local pose is not identity")
        det = np.linalg.det(poses[:, :3, :3])
        if not np.allclose(det, 1, atol=1e-3): raise ValueError("invalid rotation determinant")
    return True
