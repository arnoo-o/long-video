from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class Sim3Alignment:
    scale: float
    rotation: np.ndarray
    translation: np.ndarray
    camera_alignment_error: float
    camera_alignment_error_ratio: float
    median_rotation_error_degrees: float
    max_rotation_error_degrees: float

    def as_dict(self):
        return {
            "scale": float(self.scale), "rotation": self.rotation.tolist(),
            "translation": self.translation.tolist(),
            "camera_alignment_error": float(self.camera_alignment_error),
            "camera_alignment_error_ratio": float(self.camera_alignment_error_ratio),
            "median_rotation_error_degrees": float(self.median_rotation_error_degrees),
            "max_rotation_error_degrees": float(self.max_rotation_error_degrees),
        }


def validate_c2w(poses, frame_count):
    poses = np.asarray(poses, dtype=np.float64)
    if poses.shape != (frame_count, 4, 4):
        raise ValueError(f"expected c2w shape {(frame_count, 4, 4)}, got {poses.shape}")
    if not np.isfinite(poses).all() or not np.allclose(poses[:, 3], (0, 0, 0, 1), atol=1e-5):
        raise ValueError("invalid camera poses")
    eye = np.eye(3)
    ortho = np.max(np.abs(poses[:, :3, :3].transpose(0, 2, 1) @ poses[:, :3, :3] - eye))
    det = np.linalg.det(poses[:, :3, :3])
    if ortho > 2e-3 or np.max(np.abs(det - 1.0)) > 2e-3:
        raise ValueError("camera poses contain invalid rotations")
    return poses


def estimate_camera_sim3(recal_c2w, target_c2w):
    recal = validate_c2w(recal_c2w, len(target_c2w))
    target = validate_c2w(target_c2w, len(recal_c2w))
    src, dst = recal[:, :3, 3], target[:, :3, 3]
    src_mean, dst_mean = src.mean(0), dst.mean(0)
    src_centered, dst_centered = src - src_mean, dst - dst_mean
    variance = np.mean(np.sum(src_centered * src_centered, axis=1))
    if variance < 1e-10:
        raise ValueError("degenerate ReCal3R camera trajectory")
    covariance = dst_centered.T @ src_centered / len(src)
    u, singular, vt = np.linalg.svd(covariance)
    sign = np.ones(3)
    if np.linalg.det(u @ vt) < 0:
        sign[-1] = -1
    rotation = u @ np.diag(sign) @ vt
    scale = float(np.sum(singular * sign) / variance)
    translation = dst_mean - scale * (rotation @ src_mean)
    aligned = scale * (src @ rotation.T) + translation
    residual = np.linalg.norm(aligned - dst, axis=1)
    rmse = float(np.sqrt(np.mean(residual * residual)))
    extent = float(np.percentile(np.linalg.norm(dst - dst_mean, axis=1), 90))
    aligned_rot = rotation[None] @ recal[:, :3, :3]
    relative = target[:, :3, :3].transpose(0, 2, 1) @ aligned_rot
    cosine = np.clip((np.trace(relative, axis1=1, axis2=2) - 1) / 2, -1, 1)
    angles = np.degrees(np.arccos(cosine))
    return Sim3Alignment(scale, rotation, translation, rmse, rmse / max(extent, 1e-8),
                         float(np.median(angles)), float(np.max(angles)))


def validate_alignment(alignment, max_error_ratio=0.15, max_median_rotation_error_degrees=45.0):
    if not (np.isfinite(alignment.scale) and 1e-4 < alignment.scale < 1e4):
        raise ValueError(f"invalid Sim(3) scale {alignment.scale}")
    if alignment.camera_alignment_error_ratio > max_error_ratio:
        raise ValueError(f"camera alignment error ratio {alignment.camera_alignment_error_ratio:.4f} exceeds {max_error_ratio:.4f}")
    if alignment.median_rotation_error_degrees > max_median_rotation_error_degrees:
        raise ValueError(f"median camera rotation error {alignment.median_rotation_error_degrees:.2f} exceeds {max_median_rotation_error_degrees:.2f}")


def apply_sim3_points(points, alignment):
    return alignment.scale * (np.asarray(points) @ alignment.rotation.T) + alignment.translation


def apply_sim3_c2w(poses, alignment):
    result = np.asarray(poses, dtype=np.float64).copy()
    result[:, :3, :3] = alignment.rotation[None] @ result[:, :3, :3]
    result[:, :3, 3] = alignment.scale * (result[:, :3, 3] @ alignment.rotation.T) + alignment.translation
    return result.astype(np.float32)


def official_resize_crop(height, width, size=512):
    factor = float(size) / max(height, width)
    resized_w, resized_h = int(round(width * factor)), int(round(height * factor))
    cx, cy = resized_w // 2, resized_h // 2
    half_w, half_h = ((2 * cx) // 16) * 8, ((2 * cy) // 16) * 8
    if resized_w == resized_h:
        half_h = int(3 * half_w / 4)
    return {
        "original_height": height, "original_width": width,
        "resized_height": resized_h, "resized_width": resized_w,
        "crop_top": int(cy - half_h), "crop_left": int(cx - half_w),
        "crop_height": int(2 * half_h), "crop_width": int(2 * half_w),
    }


def remap_model_map(array, transform, interpolation):
    height, width = transform["original_height"], transform["original_width"]
    sx, sy = transform["resized_width"] / width, transform["resized_height"] / height
    x = (np.arange(width, dtype=np.float32) + .5) * sx - .5 - transform["crop_left"]
    y = (np.arange(height, dtype=np.float32) + .5) * sy - .5 - transform["crop_top"]
    map_x, map_y = np.meshgrid(x, y)
    inside = ((map_x >= 0) & (map_x <= transform["crop_width"] - 1)
              & (map_y >= 0) & (map_y <= transform["crop_height"] - 1))
    remapped = cv2.remap(np.asarray(array), map_x, map_y, interpolation=interpolation,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return remapped, inside


def fuse_voxel_observations(per_frame, voxel_size):
    keys_all, xyz_all, rgb_all, conf_all = [], [], [], []
    for xyz, rgb, confidence in per_frame:
        if len(xyz) == 0:
            continue
        keys = np.floor(xyz / voxel_size).astype(np.int64)
        unique, inverse = np.unique(keys, axis=0, return_inverse=True)
        conf = np.asarray(confidence, dtype=np.float64)
        weights = np.maximum(conf, 1e-8)
        weight_sum = np.bincount(inverse, weights=weights)
        frame_xyz = np.stack([np.bincount(inverse, weights=weights * xyz[:, a]) / weight_sum for a in range(3)], 1)
        frame_rgb = np.stack([np.bincount(inverse, weights=weights * rgb[:, a]) / weight_sum for a in range(3)], 1)
        frame_conf = np.bincount(inverse, weights=weights * conf) / weight_sum
        keys_all.append(unique); xyz_all.append(frame_xyz); rgb_all.append(frame_rgb); conf_all.append(frame_conf)
    if not keys_all:
        return {"points_xyz": np.empty((0, 3), np.float32), "points_rgb": np.empty((0, 3), np.uint8),
                "points_confidence": np.empty(0, np.float32), "observation_count": np.empty(0, np.uint16)}
    keys, xyz, rgb, conf = map(np.concatenate, (keys_all, xyz_all, rgb_all, conf_all))
    _, inverse = np.unique(keys, axis=0, return_inverse=True)
    weights = np.maximum(conf.astype(np.float64), 1e-8)
    weight_sum = np.bincount(inverse, weights=weights)
    points_xyz = np.stack([np.bincount(inverse, weights=weights * xyz[:, a]) / weight_sum for a in range(3)], 1)
    points_rgb = np.stack([np.bincount(inverse, weights=weights * rgb[:, a]) / weight_sum for a in range(3)], 1)
    points_conf = np.bincount(inverse, weights=weights * conf) / weight_sum
    count = np.bincount(inverse)
    return {"points_xyz": points_xyz.astype(np.float32),
            "points_rgb": np.clip(np.rint(points_rgb), 0, 255).astype(np.uint8),
            "points_confidence": points_conf.astype(np.float32),
            "observation_count": np.minimum(count, 65535).astype(np.uint16)}


def resolve_record_paths(record, dataset_root):
    result = {}
    for key in ("rgb_dir", "target_c2w_local", "intrinsics", "timestamps"):
        path = Path(record[key])
        result[key] = path if path.is_absolute() else Path(dataset_root) / path
    return result


def list_rgb_frames(rgb_dir, expected=193):
    frames = sorted(p for p in Path(rgb_dir).iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"})
    if len(frames) != expected:
        raise ValueError(f"expected {expected} RGB frames in {rgb_dir}, found {len(frames)}")
    return frames


def replace_directory(source, destination):
    if destination.exists():
        shutil.rmtree(destination)
    source.replace(destination)


def write_json(path, payload):
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")