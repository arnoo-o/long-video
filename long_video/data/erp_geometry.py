"""Canonical ERP rays and RGB-D projection for OpenCV camera coordinates."""

from __future__ import annotations

import numpy as np

from ..types import RAY_DISTANCE, Z_DEPTH


def erp_unit_rays(height: int, width: int, pixel_center: float = 0.5) -> np.ndarray:
    """Return [H,W,3] unit rays where ERP center is OpenCV +z."""
    height, width = int(height), int(width)
    if height <= 0 or width <= 0 or width != 2 * height:
        raise ValueError(f"ERP must have a positive 2:1 resolution, got {(height, width)}")
    if float(pixel_center) not in {0.0, 0.5}:
        raise ValueError("pixel_center must be 0.0 or 0.5")
    yy, xx = np.indices((height, width), dtype=np.float32)
    longitude = ((xx + pixel_center) / width - 0.5) * (2.0 * np.pi)
    latitude = (0.5 - (yy + pixel_center) / height) * np.pi
    rays = np.stack(
        (
            np.cos(latitude) * np.sin(longitude),
            -np.sin(latitude),
            np.cos(latitude) * np.cos(longitude),
        ),
        axis=-1,
    )
    return (rays / np.linalg.norm(rays, axis=-1, keepdims=True).clip(1e-8)).astype(np.float32)


def perspective_unit_rays(intrinsics: np.ndarray, height: int, width: int) -> np.ndarray:
    yy, xx = np.indices((int(height), int(width)), dtype=np.float32)
    pixels = np.stack((xx, yy, np.ones_like(xx)), axis=-1)
    rays = pixels @ np.linalg.inv(np.asarray(intrinsics, np.float32)).T
    return (rays / np.linalg.norm(rays, axis=-1, keepdims=True).clip(1e-8)).astype(np.float32)


def ray_distance_to_z_depth(ray_distance: np.ndarray, unit_rays: np.ndarray) -> np.ndarray:
    distance = np.asarray(ray_distance, np.float32)
    rays = np.asarray(unit_rays, np.float32)
    if distance.shape != rays.shape[:-1]:
        raise ValueError("ray distance and ray grid shapes do not match")
    result = distance * rays[..., 2]
    result[~np.isfinite(distance) | (distance <= 0) | (rays[..., 2] <= 0)] = np.nan
    return result.astype(np.float32)


def z_depth_to_ray_distance(z_depth: np.ndarray, unit_rays: np.ndarray) -> np.ndarray:
    depth = np.asarray(z_depth, np.float32)
    rays = np.asarray(unit_rays, np.float32)
    result = depth / np.maximum(rays[..., 2], 1e-8)
    result[~np.isfinite(depth) | (depth <= 0) | (rays[..., 2] <= 0)] = np.nan
    return result.astype(np.float32)


def backproject_erp_ray_distance(
    rgb: np.ndarray,
    ray_distance: np.ndarray,
    valid_mask: np.ndarray,
    c2w: np.ndarray,
    *,
    pixel_center: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    rgb = np.asarray(rgb)
    depth = np.asarray(ray_distance, np.float32)
    mask = np.asarray(valid_mask, bool)
    if rgb.shape[:2] != depth.shape or mask.shape != depth.shape:
        raise ValueError("ERP RGB, ray-distance depth, and mask shapes must match")
    rays = erp_unit_rays(*depth.shape, pixel_center=pixel_center)
    valid = mask & np.isfinite(depth) & (depth > 0)
    camera_points = rays[valid] * depth[valid, None]
    pose = np.asarray(c2w, np.float32)
    world_points = camera_points @ pose[:3, :3].T + pose[:3, 3]
    return world_points.astype(np.float32), rgb[valid]


def source_relative_c2w(source_c2w_world: np.ndarray, target_c2w_world: np.ndarray) -> np.ndarray:
    source = np.asarray(source_c2w_world, np.float32)
    target = np.asarray(target_c2w_world, np.float32)
    result = np.linalg.inv(source) @ target
    if not np.isfinite(result).all():
        raise ValueError("source-relative c2w contains non-finite values")
    return result.astype(np.float32)


def assert_depth_convention(value: str) -> str:
    if value not in {RAY_DISTANCE, Z_DEPTH}:
        raise ValueError(f"Unsupported depth convention: {value}")
    return value
