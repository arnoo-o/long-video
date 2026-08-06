"""Camera and image-array helpers shared by initialization and online inference."""
from __future__ import annotations

import numpy as np


def resize_intrinsics(intrinsics, source_hw, target_hw):
    """Scale OpenCV intrinsics for a pixel-center preserving resize."""
    value = np.asarray(intrinsics, np.float32).copy()
    source_h, source_w = map(int, source_hw)
    target_h, target_w = map(int, target_hw)
    if min(source_h, source_w, target_h, target_w) <= 0:
        raise ValueError("image dimensions must be positive")
    sx, sy = target_w / source_w, target_h / source_h
    value[..., 0, 0] *= sx
    value[..., 1, 1] *= sy
    value[..., 0, 2] = (value[..., 0, 2] + 0.5) * sx - 0.5
    value[..., 1, 2] = (value[..., 1, 2] + 0.5) * sy - 0.5
    return value


def rgb_to_float01(rgb):
    value = np.asarray(rgb)
    if value.dtype == np.uint8:
        return value.astype(np.float32) / 255.0
    value = value.astype(np.float32)
    if not np.isfinite(value).all():
        raise ValueError("RGB contains NaN or infinity")
    if value.size and (value.min() < 0 or value.max() > 1):
        raise ValueError("floating RGB must be in [0,1]")
    return value


def rgb_to_uint8(rgb):
    value = np.asarray(rgb)
    if value.dtype == np.uint8:
        return value.copy()
    return np.rint(rgb_to_float01(value) * 255.0).clip(0, 255).astype(np.uint8)
