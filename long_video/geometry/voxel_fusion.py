"""One canonical 0.02 world-unit voxel fusion implementation."""
from __future__ import annotations

import numpy as np


def fuse_voxels(points_xyz, points_rgb, confidence, observation_count=None, voxel_size=0.02):
    """Fuse observations with weight_sum=confidence_mean*observation_count.

    This is deliberately shared by Pi3X initialization and ReCal3R publishing:
    a point entering the same world-space voxel follows exactly the same rule.
    """
    if float(voxel_size) != 0.02:
        raise ValueError("persistent PointWorld voxel size is exactly 0.02")
    xyz = np.asarray(points_xyz, np.float32)
    rgb = np.asarray(points_rgb, np.uint8)
    conf = np.asarray(confidence, np.float32)
    obs = np.ones(len(xyz), np.int32) if observation_count is None else np.asarray(observation_count, np.int32).clip(1)
    valid = np.isfinite(xyz).all(1) & np.isfinite(conf) & (conf > 0)
    xyz, rgb, conf, obs = xyz[valid], rgb[valid], conf[valid], obs[valid]
    if not len(xyz):
        return (np.empty((0, 3), np.float32), np.empty((0, 3), np.uint8),
                np.empty(0, np.float32), np.empty(0, np.uint16), np.empty((0, 3), np.int64))
    keys = np.floor(xyz / float(voxel_size)).astype(np.int64)
    unique, inverse = np.unique(keys, axis=0, return_inverse=True)
    weight_sum = conf * obs.astype(np.float32)
    total = np.bincount(inverse, weights=weight_sum, minlength=len(unique)).astype(np.float32)
    count = np.bincount(inverse, weights=obs, minlength=len(unique)).astype(np.int32)
    out_xyz = np.stack([np.bincount(inverse, weights=weight_sum * xyz[:, axis], minlength=len(unique)) / total for axis in range(3)], 1)
    out_rgb = np.stack([np.bincount(inverse, weights=weight_sum * rgb[:, axis], minlength=len(unique)) / total for axis in range(3)], 1)
    return (out_xyz.astype(np.float32), np.rint(np.clip(out_rgb, 0, 255)).astype(np.uint8),
            (total / count.clip(1)).astype(np.float32), np.minimum(count, 65535).astype(np.uint16), unique)
