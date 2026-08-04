import numpy as np
from ..types import RAY_DISTANCE, Z_DEPTH

def _pixel_rays(intrinsics, height, width):
    y, x = np.indices((height, width), dtype=np.float32)
    pixels = np.stack((x, y, np.ones_like(x)), -1)
    rays = pixels @ np.linalg.inv(np.asarray(intrinsics, np.float32)).T
    return rays

def backproject_z_depth(depth, intrinsics):
    rays = _pixel_rays(intrinsics, *depth.shape)
    return rays * np.asarray(depth, np.float32)[...,None]

def backproject_ray_distance(depth, intrinsics):
    rays = _pixel_rays(intrinsics, *depth.shape)
    rays /= np.linalg.norm(rays, axis=-1, keepdims=True).clip(1e-8)
    return rays * np.asarray(depth, np.float32)[...,None]

def backproject(depth, rgb, c2w, intrinsics, confidence=None, source=None, depth_convention=RAY_DISTANCE):
    depth = np.asarray(depth, np.float32)
    valid = np.isfinite(depth) & (depth > 0)
    if confidence is None: confidence = np.ones_like(depth, np.float32)
    if source is None: source = np.zeros_like(depth, np.int8)
    camera = backproject_ray_distance(depth, intrinsics) if depth_convention == RAY_DISTANCE else backproject_z_depth(depth, intrinsics)
    cam = camera[valid]
    world = cam @ np.asarray(c2w, np.float32)[:3,:3].T + np.asarray(c2w, np.float32)[:3,3]
    return world.astype(np.float32), np.asarray(rgb)[valid], np.asarray(confidence)[valid], np.asarray(source)[valid]

def backproject_views(view_set):
    clouds = [backproject(view_set.depth[i], view_set.rgb[i], view_set.c2w[i], view_set.intrinsics[i], view_set.image_confidence[i], view_set.source[i], view_set.depth_convention) for i in range(len(view_set.rgb))]
    return clouds
