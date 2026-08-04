import numpy as np

def backproject(depth, rgb, c2w, intrinsics, confidence=None, source=None):
    h, w = depth.shape
    yy, xx = np.indices((h, w), dtype=np.float32)
    valid = np.isfinite(depth) & (depth > 0)
    if confidence is None: confidence = np.ones_like(depth, dtype=np.float32)
    if source is None: source = np.zeros_like(depth, dtype=np.int8)
    pix = np.stack((xx[valid], yy[valid], np.ones(valid.sum(), dtype=np.float32)), 1)
    cam = pix * depth[valid, None]
    world = (c2w[:3, :3] @ cam.T).T + c2w[:3, 3]
    return world, rgb[valid], confidence[valid], source[valid]
