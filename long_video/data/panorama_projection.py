"""Equirectangular and OpenCV perspective projections.

Camera convention is OpenCV: +x right, +y down, +z forward.  Equirectangular
latitude is positive up, so latitude equals -asin(direction_y).
"""
import numpy as np
from .erp_geometry import perspective_unit_rays

def intrinsics_from_fov(fov_degrees, width, height):
    f = 0.5 * width / np.tan(np.deg2rad(float(fov_degrees)) * 0.5)
    return np.array([[f, 0., (width - 1.) * .5], [0., f, (height - 1.) * .5], [0., 0., 1.]], np.float32)

def rotation_yaw_pitch(yaw, pitch=0.0):
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    yaw_r = np.array([[cy, 0., sy], [0., 1., 0.], [-sy, 0., cy]], np.float32)
    # Positive pitch looks upward in an image coordinate system whose +y points down.
    pitch_r = np.array([[1., 0., 0.], [0., cp, sp], [0., -sp, cp]], np.float32)
    return yaw_r @ pitch_r

def _sample_equirectangular(image, u, v, nearest):
    h, w = image.shape[:2]
    if nearest:
        return image[np.clip(np.rint(v).astype(np.int64), 0, h - 1), np.mod(np.rint(u).astype(np.int64), w)]
    u0 = np.floor(u).astype(np.int64); v0 = np.floor(v).astype(np.int64)
    u1 = u0 + 1; v1 = np.clip(v0 + 1, 0, h - 1)
    u0 = np.mod(u0, w); u1 = np.mod(u1, w); v0 = np.clip(v0, 0, h - 1)
    wu = (u - np.floor(u))[..., None] if image.ndim == 3 else (u - np.floor(u))
    wv = (v - np.floor(v))[..., None] if image.ndim == 3 else (v - np.floor(v))
    return ((1-wu)*(1-wv)*image[v0,u0] + wu*(1-wv)*image[v0,u1] + (1-wu)*wv*image[v1,u0] + wu*wv*image[v1,u1])

def equirectangular_to_perspective(image, yaw, pitch=0., fov_degrees=90., height=512, width=512, interpolation="bilinear"):
    image = np.asarray(image)
    ph, pw = image.shape[:2]
    k = intrinsics_from_fov(fov_degrees, width, height)
    rays = perspective_unit_rays(k, height, width)
    direction = rays @ rotation_yaw_pitch(yaw, pitch).T
    longitude = np.arctan2(direction[...,0], direction[...,2])
    latitude = -np.arcsin(np.clip(direction[...,1], -1., 1.))
    u = (longitude / (2*np.pi) + .5) * pw - .5
    v = (.5 - latitude / np.pi) * ph - .5
    sampled = _sample_equirectangular(image.astype(np.float32), u, v, interpolation == "nearest")
    if interpolation == "nearest": return sampled.astype(image.dtype)
    if np.issubdtype(image.dtype, np.integer): return np.clip(np.rint(sampled), 0, np.iinfo(image.dtype).max).astype(image.dtype)
    return sampled.astype(image.dtype)

def perspective_to_equirectangular(image, yaw, pitch=0., fov_degrees=90., height=512, width=1024, fill_value=0):
    image = np.asarray(image); ih, iw = image.shape[:2]; k = intrinsics_from_fov(fov_degrees, iw, ih)
    yy, xx = np.indices((height, width), dtype=np.float32)
    lon = ((xx + .5) / width - .5) * 2*np.pi; lat = (.5 - (yy + .5) / height) * np.pi
    direction = np.stack((np.cos(lat)*np.sin(lon), -np.sin(lat), np.cos(lat)*np.cos(lon)), -1)
    local = direction @ rotation_yaw_pitch(yaw, pitch)
    valid = local[...,2] > 1e-6
    u = k[0,0] * local[...,0] / np.maximum(local[...,2], 1e-6) + k[0,2]
    v = k[1,1] * local[...,1] / np.maximum(local[...,2], 1e-6) + k[1,2]
    valid &= (u >= 0) & (u < iw-1) & (v >= 0) & (v < ih-1)
    out = np.full((height,width)+image.shape[2:], fill_value, image.dtype)
    if valid.any():
        vals = _sample_equirectangular(np.pad(image, ((0,0),(0,0))+((0,0),)*(image.ndim-2)), u, v, False)
        out[valid] = vals[valid]
    return out

def build_canonical_view_cameras(panorama_center_c2w, fov_degrees, width, height, yaws_degrees=(0,45,90,135,180,225,270,315), pitch_degrees=0.):
    center = np.asarray(panorama_center_c2w, np.float32)
    c2w = []; k = intrinsics_from_fov(fov_degrees, width, height)
    for yaw in yaws_degrees:
        local = np.eye(4, dtype=np.float32)
        local[:3,:3] = rotation_yaw_pitch(np.deg2rad(yaw), np.deg2rad(pitch_degrees))
        c2w.append(center @ local)
    return np.stack(c2w), np.repeat(k[None], len(yaws_degrees), axis=0)

# Backward-compatible name.
panorama_to_perspective = equirectangular_to_perspective
