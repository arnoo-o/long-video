"""Canonical RGB-D geometry and causal correspondence construction.

All poses exposed by this module are OpenCV camera-to-world transforms: +x right,
+y down, +z forward.  No dense world-coordinate tensor is materialized.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import cv2
import numpy as np

HEIGHT, WIDTH = 480, 832
CHUNK_STRIDE = 32
# Compatibility defaults only.  New code must take geometry from each record.
FRAME_COUNT, CHUNK_COUNT = 97, 3
LATENT_LOCAL_FRAMES = np.asarray((0, 4, 8, 12, 16, 20, 24, 28, 32), dtype=np.int16)


def read_timestamp_file(path: str | Path, columns: int) -> list[tuple]:
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        values = line.split()
        if len(values) < columns:
            continue
        rows.append((float(values[0]), *values[1:columns]))
    return rows


def quaternion_c2w(tx: float, ty: float, tz: float, qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    quaternion = np.asarray((qx, qy, qz, qw), dtype=np.float64)
    norm = np.linalg.norm(quaternion)
    if not np.isfinite(norm) or norm < 1e-12:
        raise ValueError("invalid pose quaternion")
    x, y, z, w = quaternion / norm
    rotation = np.asarray([
        [1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)],
        [2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)],
        [2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)],
    ])
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = rotation
    result[:3, 3] = (tx, ty, tz)
    return result


def associate_timestamp_streams(rgb_rows: list[tuple], depth_rows: list[tuple], pose_rows: list[tuple], *, max_difference: float = 0.02) -> tuple[list[dict], dict]:
    """Associate RGB/depth/pose by nearest timestamp without reusing observations."""
    def nearest_unique(left, right, threshold):
        candidates = sorted((abs(a[0] - b[0]), i, j) for i, a in enumerate(left) for j, b in enumerate(right) if abs(a[0] - b[0]) <= threshold)
        used_left, used_right, pairs = set(), set(), []
        for error, i, j in candidates:
            if i not in used_left and j not in used_right:
                used_left.add(i); used_right.add(j); pairs.append((i, j, error))
        return sorted(pairs)

    rgb_depth = nearest_unique(rgb_rows, depth_rows, max_difference)
    paired = [(rgb_rows[i][0], i, j, error) for i, j, error in rgb_depth]
    pose_matches = nearest_unique(paired, pose_rows, max_difference)
    observations, rgb_depth_errors, pose_errors = [], [], []
    for pair_index, pose_index, pose_error in pose_matches:
        _, rgb_index, depth_index, depth_error = paired[pair_index]
        pose = pose_rows[pose_index]
        observations.append({
            "timestamp": float(rgb_rows[rgb_index][0]),
            "rgb": str(rgb_rows[rgb_index][1]),
            "depth": str(depth_rows[depth_index][1]),
            "c2w": quaternion_c2w(*map(float, pose[1:8])),
            "rgb_depth_error": float(depth_error),
            "rgb_pose_error": float(pose_error),
        })
        rgb_depth_errors.append(depth_error); pose_errors.append(pose_error)
    observations.sort(key=lambda row: row["timestamp"])
    stats = {
        "rgb_input": len(rgb_rows), "depth_input": len(depth_rows), "pose_input": len(pose_rows),
        "associated": len(observations), "dropped_rgb": len(rgb_rows) - len(observations),
        "max_rgb_depth_dt": max(rgb_depth_errors, default=0.0), "max_rgb_pose_dt": max(pose_errors, default=0.0),
    }
    return observations, stats


def center_crop_resize_geometry(height: int, width: int, K: np.ndarray, *, target_height: int = HEIGHT, target_width: int = WIDTH) -> tuple[tuple[int, int, int, int], np.ndarray]:
    """Return an integer center crop and the exactly transformed intrinsics."""
    source_aspect, target_aspect = width / height, target_width / target_height
    if source_aspect < target_aspect:
        crop_width = width
        crop_height = max(1, int(round(width / target_aspect)))
        left, top = 0, (height - crop_height) // 2
    else:
        crop_height = height
        crop_width = max(1, int(round(height * target_aspect)))
        top, left = 0, (width - crop_width) // 2
    right, bottom = left + crop_width, top + crop_height
    scale_x, scale_y = target_width / crop_width, target_height / crop_height
    transformed = np.asarray(K, dtype=np.float64).copy()
    transformed[0, 0] *= scale_x; transformed[1, 1] *= scale_y
    transformed[0, 2] = (transformed[0, 2] - left) * scale_x
    transformed[1, 2] = (transformed[1, 2] - top) * scale_y
    return (left, top, right, bottom), transformed


def transform_rgb_depth(rgb: np.ndarray, depth_m: np.ndarray, K: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    if rgb.shape[:2] != depth_m.shape[:2]:
        raise ValueError(f"RGB/depth dimensions differ: {rgb.shape[:2]} vs {depth_m.shape[:2]}")
    crop, K_new = center_crop_resize_geometry(*rgb.shape[:2], K)
    left, top, right, bottom = crop
    rgb_out = cv2.resize(rgb[top:bottom, left:right], (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
    depth_out = cv2.resize(depth_m[top:bottom, left:right], (WIDTH, HEIGHT), interpolation=cv2.INTER_NEAREST)
    depth_mm = np.where(np.isfinite(depth_out) & (depth_out > 0) & (depth_out < 65.535), np.rint(depth_out * 1000), 0).astype(np.uint16)
    return rgb_out, depth_mm, K_new, {"crop_xyxy": crop, "source_hw": list(rgb.shape[:2]), "target_hw": [HEIGHT, WIDTH]}


def localize_c2w(c2w_abs: np.ndarray) -> np.ndarray:
    poses = np.asarray(c2w_abs, dtype=np.float64)
    return np.linalg.inv(poses[0]) @ poses


def latent_membership(frame: int, chunk: int) -> int:
    local = frame - chunk * CHUNK_STRIDE
    indices = np.flatnonzero(LATENT_LOCAL_FRAMES == local)
    if len(indices) != 1:
        raise ValueError(f"frame {frame} is not a latent representative of chunk {chunk}")
    return int(indices[0])


def _depth(path: Path) -> np.ndarray:
    value = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if value is None or value.shape != (HEIGHT, WIDTH):
        raise ValueError(f"invalid processed depth: {path}")
    return value.astype(np.float32) / 1000.0


def _project_world(world: np.ndarray, c2w: np.ndarray, K: np.ndarray):
    camera = (np.linalg.inv(c2w)[:3] @ np.concatenate((world, np.ones((len(world), 1))), axis=1).T).T
    z = camera[:, 2]
    uvw = (K @ camera.T).T
    uv = uvw[:, :2] / np.maximum(uvw[:, 2:3], 1e-12)
    return z, uv


def build_causal_correspondence_cache(depth_paths: list[Path], c2w: np.ndarray, K: np.ndarray, output: str | Path, *, chunk_count: int | None = None, pixel_stride: int = 4, depth_abs_tolerance: float = 0.03, depth_rel_tolerance: float = 0.02, cycle_pixels: float = 2.0, token_height: int = 30, token_width: int = 52) -> dict:
    """Build sparse causal correspondences for every query_chunk/key_chunk pair."""
    if chunk_count is None:
        if (len(depth_paths) - 1) % CHUNK_STRIDE:
            raise ValueError("frame count is not a shared-boundary chunk sequence")
        chunk_count = (len(depth_paths) - 1) // CHUNK_STRIDE
    expected = 1 + CHUNK_STRIDE * int(chunk_count)
    if not 1 <= int(chunk_count) <= 6 or len(depth_paths) != expected or c2w.shape != (expected, 4, 4) or K.shape != (expected, 3, 3):
        raise ValueError("correspondence inputs must be a 1..6 chunk shared-boundary sequence")
    depth_cache = {index: _depth(depth_paths[index]) for index in sorted(set(int(chunk * CHUNK_STRIDE + f) for chunk in range(chunk_count) for f in LATENT_LOCAL_FRAMES))}
    yy, xx = np.mgrid[0:HEIGHT:pixel_stride, 0:WIDTH:pixel_stride]
    pixels = np.stack((xx.ravel(), yy.ravel()), axis=1)
    batches: dict[str, list[np.ndarray]] = {key: [] for key in ("query_frame", "key_frame", "query_chunk", "key_chunk", "query_t", "key_t", "query_y", "query_x", "key_y", "key_x", "matched_count", "valid_count", "coverage", "vote", "weight")}
    pair_stats, raw_matches = {}, 0
    for query_chunk in range(1, chunk_count):
        for key_chunk in range(query_chunk):
            for query_t, local_query in enumerate(LATENT_LOCAL_FRAMES):
                query_frame = int(query_chunk * CHUNK_STRIDE + local_query)
                for key_t, local_key in enumerate(LATENT_LOCAL_FRAMES):
                    key_frame = int(key_chunk * CHUNK_STRIDE + local_key)
                    if key_frame >= query_frame:
                        continue
                    query_depth = depth_cache[query_frame]
                    key_depth = depth_cache[key_frame]
                    zq = query_depth[pixels[:, 1], pixels[:, 0]]
                    valid = np.isfinite(zq) & (zq > 0)
                    p = pixels[valid]; zq = zq[valid]
                    if not len(p):
                        continue
                    inv_kq = np.linalg.inv(K[query_frame])
                    camera = (inv_kq @ np.stack((p[:, 0] * zq, p[:, 1] * zq, zq), axis=0)).T
                    world = (c2w[query_frame][:3] @ np.concatenate((camera, np.ones((len(camera), 1))), axis=1).T).T
                    zk, uvk = _project_world(world, c2w[key_frame], K[key_frame])
                    finite_uv = np.isfinite(uvk).all(axis=1)
                    safe_uv = np.where(finite_uv[:, None], uvk, -1.0)
                    rounded = np.rint(np.clip(safe_uv, -1.0, max(WIDTH, HEIGHT) + 1.0)).astype(np.int32)
                    inside = finite_uv & (zk > 0) & (rounded[:, 0] >= 0) & (rounded[:, 0] < WIDTH) & (rounded[:, 1] >= 0) & (rounded[:, 1] < HEIGHT)
                    safe_x = np.clip(rounded[:, 0], 0, WIDTH - 1); safe_y = np.clip(rounded[:, 1], 0, HEIGHT - 1)
                    observed_key = key_depth[safe_y, safe_x]
                    tolerance = depth_abs_tolerance + depth_rel_tolerance * np.maximum(zk, observed_key)
                    consistent = inside & (observed_key > 0) & (np.abs(zk - observed_key) <= tolerance)
                    selected = np.flatnonzero(consistent)
                    if not len(selected):
                        pair_stats[f"{query_frame}->{key_frame}"] = 0
                        continue
                    key_pixels = rounded[selected]
                    key_z = observed_key[selected]
                    key_camera = (np.linalg.inv(K[key_frame]) @ np.stack((key_pixels[:, 0] * key_z, key_pixels[:, 1] * key_z, key_z), axis=0)).T
                    key_world = (c2w[key_frame][:3] @ np.concatenate((key_camera, np.ones((len(key_camera), 1))), axis=1).T).T
                    z_back, uv_back = _project_world(key_world, c2w[query_frame], K[query_frame])
                    query_selected = p[selected]
                    cycle = np.linalg.norm(uv_back - query_selected, axis=1) <= cycle_pixels
                    cycle &= np.abs(z_back - zq[selected]) <= (depth_abs_tolerance + depth_rel_tolerance * zq[selected])
                    selected = selected[cycle]
                    pair_stats[f"{query_frame}->{key_frame}"] = int(len(selected))
                    raw_matches += len(selected)
                    qp = p[selected]; kp = rounded[selected]
                    qtx = np.minimum(token_width - 1, qp[:, 0] * token_width // WIDTH)
                    qty = np.minimum(token_height - 1, qp[:, 1] * token_height // HEIGHT)
                    ktx = np.minimum(token_width - 1, kp[:, 0] * token_width // WIDTH)
                    kty = np.minimum(token_height - 1, kp[:, 1] * token_height // HEIGHT)
                    token_count = token_height * token_width
                    pair_id = (qty * token_width + qtx) * token_count + (kty * token_width + ktx)
                    unique, inverse, counts = np.unique(pair_id, return_inverse=True, return_counts=True)
                    residual = np.abs(zk[selected] - observed_key[selected])
                    point_weight = np.exp(-residual / np.maximum(tolerance[selected], 1e-6))
                    votes = np.bincount(inverse, weights=point_weight, minlength=len(unique)).astype(np.float32)
                    query_token, key_token = np.divmod(unique, token_count)
                    u_qty, u_qtx = np.divmod(query_token, token_width); u_kty, u_ktx = np.divmod(key_token, token_width)
                    size = len(unique); constant = lambda value: np.full(size, value, dtype=np.int32)
                    for key, value in (("query_frame", constant(query_frame)), ("key_frame", constant(key_frame)), ("query_chunk", constant(query_chunk)), ("key_chunk", constant(key_chunk)), ("query_t", constant(query_t)), ("key_t", constant(key_t)), ("query_y", u_qty.astype(np.int32)), ("query_x", u_qtx.astype(np.int32)), ("key_y", u_kty.astype(np.int32)), ("key_x", u_ktx.astype(np.int32)), ("matched_count", counts.astype(np.int32)), ("valid_count", counts.astype(np.int32)), ("coverage", np.ones(size, np.float32)), ("vote", votes), ("weight", votes / np.maximum(counts, 1))):
                        batches[key].append(value)
    integer_keys = {"query_frame", "key_frame", "query_chunk", "key_chunk", "query_t", "key_t", "query_y", "query_x", "key_y", "key_x", "matched_count", "valid_count"}
    arrays = {key: (np.concatenate(value).astype(np.int32 if key in integer_keys else np.float32, copy=False) if value else np.asarray([], dtype=np.int32 if key in integer_keys else np.float32)) for key, value in batches.items()}
    for key in integer_keys | {"coverage", "vote", "weight"}:
        arrays.setdefault(key, np.asarray([], dtype=np.int32 if key in integer_keys else np.float32))
    destination = Path(output); destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, **arrays)
    return {"row_count": len(arrays["query_frame"]), "raw_match_count": int(raw_matches), "pair_counts": pair_stats, "pixel_stride": pixel_stride, "token_grid": [token_height, token_width]}


def sequence_split(dataset: str, sequence_id: str, *, val_fraction: float = 0.2) -> str:
    value = int(hashlib.sha256(f"{dataset}/{sequence_id}".encode()).hexdigest()[:8], 16) / 2**32
    return "val" if value < val_fraction else "train"
