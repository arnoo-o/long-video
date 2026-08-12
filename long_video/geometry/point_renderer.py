"""Deterministic point-cloud rendering with causal parent-first composition."""

from __future__ import annotations

import copy

import numpy as np

from ..types import WarpBatch


INVALID_SOURCE = 4
_POINT_FIELDS = (
    "points_xyz", "points_rgb", "points_confidence", "points_source",
)
_OPTIONAL_WARP_FIELDS = (
    "rgb_content_origin", "depth_content_origin", "evidence_role",
    "rgb_evidence_role", "depth_evidence_role",
    "point_index", "winning_xyz_world",
)


def _coverage_metric(node, cameras, near, far, resolution=64):
    """Fixed angular occupancy, independent of output pixels and splat radius."""
    xyz = np.asarray(node.points_xyz, np.float32)
    result = []
    homogeneous = np.c_[xyz, np.ones(len(xyz), np.float32)]
    for pose, k in zip(cameras.c2w, cameras.intrinsics):
        cam = (np.linalg.inv(pose) @ homogeneous.T).T[:, :3]
        z = cam[:, 2]
        projected = cam @ np.asarray(k, np.float32).T
        u = projected[:, 0] / np.maximum(z, 1e-8) / cameras.width
        v = projected[:, 1] / np.maximum(z, 1e-8) / cameras.height
        valid = (z > near) & (z < far) & (u >= 0) & (u < 1) & (v >= 0) & (v < 1)
        x = np.floor(u[valid] * resolution).astype(np.int64)
        y = np.floor(v[valid] * resolution).astype(np.int64)
        occupied = np.unique(y * resolution + x).size
        result.append(occupied / (resolution * resolution))
    return np.asarray(result, np.float32)


def _effective_parent_point_count(node):
    """Read the cumulative parent split from new or legacy node schemas."""
    value = getattr(node, "parent_point_count", None)
    if value is None:
        metrics = getattr(node, "quality_metrics", {}) or {}
        value = metrics.get("parent_point_count")
    if value is None:
        return None
    try:
        value = int(value)
    except (TypeError, ValueError):
        return None
    return value if 0 < value < len(np.asarray(node.points_xyz)) else None


def _slice_node(node, start, stop):
    """Create a shallow node view containing only a point subset."""
    view = copy.copy(node)
    for name in _POINT_FIELDS:
        setattr(view, name, np.asarray(getattr(node, name))[start:stop])
    view._point_index_offset = int(getattr(node, "_point_index_offset", 0)) + int(start)
    # Prevent a subset from recursively entering the parent-first path if it
    # is passed to a public wrapper by a future caller.
    if hasattr(view, "parent_point_count"):
        view.parent_point_count = None
    metrics = getattr(view, "quality_metrics", None)
    if metrics is not None:
        view.quality_metrics = dict(metrics)
        view.quality_metrics.pop("parent_point_count", None)
    return view


def _optional_composite(parent_value, delta_value, allowed):
    if parent_value is None and delta_value is None:
        return None
    if parent_value is None:
        result = np.zeros_like(np.asarray(delta_value))
        result[allowed] = np.asarray(delta_value)[allowed]
        return result
    result = np.array(parent_value, copy=True)
    if delta_value is not None:
        result[allowed] = np.asarray(delta_value)[allowed]
    return result


def _compose_parent_first(parent, delta, node, cameras, near, far):
    """Composite stable parent pixels before the newest point delta.

    The delta is intentionally not submitted to the same z-buffer as the
    parent.  A strict parent-pixel exclusion prevents newly inferred
    points from replacing parent samples while still allowing them
    to fill genuinely unknown regions.
    """
    parent_visibility = np.asarray(parent.visibility, bool)
    delta_visibility = np.asarray(delta.visibility, bool)
    parent_protection = parent_visibility
    delta_allowed = delta_visibility & ~parent_protection
    delta_output_on_parent_visible = int(np.count_nonzero(delta_allowed & parent_visibility))
    delta_output_on_parent_protection_mask = int(np.count_nonzero(delta_allowed & parent_protection))
    if delta_output_on_parent_visible or delta_output_on_parent_protection_mask:
        raise RuntimeError("parent-first delta escaped the parent exclusion mask")

    def composite_field(name):
        parent_value = np.asarray(getattr(parent, name))
        delta_value = np.asarray(getattr(delta, name))
        result = np.array(parent_value, copy=True)
        result[delta_allowed] = delta_value[delta_allowed]
        return result

    fields = {
        "rgb": composite_field("rgb"),
        "depth": composite_field("depth"),
        "visibility": parent_visibility.copy(),
        "confidence": composite_field("confidence"),
        "source": composite_field("source"),
        # Coverage is defined over all committed points, not only visible
        # output pixels or the splat radius.
        "coverage_per_frame": _coverage_metric(node, cameras, near, far),
    }
    fields["visibility"] |= delta_allowed
    for name in _OPTIONAL_WARP_FIELDS:
        fields[name] = _optional_composite(
            getattr(parent, name, None), getattr(delta, name, None), delta_allowed,
        )
    result = WarpBatch(**fields)
    # Keep a small machine-readable diagnostic without changing the public
    # WarpBatch schema.  Consumers that do not inspect it remain compatible.
    result.parent_first = True
    result.parent_visibility = parent_visibility
    result.parent_protection_visibility = parent_protection
    result.delta_allowed_visibility = delta_allowed
    result.parent_point_count = int(_effective_parent_point_count(node))
    result.delta_output_on_parent_visible = delta_output_on_parent_visible
    result.delta_output_on_parent_protection_mask = delta_output_on_parent_protection_mask
    return result


def _render_numpy_single(node, cameras, near=.05, far=100., point_radius=0, depth_epsilon=1e-5):
    t, h, w = len(cameras.c2w), cameras.height, cameras.width
    rgb = np.zeros((t, h, w, 3), np.float32)
    depth = np.full((t, h, w), np.nan, np.float32)
    vis = np.zeros((t, h, w), bool)
    conf = np.zeros((t, h, w), np.float32)
    src = np.full((t, h, w), INVALID_SOURCE, np.int8)
    point_index = np.full((t, h, w), -1, np.int64)
    winning_xyz = np.zeros((t, h, w, 3), np.float32)
    offset = int(getattr(node, "_point_index_offset", 0))
    xyz = np.asarray(node.points_xyz, np.float32)
    ph = np.c_[xyz, np.ones(len(xyz), np.float32)]
    for ti in range(t):
        cam = (np.linalg.inv(cameras.c2w[ti]) @ ph.T).T[:, :3]
        z = cam[:, 2]
        k = cameras.intrinsics[ti]
        uv = cam @ k.T
        uv = uv[:, :2] / np.maximum(uv[:, 2:3], 1e-8)
        for dy in range(-point_radius, point_radius + 1):
            for dx in range(-point_radius, point_radius + 1):
                x = np.rint(uv[:, 0] + dx).astype(int)
                y = np.rint(uv[:, 1] + dy).astype(int)
                valid = (z > near) & (z < far) & (x >= 0) & (x < w) & (y >= 0) & (y < h)
                for j in np.flatnonzero(valid):
                    if not vis[ti, y[j], x[j]] or z[j] < depth[ti, y[j], x[j]] - depth_epsilon:
                        depth[ti, y[j], x[j]] = z[j]
                        rgb[ti, y[j], x[j]] = node.points_rgb[j] / 255.
                        conf[ti, y[j], x[j]] = node.points_confidence[j]
                        src[ti, y[j], x[j]] = node.points_source[j]
                        point_index[ti, y[j], x[j]] = offset + j
                        winning_xyz[ti, y[j], x[j]] = xyz[j]
                        vis[ti, y[j], x[j]] = True
    return WarpBatch(
        rgb, depth, vis, conf, src, _coverage_metric(node, cameras, near, far),
        point_index=point_index, winning_xyz_world=winning_xyz,
    )


def render_numpy_reference(node, cameras, near=.05, far=100., point_radius=0, depth_epsilon=1e-5):
    parent_count = _effective_parent_point_count(node)
    if parent_count is None:
        return _render_numpy_single(node, cameras, near, far, point_radius, depth_epsilon)
    parent = _render_numpy_single(
        _slice_node(node, 0, parent_count), cameras, near, far, point_radius, depth_epsilon,
    )
    delta = _render_numpy_single(
        _slice_node(node, parent_count, None), cameras, near, far, point_radius, depth_epsilon,
    )
    return _compose_parent_first(parent, delta, node, cameras, near, far)


def _render_gpu_single(
    node, cameras, near=.05, far=100., point_radius=0, depth_epsilon=1e-5,
    device="cuda:0", chunk_points=1_000_000,
):
    import torch

    dev = torch.device(str(device))
    t, h, w = len(cameras.c2w), cameras.height, cameras.width
    xyz = torch.as_tensor(node.points_xyz, dtype=torch.float32, device=dev)
    prgb = torch.as_tensor(node.points_rgb, dtype=torch.float32, device=dev) / 255.
    pconf = torch.as_tensor(node.points_confidence, dtype=torch.float32, device=dev)
    psrc = torch.as_tensor(node.points_source, dtype=torch.int8, device=dev)
    point_offset = int(getattr(node, "_point_index_offset", 0))
    poses = torch.as_tensor(cameras.c2w, dtype=torch.float32, device=dev)
    ks = torch.as_tensor(cameras.intrinsics, dtype=torch.float32, device=dev)
    outputs = []
    for ti in range(t):
        inv = torch.linalg.inv(poses[ti])
        zbuf = torch.full((h * w,), float("inf"), device=dev)
        best = torch.full((h * w,), len(xyz), dtype=torch.long, device=dev)
        for start in range(0, len(xyz), chunk_points):
            stop = min(start + chunk_points, len(xyz))
            cam = xyz[start:stop] @ inv[:3, :3].T + inv[:3, 3]
            z = cam[:, 2]
            uv = cam @ ks[ti].T
            uv = uv[:, :2] / z[:, None].clamp_min(1e-8)
            for dy in range(-point_radius, point_radius + 1):
                for dx in range(-point_radius, point_radius + 1):
                    x = torch.round(uv[:, 0] + dx).long()
                    y = torch.round(uv[:, 1] + dy).long()
                    ok = (z > near) & (z < far) & (x >= 0) & (x < w) & (y >= 0) & (y < h)
                    idx = torch.nonzero(ok).flatten()
                    if len(idx):
                        zbuf.scatter_reduce_(0, y[idx] * w + x[idx], z[idx], reduce="amin", include_self=True)
        for start in range(0, len(xyz), chunk_points):
            stop = min(start + chunk_points, len(xyz))
            cam = xyz[start:stop] @ inv[:3, :3].T + inv[:3, 3]
            z = cam[:, 2]
            uv = cam @ ks[ti].T
            uv = uv[:, :2] / z[:, None].clamp_min(1e-8)
            for dy in range(-point_radius, point_radius + 1):
                for dx in range(-point_radius, point_radius + 1):
                    x = torch.round(uv[:, 0] + dx).long()
                    y = torch.round(uv[:, 1] + dy).long()
                    ok = (z > near) & (z < far) & (x >= 0) & (x < w) & (y >= 0) & (y < h)
                    idx = torch.nonzero(ok).flatten()
                    if not len(idx):
                        continue
                    flat = y[idx] * w + x[idx]
                    selected = idx[z[idx] <= zbuf[flat] + depth_epsilon]
                    if len(selected):
                        flat_selected = y[selected] * w + x[selected]
                        best.scatter_reduce_(0, flat_selected, selected + start, reduce="amin", include_self=True)
        valid = (best >= 0) & (best < len(xyz))
        out_rgb = torch.zeros((h * w, 3), device=dev)
        out_cf = torch.zeros(h * w, device=dev)
        out_src = torch.full((h * w,), INVALID_SOURCE, dtype=torch.int8, device=dev)
        out_z = torch.full((h * w,), float("nan"), device=dev)
        chosen = best[valid]
        out_rgb[valid] = prgb[chosen]
        out_cf[valid] = pconf[chosen]
        out_src[valid] = psrc[chosen]
        out_z[valid] = zbuf[valid]
        out_index = torch.full((h * w,), -1, dtype=torch.long, device=dev)
        out_xyz = torch.zeros((h * w, 3), dtype=torch.float32, device=dev)
        out_index[valid] = chosen + point_offset
        out_xyz[valid] = xyz[chosen]
        outputs.append((
            out_rgb.reshape(h, w, 3), out_z.reshape(h, w), valid.reshape(h, w),
            out_cf.reshape(h, w), out_src.reshape(h, w), out_index.reshape(h, w),
            out_xyz.reshape(h, w, 3),
        ))
    rgb, depth, vis, conf, src, point_index, winning_xyz = zip(*outputs)
    rgb = torch.stack(rgb).cpu().numpy()
    depth = torch.stack(depth).cpu().numpy()
    vis = torch.stack(vis).cpu().numpy()
    conf = torch.stack(conf).cpu().numpy()
    src = torch.stack(src).cpu().numpy()
    point_index = torch.stack(point_index).cpu().numpy()
    winning_xyz = torch.stack(winning_xyz).cpu().numpy()
    return WarpBatch(
        rgb, depth, vis, conf, src, _coverage_metric(node, cameras, near, far),
        point_index=point_index, winning_xyz_world=winning_xyz,
    )


def render(
    node, cameras, near=.05, far=100., point_radius=0, depth_epsilon=1e-5,
    device="cpu", chunk_points=1_000_000,
):
    """Render a node, splitting cumulative parent points from its latest delta."""
    parent_count = _effective_parent_point_count(node)
    try:
        import torch  # noqa: F401
    except ImportError:
        if parent_count is None:
            return _render_numpy_single(node, cameras, near, far, point_radius, depth_epsilon)
        parent = _render_numpy_single(
            _slice_node(node, 0, parent_count), cameras, near, far, point_radius, depth_epsilon,
        )
        delta = _render_numpy_single(
            _slice_node(node, parent_count, None), cameras, near, far, point_radius, depth_epsilon,
        )
        return _compose_parent_first(parent, delta, node, cameras, near, far)
    if device is None:
        raise ValueError("renderer device must be explicitly configured")
    if str(device) == "cpu":
        return render_numpy_reference(node, cameras, near, far, point_radius, depth_epsilon)
    if parent_count is None:
        return _render_gpu_single(
            node, cameras, near, far, point_radius, depth_epsilon, device, chunk_points,
        )
    parent = _render_gpu_single(
        _slice_node(node, 0, parent_count), cameras, near, far,
        point_radius, depth_epsilon, device, chunk_points,
    )
    delta = _render_gpu_single(
        _slice_node(node, parent_count, None), cameras, near, far,
        point_radius, depth_epsilon, device, chunk_points,
    )
    return _compose_parent_first(parent, delta, node, cameras, near, far)
