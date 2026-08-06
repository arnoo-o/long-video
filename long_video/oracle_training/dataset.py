"""Construct leak-free Holo360D Oracle-M0 WAH sequences."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image

from ..data.camera import rgb_to_uint8
from ..data.erp_geometry import perspective_unit_rays, source_relative_c2w
from ..data.holo360d import Holo360DReader
from ..data.panorama_projection import equirectangular_to_perspective, intrinsics_from_fov
from ..geometry.point_renderer import render
from ..memory.node_store import NodeStore
from ..types import CameraBatch, WarpBatch
from .oracle_node import build_oracle_erp_node
from .temporal import ChunkContract, build_primary_loss_masks


def _nearest_resize(array: np.ndarray, height: int, width: int) -> np.ndarray:
    source_h, source_w = array.shape[:2]
    yy = np.clip(np.floor((np.arange(height) + 0.5) * source_h / height).astype(np.int64), 0, source_h - 1)
    xx = np.mod(np.floor((np.arange(width) + 0.5) * source_w / width).astype(np.int64), source_w)
    return np.asarray(array)[yy[:, None], xx[None, :]]


def _resize_erp(rgb, depth, mask, resolution):
    height, width = map(int, resolution)
    if width != 2 * height:
        raise ValueError(f"ERP resolution must be 2:1, got {(height, width)}")
    if tuple(rgb.shape[:2]) == (height, width):
        return rgb, depth, mask
    resized_rgb = np.asarray(Image.fromarray(rgb_to_uint8(rgb)).resize((width, height), Image.Resampling.BILINEAR))
    # Geometry uses nearest sampling; bilinear depth would create foreground/background surfaces.
    return resized_rgb, _nearest_resize(depth, height, width).astype(np.float32), _nearest_resize(mask, height, width).astype(bool)


def _perspective(frame, *, fov_degrees, height, width):
    rgb = equirectangular_to_perspective(
        frame.rgb, 0.0, 0.0, fov_degrees, height, width, interpolation="bilinear"
    )
    ray_distance = equirectangular_to_perspective(
        frame.depth, 0.0, 0.0, fov_degrees, height, width, interpolation="nearest"
    ).astype(np.float32)
    valid = equirectangular_to_perspective(
        frame.mask.astype(np.uint8), 0.0, 0.0, fov_degrees, height, width, interpolation="nearest"
    ).astype(bool)
    intrinsics = intrinsics_from_fov(fov_degrees, width, height)
    rays = perspective_unit_rays(intrinsics, height, width)
    z_depth = ray_distance * rays[..., 2]
    z_depth[~valid | ~np.isfinite(ray_distance) | (ray_distance <= 0)] = np.nan
    ray_distance[~valid | ~np.isfinite(ray_distance) | (ray_distance <= 0)] = np.nan
    return rgb_to_uint8(rgb), z_depth.astype(np.float32), ray_distance, valid, intrinsics


def _validate_trajectory(frames, assumed_fps):
    poses = np.stack([frame.raw_c2w for frame in frames]).astype(np.float32)
    if not np.isfinite(poses).all():
        raise ValueError("trajectory contains non-finite poses")
    rotations = poses[:, :3, :3]
    orthogonality = np.max(np.abs(rotations @ np.swapaxes(rotations, 1, 2) - np.eye(3)))
    determinants = np.linalg.det(rotations)
    if orthogonality > 1e-3 or np.max(np.abs(determinants - 1.0)) > 1e-3:
        raise ValueError("trajectory rotations are not proper orthonormal matrices")
    timestamps=np.asarray([float(frame.frame_id) for frame in frames],np.float64)
    time_steps=np.diff(timestamps)
    median_dt=float(np.median(time_steps))
    if median_dt<=0 or np.any(time_steps<=0):
        raise ValueError("frame IDs must be finite and strictly increasing")
    acquisition_gap_mask=time_steps>2.5*median_dt
    translations = np.linalg.norm(np.diff(poses[:, :3, 3], axis=0), axis=1)
    relative = np.swapaxes(rotations[:-1], 1, 2) @ rotations[1:]
    angles = np.arccos(np.clip((np.trace(relative, axis1=1, axis2=2) - 1) * 0.5, -1, 1))
    return {
        "assumed_fps": float(assumed_fps),
        "estimated_fps_from_frame_ids":float(1.0/median_dt),
        "frame_id_delta_min":float(time_steps.min()),
        "frame_id_delta_max":float(time_steps.max()),
        "acquisition_gap_count":int(acquisition_gap_mask.sum()),
        "constant_rate_continuous":bool(not acquisition_gap_mask.any()),
        "adjacent_translation_min": float(translations.min(initial=0.0)),
        "adjacent_translation_max": float(translations.max(initial=0.0)),
        "adjacent_rotation_degrees_max": float(np.rad2deg(angles).max(initial=0.0)),
        "rotation_orthogonality_max_error": float(orthogonality),
    }


def _write_png_frames(root: Path, frames: np.ndarray):
    root.mkdir(parents=True, exist_ok=True)
    for index, frame in enumerate(frames):
        Image.fromarray(rgb_to_uint8(frame)).save(root / f"{index:06d}.png")


def _depth_preview(depth):
    depth = np.asarray(depth, np.float32)
    valid = np.isfinite(depth) & (depth > 0)
    output = np.zeros(depth.shape, np.uint8)
    if valid.any():
        low, high = np.percentile(depth[valid], (2, 98))
        output[valid] = np.rint(np.clip((depth[valid] - low) / max(high - low, 1e-6), 0, 1) * 255).astype(np.uint8)
    return output


def attach_warp_provenance(warp: WarpBatch, node) -> WarpBatch:
    visible = np.asarray(warp.visibility, bool)
    source = np.asarray(warp.source)
    oracle = visible & (source == 0)
    generated = visible & np.isin(source, (2, 3))
    rgb_origin = np.full(source.shape, "", dtype="U24")
    depth_origin = np.full(source.shape, "", dtype="U24")
    rgb_role = np.full(source.shape, "", dtype="U24")
    depth_role = np.full(source.shape, "", dtype="U24")
    rgb_origin[oracle] = "oracle_source"
    depth_origin[oracle] = "oracle_source"
    rgb_origin[generated] = "model_generated"
    depth_origin[generated] = "pi3_prediction"
    if node.parent_id is None:
        rgb_role[visible] = "direct_source"
        depth_role[visible] = "direct_source"
    else:
        rgb_role[oracle] = "parent_warp"
        depth_role[oracle] = "parent_warp"
        rgb_role[generated] = "current_generation"
        depth_role[generated] = "geometry_prediction"
    warp.rgb_content_origin = rgb_origin
    warp.depth_content_origin = depth_origin
    warp.evidence_role = rgb_role
    warp.rgb_evidence_role = rgb_role
    warp.depth_evidence_role = depth_role
    return warp


def build_oracle_sequence(
    scene_root,
    output_root,
    *,
    sequence_id: str,
    split: str,
    source_index: int,
    frame_stride: int,
    num_chunks: int,
    contract: ChunkContract,
    erp_resolution=(1024, 2048),
    perspective_resolution=(384, 640),
    fov_degrees=90.0,
    pixel_center=0.5,
    prompt="an indoor scene",
    assumed_fps=16.0,
    voxel_size=0.01,
    renderer_kwargs=None,
    base_commit_sha="unknown",
    worktree_dirty=False,
    source_diff_sha256="clean",
):
    """Build one independent source window; future GT never enters M0 or warp."""
    reader = Holo360DReader(scene_root, normalize_first_pose=False)
    global_count = 1 + int(num_chunks) * (contract.window_num_frames - 1)
    indices = [int(source_index) + i * int(frame_stride) for i in range(global_count)]
    if indices[-1] >= len(reader.frame_ids):
        raise IndexError(f"sequence {sequence_id} needs frame {indices[-1]}, dataset has {len(reader.frame_ids)}")
    frames = [reader.read(index) for index in indices]
    trajectory_metrics = _validate_trajectory(frames, assumed_fps)
    source = frames[0]
    source_rgb, source_depth, source_mask = _resize_erp(
        source.rgb, source.depth, source.mask, erp_resolution
    )
    source_c2w_world = np.asarray(source.raw_c2w, np.float32)
    source_c2w_local = np.eye(4, dtype=np.float32)
    node = build_oracle_erp_node(
        source_rgb, source_depth, source_mask,
        source_c2w_local=source_c2w_local,
        voxel_size=voxel_size,
        pixel_center=pixel_center,
        model_versions={"geometry": "Holo360D_mesh_depth", "builder": "oracle_erp_v1"},
    )
    out = Path(output_root) / sequence_id
    out.mkdir(parents=True, exist_ok=True)
    NodeStore(out / "session").save(node)

    height, width = map(int, perspective_resolution)
    target_rgb, target_z, target_ray, target_mask, intrinsics = [], [], [], [], []
    target_c2w_world, target_c2w_local = [], []
    for frame in frames:
        rgb, z, ray, valid, k = _perspective(
            frame, fov_degrees=fov_degrees, height=height, width=width
        )
        target_rgb.append(rgb); target_z.append(z); target_ray.append(ray)
        target_mask.append(valid); intrinsics.append(k)
        target_c2w_world.append(frame.raw_c2w)
        target_c2w_local.append(source_relative_c2w(source_c2w_world, frame.raw_c2w))
    target_rgb = np.stack(target_rgb)
    target_z = np.stack(target_z).astype(np.float32)
    target_ray = np.stack(target_ray).astype(np.float32)
    target_mask = np.stack(target_mask)
    intrinsics = np.stack(intrinsics).astype(np.float32)
    target_c2w_world = np.stack(target_c2w_world).astype(np.float32)
    target_c2w_local = np.stack(target_c2w_local).astype(np.float32)
    np.testing.assert_array_equal(target_rgb[0], _perspective(source, fov_degrees=fov_degrees, height=height, width=width)[0])
    np.testing.assert_allclose(target_c2w_local[0], np.eye(4), atol=1e-5)

    # The training chunk alone may use a precomputed M0-only external warp.
    chunk_slice = slice(0, contract.window_num_frames)
    cameras = CameraBatch(target_c2w_local[chunk_slice], intrinsics[chunk_slice], height, width)
    render_options = {"near": 0.05, "far": 100.0, "point_radius": 1, "device": "cpu", **dict(renderer_kwargs or {})}
    single_warp = attach_warp_provenance(render(node, cameras, **render_options), node)
    if len(single_warp.rgb) != contract.window_num_frames:
        raise RuntimeError("precomputed warp length differs from WAH chunk contract")

    source_dir = out / "source"; target_dir = out / "target"; warp_dir = out / "single_chunk_warp"
    source_dir.mkdir(exist_ok=True); target_dir.mkdir(exist_ok=True); warp_dir.mkdir(exist_ok=True)
    Image.fromarray(source_rgb).save(source_dir / "source_erp_rgb.png")
    np.save(source_dir / "source_erp_depth_ray_distance.npy", source_depth)
    Image.fromarray(source_mask.astype(np.uint8) * 255).save(source_dir / "source_erp_mask.png")
    np.save(source_dir / "source_c2w_world.npy", source_c2w_world)
    Image.fromarray(target_rgb[0]).save(source_dir / "source_perspective.png")
    _write_png_frames(target_dir / "target_rgb_for_loss", target_rgb)
    np.save(target_dir / "target_z_depth_for_eval.npy", target_z)
    np.save(target_dir / "target_ray_distance_for_reference.npy", target_ray)
    np.save(target_dir / "target_valid_mask.npy", target_mask)
    np.save(target_dir / "target_c2w_world.npy", target_c2w_world)
    np.save(target_dir / "target_c2w_local.npy", target_c2w_local)
    np.save(target_dir / "intrinsics.npy", intrinsics)
    _write_png_frames(warp_dir / "warp_rgb", single_warp.rgb)
    np.save(warp_dir / "warp_z_depth.npy", single_warp.depth)
    np.save(warp_dir / "warp_visibility.npy", single_warp.visibility)
    np.save(warp_dir / "warp_confidence.npy", single_warp.confidence)
    np.save(warp_dir / "rgb_content_origin.npy", single_warp.rgb_content_origin)
    np.save(warp_dir / "depth_content_origin.npy", single_warp.depth_content_origin)
    np.save(warp_dir / "evidence_role.npy", single_warp.evidence_role)
    np.save(warp_dir / "rgb_evidence_role.npy", single_warp.rgb_evidence_role)
    np.save(warp_dir / "depth_evidence_role.npy", single_warp.depth_evidence_role)

    valid_frames = target_mask[: contract.window_num_frames].reshape(contract.window_num_frames, -1).any(1)
    primary_rgb, primary_latent = build_primary_loss_masks(contract, valid_target_frames=valid_frames)
    np.save(out / "primary_loss_mask_rgb.npy", primary_rgb)
    np.save(out / "primary_loss_mask_latent.npy", primary_latent)
    (out / "prompt.txt").write_text(str(prompt), encoding="utf-8")
    metadata = {
        "sequence_id": sequence_id, "scene_id": Path(scene_root).name, "dataset_split": split,
        "source_frame_id": source.frame_id, "target_frame_ids": [frame.frame_id for frame in frames],
        "frame_stride": int(frame_stride), "assumed_fps": float(assumed_fps),

        "erp_resolution": list(map(int, erp_resolution)),
        "perspective_resolution": [height, width], "fov_degrees": float(fov_degrees),
        "pixel_center": float(pixel_center), "source_depth_convention": "RAY_DISTANCE",
        "target_evaluation_depth_convention": "Z_DEPTH", "renderer_depth_convention": "Z_DEPTH",
        "coordinate_convention": "OpenCV c2w: +x right, +y down, +z forward",
        "geometry_source_frame_ids": [source.frame_id], "future_geometry_used": False,
        "base_commit_sha": base_commit_sha, "worktree_dirty": bool(worktree_dirty),
        "num_chunks": int(num_chunks),
        "source_diff_sha256": source_diff_sha256, "chunk_frames": contract.window_num_frames,
        "chunk_stride": contract.window_num_frames - 1,
        "shared_boundary_rule": "chunk k>0 reuses previous global boundary as warp frame 0; decoded duplicate is dropped",
        "source_prefix_length_rgb": contract.source_prefix_length_rgb,
        "valid_target_frames": int(valid_frames.sum()),
        "valid_loss_latent_frames": int(primary_latent.sum()),
        "actual_loss_latent_count": int(primary_latent.sum()),
        "scale_mode": "dataset_calibrated", "meters_per_world_unit": 1.0,
        "trajectory": trajectory_metrics,
    }
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    debug = out / "debug"; debug.mkdir(exist_ok=True)
    for name, image in {
        "source_erp.png": source_rgb, "source_perspective.png": target_rgb[0],
        "target_rgb.png": target_rgb[min(1, len(target_rgb)-1)],
        "warp_rgb.png": rgb_to_uint8(single_warp.rgb[min(1, len(single_warp.rgb)-1)]),
        "warp_visibility.png": single_warp.visibility[min(1, len(single_warp.rgb)-1)].astype(np.uint8) * 255,
        "warp_confidence.png": np.rint(single_warp.confidence[min(1, len(single_warp.rgb)-1)] * 255).astype(np.uint8),
        "warp_z_depth.png": _depth_preview(single_warp.depth[min(1, len(single_warp.rgb)-1)]),
        "target_z_depth.png": _depth_preview(target_z[min(1, len(target_z)-1)]),
    }.items():
        Image.fromarray(image).save(debug / name)
    overlay_index = min(1, len(target_rgb) - 1)
    rgb_overlay = np.rint(0.5 * target_rgb[overlay_index] + 0.5 * rgb_to_uint8(single_warp.rgb[overlay_index])).astype(np.uint8)
    depth_overlay = np.stack((_depth_preview(target_z[overlay_index]), _depth_preview(single_warp.depth[overlay_index]), np.zeros((height, width), np.uint8)), -1)
    Image.fromarray(rgb_overlay).save(debug / "source_to_target_rgb_overlay.png")
    Image.fromarray(depth_overlay).save(debug / "source_to_target_depth_overlay.png")
    return out, metadata
