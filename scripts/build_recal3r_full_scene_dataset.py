#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
import sys
import time
import traceback
from pathlib import Path

import cv2
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from long_video.data.recal3r_full_scene import (
    apply_sim3_c2w,
    apply_sim3_points,
    calibrate_recal3r_confidence,
    estimate_camera_sim3,
    list_rgb_frames,
    official_resize_crop,
    remap_model_map,
    replace_directory,
    resolve_record_paths,
    validate_alignment,
    validate_c2w,
    write_json,
)
from long_video.geometry.voxel_fusion import fuse_voxels
from long_video.training.stage2_cleanup import select_balanced_training_records


def parse_args():
    parser = argparse.ArgumentParser(description="Build aligned full-scene ReCal3R geometry for the fixed WPF training subset.")
    parser.add_argument("--recal3r-repo", type=Path, required=True)
    parser.add_argument("--recal3r-checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--record-count", type=int, default=100)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--recal3r-reset-interval", type=int, default=64)
    parser.add_argument("--confidence-threshold", type=float, default=1.5)
    parser.add_argument("--voxel-size", type=float, default=0.02)
    parser.add_argument("--max-camera-alignment-error-ratio", type=float, default=0.15)
    parser.add_argument("--max-median-rotation-error-degrees", type=float, default=45.0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--min-free-gb", type=float, default=8.0)
    parser.add_argument("--fail-fast", action="store_true")
    args = parser.parse_args()
    if args.shard_count <= 0 or not 0 <= args.shard_index < args.shard_count:
        parser.error("--shard-index must satisfy 0 <= index < --shard-count")
    return args


def prepare_views(frame_paths, load_images, reset_interval):
    images = load_images([str(p) for p in frame_paths], size=512, verbose=False)
    views = []
    for index, image in enumerate(images):
        tensor = image["img"]
        view = {
            "img": tensor,
            "ray_map": torch.full((tensor.shape[0], 6, tensor.shape[-2], tensor.shape[-1]), torch.nan),
            "true_shape": torch.from_numpy(image["true_shape"]),
            "idx": index,
            "instance": str(index),
            "camera_pose": torch.eye(4, dtype=torch.float32).unsqueeze(0),
            "img_mask": torch.tensor([True]),
            "ray_mask": torch.tensor([False]),
            "update": torch.tensor([True]),
            "reset": torch.tensor([(index + 1) % reset_interval == 0]),
        }
        views.append(view)
        if (index + 1) % reset_interval == 0:
            overlap = {key: value.clone() if torch.is_tensor(value) else value for key, value in view.items()}
            overlap["reset"] = torch.tensor([False])
            views.append(overlap)
    return views


def load_official_model(args):
    repo = str(args.recal3r_repo.resolve())
    for import_root in (repo, str(args.recal3r_repo.resolve() / "src")):
        if import_root not in sys.path:
            sys.path.insert(0, import_root)
    from src.dust3r.inference import inference_recurrent_lighter
    from src.dust3r.model import ARCroco3DStereo
    from src.dust3r.utils.camera import pose_encoding_to_camera
    from src.dust3r.utils.image import load_images

    model = ARCroco3DStereo.from_pretrained(str(args.recal3r_checkpoint)).to(args.device)
    model.eval()
    model.config.model_update_type = "recal3r"
    model.beta_base = 0.1
    model.config.beta_base = 0.1
    return model, inference_recurrent_lighter, pose_encoding_to_camera, load_images


def infer_full_trajectory(frame_paths, model_bundle, device, reset_interval):
    model, inference_recurrent_lighter, pose_encoding_to_camera, load_images = model_bundle
    views = prepare_views(frame_paths, load_images, reset_interval)
    outputs, _ = inference_recurrent_lighter(views, model, device, verbose=True)
    predictions = outputs["pred"]
    output_views = outputs["views"]
    reset_mask = np.array([bool(view["reset"].detach().cpu().item()) for view in output_views])
    shifted_reset = np.concatenate(([False], reset_mask[:-1]))
    predictions = [prediction for prediction, remove in zip(predictions, shifted_reset) if not remove]
    output_views = [view for view, remove in zip(output_views, shifted_reset) if not remove]
    reset_mask = reset_mask[~shifted_reset]
    if len(predictions) != len(frame_paths):
        raise RuntimeError(
            f"ReCal3R reset stitching returned {len(predictions)} predictions "
            f"for {len(frame_paths)} frames")
    local_poses = []
    for prediction in predictions:
        pose = pose_encoding_to_camera(prediction["camera_pose"].clone())
        local_poses.append(pose.detach().cpu().numpy()[0])
    local_poses = np.stack(local_poses).astype(np.float64)
    reset_poses = np.repeat(np.eye(4)[None], len(local_poses), axis=0)
    reset_poses[reset_mask] = local_poses[reset_mask]
    cumulative = np.empty_like(reset_poses)
    cumulative[0] = reset_poses[0]
    for index in range(1, len(cumulative)):
        cumulative[index] = cumulative[index - 1] @ reset_poses[index]
    bases = np.concatenate((np.eye(4)[None], cumulative[:-1]), axis=0)
    poses = bases @ local_poses
    return predictions, poses.astype(np.float32)


def free_gb(path):
    return shutil.disk_usage(path).free / (1024 ** 3)


def write_invalid(destination, record, error, started_at):
    temporary = destination.parent / f".{destination.name}.tmp-{os.getpid()}"
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    write_json(temporary / "metadata.json", {
        "trajectory_id": record["trajectory_id"],
        "valid": False,
        "error": str(error),
        "traceback": traceback.format_exc(),
        "elapsed_seconds": time.time() - started_at,
    })
    replace_directory(temporary, destination)


def process_record(record, args, model_bundle):
    started_at = time.time()
    trajectory_id = record["trajectory_id"]
    destination = args.output_root / trajectory_id
    if args.resume and (destination / "metadata.json").exists():
        metadata = json.loads((destination / "metadata.json").read_text())
        if (metadata.get("valid") and
                metadata.get("geometry_implementation_version") == "recal-full-teacher-world-v4-rgb-anchor"):
            print(f"[skip] {trajectory_id}")
            return metadata

    temporary = destination.parent / f".{destination.name}.tmp-{os.getpid()}"
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    try:
        paths = resolve_record_paths(record, args.dataset_root)
        frames = list_rgb_frames(paths["rgb_dir"], expected=193)
        target_c2w = validate_c2w(np.load(paths["target_c2w_local"]), 193)
        intrinsics = np.load(paths["intrinsics"])
        timestamps = np.load(paths["timestamps"])
        if len(intrinsics) not in (1, 193) or len(timestamps) != 193:
            raise ValueError("intrinsics/timestamps do not match the 193-frame trajectory")

        first_rgb = cv2.imread(str(frames[0]), cv2.IMREAD_COLOR)
        if first_rgb is None:
            raise ValueError(f"cannot read {frames[0]}")
        height, width = first_rgb.shape[:2]
        transform = official_resize_crop(height, width, 512)

        predictions, recal_c2w = infer_full_trajectory(
            frames, model_bundle, args.device, args.recal3r_reset_interval)
        alignment = estimate_camera_sim3(recal_c2w, target_c2w)
        validate_alignment(alignment, args.max_camera_alignment_error_ratio,
                           args.max_median_rotation_error_degrees)
        aligned_c2w = apply_sim3_c2w(recal_c2w, alignment)

        xyz_maps = np.lib.format.open_memmap(
            temporary / "xyz_world.npy", mode="w+", dtype=np.float32,
            shape=(193, height, width, 3))
        valid_maps = np.lib.format.open_memmap(
            temporary / "valid.npy", mode="w+", dtype=np.bool_,
            shape=(193, height, width))
        confidence_maps = np.lib.format.open_memmap(
            temporary / "confidence.npy", mode="w+", dtype=np.float32,
            shape=(193, height, width))
        observations = []
        valid_pixels = 0

        for frame_index, (frame_path, prediction) in enumerate(zip(frames, predictions)):
            point_self = prediction["pts3d_in_self_view"].detach().cpu().numpy()[0]
            confidence = prediction["conf_self"].detach().cpu().numpy()[0]
            pose = recal_c2w[frame_index]
            point_recal_world = point_self @ pose[:3, :3].T + pose[:3, 3]
            point_target_world = apply_sim3_points(point_recal_world, alignment).astype(np.float32)
            xyz, inside = remap_model_map(point_target_world, transform, cv2.INTER_LINEAR)
            raw_conf, _ = remap_model_map(confidence.astype(np.float32), transform, cv2.INTER_LINEAR)
            conf = calibrate_recal3r_confidence(raw_conf, args.confidence_threshold)
            finite = np.isfinite(xyz).all(axis=-1) & np.isfinite(conf)
            valid = inside & finite & (raw_conf >= args.confidence_threshold)
            xyz[~valid] = 0
            conf = np.where(np.isfinite(conf), conf, 0).astype(np.float32)
            xyz_maps[frame_index] = xyz
            valid_maps[frame_index] = valid
            confidence_maps[frame_index] = conf
            valid_pixels += int(valid.sum())

            rgb_bgr = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
            if rgb_bgr is None:
                raise ValueError(f"cannot read {frame_path}")
            rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
            observations.append((xyz[valid].copy(), rgb[valid].copy(), conf[valid].copy()))
            del prediction
            if (frame_index + 1) % 16 == 0:
                print(f"  maps {frame_index + 1}/193")

        xyz_maps.flush(); valid_maps.flush(); confidence_maps.flush()
        if observations:
            all_xyz, all_rgb, all_conf = map(np.concatenate, zip(*observations))
            points_xyz, points_rgb, points_confidence, observation_count, _, anchors = fuse_voxels(
                all_xyz, all_rgb, all_conf, voxel_size=args.voxel_size,
                anchor_frame=np.repeat(np.arange(193, dtype=np.int32), [len(x) for x, _, _ in observations]),
                return_anchors=True)
            scene = {"points_xyz": points_xyz, "points_rgb": points_rgb,
                     "points_confidence": points_confidence, "observation_count": observation_count, **anchors}
        else:
            scene = {"points_xyz": np.empty((0,3),np.float32), "points_rgb": np.empty((0,3),np.uint8),
                     "points_confidence": np.empty(0,np.float32), "observation_count": np.empty(0,np.uint16)}
        np.savez_compressed(temporary / "scene_points.npz", **scene)
        np.save(temporary / "recal3r_c2w_aligned.npy", aligned_c2w)
        np.save(temporary / "target_c2w_local.npy", target_c2w.astype(np.float32))
        np.save(temporary / "intrinsics.npy", intrinsics)
        np.save(temporary / "timestamps.npy", timestamps)
        np.savez(temporary / "alignment.npz", scale=np.float64(alignment.scale),
                 rotation=alignment.rotation, translation=alignment.translation,
                 camera_alignment_error=np.float64(alignment.camera_alignment_error))
        metadata = {
            "schema_version": 3,
            "geometry_implementation_version": "recal-full-teacher-world-v4-rgb-anchor",
            "alignment_version": "offline-full-trajectory-recal-to-dataset-v3",
            "confidence_calibration": {"kind": "sigmoid", "threshold": args.confidence_threshold, "temperature": 0.35},
            "trajectory_id": trajectory_id,
            "rgb_dir": record["rgb_dir"],
            "scene_hash": record.get("scene_hash"),
            "split": record.get("split"),
            "environment": record.get("environment"),
            "valid": True,
            "frame_count": 193,
            "height": height,
            "width": width,
            "confidence_threshold": args.confidence_threshold,
            "voxel_size": args.voxel_size,
            "valid_pixel_ratio": valid_pixels / (193 * height * width),
            "scene_point_count": int(len(scene["points_xyz"])),
            "recal3r_repo_commit": os.popen(f"git -C {args.recal3r_repo} rev-parse HEAD").read().strip(),
            "recal3r_checkpoint": str(args.recal3r_checkpoint),
            "alignment": alignment.as_dict(),
            "official_input_transform": transform,
            "elapsed_seconds": time.time() - started_at,
        }
        write_json(temporary / "metadata.json", metadata)
        replace_directory(temporary, destination)
        print(f"[valid] {trajectory_id}: {metadata['scene_point_count']} points, "
              f"alignment={alignment.camera_alignment_error_ratio:.4f}")
        return metadata
    except Exception as error:
        if temporary.exists():
            shutil.rmtree(temporary)
        if args.fail_fast:
            raise
        write_invalid(destination, record, error, started_at)
        print(f"[invalid] {trajectory_id}: {error}")
        return {"trajectory_id": trajectory_id, "valid": False, "error": str(error)}
    finally:
        try:
            torch.cuda.empty_cache()
        except RuntimeError:
            pass


def main():
    args = parse_args()
    args.dataset_root = args.dataset_root.resolve()
    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = args.dataset_root / "dl3dv_24fps_manifest.json"
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    selected = select_balanced_training_records(manifest["records"], count=args.record_count)
    selection = {
        "selector": "long_video.training.stage2_cleanup.select_balanced_training_records",
        "record_count": args.record_count,
        "source_manifest": str(manifest_path),
        "source_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "trajectory_ids": [record["trajectory_id"] for record in selected],
        "environment_counts": {
            name: sum(record.get("environment") == name for record in selected)
            for name in ("indoor", "outdoor")
        },
    }
    write_json(args.output_root / "selection_manifest.json", selection)
    records = selected[args.shard_index::args.shard_count]
    if args.limit:
        records = records[:args.limit]
    summary_name = (
        "build_summary.json"
        if args.shard_count == 1
        else f"build_summary_shard_{args.shard_index:02d}.json"
    )

    random.seed(0); np.random.seed(0); torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    model_bundle = load_official_model(args)
    results = []
    for index, record in enumerate(records):
        if free_gb(args.output_root) < args.min_free_gb:
            raise RuntimeError(f"free disk below {args.min_free_gb} GiB; refusing to start another trajectory")
        print(f"[{index + 1}/{len(records)}] {record['trajectory_id']} free={free_gb(args.output_root):.1f} GiB")
        results.append(process_record(record, args, model_bundle))
        write_json(args.output_root / summary_name, {
            "requested": len(records), "completed": len(results),
            "valid": sum(bool(item.get("valid")) for item in results),
            "invalid": sum(not bool(item.get("valid")) for item in results),
            "results": results,
        })


if __name__ == "__main__":
    main()
