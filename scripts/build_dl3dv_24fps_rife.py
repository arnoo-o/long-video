#!/usr/bin/env python3
"""Build timestamp-aligned DL3DV RGB/camera trajectories with Practical-RIFE."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from long_video.data.dl3dv import (TARGET_FPS, TARGET_HW, build_interpolated_timeline,
    center_crop_resize_geometry, interpolate_timeline_c2w, load_dl3dv_scene)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--scene-root", type=Path, required=True)
    p.add_argument("--duration", type=float, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--start-real-index", type=int, required=True)
    p.add_argument("--frame-count", type=int, default=33)
    p.add_argument("--rife-root", type=Path, required=True)
    p.add_argument("--rife-checkpoint", type=Path, required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--jpeg-quality", type=int, default=95)
    return p.parse_args()

def main():
    args = parse_args()
    import torch
    from PIL import Image
    scene = load_dl3dv_scene(args.scene_root, duration=args.duration)
    timeline = build_interpolated_timeline(scene.frame_times, args.start_real_index,
                                           args.frame_count, TARGET_FPS)
    crop, resized_k = center_crop_resize_geometry(scene.source_hw, scene.intrinsics)
    dense_c2w = interpolate_timeline_c2w(scene.c2w_opencv, timeline)
    args.output.mkdir(parents=True, exist_ok=True)
    rgb_dir = args.output / "rgb_24fps"; rgb_dir.mkdir(exist_ok=True)
    source_dir = args.output / "source"; source_dir.mkdir(exist_ok=True)
    pi3_dir = args.output / "pi3_initial_real"; pi3_dir.mkdir(exist_ok=True)
    sys.path.insert(0, str(args.rife_root.resolve()))
    sys.path.insert(0, str(args.rife_checkpoint.resolve()))
    from train_log.RIFE_HDv3 import Model
    model_dir = args.rife_checkpoint.resolve()
    if not (model_dir / "flownet.pkl").is_file():
        model_dir = model_dir / "train_log"
    model = Model(); model.load_model(str(model_dir), -1)
    model.eval(); model.device()
    cache = {}
    def image(real_index):
        key = int(real_index)
        if key not in cache:
            with Image.open(scene.image_paths[key]) as im:
                cache[key] = np.asarray(im.convert("RGB").crop(crop).resize(
                    (TARGET_HW[1], TARGET_HW[0]), Image.Resampling.LANCZOS), np.uint8)
        return cache[key]
    def tensor(value):
        return torch.from_numpy(value.astype(np.float32) / 255).permute(2,0,1).unsqueeze(0).to(args.device)
    sources = []
    with torch.inference_mode():
        rows = zip(timeline["left_real_indices"], timeline["right_real_indices"],
                   timeline["alpha"], timeline["is_real"], timeline["rgb_real_indices"])
        for j, (left, right, alpha, real, real_index) in enumerate(rows):
            if real:
                frame = image(real_index); kind = "real"
            else:
                pred = model.inference(tensor(image(left)), tensor(image(right)), timestep=float(alpha))
                frame = np.rint(pred[0, :, :TARGET_HW[0], :TARGET_HW[1]].clamp(0,1)
                                 .permute(1,2,0).float().cpu().numpy() * 255).astype(np.uint8)
                kind = "rife"
            Image.fromarray(frame).save(rgb_dir / f"{j:06d}.jpg", quality=args.jpeg_quality, subsampling=0)
            sources.append({"output_index": j, "timestamp": float(timeline["timestamps"][j]),
                "source": kind, "left_real_index": int(left), "right_real_index": int(right),
                "alpha": float(alpha),
                "rgb_real_index": int(real_index) if real else None})
    Image.fromarray(image(timeline["left_real_indices"][0])).save(source_dir / "source.png")
    source_real = int(timeline["left_real_indices"][0])
    pi3_indices = np.arange(max(0, source_real - 7), source_real + 1, dtype=np.int64)
    for slot, real_index in enumerate(pi3_indices):
        Image.fromarray(image(real_index)).save(pi3_dir / f"{slot:02d}.jpg",
                                                quality=args.jpeg_quality, subsampling=0)
    np.save(args.output / "pi3_initial_real_frame_indices.npy", pi3_indices)
    source_world = scene.c2w_opencv[source_real]
    pi3_local = np.linalg.inv(source_world) @ scene.c2w_opencv[pi3_indices]
    pi3_local[:, 3] = np.array([0, 0, 0, 1], np.float32)
    np.save(args.output / "pi3_initial_c2w_local.npy", pi3_local.astype(np.float32))
    np.save(args.output / "pi3_initial_intrinsics.npy", resized_k[pi3_indices].astype(np.float32))
    intrinsics = np.stack([(1-a)*resized_k[l] + a*resized_k[r] for l,r,a in zip(
        timeline["left_real_indices"], timeline["right_real_indices"], timeline["alpha"])]).astype(np.float32)
    fx, fy, cx, cy = (intrinsics[:, 0, 0], intrinsics[:, 1, 1],
                      intrinsics[:, 0, 2], intrinsics[:, 1, 2])
    intrinsics_valid = bool(np.isfinite(intrinsics).all() and (fx > 1).all() and
        (fy > 1).all() and (fx < 10 * TARGET_HW[1]).all() and
        (fy < 10 * TARGET_HW[0]).all() and (cx > 0).all() and
        (cx < TARGET_HW[1]).all() and (cy > 0).all() and (cy < TARGET_HW[0]).all())
    if not intrinsics_valid:
        raise ValueError("crop/resize intrinsics are invalid for 384x640 renderer input")
    np.save(args.output / "target_c2w_local.npy", dense_c2w)
    np.save(args.output / "intrinsics.npy", intrinsics)
    np.save(args.output / "timestamps.npy", timeline["timestamps"])
    np.save(args.output / "source_frame_indices.npy", np.stack([timeline["left_real_indices"],
        timeline["right_real_indices"]], axis=1))
    np.save(args.output / "real_keyframe_indices.npy", np.unique(np.concatenate([
        timeline["left_real_indices"], timeline["right_real_indices"]])).astype(np.int64))
    (args.output / "frame_sources.json").write_text(json.dumps(sources, indent=2))
    validation = {"valid": True, "fps": TARGET_FPS, "frame_count": len(sources),
        "resolution": list(TARGET_HW),
        "source_local_identity_max_error": float(np.max(np.abs(dense_c2w[0]-np.eye(4)))),
        "timestamp_step_max_error": float(np.max(np.abs(np.diff(timeline["timestamps"])-1/TARGET_FPS))),
        "real_count": int(timeline["is_real"].sum()),
        "rife_count": int((~timeline["is_real"]).sum()),
        "rgb_pose_timestamp_aligned": True, "extrapolation": False,
        "rife_arbitrary_timestep": True,
        "intrinsics_valid_for_384x640": intrinsics_valid,
        "intrinsics_range": {"fx": [float(fx.min()), float(fx.max())],
            "fy": [float(fy.min()), float(fy.max())],
            "cx": [float(cx.min()), float(cx.max())],
            "cy": [float(cy.min()), float(cy.max())]},
        "pi3_initial_views_are_real": True, "pi3_initial_uses_future_gt": False}
    (args.output / "validation.json").write_text(json.dumps(validation, indent=2))
    print(json.dumps(validation, indent=2))
if __name__ == "__main__": main()
