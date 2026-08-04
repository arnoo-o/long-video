#!/usr/bin/env python3
"""Run official MVDiffusion and restore all observed pixels afterward."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image


def intrinsics(fov_degrees, width, height):
    focal = 0.5 * width / np.tan(np.deg2rad(fov_degrees) * 0.5)
    return np.array(
        [[focal, 0, (width - 1) * 0.5], [0, focal, (height - 1) * 0.5], [0, 0, 1]],
        np.float32,
    )


def rotation(yaw_degrees, pitch_degrees=0):
    yaw = np.deg2rad(yaw_degrees)
    pitch = np.deg2rad(pitch_degrees)
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    yaw_matrix = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], np.float32)
    pitch_matrix = np.array([[1, 0, 0], [0, cp, sp], [0, -sp, cp]], np.float32)
    return yaw_matrix @ pitch_matrix


def target_rays(yaw, pitch, fov, height, width):
    k = intrinsics(fov, width, height)
    yy, xx = np.indices((height, width), np.float32)
    local = np.stack(
        ((xx - k[0, 2]) / k[0, 0], (yy - k[1, 2]) / k[1, 1], np.ones_like(xx)),
        axis=-1,
    )
    local /= np.linalg.norm(local, axis=-1, keepdims=True)
    return local @ rotation(yaw, pitch).T


def project_observation(image, source_spec, target_spec, height, width):
    image = np.asarray(image)
    source_height, source_width = image.shape[:2]
    world = target_rays(
        target_spec["yaw_degrees"], target_spec["pitch_degrees"],
        target_spec["fov_degrees"], height, width,
    )
    source_local = world @ rotation(
        source_spec["yaw_degrees"], source_spec.get("pitch_degrees", 0)
    )
    source_k = intrinsics(source_spec["fov_degrees"], source_width, source_height)
    z = source_local[..., 2]
    map_x = source_k[0, 0] * source_local[..., 0] / np.maximum(z, 1e-6) + source_k[0, 2]
    map_y = source_k[1, 1] * source_local[..., 1] / np.maximum(z, 1e-6) + source_k[1, 2]
    valid = (
        (z > 1e-5) & (map_x >= 0) & (map_x <= source_width - 1)
        & (map_y >= 0) & (map_y <= source_height - 1)
    )
    sampled = cv2.remap(
        image, map_x.astype(np.float32), map_y.astype(np.float32),
        cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
    )
    center_weight = np.clip(z / np.linalg.norm(source_local, axis=-1), 0, 1) ** 4
    return sampled, valid, center_weight.astype(np.float32)


def project_all_observations(observed, targets, height, width):
    rgb_sum = np.zeros((len(targets), height, width, 3), np.float32)
    weight_sum = np.zeros((len(targets), height, width), np.float32)
    square_sum = np.zeros_like(rgb_sum)
    source_coverage = np.zeros(len(observed), np.float64)
    for source_index, item in enumerate(observed):
        image = np.asarray(Image.open(item["image_path"]).convert("RGB"), np.float32) / 255
        for target_index, target in enumerate(targets):
            projected, valid, weight = project_observation(image, item, target, height, width)
            effective = valid.astype(np.float32) * weight
            rgb_sum[target_index] += projected * effective[..., None]
            square_sum[target_index] += projected ** 2 * effective[..., None]
            weight_sum[target_index] += effective
            source_coverage[source_index] += valid.sum()
    valid = weight_sum > 1e-6
    rgb = rgb_sum / np.maximum(weight_sum[..., None], 1e-6)
    variance = square_sum / np.maximum(weight_sum[..., None], 1e-6) - rgb ** 2
    conflict = np.sqrt(np.maximum(variance, 0).mean(axis=-1)) > 0.12
    return rgb, valid, conflict, source_coverage


def build_preview(views, target_yaws, target_pitch, target_fov):
    height, width = 512, 1024
    yy, xx = np.indices((height, width), np.float32)
    longitude = ((xx + 0.5) / width - 0.5) * 360
    latitude = (0.5 - (yy + 0.5) / height) * 180
    world = np.stack(
        (
            np.cos(np.deg2rad(latitude)) * np.sin(np.deg2rad(longitude)),
            -np.sin(np.deg2rad(latitude)),
            np.cos(np.deg2rad(latitude)) * np.cos(np.deg2rad(longitude)),
        ),
        axis=-1,
    )
    best_z = np.full((height, width), -np.inf, np.float32)
    preview = np.zeros((height, width, 3), np.uint8)
    for image, yaw in zip(views, target_yaws):
        local = world @ rotation(yaw, target_pitch)
        z = local[..., 2]
        k = intrinsics(target_fov, image.shape[1], image.shape[0])
        map_x = k[0, 0] * local[..., 0] / np.maximum(z, 1e-6) + k[0, 2]
        map_y = k[1, 1] * local[..., 1] / np.maximum(z, 1e-6) + k[1, 2]
        valid = (
            (z > best_z) & (z > 0) & (map_x >= 0) & (map_x <= image.shape[1] - 1)
            & (map_y >= 0) & (map_y <= image.shape[0] - 1)
        )
        sampled = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR)
        preview[valid] = sampled[valid]
        best_z[valid] = z[valid]
    return preview


def run_model(manifest, main_index):
    import torch
    import yaml

    repo = Path(manifest["mvdiffusion_repo"])
    sys.path.insert(0, str(repo))
    from src.lightning_pano_outpaint import PanoOutpaintGenerator

    base_model = manifest.get("base_model_path")
    if not base_model or not Path(base_model).exists():
        raise FileNotFoundError(
            "Stable Diffusion 2 inpainting base model is missing. Download "
            "stabilityai/stable-diffusion-2-inpainting after accepting its Hugging Face license "
            f"to: {base_model}"
        )
    config = yaml.safe_load((repo / "configs" / "pano_generation_outpaint.yaml").read_text())
    config["model"]["model_id"] = str(Path(base_model).resolve())
    model = PanoOutpaintGenerator(config)
    checkpoint = torch.load(manifest["checkpoint"], map_location="cpu")
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model = model.cuda().eval()

    source = np.asarray(Image.open(manifest["observed_views"][main_index]["image_path"]).convert("RGB"))
    source = cv2.resize(source, (512, 512), interpolation=cv2.INTER_AREA)
    source = torch.from_numpy(source.astype(np.float32) / 127.5 - 1).cuda()
    images = torch.zeros((1, 8, 512, 512, 3), device="cuda")
    images[0, 0] = source
    k_values = []
    r_values = []
    for index in range(8):
        k_values.append(intrinsics(90, 512, 512))
        r_values.append(rotation(index * 45, 0))
    prompts = [manifest["prompt"]] * 8
    batch = {
        "images": images,
        "prompt": prompts,
        "R": torch.from_numpy(np.stack(r_values))[None].cuda(),
        "K": torch.from_numpy(np.stack(k_values))[None].cuda(),
    }
    return model.inference(batch)[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    output = Path(manifest["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    target_yaws = manifest["target_yaws_degrees"]
    targets = [
        {
            "yaw_degrees": yaw,
            "pitch_degrees": manifest["target_pitch_degrees"],
            "fov_degrees": manifest["target_fov_degrees"],
        }
        for yaw in target_yaws
    ]
    height, width = int(manifest["height"]), int(manifest["width"])
    observed_rgb, observed_mask, conflict, coverage = project_all_observations(
        manifest["observed_views"], targets, height, width
    )
    main_index = int(np.argmax(coverage))
    if args.prepare_only:
        generated = np.zeros((8, height, width, 3), np.uint8)
    else:
        relative = run_model(manifest, main_index)
        main_yaw = float(manifest["observed_views"][main_index]["yaw_degrees"])
        ordered = []
        for yaw in target_yaws:
            relative_index = int(round(((float(yaw) - main_yaw) % 360) / 45)) % 8
            ordered.append(cv2.resize(relative[relative_index], (width, height), interpolation=cv2.INTER_LANCZOS4))
        generated = np.stack(ordered)

    final = generated.astype(np.float32) / 255
    final[observed_mask] = observed_rgb[observed_mask]
    final = (np.clip(final, 0, 1) * 255).round().astype(np.uint8)
    source = np.ones((8, height, width), np.int8)
    source[observed_mask] = 0
    confidence = np.full((8, height, width), float(manifest.get("synthesized_confidence", 0.4)), np.float32)
    confidence[observed_mask] = 1.0
    confidence[conflict & observed_mask] = 0.5
    boundary = np.stack([
        cv2.dilate(mask.astype(np.uint8), np.ones((9, 9), np.uint8)) > mask
        for mask in observed_mask
    ])
    confidence[boundary & ~observed_mask] *= 0.65
    c2w = np.repeat(np.eye(4, dtype=np.float32)[None], 8, axis=0)
    for index, yaw in enumerate(target_yaws):
        c2w[index, :3, :3] = rotation(yaw, manifest["target_pitch_degrees"])
    k = intrinsics(manifest["target_fov_degrees"], width, height)
    intrinsics_array = np.repeat(k[None], 8, axis=0)
    np.save(output / "views_rgb.npy", final)
    np.save(output / "view_poses.npy", c2w)
    np.save(output / "intrinsics.npy", intrinsics_array)
    np.save(output / "observed_masks.npy", observed_mask)
    np.save(output / "source_maps.npy", source)
    np.save(output / "image_confidence.npy", confidence)
    Image.fromarray(build_preview(final, target_yaws, manifest["target_pitch_degrees"],
                                  manifest["target_fov_degrees"])).save(output / "preview_panorama.png")
    metadata = {
        "backend": "MVDiffusion panorama outpainting",
        "main_condition_index": main_index,
        "main_condition_coverage": float(coverage[main_index]),
        "prepare_only": args.prepare_only,
        "manifest": manifest,
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
