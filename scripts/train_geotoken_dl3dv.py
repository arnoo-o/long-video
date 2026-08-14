#!/usr/bin/env python3
"""Train geometry-only GeoTokens with clean, partial, and online Pi3 worlds."""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import torch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wah-root", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--recal3r-root", type=Path, required=True)
    parser.add_argument("--initial-world-cache-root", type=Path, required=True)
    parser.add_argument("--pi3-repo", type=Path, required=True)
    parser.add_argument("--pi3-checkpoint", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--total-steps", type=int, default=2000)
    parser.add_argument("--record-count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260814)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--point-dropout", type=float, default=0.10)
    parser.add_argument("--confidence-dropout", type=float, default=0.05)
    parser.add_argument("--depth-noise", type=float, default=0.01)
    parser.add_argument("--xyz-jitter", type=float, default=0.005)
    parser.add_argument("--prompt", default="A stable realistic view of the same scene.")
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument("--smoke-only", action="store_true")
    return parser.parse_args()


def read_rgb(path):
    return np.asarray(Image.open(path).convert("RGB"), np.uint8)


def load_arrays(root, record):
    paths = sorted((root / record["rgb_dir"]).glob("*"))
    if len(paths) != 193:
        raise ValueError(f"GeoToken requires 193-frame trajectories, got {len(paths)}")
    if bool(record.get("uses_future_gt", True)):
        raise ValueError("future-GT records are forbidden")
    return {
        "rgb_paths": paths,
        "c2w": np.load(root / record["target_c2w_local"]).astype(np.float32),
        "k": np.load(root / record["intrinsics"]).astype(np.float32),
    }


def deterministic_latent(pipe, frames, device):
    tensor = pipe._coerce_warp_video_tensor(
        np.asarray(frames), height=384, width=640, device=torch.device(device),
    ).to(dtype=pipe.vae.dtype)
    posterior = pipe.vae.encode(tensor)
    posterior = getattr(posterior, "latent_dist", posterior)
    mode = getattr(posterior, "mode", None)
    clean = mode() if callable(mode) else posterior.mean
    mean = torch.tensor(pipe.vae.config.latents_mean, device=device, dtype=clean.dtype).view(1, -1, 1, 1, 1)
    std = 1 / torch.tensor(pipe.vae.config.latents_std, device=device, dtype=clean.dtype).view(1, -1, 1, 1, 1)
    return ((clean - mean) * std).detach()


def scene_scale_from_recal(root, c2w):
    xyz = np.load(root / "xyz_world.npy", mmap_mode="r")[0]
    valid = np.load(root / "valid.npy", mmap_mode="r")[0].astype(bool)
    points = np.asarray(xyz[valid], np.float32)
    camera = (np.linalg.inv(c2w[0])[:3, :3] @ points.T).T + np.linalg.inv(c2w[0])[:3, 3]
    depth = camera[:, 2]
    depth = depth[np.isfinite(depth) & (depth > 0)]
    if not len(depth):
        raise RuntimeError(f"no source-visible ReCal3R depth in {root}")
    return float(np.median(depth))


def full_scene_points(root):
    value = np.load(root / "scene_points.npz")
    return value["points_xyz"].astype(np.float32), value["points_confidence"].astype(np.float32)


def causal_partial_points(root, frame_limit, voxel_size=0.02):
    xyz = np.load(root / "xyz_world.npy", mmap_mode="r")[: int(frame_limit) + 1]
    valid = np.load(root / "valid.npy", mmap_mode="r")[: int(frame_limit) + 1].astype(bool)
    confidence = np.load(root / "confidence.npy", mmap_mode="r")[: int(frame_limit) + 1]
    points = np.asarray(xyz[valid], np.float32)
    weights = np.asarray(confidence[valid], np.float32)
    finite = np.isfinite(points).all(1) & np.isfinite(weights) & (weights > 0)
    points, weights = points[finite], weights[finite]
    if not len(points):
        raise RuntimeError("causal ReCal3R subset contains no valid observations")
    voxels = np.floor(points / float(voxel_size)).astype(np.int64)
    _, first, inverse = np.unique(voxels, axis=0, return_index=True, return_inverse=True)
    sums = np.zeros((len(first), 3), np.float64)
    sum_weights = np.zeros(len(first), np.float64)
    np.add.at(sums, inverse, points * weights[:, None])
    np.add.at(sum_weights, inverse, weights)
    fused = (sums / np.maximum(sum_weights[:, None], 1e-8)).astype(np.float32)
    fused_confidence = np.minimum(1.0, sum_weights / np.maximum(1, np.bincount(inverse))).astype(np.float32)
    return fused, fused_confidence


def augment_partial_point_world(points, confidence, *, source_center, scene_scale, args, seed):
    """Apply one trajectory-consistent perturbation before all camera renders."""
    rng = np.random.default_rng(int(seed))
    keep = rng.random(len(points)) >= float(args.point_dropout)
    points = np.asarray(points, np.float32)[keep].copy()
    confidence = np.asarray(confidence, np.float32)[keep].copy()
    confidence[rng.random(len(confidence)) < float(args.confidence_dropout)] = 0
    if args.xyz_jitter > 0:
        points += rng.normal(
            0, float(args.xyz_jitter) * float(scene_scale), points.shape,
        ).astype(np.float32)
    if args.depth_noise > 0:
        ray = points - np.asarray(source_center, np.float32)
        ray /= np.maximum(np.linalg.norm(ray, axis=1, keepdims=True), 1e-8)
        offset = rng.normal(
            0, float(args.depth_noise) * float(scene_scale), (len(points), 1),
        ).astype(np.float32)
        points += ray * offset
    valid = np.isfinite(points).all(1) & np.isfinite(confidence) & (confidence > 0)
    return points[valid], confidence[valid]


def build_online(args, pipe, record, source, geometry_backend=None):
    from long_video.config import load_yaml
    from long_video.memory.memory_manager import MemoryManager
    from long_video.memory.node_store import NodeStore
    from long_video.online.pipeline import OnlineSpatialHistoryPipeline

    cache = args.initial_world_cache_root / record["trajectory_id"]
    node = NodeStore(cache).load("node_000")
    manager = None
    if geometry_backend is not None:
        manager = MemoryManager.from_config(
            load_yaml("configs/online_memory.yaml"), geometry_backend=geometry_backend,
        )
    online = OnlineSpatialHistoryPipeline(
        wah_pipeline=pipe, active_node=node, memory_manager=manager, prompt=args.prompt,
        renderer_kwargs={"device": args.device, "point_radius": 0},
        wah_state_kwargs={
            "height": 384, "width": 640, "num_frames": 33, "output_type": "np",
            "pyramid_num_inference_steps_list": [2, 2, 2],
        },
    )
    online.wah_fill_frame = source.copy()
    online.autoregressive_state = pipe.init_autoregressive_state(
        prompt=args.prompt, image=Image.fromarray(source), conditioning_type="warp",
        warp_history_downsample_mode="short", rope_alignment=True,
        height=384, width=640, num_frames=33, output_type="np",
        pyramid_num_inference_steps_list=[2, 2, 2],
    )
    online.wah_adapter.configure_state(online.autoregressive_state)
    online.autoregressive_state["is_amplify_first_chunk"] = False
    return online


class ForwardCapture:
    def __init__(self):
        self.by_shape = {}

    def __call__(self, _module, args, kwargs):
        hidden = kwargs.get("hidden_states")
        if torch.is_tensor(hidden):
            key = tuple(hidden.shape[-3:])
            self.by_shape.setdefault(key, {
                name: value for name, value in kwargs.items()
                if name != "hidden_states"
            })
        return args, kwargs


def training_exact_args():
    return SimpleNamespace(
        pyramid_num_inference_steps_list=[2, 2, 2],
        flow_matching_stage_sampling="fixed",
        flow_matching_stage_id=-1,
        flow_matching_train_exact_timestep_sampling="training_density",
        # The formal inference transformer consumes discrete scheduler
        # timesteps. Keep train_exact on that same native scheduler domain.
        flow_matching_use_dynamic_shifting="off",
        weighting_scheme="none",
        is_amplify_first_chunk=False,
    )


def native_flow_backward(pipe, z_gt, capture, provider):
    from warp_as_history.training.core import (
        compute_loss_weighting_for_sd3, flow_matching_train_exact_items,
    )
    items = flow_matching_train_exact_items(pipe, z_gt, training_exact_args(), z_gt.device)
    losses, stage_stats = [], {}
    for item in items:
        stage_id = int(item["stage_id"])
        key = tuple(item["noisy_latents"].shape[-3:])
        if key not in capture.by_shape:
            raise RuntimeError(f"formal inference did not capture Stage{stage_id} shape {key}")
        kwargs = dict(capture.by_shape[key])
        kwargs["hidden_states"] = (
            item["noisy_latents"].to(dtype=pipe.transformer.dtype).detach().requires_grad_(True)
        )
        kwargs["timestep"] = item["timesteps"]
        kwargs["return_dict"] = False
        if stage_id == 0:
            pipe._set_wah_lora_enabled(True)
        else:
            pipe._set_wah_lora_enabled(False)
        # PEFT's adapter toggles recursively clear requires_grad on modules
        # outside the adapter too. Restore the sole permitted trainable set.
        for name, parameter in pipe.transformer.named_parameters():
            parameter.requires_grad_("geotoken." in name)
        finite_inputs = {
            "noisy_latents": bool(torch.isfinite(kwargs["hidden_states"]).all()),
            "target": bool(torch.isfinite(item["target"]).all()),
            "timesteps": bool(torch.isfinite(item["timesteps"]).all()),
            "sigmas": bool(torch.isfinite(item["sigmas"]).all()),
            "prompt": bool(torch.isfinite(kwargs["encoder_hidden_states"]).all()),
        }
        for name in ("latents_history_short", "latents_history_mid", "latents_history_long"):
            value = kwargs.get(name)
            if value is not None:
                finite_inputs[name] = bool(torch.isfinite(value).all())
        prediction = pipe.transformer(**kwargs)[0]
        if not prediction.requires_grad or not bool(torch.isfinite(prediction).all()):
            geo_state = {
                name: (parameter.requires_grad, bool(torch.isfinite(parameter).all()))
                for name, parameter in pipe.transformer.named_parameters() if "geotoken." in name
            }
            raise RuntimeError(
                f"Stage{stage_id} GeoToken prediction invalid: requires_grad={prediction.requires_grad}, "
                f"finite={bool(torch.isfinite(prediction).all())}, inputs={finite_inputs}, geotoken={geo_state}"
            )
        sigma = item["sigmas"]
        weighting = compute_loss_weighting_for_sd3("none", sigma)
        loss = torch.mean(
            (weighting.float() * (prediction.float() - item["target"].float()).square())
            .reshape(prediction.shape[0], -1), dim=1,
        ).mean()
        if not bool(torch.isfinite(loss)):
            raise RuntimeError(f"Stage{stage_id} native flow loss is non-finite")
        losses.append(loss)
        stage_stats[f"stage{stage_id}_flow_mse"] = float(loss.detach())
        (loss / len(items)).backward()
        broken = [
            name for name, parameter in pipe.transformer.named_parameters()
            if "geotoken." in name and parameter.grad is not None
            and not bool(torch.isfinite(parameter.grad).all())
        ]
        if broken:
            raise RuntimeError(f"Stage{stage_id} produced non-finite GeoToken gradients: {broken}")
    return torch.stack([loss.detach() for loss in losses]).mean(), stage_stats


def lr_scale(step, total, warmup):
    if step < warmup:
        return float(step + 1) / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))


def main():
    args = parse_args()
    if args.total_steps != 2000 and not args.smoke_only:
        raise ValueError("formal GeoToken training is strictly 2000 optimizer steps")
    sys.path.insert(0, str(args.wah_root))
    from warp_as_history import WarpAsHistoryPipeline
    from long_video.geometry.geotoken import assert_geotoken_only_trainable, install_geotoken
    from long_video.geometry.geotoken_runtime import PointWorldGeoTokenProvider
    from long_video.training.geotoken import (
        BalancedRolloutSampler, checkpoint_names, load_geotoken_checkpoint,
        phase_for_step, save_geotoken_checkpoint,
    )
    from long_video.training.wpf_adaptation import select_balanced_training_records

    if args.run_dir.exists() and any(args.run_dir.iterdir()):
        raise FileExistsError(f"run directory must be new: {args.run_dir}")
    metrics_dir = args.run_dir / "metrics"
    checkpoints_dir = args.run_dir / "checkpoints"
    metrics_dir.mkdir(parents=True)
    checkpoints_dir.mkdir()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    manifest = json.loads((args.dataset_root / "dl3dv_24fps_manifest.json").read_text())
    selected_records = select_balanced_training_records(manifest["records"], args.record_count)
    missing = [record["trajectory_id"] for record in selected_records if not (
        args.recal3r_root / record["trajectory_id"] / "metadata.json"
    ).is_file()]
    if missing:
        raise RuntimeError(f"ReCal3R geometry is incomplete for {len(missing)} selected trajectories")
    records = [record for record in selected_records if json.loads(
        (args.recal3r_root / record["trajectory_id"] / "metadata.json").read_text()
    ).get("valid", False)]
    if len(records) < 90:
        raise RuntimeError(
            f"only {len(records)}/{len(selected_records)} selected ReCal3R trajectories are valid"
        )

    pipe = WarpAsHistoryPipeline.from_pretrained(args.model, torch_dtype=torch.bfloat16).to(args.device)
    if not hasattr(pipe.transformer.config, "image_dim"):
        pipe.transformer.register_to_config(image_dim=None)
    official = args.wah_root / "checkpoints/warp-as-history/visible_lora_state_step1000.safetensors"
    pipe._configure_wah_lora(str(official))
    for module in (pipe.vae, pipe.text_encoder, pipe.transformer):
        for parameter in module.parameters():
            parameter.requires_grad_(False)
    conditioner = install_geotoken(pipe.transformer).to(device=args.device)
    if hasattr(pipe.transformer, "enable_gradient_checkpointing"):
        pipe.transformer.enable_gradient_checkpointing()
    trainable = assert_geotoken_only_trainable(pipe.transformer)
    if hasattr(pipe, "_world_projection_context") or hasattr(pipe, "_pyramid_training_adapter_name"):
        raise RuntimeError("GeoToken training requires WPF and wpf_adaptation to be absent")
    for module in (pipe.vae, pipe.text_encoder):
        if any(parameter.requires_grad for parameter in module.parameters()):
            raise RuntimeError("VAE/text encoder must remain frozen")
    optimizer = torch.optim.AdamW(
        [parameter for _, parameter in trainable], lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda value: lr_scale(value, args.total_steps, args.warmup_steps),
    )
    sampler = BalancedRolloutSampler(args.seed)
    start_step = 0
    if args.resume_checkpoint:
        start_step = load_geotoken_checkpoint(
            args.resume_checkpoint, transformer=pipe.transformer, optimizer=optimizer,
            lr_scheduler=scheduler, sampler=sampler,
        )
    geometry_backend = None
    run_started = time.time()
    steps_to_run = [start_step + 1] if args.smoke_only else range(start_step + 1, 2001)
    for step in steps_to_run:
        phase = phase_for_step(step)
        record = sampler.choose_record(records, step)
        rollout_length = 1 if args.smoke_only else sampler.choose_length(step)
        arrays = load_arrays(args.dataset_root, record)
        geometry_root = args.recal3r_root / record["trajectory_id"]
        metadata = json.loads((geometry_root / "metadata.json").read_text())
        if not metadata.get("valid", False):
            raise RuntimeError(f"invalid ReCal3R geometry selected: {record['trajectory_id']}")
        source = read_rgb(args.dataset_root / record["source"])
        if phase == "C" and geometry_backend is None:
            from long_video.initialization.geometry_backend import Pi3GeometryBackend
            geometry_backend = Pi3GeometryBackend(args.pi3_checkpoint, args.pi3_repo, args.device)
            pi3_model = getattr(geometry_backend, "model", None)
            if pi3_model is not None and hasattr(pi3_model, "parameters"):
                for parameter in pi3_model.parameters():
                    parameter.requires_grad_(False)
        online = build_online(args, pipe, record, source, geometry_backend if phase == "C" else None)
        source_center = arrays["c2w"][0, :3, 3]
        scale = scene_scale_from_recal(geometry_root, arrays["c2w"])
        provider = PointWorldGeoTokenProvider(
            conditioner, device=args.device, source_center=source_center, scene_scale=scale,
        )
        provider.attach(pipe.transformer)
        history_cameras = [(np.repeat(arrays["c2w"][:1], 33, 0), np.repeat(arrays["k"][:1], 33, 0))]
        capture = ForwardCapture()
        capture_handle = pipe.transformer.register_forward_pre_hook(capture, with_kwargs=True)
        step_started = time.time()
        try:
            for chunk_index in range(rollout_length):
                frame_slice = slice(chunk_index * 32, chunk_index * 32 + 33)
                poses, intrinsics = arrays["c2w"][frame_slice], arrays["k"][frame_slice]
                if phase == "A":
                    points, confidence = full_scene_points(geometry_root)
                elif phase == "B":
                    points, confidence = causal_partial_points(geometry_root, chunk_index * 32)
                    points, confidence = augment_partial_point_world(
                        points, confidence, source_center=source_center, scene_scale=scale,
                        args=args, seed=args.seed + step * 17 + chunk_index,
                    )
                else:
                    points = np.asarray(online.active_node.points_xyz, np.float32).copy()
                    confidence = np.asarray(online.active_node.points_confidence, np.float32).copy()
                provider.configure_world(points, confidence)
                provider.configure_chunk(poses, intrinsics, history_cameras)
                capture.by_shape.clear()
                with torch.no_grad():
                    online.generate_chunk_at_cameras(poses, intrinsics, 384, 640)
                history_cameras.append((poses, intrinsics))
            gt = [read_rgb(path) for path in arrays["rgb_paths"][frame_slice]]
            with torch.no_grad():
                z_gt = deterministic_latent(pipe, gt, args.device)
            optimizer.zero_grad(set_to_none=True)
            with torch.enable_grad():
                loss, stage_stats = native_flow_backward(pipe, z_gt, capture, provider)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                [parameter for _, parameter in trainable], args.max_grad_norm,
            )
            if not torch.isfinite(loss) or not torch.isfinite(grad_norm):
                raise RuntimeError("non-finite GeoToken loss/gradient")
            before = {name: value.detach().clone() for name, value in trainable}
            optimizer.step(); scheduler.step()
            changed = [name for name, value in trainable if not torch.equal(before[name], value.detach())]
            if not changed:
                raise RuntimeError("optimizer step did not update GeoToken")
        finally:
            capture_handle.remove()
            if provider._handle is not None:
                provider._handle.remove()
            conditioner.clear_active()
        metrics = {
            "global_step": int(step), "total_steps": 2000, "phase": phase,
            "selected_trajectory_count": len(selected_records),
            "valid_geometry_trajectory_count": len(records),
            "trajectory_id": record["trajectory_id"], "rollout_length": rollout_length,
            "supervision_chunk": rollout_length - 1, "flow_loss": float(loss.detach()),
            "grad_norm": float(grad_norm), "learning_rate": optimizer.param_groups[0]["lr"],
            "geotoken_parameter_norm": float(torch.sqrt(sum(
                parameter.detach().float().square().sum() for _, parameter in trainable
            ))),
            "injection_gates": {
                key: float(value.detach()) for key, value in conditioner.injection_gates.items()
            },
            "optimizer_step_seconds": time.time() - step_started,
            "elapsed_seconds": time.time() - run_started,
            "uses_future_gt": False,
            **stage_stats,
        }
        (metrics_dir / f"step_{step:04d}.json").write_text(json.dumps(metrics, indent=2))
        for name in checkpoint_names(step):
            save_geotoken_checkpoint(
                checkpoints_dir / name, transformer=pipe.transformer, optimizer=optimizer,
                lr_scheduler=scheduler, step=step, sampler=sampler,
            )
        print(json.dumps(metrics), flush=True)
    if args.smoke_only:
        save_geotoken_checkpoint(
            checkpoints_dir / "smoke_step_0001.pt", transformer=pipe.transformer,
            optimizer=optimizer, lr_scheduler=scheduler, step=1, sampler=sampler,
        )


if __name__ == "__main__":
    main()
