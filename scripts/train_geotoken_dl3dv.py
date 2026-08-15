#!/usr/bin/env python3
"""Train geometry-only GeoTokens with clean, partial, and online ReCal3R worlds."""
from __future__ import annotations

import argparse
import hashlib
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
    parser.add_argument("--causal-world-cache-root", type=Path, required=True)
    parser.add_argument("--gt-latent-cache-root", type=Path, required=True)
    parser.add_argument("--initial-pi3x-world-cache-root", type=Path, required=True)
    parser.add_argument("--pi3x-repo", type=Path, required=True)
    parser.add_argument("--pi3x-checkpoint", type=Path, required=True)
    parser.add_argument("--recal3r-repo", type=Path, required=True)
    parser.add_argument("--recal3r-checkpoint", type=Path, required=True)
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
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--geotoken-strength", type=float, default=1.0)
    parser.add_argument("--camera-strength", type=float, default=1.0)
    parser.add_argument("--world-strength", type=float, default=1.0)
    return parser.parse_args()


def read_rgb(path):
    return np.asarray(Image.open(path).convert("RGB"), np.uint8)

def file_sha256(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024*1024),b''): digest.update(block)
    return digest.hexdigest()


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


def latent_cache_identities(pipe, model):
    import hashlib
    vae_identity = hashlib.sha256(json.dumps({
        "class": type(pipe.vae).__qualname__,
        "latents_mean": [float(value) for value in pipe.vae.config.latents_mean],
        "latents_std": [float(value) for value in pipe.vae.config.latents_std],
    }, sort_keys=True).encode()).hexdigest()
    model_identity = hashlib.sha256(str(Path(model).resolve()).encode()).hexdigest()
    return vae_identity, model_identity


def cached_gt_latent(root, trajectory_id, chunk_index, device, *, expected_shape, vae_identity, model_identity):
    path = Path(root) / trajectory_id / f"chunk_{int(chunk_index):02d}.pt"
    if not path.is_file():
        raise FileNotFoundError(f"frozen GT latent cache is required: {path}")
    payload = torch.load(path, map_location=device, weights_only=False)
    latent = payload.get("latent")
    if not torch.is_tensor(latent) or tuple(latent.shape) != tuple(expected_shape):
        raise RuntimeError(f"invalid GT latent cache: {path}")
    if tuple(payload.get("shape", ())) != tuple(expected_shape):
        raise RuntimeError(f"GT latent shape metadata mismatch: {path}")
    if payload.get("vae_identity") != vae_identity or payload.get("model_identity") != model_identity:
        raise RuntimeError(f"GT latent model identity mismatch: {path}")
    return latent.to(device=device, dtype=torch.bfloat16).detach()


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
    return (value["points_xyz"].astype(np.float32), value["points_rgb"].astype(np.uint8),
            value["points_confidence"].astype(np.float32), value["observation_count"].astype(np.uint16))


def teacher_world_node(source, c2w, intrinsics, points_xyz, points_rgb, confidence, observation_count):
    """A/B ReCal teacher world; intentionally never touches Pi3X cache."""
    from long_video.types import ScaleMetadata, SpatialNode
    xyz = np.asarray(points_xyz, np.float32)
    return SpatialNode("recal_teacher_world", "active", None, np.asarray(c2w[0], np.float32), 0,
        float(np.linalg.norm(xyz.max(0)-xyz.min(0))*0.5), xyz.min(0), xyz.max(0),
        np.asarray(source, np.uint8)[None], np.full((1, *source.shape[:2]), np.nan, np.float32),
        np.asarray(c2w[:1], np.float32), np.asarray(intrinsics[:1], np.float32), xyz,
        np.asarray(points_rgb, np.uint8), np.asarray(confidence, np.float32), np.ones(len(xyz), np.int8),
        np.asarray(observation_count, np.uint16), scale=ScaleMetadata(),
        quality_metrics={"world_backend": "recal_teacher", "voxel_size": 0.02})


def build_online(args, pipe, record, source, geometry_backend=None, pre_render_world_hook=None, initial_node=None):
    from long_video.memory.node_store import NodeStore
    from long_video.online.pipeline import OnlineSpatialHistoryPipeline
    from long_video.initialization.recal3r_world_accumulator import ReCal3RWorldAccumulator

    if initial_node is not None:
        node = initial_node
    else:
        cache = args.initial_pi3x_world_cache_root / record["trajectory_id"]
        metadata_path = cache / "cache_metadata.json"
        if not metadata_path.is_file():
            raise FileNotFoundError(f"Pi3X initial world cache is required: {metadata_path}")
        metadata = json.loads(metadata_path.read_text())
        if (metadata.get("schema_version") != 3 or metadata.get("geometry_implementation_version") != "pi3x-source-only-official-resize-v3"
            or not metadata.get("uses_only_source") or float(metadata.get("voxel_size", -1)) != 0.02):
            raise RuntimeError(f"stale or non-source-only Pi3X initial-world cache: {cache}")
        source_path=sorted((args.dataset_root / record["rgb_dir"]).glob("*"))[0]
        current_repo_commit=__import__('subprocess').check_output(['git','-C',str(args.pi3x_repo),'rev-parse','HEAD'],text=True).strip()
        if (metadata.get('trajectory_id') != record['trajectory_id'] or metadata.get('source_frame_index') != 0
                or metadata.get('source_rgb_sha256') != file_sha256(source_path)
                or metadata.get('pi3x_repo_commit') != current_repo_commit
                or metadata.get('pi3x_checkpoint_sha256') != file_sha256(args.pi3x_checkpoint)):
            raise RuntimeError(f"stale Pi3X W0 provenance: {cache}")
        node = NodeStore(cache).load("node_000")
        # Pi3X cache v3 remains valid. Appearance-anchor state is runtime
        # metadata and must not alter the cached source RGB values.
        node.appearance_anchors = {
            "anchor_rgb": np.asarray(node.points_rgb, np.uint8).copy(),
            "anchor_confidence": np.asarray(node.points_confidence, np.float32).copy(),
            "anchor_frame": np.zeros(len(node.points_xyz), np.int32),
            "source_locked": np.ones(len(node.points_xyz), bool),
        }
    accumulator = None
    if geometry_backend is not None:
        accumulator = ReCal3RWorldAccumulator(
            geometry_backend, node, trajectory_id=record["trajectory_id"], voxel_size=0.02,
        )
    online = OnlineSpatialHistoryPipeline(
        wah_pipeline=pipe, active_node=node, memory_manager=None, world_accumulator=accumulator, prompt=args.prompt,
        renderer_kwargs={"device": args.device, "point_radius": 0},
        wah_state_kwargs={
            "height": 384, "width": 640, "num_frames": 33, "output_type": "np",
            "pyramid_num_inference_steps_list": [2, 2, 2],
        },
        pre_render_world_hook=pre_render_world_hook,
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
    def __init__(self, pipe):
        self.pipe = pipe
        self.by_shape = {}

    def __call__(self, _module, args, kwargs):
        hidden = kwargs.get("hidden_states")
        if torch.is_tensor(hidden):
            key = tuple(hidden.shape[-3:])
            if key in self.by_shape:
                return args, kwargs
            self.by_shape[key] = {
                name: value for name, value in kwargs.items()
                if name != "hidden_states"
            }
            self.by_shape[key]["sample"] = hidden.detach()
            sigmas = getattr(self.pipe.scheduler, "sigmas", None)
            if sigmas is None or not len(sigmas):
                raise RuntimeError("formal Helios scheduler did not expose sigma")
            self.by_shape[key]["sigma"] = torch.as_tensor(sigmas[0]).detach()
        return args, kwargs


def native_flow_backward(pipe, z_gt, capture, provider, exact_args):
    """Use official train_exact samples; rollout supplies history only."""
    from warp_as_history.training.core import flow_matching_train_exact_items
    items = flow_matching_train_exact_items(pipe, z_gt, exact_args, z_gt.device)
    losses, stage_stats = [], {}
    if {item["stage_id"] for item in items} != {0, 1, 2}:
        raise RuntimeError("GeoToken training requires official train_exact samples for all stages")
    for item in items:
        stage_id = int(item["stage_id"])
        from long_video.geometry.geotoken import progress_from_sigma
        pipe.transformer.geotoken.set_timing(
            stage_index=stage_id,
            denoise_progress=progress_from_sigma(item["sigmas"]),
        )
        key = tuple(item["noisy_latents"].shape[-3:])
        if key not in capture.by_shape:
            raise RuntimeError(f"formal inference did not capture Stage{stage_id} shape {key}")
        kwargs = dict(capture.by_shape[key])
        kwargs.pop("sample", None); kwargs.pop("sigma", None)
        x_t = item["noisy_latents"].to(device=z_gt.device, dtype=pipe.transformer.dtype).detach()
        kwargs["hidden_states"] = x_t
        kwargs["timestep"] = item["timesteps"].to(device=z_gt.device)
        kwargs["return_dict"] = False
        target = item["target"].to(device=z_gt.device, dtype=torch.float32)
        if target.shape != x_t.shape:
            raise RuntimeError(f"Stage{stage_id} official train_exact state/target mismatch")
        if stage_id == 0:
            pipe._set_wah_lora_enabled(True)
        else:
            pipe._set_wah_lora_enabled(False)
        # PEFT's adapter toggles recursively clear requires_grad on modules
        # outside the adapter too. Restore the sole permitted trainable set.
        for name, parameter in pipe.transformer.named_parameters():
            parameter.requires_grad_("geotoken." in name)
        finite_inputs = {
            "x_t": bool(torch.isfinite(kwargs["hidden_states"]).all()),
            "target": bool(torch.isfinite(target).all()),
            "timesteps": bool(torch.isfinite(kwargs["timestep"]).all()),
            "sigma": bool(torch.isfinite(item["sigmas"]).all()),
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
        loss = torch.mean(
            (prediction.float() - target.float()).square()
            .reshape(prediction.shape[0], -1), dim=1,
        ).mean()
        if not bool(torch.isfinite(loss)):
            raise RuntimeError(f"Stage{stage_id} native flow loss is non-finite")
        losses.append(loss)
        stage_stats[f"stage{stage_id}_flow_mse"] = float(loss.detach())
        (loss / len(items)).backward()
        # A later stage can share a token-grid resolution with an earlier
        # history branch. Do not retain an already-backpropagated encoder graph.
        if kwargs["hidden_states"].requires_grad or kwargs["hidden_states"].grad is not None:
            raise RuntimeError("native_flow_backward must not build gradients for x_t")
        provider.clear_feature_cache()
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
    from long_video.wah.upstream import assert_wah_upstream
    assert_wah_upstream(args.wah_root)
    if args.total_steps != 2000 and not args.smoke_only:
        raise ValueError("formal GeoToken training is strictly 2000 optimizer steps")
    sys.path.insert(0, str(args.wah_root))
    from warp_as_history import WarpAsHistoryPipeline
    from long_video.geometry.geotoken import assert_geotoken_only_trainable, install_geotoken
    from long_video.geometry.geotoken_runtime import (
        PointWorldGeoTokenProvider, source_scene_scale_from_active_node,
    )
    from long_video.training.geotoken import (
        BalancedRolloutSampler, checkpoint_names, load_geotoken_checkpoint,
        phase_for_step, save_geotoken_checkpoint, load_causal_world_cache,
        augment_partial_voxels,
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
    records = [record for record in selected_records if (
        (lambda m: m.get("valid", False) and m.get("schema_version") == 3
         and m.get("geometry_implementation_version") == "recal-full-teacher-world-v4-rgb-anchor")(
            json.loads((args.recal3r_root / record["trajectory_id"] / "metadata.json").read_text())
        ))]
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
    conditioner.configure_strengths(
        geotoken=args.geotoken_strength, camera=args.camera_strength, world=args.world_strength,
    )
    vae_identity, model_identity = latent_cache_identities(pipe, args.model)
    if args.gradient_checkpointing and hasattr(pipe.transformer, "enable_gradient_checkpointing"):
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
        total_started = time.time()
        phase = phase_for_step(step)
        record = sampler.choose_record(records, step)
        rollout_length = 1 if args.smoke_only else sampler.choose_length(step)
        arrays = load_arrays(args.dataset_root, record)
        geometry_root = args.recal3r_root / record["trajectory_id"]
        metadata = json.loads((geometry_root / "metadata.json").read_text())
        if (not metadata.get("valid", False) or metadata.get("schema_version") != 3
                or metadata.get("geometry_implementation_version") != "recal-full-teacher-world-v4-rgb-anchor"):
            raise RuntimeError(f"invalid ReCal3R geometry selected: {record['trajectory_id']}")
        source = read_rgb(arrays["rgb_paths"][0])
        data_load_seconds = time.time() - total_started
        if phase == "C" and geometry_backend is None:
            from long_video.initialization.recal3r_geometry_backend import ReCal3RGeometryBackend
            geometry_backend = ReCal3RGeometryBackend(
                args.recal3r_checkpoint, args.recal3r_repo, args.device,
            )
        source_center = arrays["c2w"][0, :3, 3]
        # ReCal3R normalization is confined to the offline A/B phases.
        bootstrap_online = None
        if phase == "C":
            # The model weights may persist, but recurrent observations are
            # trajectory-owned and must never cross an optimizer step.
            geometry_backend.reset()
            initial_recal_state = geometry_backend.get_state()
            if (initial_recal_state.get("frame_count") != 0
                    or initial_recal_state.get("sequence_version") != 0
                    or initial_recal_state.get("has_recurrent_state")):
                raise RuntimeError(
                    f"Phase C ReCal3R state did not reset for step {step}: {initial_recal_state}"
                )
            bootstrap_online = build_online(args, pipe, record, source, geometry_backend)
            scale = source_scene_scale_from_active_node(
                bootstrap_online.active_node, arrays["c2w"][0], arrays["k"][0], device=args.device,
            )
        else:
            scale = scene_scale_from_recal(geometry_root, arrays["c2w"])
            if phase == "A":
                a_xyz, a_rgb, a_conf, a_obs = full_scene_points(geometry_root)
                teacher_node = teacher_world_node(source, arrays["c2w"], arrays["k"], a_xyz, a_rgb, a_conf, a_obs)
            else:
                b_xyz, b_rgb, b_conf, b_obs, _b_keys, b_max = load_causal_world_cache(
                    args.causal_world_cache_root / record["trajectory_id"], 0)
                if int(b_max.max()) > 0: raise RuntimeError("Phase B initial world leaks future observations")
                teacher_node = teacher_world_node(source, arrays["c2w"], arrays["k"], b_xyz, b_rgb, b_conf, b_obs)
        provider = PointWorldGeoTokenProvider(
            conditioner, device=args.device, source_center=source_center, scene_scale=scale, render_height=384, render_width=640,
        )
        provider.set_world_slot_dropout(0.15 if phase == "B" else 0.0)
        provider.attach(pipe.transformer)
        online = bootstrap_online if phase == "C" else build_online(args, pipe, record, source, initial_node=teacher_node)

        def pre_render_world_hook(active_node, cameras):
            active_world = provider.configure_active_node(active_node)
            source_geometry = provider.ensure_source_geometry(arrays["c2w"][0], arrays["k"][0])
            existing_source = online.autoregressive_state.setdefault("_geotoken_source_geometry", source_geometry)
            if existing_source is not source_geometry:
                raise RuntimeError("GeoToken source geometry changed within one trajectory")
            provider.configure_chunk(
                cameras.c2w, cameras.intrinsics,
                online.autoregressive_state.get("_geotoken_history_snapshots", ()),
                history_window=online.autoregressive_state.get("_wah_geometry_slot_refs", ()),
                source_geometry=online.autoregressive_state["_geotoken_source_geometry"],
            )
            from long_video.online.pipeline import point_world_snapshot_identity
            return {
                "world_identity": point_world_snapshot_identity(active_node),
                "freeze_history": provider.freeze_current_snapshot,
            }

        online.pre_render_world_hook = pre_render_world_hook
        capture = ForwardCapture(pipe)
        capture_handle = pipe.transformer.register_forward_pre_hook(capture, with_kwargs=True)
        prefix_rollout_seconds = 0.0
        gt_latent_seconds = 0.0
        try:
            corruption_seed = (int(args.seed) << 32) ^ int(step)
            phase_a_world = full_scene_points(geometry_root) if phase == "A" else None
            for chunk_index in range(rollout_length):
                frame_slice = slice(chunk_index * 32, chunk_index * 32 + 33)
                poses, intrinsics = arrays["c2w"][frame_slice], arrays["k"][frame_slice]
                if phase == "A":
                    points, colors, confidence, observations = phase_a_world
                elif phase == "B":
                    points, colors, confidence, observations, voxel_keys, max_frame = load_causal_world_cache(
                        args.causal_world_cache_root / record["trajectory_id"], chunk_index * 32,
                    )
                    if int(max_frame.max()) > chunk_index * 32: raise RuntimeError("Phase B causal world contains future observation")
                    points, confidence, kept = augment_partial_voxels(
                        points, confidence, voxel_keys, source_center=source_center, scene_scale=scale,
                        args=args, seed=corruption_seed, return_indices=True,
                    )
                    colors, observations = colors[kept], observations[kept]
                if phase != "C":
                    online.active_node = teacher_world_node(source, poses, intrinsics, points, colors, confidence, observations)
                    world_version = (
                        ("A", record["trajectory_id"])
                        if phase == "A" else ("B", record["trajectory_id"], chunk_index)
                    )
                    provider.configure_world(points, confidence, world_version=world_version)
                capture.by_shape.clear()
                rollout_started = time.time()
                with torch.no_grad():
                    online.generate_chunk_at_cameras(poses, intrinsics, 384, 640)
                prefix_rollout_seconds += time.time() - rollout_started
            if phase == "C":
                final_recal_state = geometry_backend.get_state()
                # The first chunk contributes all 33 frames; subsequent
                # chunks contribute 32 because their boundary is shared.
                # Candidate validation may re-evaluate existing frames but
                # cannot add observations beyond this trajectory's prefix.
                # One source-only recurrent priming observation plus each
                # generated frame (shared chunk boundaries are not replayed).
                expected_recal_frames = 1 + 32 * rollout_length
                if int(final_recal_state.get("frame_count", 0)) != expected_recal_frames:
                    raise RuntimeError(
                        f"Phase C ReCal3R state leaked observations at step {step}: "
                        f"{final_recal_state}, rollout_length={rollout_length}"
                    )
            gt_started = time.time()
            z_gt = cached_gt_latent(
                args.gt_latent_cache_root, record["trajectory_id"], rollout_length - 1, args.device,
                expected_shape=(1, 16, 9, 48, 80), vae_identity=vae_identity, model_identity=model_identity,
            )
            gt_latent_seconds = time.time() - gt_started
            optimizer.zero_grad(set_to_none=True)
            train_started = time.time()
            with torch.enable_grad():
                exact_args = SimpleNamespace(
                    pyramid_num_inference_steps_list=[2, 2, 2],
                    flow_matching_stage_sampling="all",
                    flow_matching_stage_id=0,
                    flow_matching_train_exact_timestep_sampling="training_density",
                    flow_matching_use_dynamic_shifting="off",
                    weighting_scheme="none",
                    is_amplify_first_chunk=False,
                )
                loss, stage_stats = native_flow_backward(pipe, z_gt, capture, provider, exact_args)
            training_forward_backward_seconds = time.time() - train_started
            optimizer_started = time.time()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                [parameter for _, parameter in trainable], args.max_grad_norm,
            )
            if not torch.isfinite(loss) or not torch.isfinite(grad_norm):
                raise RuntimeError("non-finite GeoToken loss/gradient")
            before = {name: value.detach().clone() for name, value in trainable}
            optimizer.step(); scheduler.step()
            optimizer_seconds = time.time() - optimizer_started
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
            "camera_strength": conditioner.camera_strength, "world_strength": conditioner.world_strength,
            "geotoken_injection": conditioner.diagnostics,
            "data_load_seconds": data_load_seconds,
            "point_render_seconds": provider.point_render_seconds(),
            "prefix_rollout_seconds": prefix_rollout_seconds,
            "gt_latent_seconds": gt_latent_seconds,
            "training_forward_backward_seconds": training_forward_backward_seconds,
            "optimizer_seconds": optimizer_seconds,
            "total_step_seconds": time.time() - total_started,
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
