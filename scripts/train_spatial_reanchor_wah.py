#!/usr/bin/env python3
"""Two-phase Spatially Re-Anchored WAH + Plucker training."""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import random
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def _args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/spatial_reanchor_training.yaml")
    parser.add_argument("--manifest", required=True)
    parser.add_argument(
        "--mode",
        choices=("smoke", "phase-b-smoke", "train", "phase-b-v2-smoke", "phase-b-v2-train"),
        required=True,
    )
    parser.add_argument("--resume", default="")
    parser.add_argument("--phase-b-v2-resume", default="")
    parser.add_argument("--phase-b-smoke-steps", type=int, default=3)
    parser.add_argument("--set", action="append", default=[], dest="overrides")
    return parser.parse_args()


def _atomic_json(path, payload):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def _atomic_torch_save(path, payload):
    import torch

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _frames(path, start=0, count=None):
    from PIL import Image
    files = sorted(Path(path).glob("*.png"))
    selected = files[int(start):] if count is None else files[int(start):int(start) + int(count)]
    return [Image.open(item).convert("RGB") for item in selected]


def _mask_frames(values):
    import numpy as np
    from PIL import Image
    return [Image.fromarray(np.rint(np.clip(frame, 0, 1) * 255).astype(np.uint8), mode="L") for frame in values]


def _tree_to(value, device):
    import torch
    if isinstance(value, torch.Tensor):
        return value.detach().to(device=device)
    if isinstance(value, dict):
        return {key: _tree_to(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [_tree_to(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_tree_to(item, device) for item in value)
    return value


def _gpu_snapshot():
    lines = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
         "--format=csv,noheader,nounits"], text=True, capture_output=True, check=False,
    ).stdout.splitlines()
    return [line.strip() for line in lines if line.strip()]


def main():
    args = _args()
    repo = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo))
    from long_video.config import load_yaml
    config = load_yaml(args.config, args.overrides)
    if int(config.get("physical_gpu", 1)) != 1:
        raise ValueError("training is restricted to physical GPU 1")
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    os.environ.setdefault("XFORMERS_DISABLED", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    import imageio.v2 as imageio
    import numpy as np
    import torch
    from PIL import Image
    from long_video.memory.node_store import NodeStore
    from long_video.types import CameraBatch
    from long_video.oracle_training.wah_training import (
        assert_only_lora_gradients,
        load_source_trainable_state,
        load_phase_b_v2_base_checkpoint,
        load_phase_b_v2_checkpoint,
        load_spatial_training_checkpoint,
        masked_flow_matching_loss,
        save_phase_b_v2_checkpoint,
        save_spatial_training_checkpoint,
    )
    from long_video.oracle_training.history_bank import (
        HistoryBankKey, history_bank_cache_key, validate_history_bank_entry,
    )
    from long_video.oracle_training.round_robin import RoundRobinChunkScheduler, eligible_current_chunks
    from long_video.oracle_training.supervision import validate_current_chunk_supervision
    from long_video.oracle_training.spatial_memory_warp import SpatialMemoryWarpBank
    from long_video.oracle_training.causal_warp import CausalActiveNodeRenderer
    from long_video.wah.spatial_reanchor import (
        install_spatial_reanchor, plucker_camera_rays, resize_latents_spatial,
        visibility_to_target_tokens,
    )

    if torch.cuda.device_count() != 1:
        raise RuntimeError("training must see exactly physical GPU 1")
    torch.cuda.set_device(0)
    torch.cuda.reset_peak_memory_stats(0)
    required = ("wah_root", "wah_model", "run_dir", "source_checkpoint")
    missing = [key for key in required if not config.get(key)]
    if missing:
        raise ValueError(f"missing machine path overrides: {missing}")
    run_dir = Path(config["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "train.log"
    status_path = run_dir / "training_status.json"
    manifest_path = Path(args.manifest)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = manifest["sequences"]
    phase_a = [item for item in records if item["phase"] == "A" and item["split"] == "train"]
    phase_a_diag = [item for item in records if item["phase"] == "A" and item["split"] == "diagnostic"]
    phase_b = [item for item in records if item["phase"] == "B" and item["split"] == "train"]
    phase_a_scenes = sorted({item["scene_id"] for item in phase_a})
    scenes = sorted({item["scene_id"] for item in phase_b})
    phase_a_train_counts = {
        scene: sum(item["scene_id"] == scene for item in phase_a)
        for scene in phase_a_scenes
    }
    phase_a_diag_counts = {
        scene: sum(item["scene_id"] == scene for item in phase_a_diag)
        for scene in phase_a_scenes
    }
    if (
        len(phase_a_scenes) < 2
        or any(count < 4 for count in phase_a_train_counts.values())
        or any(phase_a_diag_counts.get(scene, 0) < 1 for scene in phase_a_scenes)
        or not scenes
        or set(phase_a_scenes) != set(scenes)
    ):
        raise ValueError(
            "manifest must contain at least two Phase A scenes with four train and one "
            "diagnostic window each, plus Phase B windows"
        )
    available_phase_b_chunks = {
        scene: sorted({
            int(item["chunk_count"]) for item in phase_b if item["scene_id"] == scene
        })
        for scene in scenes
    }
    if any(not values for values in available_phase_b_chunks.values()):
        raise ValueError("each scene must contain at least one gap-safe Phase B window")

    def curriculum_chunk(scene, target):
        available = available_phase_b_chunks[scene]
        eligible = [value for value in available if value <= int(target)]
        return max(eligible) if eligible else min(available)

    sys.path.insert(0, str(Path(config["wah_root"])))
    from warp_as_history.training import core as opt
    from warp_as_history import WarpAsHistoryPipeline

    training = config["training"]
    phase_b_mix = {key: float(training["phase_b_mix"][key]) for key in ("revisit", "large_motion", "corruption")}
    if any(value < 0 for value in phase_b_mix.values()) or not np.isclose(sum(phase_b_mix.values()), 1.0):
        raise ValueError(f"Phase B mix must be non-negative and sum to one: {phase_b_mix}")
    exact = opt.parse_args([])
    exact.base_model_path = str(config["wah_model"])
    exact.transformer_path = str(config["wah_model"])
    exact.height, exact.width = map(int, config["perspective_resolution"])
    exact.num_frames = int(config["chunk_frames"])
    exact.num_latent_frames_per_chunk = (exact.num_frames - 1) // int(config["vae_temporal_scale"]) + 1
    exact.history_sizes = [16, 2, 1]
    exact.history_temporal_layout = "long_mid_short"
    exact.pyramid_num_inference_steps_list = list(training["pyramid_num_inference_steps_list"])
    exact.attention_backend = "native"
    exact.use_warp_as_history = True
    exact.warp_history_downsample_mode = "short"
    exact.history_positioning = "last_n_same_order"
    exact.history_position_count = exact.num_latent_frames_per_chunk
    exact.history_position_delta = 0
    exact.history_visible_token_drop = True
    exact.visible_token_mode = "drop"
    exact.history_visible_token_threshold = 0.05
    exact.history_confidence_threshold = 0.1
    exact.history_confidence_lambda = 1.0
    exact.history_confidence_epsilon = 1e-6
    exact.add_noise_to_video_latents = False
    exact.add_noise_to_image_latents = False
    exact.flow_matching_mode = "train_exact"
    exact.flow_matching_stage_sampling = "fixed"
    exact.flow_matching_stage_id = int(training["flow_matching_stage_id"])
    exact.flow_matching_train_exact_timestep_sampling = "training_density"
    exact.flow_matching_use_dynamic_shifting = "off"
    exact.weighting_scheme = "none"
    exact.seed = int(config["seed"])
    exact.lora_rank = int(training["lora_rank"])
    exact.lora_alpha = int(training["lora_alpha"])
    exact.lora_dropout = float(training["lora_dropout"])
    exact.lora_target_modules = str(training["lora_target_modules"])
    exact.lora_adapter_name = "oracle_wah_24fps"
    exact.iters = 1200
    exact.gradient_checkpointing = bool(training["gradient_checkpointing"])
    opt.validate_args(exact)

    random.seed(exact.seed)
    np.random.seed(exact.seed)
    opt.seed_global_rng(exact.seed)
    device = torch.device("cuda:0")
    started = time.perf_counter()
    git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    pipe = opt.load_pipeline(exact, device)
    if not isinstance(pipe, WarpAsHistoryPipeline):
        # Keep the exact training transformer/VAE/scheduler loaded by
        # train_exact, but use the same autoregressive state implementation as
        # formal inference. WarpAsHistoryPipeline is a state-compatible
        # HeliosPipeline subclass and adds no model parameters.
        pipe.__class__ = WarpAsHistoryPipeline
    required_ar_methods = (
        "init_autoregressive_state", "generate_next_chunk",
        "_prepare_autoregressive_warp_chunk", "_build_pyramid_base_histories",
    )
    missing_ar_methods = [name for name in required_ar_methods if not callable(getattr(pipe, name, None))]
    if missing_ar_methods:
        raise TypeError(f"training pipeline is missing formal WAH AR methods: {missing_ar_methods}")

    mean, std = opt.latent_stats(pipe, device)
    adapter_name, lora_params, lora_stats = opt.setup_visible_lora(
        pipe.transformer, exact, "oracle_wah_24fps"
    )
    source_info = load_source_trainable_state(config["source_checkpoint"], pipe.transformer)
    controller = install_spatial_reanchor(
        pipe.transformer,
        rank=int(training["anchor_rank"]),
        refresh_blocks=tuple(training["anchor_blocks"]),
        gate_init=float(training["gate_init"]),
        spatial_rank=8,
    ).to(device)
    phase_b_v2 = args.mode in ("phase-b-v2-smoke", "phase-b-v2-train")
    base_v2_info = None
    if phase_b_v2:
        if not args.resume:
            raise ValueError("Phase-B v2 requires --resume pointing to checkpoint_step600_phaseA.pt")
        base_v2_info = load_phase_b_v2_base_checkpoint(args.resume, pipe.transformer)
    spatial_v2_prefixes = (
        "spatial_reanchor.spatial_k_lora.",
        "spatial_reanchor.spatial_v_lora.",
    )
    spatial_v2_exact = {
        "spatial_reanchor.spatial_memory_role", "spatial_reanchor.spatial_gate",
    }
    named_parameters = dict(pipe.transformer.named_parameters())
    spatial_v2_names = sorted(
        name for name in named_parameters
        if name.startswith(spatial_v2_prefixes) or name in spatial_v2_exact
    )
    spatial_v2_params = [named_parameters[name] for name in spatial_v2_names]
    if phase_b_v2:
        trainable = spatial_v2_params
        new_params = spatial_v2_params
    else:
        new_params = [
            parameter for name, parameter in controller.named_parameters()
            if not name.startswith(("spatial_k_lora.", "spatial_v_lora."))
            and name not in {"spatial_memory_role", "spatial_gate"}
        ]
        trainable = list(lora_params) + new_params
    trainable_ids = {id(item) for item in trainable}
    for name, parameter in pipe.transformer.named_parameters():
        parameter.requires_grad_(id(parameter) in trainable_ids)
    optimizer_groups = ([{
        "params": spatial_v2_params, "lr": float(training["new_module_learning_rate"]),
        "name": "spatial_memory_attention_v2",
    }] if phase_b_v2 else [
        {"params": lora_params, "lr": float(training["lora_learning_rate"]), "name": "wah_lora"},
        {"params": new_params, "lr": float(training["new_module_learning_rate"]), "name": "spatial_new"},
    ])
    optimizer = torch.optim.AdamW(optimizer_groups, weight_decay=float(training["weight_decay"]))
    warmup = max(1, int(training["warmup_steps"]))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: min(1.0, (step + 1) / warmup)
    )
    global_step = 600 if phase_b_v2 else 0
    phase_step = 0
    round_robin = RoundRobinChunkScheduler()
    phase = "B" if phase_b_v2 else "A"
    restored = None
    restored_rng_state = None
    if phase_b_v2 and args.phase_b_v2_resume:
        restored = load_phase_b_v2_checkpoint(
            args.phase_b_v2_resume, pipe.transformer, optimizer, scheduler,
        )
        global_step = restored["global_step"]
        phase, phase_step = restored["phase"], restored["phase_step"]
        round_robin.restore(restored.get("metadata", {}).get("round_robin"))
        restored_rng_state = {
            "python": random.getstate(), "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state(), "torch_cuda": torch.cuda.get_rng_state_all(),
        }
    elif args.resume and not phase_b_v2:
        restored = load_spatial_training_checkpoint(
            args.resume, pipe.transformer, optimizer, scheduler
        )
        global_step = restored["global_step"]
        phase, phase_step = restored["phase"], restored["phase_step"]
        restored_rng_state = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all(),
        }
        restored_trainable = {
            name for name, parameter in pipe.transformer.named_parameters()
            if parameter.requires_grad
        }
        restored_lora = {name for name in restored_trainable if ".lora_" in name}
        restored_spatial = {name for name in restored_trainable if name.startswith("spatial_reanchor.")}
        if len(restored_lora) != 320 or len(restored_spatial) != 17:
            raise RuntimeError(
                "step600 resume must restore exactly 320 LoRA and 17 Spatial tensors; "
                f"got LoRA={len(restored_lora)} Spatial={len(restored_spatial)}"
            )
        # New Phase B checkpoints carry the scheduler cursor/counts.  The
        # original step-600 checkpoint predates this metadata and intentionally
        # starts from all-zero counts.
        round_robin.restore(restored.get("metadata", {}).get("round_robin"))
    if args.mode in ("phase-b-smoke", "phase-b-v2-smoke"):
        if not args.resume or global_step != int(training["phase_a_steps"]):
            raise ValueError("phase-b-smoke requires the completed step600 Phase A checkpoint")
        if not 2 <= int(args.phase_b_smoke_steps) <= 5:
            raise ValueError("phase-b-smoke must run 2 to 5 optimizer steps")

    def encode_warp(frames):
        with torch.no_grad():
            return opt.encode_video_latents(pipe, frames, exact, device, mean, std).detach()

    def make_history(first, prompt, warp, visibility, confidence, sequence_id):
        exact.history_visibility_extra_mask_frames = _mask_frames(visibility)
        exact.history_confidence_extra_mask_frames = _mask_frames(confidence)
        prompt_embeds, image_latents, fake_image_latents, video_latents = opt.prepare_condition(
            pipe, first, prompt, exact, device, mean, std, history_frames=warp
        )
        histories = opt.make_histories(
            pipe, image_latents, fake_image_latents, exact, device,
            video_latents=video_latents, seq=sequence_id,
        )
        return prompt_embeds, histories

    def sample_arrays(record, chunk_index=0, *, causal_renderer=None):
        root = Path(record["path"])
        start = int(chunk_index) * int(config["chunk_stride"])
        target = _frames(root / "target" / "target_rgb_for_loss", start, exact.num_frames)
        poses = np.load(root / "target" / "target_c2w_local.npy")[start:start + exact.num_frames]
        intrinsics = np.load(root / "target" / "intrinsics.npy")[start:start + exact.num_frames]
        prompt = (root / "prompt.txt").read_text(encoding="utf-8")
        warp_provenance = None
        if causal_renderer is not None:
            cameras = CameraBatch(poses, intrinsics, exact.height, exact.width)
            causal = causal_renderer.render(cameras, frame_start=start)
            rendered = causal.warp
            warp = [Image.fromarray(np.asarray(frame).astype(np.uint8)) for frame in rendered.rgb]
            visibility = np.asarray(rendered.visibility, np.float32)
            confidence = np.asarray(rendered.confidence, np.float32)
            depth = np.asarray(rendered.depth, np.float32)
            warp_provenance = causal.provenance
        elif chunk_index == 0:
            warp = _frames(root / "single_chunk_warp" / "warp_rgb")
            visibility = np.load(root / "single_chunk_warp" / "warp_visibility.npy")
            confidence = np.load(root / "single_chunk_warp" / "warp_confidence.npy")
            depth = np.load(root / "single_chunk_warp" / "warp_z_depth.npy")
        else:
            raise RuntimeError("Phase B warp must be rendered by the causal active-node renderer")
        metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
        return {
            "root": root, "target": target, "poses": poses, "intrinsics": intrinsics,
            "prompt": prompt, "warp": warp, "visibility": visibility,
            "confidence": confidence, "metadata": metadata, "start": start,
            "depth": depth,
            "warp_provenance": warp_provenance,
        }

    spatial_geometry_cache = {}

    def spatial_conditioning(
        sample, warp_latents, stage_latents, *, anchor=True, camera=True, spatial_warp=True,
        memory_warp_latents=None, memory_visibility=None, memory_confidence=None,
    ):
        patch = tuple(int(value) for value in pipe.transformer.config.patch_size)
        if int(stage_latents.shape[2]) != int(warp_latents.shape[2]):
            raise ValueError(
                f"stage target/anchor latent T mismatch: {stage_latents.shape[2]} vs {warp_latents.shape[2]}"
            )
        latent_h, latent_w = map(int, stage_latents.shape[-2:])
        stage_warp = resize_latents_spatial(
            warp_latents, height=latent_h, width=latent_w,
        )
        token_h = latent_h // patch[-2]
        token_w = latent_w // patch[-1]
        if latent_h != token_h * patch[-2] or latent_w != token_w * patch[-1]:
            raise ValueError(
                f"stage latent {(latent_h, latent_w)} is not divisible by patch {patch[-2:]}"
            )
        scene_scale = float(sample["metadata"].get("scene_scale", 1.0))
        geometry_key = (
            str(sample["root"]), int(sample["start"]),
            str(sample.get("spatial_geometry_variant", "canonical")),
            int(stage_warp.shape[2]), latent_h, latent_w, tuple(patch), scene_scale,
        )
        geometry = spatial_geometry_cache.get(geometry_key)
        if geometry is None:
            visibility = visibility_to_target_tokens(
                sample["visibility"], latent_frames=stage_warp.shape[2],
                latent_height=latent_h, latent_width=latent_w,
                patch_height=patch[-2], patch_width=patch[-1],
                temporal_scale=int(config["vae_temporal_scale"]),
            )
            rays = plucker_camera_rays(
                sample["poses"], sample["intrinsics"],
                image_height=exact.height, image_width=exact.width,
                token_height=token_h, token_width=token_w,
                latent_frames=stage_warp.shape[2],
                temporal_scale=int(config["vae_temporal_scale"]),
                scene_scale=scene_scale,
                sequence_frame_start=int(sample["start"]),
                validate_sequence_source_origin=False,
            )
            geometry = (visibility, rays)
            spatial_geometry_cache[geometry_key] = geometry
        visibility, rays = geometry
        result = {
            "warp_latents": stage_warp,
            "visibility_tokens": visibility,
            "warp_confidence_tokens": visibility_to_target_tokens(
                np.asarray(sample["confidence"], np.float32) * np.asarray(sample["visibility"], np.float32),
                latent_frames=stage_warp.shape[2], latent_height=latent_h, latent_width=latent_w,
                patch_height=patch[-2], patch_width=patch[-1],
                temporal_scale=int(config["vae_temporal_scale"]),
            ),
            "plucker_tokens": rays,
            "anchor_enabled": anchor,
            "camera_enabled": camera,
            "spatial_warp_enabled": spatial_warp,
        }
        if memory_warp_latents is not None and int(token_h * token_w * stage_warp.shape[2]) == 540:
            memory_stage = resize_latents_spatial(
                memory_warp_latents, height=latent_h, width=latent_w,
            )
            result.update({
                "memory_warp_latents": memory_stage,
                "memory_visibility_tokens": visibility_to_target_tokens(
                    memory_visibility, latent_frames=memory_stage.shape[2],
                    latent_height=latent_h, latent_width=latent_w,
                    patch_height=patch[-2], patch_width=patch[-1],
                    temporal_scale=int(config["vae_temporal_scale"]),
                ),
                "memory_confidence_tokens": visibility_to_target_tokens(
                    np.asarray(memory_confidence, np.float32) * np.asarray(memory_visibility, np.float32),
                    latent_frames=memory_stage.shape[2], latent_height=latent_h, latent_width=latent_w,
                    patch_height=patch[-2], patch_width=patch[-1],
                    temporal_scale=int(config["vae_temporal_scale"]),
                ),
                "spatial_attention_enabled": True,
            })
        return result

    def pyramid_spatial_conditioning(
        sample, warp_latents, stage_latents_list, *, anchor=True, camera=True, spatial_warp=True,
        memory_warp_latents=None, memory_visibility=None, memory_confidence=None,
    ):
        contexts = [
            spatial_conditioning(
                sample, warp_latents, stage_latents, anchor=anchor,
                camera=camera, spatial_warp=spatial_warp,
                memory_warp_latents=memory_warp_latents,
                memory_visibility=memory_visibility,
                memory_confidence=memory_confidence,
            )
            for stage_latents in stage_latents_list
        ]
        return {
            "stage_contexts": [
                {
                    "warp_latents": context["warp_latents"],
                    "visibility_tokens": context["visibility_tokens"],
                    "plucker_tokens": context["plucker_tokens"],
                    **({
                        "warp_confidence_tokens": context["warp_confidence_tokens"],
                        "memory_warp_latents": context["memory_warp_latents"],
                        "memory_visibility_tokens": context["memory_visibility_tokens"],
                        "memory_confidence_tokens": context["memory_confidence_tokens"],
                        "spatial_attention_enabled": True,
                    } if "memory_warp_latents" in context else {
                        "warp_confidence_tokens": context["warp_confidence_tokens"],
                    }),
                }
                for context in contexts
            ],
            "anchor_enabled": anchor,
            "camera_enabled": camera,
            "spatial_warp_enabled": spatial_warp,
        }

    encoded_a = {}
    def phase_a_item(record):
        key = record["sequence_id"]
        if key not in encoded_a:
            sample = sample_arrays(record, 0)
            with torch.no_grad():
                target_latents = opt.encode_video_latents(
                    pipe, sample["target"], exact, device, mean, std
                ).detach()
            prompt, histories = make_history(
                sample["target"][0], sample["prompt"], sample["warp"],
                sample["visibility"], sample["confidence"], key,
            )
            empty_warp = [Image.new("RGB", (exact.width, exact.height)) for _ in sample["warp"]]
            empty_visibility = np.zeros_like(sample["visibility"])
            empty_confidence = np.zeros_like(sample["confidence"])
            _, empty_histories = make_history(
                sample["target"][0], sample["prompt"], empty_warp,
                empty_visibility, empty_confidence, key + "_camera_only",
            )
            warp_latents = encode_warp(sample["warp"])
            weights = np.load(sample["root"] / "primary_loss_weight_latent.npy")[:target_latents.shape[2]]
            encoded_a[key] = {
                "sample": sample,
                "target": _tree_to(target_latents, "cpu"),
                "prompt": _tree_to(prompt, "cpu"),
                "histories": _tree_to(histories, "cpu"),
                "empty_histories": _tree_to(empty_histories, "cpu"),
                "warp_latents": _tree_to(warp_latents, "cpu"),
                "weights": weights,
            }
            del target_latents, prompt, histories, empty_histories, warp_latents
        item = encoded_a[key]
        return {
            **item,
            "target": _tree_to(item["target"], device),
            "prompt": _tree_to(item["prompt"], device),
            "histories": _tree_to(item["histories"], device),
            "empty_histories": _tree_to(item["empty_histories"], device),
            "warp_latents": _tree_to(item["warp_latents"], device),
        }

    fixed = phase_a_item(phase_a_diag[0])
    if not args.resume:
        opt.seed_global_rng(exact.seed)
    fixed_stage_items = opt.flow_matching_train_exact_items(pipe, fixed["target"], exact, device)
    if len(fixed_stage_items) != 1:
        raise ValueError("the restored formal WAH fixed-stage strategy must sample exactly one stage")
    fixed_stage = fixed_stage_items[0]
    fixed_condition = spatial_conditioning(
        fixed["sample"], fixed["warp_latents"], fixed_stage["noisy_latents"],
    )
    pyramid_latents = opt.training_exact_pyramid_latents(
        fixed["target"], len(exact.pyramid_num_inference_steps_list),
    )
    pyramid_conditions = [
        spatial_conditioning(fixed["sample"], fixed["warp_latents"], stage_latents)
        for stage_latents in pyramid_latents
    ]
    startup_report = {
        "git_sha": git_sha,
        "source_checkpoint": str(config["source_checkpoint"]),
        "source_checkpoint_sha256": _sha256(config["source_checkpoint"]),
        "manifest": str(manifest_path),
        "manifest_sha256": _sha256(manifest_path),
        "scenes": scenes,
        "phase_a_scenes": phase_a_scenes,
        "selected_phase_b_trajectories": [
            {
                "sequence_id": item["sequence_id"], "scene_id": item["scene_id"],
                "sample_type": item["sample_type"], "chunk_count": item["chunk_count"],
                "training_chunk_index": item.get("training_chunk_index"),
                "revisit": item["metadata"].get("revisit"),
                "phase_b_selection": item["metadata"].get("phase_b_selection"),
            }
            for item in phase_b
        ],
        "phase_b_chunk_distribution": {
            str(count): sum(int(item["chunk_count"]) == count for item in phase_b)
            for count in sorted({int(item["chunk_count"]) for item in phase_b})
        },
        "available_phase_b_chunks_by_scene": available_phase_b_chunks,
        "phase_b_node_mode": "M0-only",
        "phase_b_uses_future_gt": False,
        "phase_b_current_chunk_policy": {
            "eligible": "1..N-1",
            "scheduler": "deterministic_round_robin_per_trajectory",
            "shared_boundary_weight": 0.0,
        },
        "phase_b_spatial_memory_warp": {
            "translation_threshold": 3.0,
            "rotation_threshold_degrees": 30.0,
            "hit_requires": "translation<=3 AND full_rotation_angle<=30",
            "rgb_origin": "previous_model_generation_boundary",
            "geometry_origin": "same_time_causal_M0_renderer",
            "uses_future_gt": False,
        },
        "source_prefix_fixed": True,
        "source_prefix_rope": 0,
        "spatial_kv_layout": {"W": 540, "R": 540, "total": 1080},
        "phase_b_v2_base_checkpoint": args.resume if phase_b_v2 else None,
        "phase_b_v2_base_load": base_v2_info,
        "phase_b_mix": phase_b_mix,
        "scene_scale": {scene: {"value": 1.0, "source": "Holo360D_dataset_calibrated_metric"} for scene in scenes},
        "trainable_parameters": {
            "lora": int(sum(item.numel() for item in lora_params)),
            "new_modules": int(sum(item.numel() for item in new_params)),
            "total": int(sum(item.numel() for item in trainable)),
        },
        "trainable_parameter_names": spatial_v2_names if phase_b_v2 else [
            name for name, parameter in pipe.transformer.named_parameters() if parameter.requires_grad
        ],
        "trainable_tensor_count": len(trainable),
        "frozen_step600_wah_lora_tensor_count": 320 if phase_b_v2 else 0,
        "frozen_step600_spatial_camera_tensor_count": 17 if phase_b_v2 else 0,
        "target_latent_shape": list(fixed["target"].shape),
        "full_warp_latent_shape": list(fixed["warp_latents"].shape),
        "actual_stage_target_shape": list(fixed_stage["noisy_latents"].shape),
        "actual_stage_anchor_shape": list(fixed_condition["warp_latents"].shape),
        "target_token_shape": list(fixed_condition["visibility_tokens"].shape[:-1])
        + [int(pipe.transformer.patch_embedding.out_channels)],
        "visibility_token_shape": list(fixed_condition["visibility_tokens"].shape),
        "plucker_shape": list(fixed_condition["plucker_tokens"].shape),
        "plucker_norm_mean": float(fixed_condition["plucker_tokens"].float().norm(dim=-1).mean()),
        "indices_hidden_states": fixed["histories"]["indices_hidden_states"].detach().cpu().tolist(),
        "indices_spatial_warp": fixed["histories"]["indices_latents_history_short"][0, -9:].detach().cpu().tolist(),
        "warp_target_rope_aligned": bool(torch.equal(
            fixed["histories"]["indices_hidden_states"][0],
            fixed["histories"]["indices_latents_history_short"][0, -9:],
        )),
        "wah_patch": "patch_short",
        "anchor_patch": "frozen_target_patch_embedding",
        "spatial_warp_role_independent": True,
        "temporal_history_weights": {"TEMP_SHORT": 1.0, "TEMP_MID": 1.0, "TEMP_LONG": 1.0},
        "flow_matching_stage_id": exact.flow_matching_stage_id,
        "pyramid_stage_shapes": [
            {
                "stage_id": stage_id,
                "target_latent": list(stage_latents.shape),
                "anchor_latent": list(condition["warp_latents"].shape),
                "visibility_tokens": list(condition["visibility_tokens"].shape),
                "plucker": list(condition["plucker_tokens"].shape),
                "token_count": int(condition["visibility_tokens"].shape[1]),
            }
            for stage_id, (stage_latents, condition) in enumerate(
                zip(pyramid_latents, pyramid_conditions)
            )
        ],
        "anchor_blocks": list(controller.refresh_blocks),
        "initial_anchor_gates": controller.anchor_gates.detach().cpu().tolist(),
        "initial_camera_gate": float(controller.camera_gate.detach().cpu()),
        "initial_spatial_gates": controller.spatial_gate.detach().cpu().tolist(),
        "gpu_name": torch.cuda.get_device_name(0),
        "gpu_uuid": str(torch.cuda.get_device_properties(0).uuid),
        "gpu_processes": _gpu_snapshot(),
    }
    if startup_report["full_warp_latent_shape"] != [1, 16, 9, 48, 80]:
        raise ValueError(
            f"unexpected current full warp latent shape: {startup_report['full_warp_latent_shape']}"
        )
    expected_stage_shapes = [
        ([1, 16, 9, 12, 20], 540),
        ([1, 16, 9, 24, 40], 2160),
        ([1, 16, 9, 48, 80], 8640),
    ]
    observed_stage_shapes = [
        (entry["target_latent"], entry["token_count"])
        for entry in startup_report["pyramid_stage_shapes"]
    ]
    if observed_stage_shapes != expected_stage_shapes:
        raise ValueError(f"unexpected train_exact pyramid shapes: {observed_stage_shapes}")
    if not startup_report["warp_target_rope_aligned"]:
        raise ValueError("SPATIAL_WARP and target RoPE indices are not aligned")
    if phase_b_v2:
        expected_v2_tensors = 40 * 4 + 2
        if len(spatial_v2_names) != expected_v2_tensors:
            raise RuntimeError(
                f"Phase-B v2 expects {expected_v2_tensors} trainable tensors, got {len(spatial_v2_names)}"
            )
        if bool((controller.spatial_gate.detach() != 0).any()):
            raise RuntimeError("all 40 spatial gates must initialize strictly to zero")
        frozen_lora = [
            name for name, parameter in pipe.transformer.named_parameters()
            if ".lora_" in name and parameter.requires_grad
        ]
        frozen_old_spatial = [
            name for name, parameter in pipe.transformer.named_parameters()
            if name in set(base_v2_info["loaded_spatial_camera_names"]) and parameter.requires_grad
        ]
        if frozen_lora or frozen_old_spatial:
            raise RuntimeError("step600 WAH LoRA and Spatial Anchor/Camera must remain frozen")
    _atomic_json(run_dir / "startup_report.json", startup_report)

    if restored_rng_state is not None:
        # Startup diagnostics construct fixed tensors and may consume global
        # randomness. Resume must begin Phase B from the checkpoint RNG state.
        random.setstate(restored_rng_state["python"])
        np.random.set_state(restored_rng_state["numpy"])
        torch.set_rng_state(restored_rng_state["torch_cpu"])
        torch.cuda.set_rng_state_all(restored_rng_state["torch_cuda"])

    def loss_for(item, *, histories=None, anchor=True, camera=True, spatial_warp=True, fixed_items=None):
        if "current_chunk_index" in item:
            raw_weights = np.asarray(item["weights"])
            validate_current_chunk_supervision(raw_weights, int(item["target"].shape[2]))
        def condition(stage_item):
            return spatial_conditioning(
                item["sample"], item["warp_latents"], stage_item["noisy_latents"],
                anchor=anchor, camera=camera, spatial_warp=spatial_warp,
                memory_warp_latents=item.get("memory_warp_latents"),
                memory_visibility=item.get("memory_visibility"),
                memory_confidence=item.get("memory_confidence"),
            )
        return masked_flow_matching_loss(
            pipe, item["prompt"], item["target"], histories or item["histories"],
            exact, device, item["weights"], fixed_stage_items=fixed_items,
            spatial_conditioning=condition,
        )[0]

    if phase_b_v2:
        compatibility_item = dict(fixed)
        compatibility_item.update({
            "memory_warp_latents": torch.zeros_like(fixed["warp_latents"]),
            "memory_visibility": np.zeros_like(fixed["sample"]["visibility"], dtype=bool),
            "memory_confidence": np.zeros_like(fixed["sample"]["confidence"], dtype=np.float32),
        })
        pipe.transformer.eval()
        with torch.no_grad():
            baseline_loss = loss_for(fixed, fixed_items=fixed_stage_items)
            gated_loss = loss_for(compatibility_item, fixed_items=fixed_stage_items)
        pipe.transformer.train()
        if not torch.equal(baseline_loss, gated_loss):
            raise RuntimeError(
                f"zero-gate spatial branch changed step600 behavior: {baseline_loss} vs {gated_loss}"
            )
        startup_report["step600_zero_gate_numerical_reproduction"] = {
            "exact": True, "baseline_loss": float(baseline_loss.cpu()),
            "spatial_branch_loss": float(gated_loss.cpu()),
        }
        _atomic_json(run_dir / "startup_report.json", startup_report)
        if restored_rng_state is None:
            random.seed(exact.seed)
            np.random.seed(exact.seed)
            opt.seed_global_rng(exact.seed)
        else:
            random.setstate(restored_rng_state["python"])
            np.random.set_state(restored_rng_state["numpy"])
            torch.set_rng_state(restored_rng_state["torch_cpu"])
            torch.cuda.set_rng_state_all(restored_rng_state["torch_cuda"])

    def diagnostic():
        pipe.transformer.eval()
        values = {}
        with torch.no_grad():
            values["correct"] = float(loss_for(fixed, fixed_items=fixed_stage_items).cpu())
            values["empty"] = float(loss_for(
                fixed, histories=fixed["empty_histories"], anchor=False,
                spatial_warp=False, fixed_items=fixed_stage_items,
            ).cpu())
            shuffled = dict(fixed)
            order = np.random.default_rng(exact.seed).permutation(exact.num_frames)
            shuffled_sample = dict(fixed["sample"])
            shuffled_sample["spatial_geometry_variant"] = (
                "shuffled_" + hashlib.sha256(order.tobytes()).hexdigest()
            )
            shuffled_sample["visibility"] = fixed["sample"]["visibility"][order]
            shuffled_sample["warp"] = [fixed["sample"]["warp"][index] for index in order]
            shuffled["sample"] = shuffled_sample
            shuffled["warp_latents"] = encode_warp(shuffled_sample["warp"])
            _, shuffled_histories = make_history(
                shuffled_sample["target"][0], shuffled_sample["prompt"],
                shuffled_sample["warp"], shuffled_sample["visibility"],
                fixed["sample"]["confidence"][order], "fixed_shuffled",
            )
            values["shuffled"] = float(loss_for(
                shuffled, histories=shuffled_histories, fixed_items=fixed_stage_items,
            ).cpu())
            values["anchor_on"] = values["correct"]
            values["anchor_off"] = float(loss_for(
                fixed, anchor=False, fixed_items=fixed_stage_items,
            ).cpu())
            values["camera_on"] = values["correct"]
            values["camera_off"] = float(loss_for(
                fixed, camera=False, fixed_items=fixed_stage_items,
            ).cpu())
        pipe.transformer.train()
        values["anchor_gates"] = controller.anchor_gates.detach().cpu().tolist()
        values["camera_gate"] = float(controller.camera_gate.detach().cpu())
        values["utilization"] = controller.metrics_snapshot()
        return values

    # Phase B cache entries are indexed by scene/sample/trajectory/current
    # chunk.  The scheduler state is checkpointed so resume preserves exact
    # round-robin coverage instead of silently starting at chunk one again.
    history_bank = {}
    history_bank_cache = {}
    bank_step = -1
    bank_stats = []
    memory_cleanup_stats = {"calls": 0, "reasons": [], "entries": 0}
    persistent_bank_dir = Path(
        config.get("history_bank_cache_dir") or (run_dir / "persistent_history_bank")
    )
    persistent_bank_dir.mkdir(parents=True, exist_ok=True)

    def controller_state_sha():
        digest = hashlib.sha256()
        for name, value in sorted(controller.state_dict().items()):
            tensor = value.detach().cpu().contiguous()
            digest.update(name.encode("utf-8"))
            digest.update(str(tuple(tensor.shape)).encode("ascii"))
            digest.update(str(tensor.dtype).encode("ascii"))
            digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        return digest.hexdigest()

    def maybe_cleanup_after_bank_entry():
        allocated = int(torch.cuda.memory_allocated(0))
        reserved = int(torch.cuda.memory_reserved(0))
        total = int(torch.cuda.get_device_properties(0).total_memory)
        reason = None
        if reserved > int(0.90 * total):
            reason = "reserved>0.90*total"
        elif reserved - allocated > 8 * 1024**3:
            reason = "allocator_fragmentation_gap>8GiB"
        memory_cleanup_stats["entries"] += 1
        if reason is not None:
            torch.cuda.empty_cache()
            memory_cleanup_stats["calls"] += 1
            memory_cleanup_stats["reasons"].append({
                "global_step": global_step, "reason": reason,
                "allocated": allocated, "reserved": reserved,
            })

    def lora_snapshot():
        folder = run_dir / "history_bank_lora"
        folder.mkdir(exist_ok=True)
        opt.save_visible_lora_state(
            pipe.transformer, folder, adapter_name, "current_lora.pt"
        )
        return folder / "current_lora.pt"

    def reset_runtime_wah_adapter():
        pipe._unfuse_wah_lora()
        pipe._delete_wah_adapter()
        pipe._wah_loaded_lora_path = None
        pipe.transformer.set_adapter(adapter_name)
        for name, parameter in pipe.transformer.named_parameters():
            parameter.requires_grad_(id(parameter) in trainable_ids)

    min_temporal_gap_frames = (19 - 1) * int(config["vae_temporal_scale"])
    if min_temporal_gap_frames != 72:
        raise RuntimeError(
            f"formal spatial-memory temporal gap must be 72 frames, got {min_temporal_gap_frames}"
        )

    def build_bank_entries(record, snapshot, snapshot_sha, *, corrupt_generated_history=False):
        """Build every current-chunk entry from one causal AR rollout.

        The old implementation restarted the model at chunk zero for every
        selector, making a 16-chunk bank an O(N^2) rollout.  We now retain a
        CPU snapshot at each current boundary while traversing the trajectory
        once; each entry still contains only its own prefix history.
        """
        chunk_count = int(record["chunk_count"])
        eligible = eligible_current_chunks(chunk_count)
        root = Path(record["path"])
        source = Image.open(root / "source" / "source_perspective.png").convert("RGB")
        prompt = (root / "prompt.txt").read_text(encoding="utf-8")
        generator = torch.Generator(device=device).manual_seed(exact.seed)
        init_kwargs = {
            "prompt": prompt, "image": source, "conditioning_type": "warp",
            "lora_path": str(snapshot), "lora_prompt_trigger": "camctl23x.",
            "visible_token_drop": True, "warp_history_downsample_mode": "short",
            "rope_alignment": True, "height": exact.height, "width": exact.width,
            "num_frames": exact.num_frames, "output_type": "np", "generator": generator,
            "add_noise_to_image_latents": False, "add_noise_to_warp_latents": True,
            "pyramid_num_inference_steps_list": exact.pyramid_num_inference_steps_list,
            "is_amplify_first_chunk": False, "prev_chunk_history_sizes": (16, 2, 1),
        }
        signature = inspect.signature(pipe.init_autoregressive_state)
        if "temp_short_acceptance_scale" in signature.parameters:
            init_kwargs["temp_short_acceptance_scale"] = 1.0
        state = pipe.init_autoregressive_state(**init_kwargs)
        if tuple(int(value) for value in state["history_sizes"]) != (16, 2, 1):
            raise RuntimeError(f"formal AR history sizes changed: {state['history_sizes']}")
        if int(state["num_history_latent_frames"]) != 19:
            raise RuntimeError("formal AR history must retain all 19 TEMP_LONG/MID/SHORT latents")
        state_min_temporal_gap_frames = (
            (int(state["num_history_latent_frames"]) - 1)
            * int(config["vae_temporal_scale"])
        )
        if state_min_temporal_gap_frames != min_temporal_gap_frames:
            raise RuntimeError(
                "formal spatial-memory temporal gap does not match the actual AR state: "
                f"expected {min_temporal_gap_frames}, got {state_min_temporal_gap_frames}"
            )
        causal_renderer = CausalActiveNodeRenderer(
            NodeStore(root / "session"),
            renderer_kwargs={"device": "cuda:0", "near": 0.05, "far": 100.0,
                             "point_radius": 1, "chunk_points": 1000000},
        )
        fixed_source_prefix = state["image_latents"].detach().clone()
        if int(state["indices_latents_history_short"][0, 0]) != 0:
            raise RuntimeError("the permanent source prefix temporal RoPE must be zero")
        memory_bank = SpatialMemoryWarpBank(
            translation_threshold=3.0, rotation_threshold_degrees=30.0,
        )
        sample_cache = {}

        def _sample(chunk_index):
            chunk_index = int(chunk_index)
            if chunk_index not in sample_cache:
                sample_cache[chunk_index] = sample_arrays(
                    record, chunk_index, causal_renderer=causal_renderer,
                )
            return sample_cache[chunk_index]

        memory_reports = []
        warp_provenance = []
        boundary_snapshots = {}
        corruption_generator = torch.Generator(device=device).manual_seed(
            exact.seed + global_step + 1009
        )
        def _snapshot_state(value, key=None):
            # Avoid retaining full encoded-warp/video caches at every boundary.
            if key in {
                "warp_latents_tensor", "online_warp_video_tensor", "online_visibility_mask",
                "history_video", "real_history_latents", "last_output",
            }:
                return None
            if isinstance(value, torch.Tensor):
                return value.detach().cpu().clone()
            if isinstance(value, torch.Generator):
                return {"__torch_generator_state__": value.get_state().cpu().clone()}
            if isinstance(value, np.ndarray):
                return value.copy()
            if isinstance(value, dict):
                return {
                    name: item for name, child in value.items()
                    if (item := _snapshot_state(child, name)) is not None
                }
            if isinstance(value, list):
                return [_snapshot_state(child) for child in value]
            if isinstance(value, tuple):
                return tuple(_snapshot_state(child) for child in value)
            return value

        def _restore_state(value):
            if isinstance(value, dict) and set(value) == {"__torch_generator_state__"}:
                generator = torch.Generator(device=device)
                generator.set_state(value["__torch_generator_state__"].detach().cpu())
                return generator
            if isinstance(value, torch.Tensor):
                return value.to(device=device)
            if isinstance(value, dict):
                return {name: _restore_state(child) for name, child in value.items()}
            if isinstance(value, list):
                return [_restore_state(child) for child in value]
            if isinstance(value, tuple):
                return tuple(_restore_state(child) for child in value)
            if isinstance(value, np.ndarray):
                return value.copy()
            return value

        def _memory_for_sample(sample):
            rendered = memory_bank.render_query(
                poses=sample["poses"], intrinsics=sample["intrinsics"],
                query_frame_id=int(sample["start"]),
                min_temporal_gap_frames=min_temporal_gap_frames,
                height=exact.height, width=exact.width, device="cuda:0",
                near=0.05, far=100.0, point_radius=1, chunk_points=1000000,
            )
            frames = [Image.fromarray(frame, mode="RGB") for frame in rendered["rgb"]]
            latent = encode_warp(frames)
            return latent, rendered

        def _last_generated_rgb(generated):
            value = np.asarray(generated)
            if value.ndim == 5:
                value = value[0]
            if value.ndim == 4 and value.shape[-1] == 3:
                frame = value[-1]
            elif value.ndim == 4 and value.shape[0] == 3:
                frame = np.moveaxis(value[:, -1], 0, -1)
            else:
                raise ValueError(f"unexpected generated video shape {value.shape}")
            if frame.dtype != np.uint8:
                frame = np.rint(np.clip(frame, 0, 1) * 255).astype(np.uint8)
            return frame

        for chunk in range(max(eligible)):
            sample = _sample(chunk)
            warp_provenance.append(sample["warp_provenance"])
            if corrupt_generated_history:
                generated_frames = max(0, int(state["total_generated_latent_frames"]) - 1)
                generated_frames = min(generated_frames, int(state["num_history_latent_frames"]))
                if generated_frames:
                    sigma = float(training["history_corruption_sigma"])
                    current = state["history_latents"][:, :, -generated_frames:]
                    noise = torch.randn(
                        current.shape, generator=corruption_generator,
                        device=current.device, dtype=current.dtype,
                    )
                    state["history_latents"][:, :, -generated_frames:] = (
                        (1.0 - sigma) * current + sigma * noise
                    )
            warp_latents = encode_warp(sample["warp"])
            if not torch.equal(state["image_latents"], fixed_source_prefix):
                raise RuntimeError("AR rollout replaced the permanent original source prefix")
            memory_warp_latents, memory_render = _memory_for_sample(sample)
            memory_report = dict(memory_render["report"])
            memory_report.update({
                "query_frame_id": int(sample["start"]), "query_chunk_id": int(chunk),
                "uses_future_gt": False,
            })
            memory_reports.append(memory_report)
            # This state is the exact no-grad AR prefix for this boundary.
            # Keep it before generating the current chunk and before adding a
            # generated memory candidate.
            boundary_snapshots[int(chunk)] = {
                "state": _snapshot_state(state),
                "sample": sample,
                "warp_latents": warp_latents.detach().cpu().clone(),
                "memory_warp_latents": memory_warp_latents.detach().cpu().clone(),
                "memory_visibility": np.asarray(memory_render["visibility"], bool),
                "memory_confidence": np.asarray(memory_render["confidence"], np.float16),
                "memory_report": memory_report,
                "warp_provenance": list(warp_provenance),
                "memory_reports": list(memory_reports),
                "memory_summary": memory_bank.summary(),
            }
            inference_pyramid = opt.training_exact_pyramid_latents(
                warp_latents, len(exact.pyramid_num_inference_steps_list),
            )
            controller.prepare_context(**pyramid_spatial_conditioning(
                sample, warp_latents, inference_pyramid,
                memory_warp_latents=memory_warp_latents,
                memory_visibility=memory_render["visibility"],
                memory_confidence=memory_render["confidence"],
            ))
            try:
                with torch.no_grad():
                    generated_video, state = pipe.generate_next_chunk(
                        state, warp_video=np.stack([np.asarray(item) for item in sample["warp"]]),
                        warp_visibility_mask=sample["visibility"][None, None],
                        warp_confidence_mask=(sample["confidence"] * sample["visibility"])[None, None],
                        output_type="np",
                    )
            finally:
                controller.clear_context()
            boundary_entry = memory_bank.add_generated_boundary(
                rgb=_last_generated_rgb(generated_video), depth=sample["depth"][-1],
                visibility=sample["visibility"][-1], confidence=sample["confidence"][-1],
                pose=sample["poses"][-1], intrinsics=sample["intrinsics"][-1],
                frame_id=int(sample["start"]) + 32, chunk_id=int(chunk),
                provenance=sample["warp_provenance"],
            )
            memory_reports.append({
                "created_entry_id": int(boundary_entry.entry_id),
                "created_frame_id": int(boundary_entry.frame_id),
                "point_count": int(len(boundary_entry.points_xyz)), "uses_future_gt": False,
            })
            del warp_latents, memory_warp_latents

        # The final eligible boundary follows chunk N-2's rollout.  Select its
        # Spatial Memory Prefix without generating the supervised chunk.
        current_chunk = int(max(eligible))
        sample = _sample(current_chunk)
        warp_provenance.append(sample["warp_provenance"])
        warp_latents = encode_warp(sample["warp"])
        if not torch.equal(state["image_latents"], fixed_source_prefix):
            raise RuntimeError("AR rollout replaced the permanent original source prefix")
        memory_warp_latents, memory_render = _memory_for_sample(sample)
        memory_report = dict(memory_render["report"])
        memory_report.update({
            "query_frame_id": int(sample["start"]), "query_chunk_id": int(current_chunk),
            "uses_future_gt": False,
        })
        memory_reports.append(memory_report)
        boundary_snapshots[current_chunk] = {
            "state": _snapshot_state(state), "sample": sample,
            "warp_latents": warp_latents.detach().cpu().clone(),
            "memory_warp_latents": memory_warp_latents.detach().cpu().clone(),
            "memory_visibility": np.asarray(memory_render["visibility"], bool),
            "memory_confidence": np.asarray(memory_render["confidence"], np.float16),
            "memory_report": memory_report,
            "warp_provenance": list(warp_provenance),
            "memory_reports": list(memory_reports),
            "memory_summary": memory_bank.summary(),
        }

        entries = {}
        all_weights = np.load(root / "primary_loss_weight_latent.npy")
        for current_chunk in eligible:
            snap = boundary_snapshots[int(current_chunk)]
            sample = snap["sample"]
            state = _restore_state(snap["state"])
            if not torch.equal(state["image_latents"], fixed_source_prefix):
                raise RuntimeError("supervised History Bank state replaced the original source prefix")
            if int(state["indices_latents_history_short"][0, 0]) != 0:
                raise RuntimeError("supervised source prefix temporal RoPE is not zero")
            generator = state.get("generator", generator)
            warp_latents = _tree_to(snap["warp_latents"], device)
            memory_warp_latents = _tree_to(snap["memory_warp_latents"], device)
            pipe._prepare_autoregressive_warp_chunk(
                state, np.stack([np.asarray(item) for item in sample["warp"]]),
                sample["visibility"][None, None],
                (sample["confidence"] * sample["visibility"])[None, None],
            )
            clean_history = state["history_latents"][:, :, -int(state["num_history_latent_frames"]):]
            _, _, short_history = clean_history.split(state["history_sizes"], dim=2)
            base_short = torch.cat([state["image_latents"], short_history], dim=2)
            histories = pipe._build_pyramid_base_histories(
                state, device, short_history.dtype, generator, base_short,
            )
            prompt_embeds = state.get("lora_prompt_embeds")
            if prompt_embeds is None:
                prompt_embeds = state["prompt_embeds"]
            with torch.no_grad():
                target_latents = opt.encode_video_latents(
                    pipe, sample["target"], exact, device, mean, std,
                ).detach()
            latent_start = int(current_chunk) * (exact.num_latent_frames_per_chunk - 1)
            weights = all_weights[latent_start:latent_start + exact.num_latent_frames_per_chunk]
            if len(weights) != exact.num_latent_frames_per_chunk:
                raise ValueError("current Phase B supervision weights are truncated")
            weights = weights.copy()
            weights[0] = 0.0
            supervised_latent_indices = validate_current_chunk_supervision(
                weights, int(target_latents.shape[2]),
            )
            bank_key = HistoryBankKey(
                checkpoint_sha=snapshot_sha, global_step=global_step,
                scene_id=record["scene_id"], source_id=record["sequence_id"],
                trajectory_id=record["sequence_id"], history_chunk_index=int(current_chunk),
                generation_config=(
                    ("pyramid_steps", tuple(exact.pyramid_num_inference_steps_list)),
                    ("history_sizes", tuple(exact.history_sizes)),
                    ("visible_token_drop", bool(exact.history_visible_token_drop)),
                    ("warp_downsample", str(exact.warp_history_downsample_mode)),
                    ("spatial_reanchor", True),
                    ("source_prefix_fixed", True),
                    ("source_prefix_rope", 0),
                    ("spatial_memory_warp_attention", True),
                    ("memory_translation_threshold", 3.0),
                    ("memory_rotation_threshold_degrees", 30.0),
                    ("memory_min_temporal_gap_frames", int(min_temporal_gap_frames)),
                    ("fixed_spatial_kv_tokens", (540, 540)),
                    ("history_corruption", bool(corrupt_generated_history)),
                    ("history_corruption_sigma", float(training["history_corruption_sigma"])),
                ),
                prompt=prompt, seed=exact.seed,
            )
            key_payload = dict(bank_key.__dict__)
            key = bank_key.digest()
            entry = {
                "key": key, "key_payload": key_payload, "record": {
                    **record, "training_chunk_index": int(current_chunk),
                },
                "sample": sample, "target": _tree_to(target_latents, "cpu"),
                "prompt": _tree_to(prompt_embeds, "cpu"),
                "histories": _tree_to(histories, "cpu"),
                "warp_latents": _tree_to(warp_latents, "cpu"), "weights": weights,
                "memory_warp_latents": _tree_to(memory_warp_latents, "cpu"),
                "memory_visibility": np.asarray(snap["memory_visibility"], bool),
                "memory_confidence": np.asarray(snap["memory_confidence"], np.float16),
                "supervised_latent_indices": supervised_latent_indices,
                "metadata": {
                    "uses_gt_future": False, "checkpoint_sha": snapshot_sha,
                    "global_step": global_step, "history_chunk_index": int(current_chunk),
                    "self_augmentation": bool(corrupt_generated_history),
                    "restoration_steps_per_pyramid_stage": tuple(exact.pyramid_num_inference_steps_list),
                    "node_mode": causal_renderer.node_mode,
                    "warp_provenance": snap["warp_provenance"],
                    "training_chunk_index": int(current_chunk),
                    "history_state_semantics": "formal_wah_autoregressive_v1",
                    "history_sizes": {
                        "TEMP_LONG": int(state["history_sizes"][0]),
                        "TEMP_MID": int(state["history_sizes"][1]),
                        "TEMP_SHORT": int(state["history_sizes"][2]),
                    },
                    "rollout_prefix_chunks": int(current_chunk),
                    "supervised_latent_indices": supervised_latent_indices,
                    "supervised_latent_count": len(supervised_latent_indices),
                    "spatial_memory_warp": {
                        "session_local": True, "translation_threshold": 3.0,
                        "rotation_threshold_degrees": 30.0,
                        "hit_requires": "translation<=3 AND full_rotation_angle<=30",
                        "entries": snap["memory_summary"], "reports": snap["memory_reports"],
                        "query": snap["memory_report"],
                        "rgb_origin": "model_generated",
                        "geometry_origin": "causal_M0_renderer",
                    },
                },
            }
            validate_history_bank_entry({
                "TEMP_LONG": entry["histories"].get("latents_history_long"),
                "TEMP_MID": entry["histories"].get("latents_history_mid"),
                "TEMP_SHORT": entry["histories"].get("latents_history_short"),
                "key": entry["key"], "metadata": entry["metadata"],
            })
            entries[int(current_chunk)] = entry
            del target_latents, histories, warp_latents, memory_warp_latents
        return entries

    def refresh_history_bank(chunk_plan):
        nonlocal history_bank, history_bank_cache, bank_step
        refresh_started = time.perf_counter()
        reset_runtime_wah_adapter()
        snapshot = lora_snapshot()
        snapshot_sha = _sha256(snapshot)
        selected = [
            item for item in phase_b
            if int(item["chunk_count"]) == int(chunk_plan[item["scene_id"]])
        ]
        cache_descriptor = {
            "schema_version": 1,
            "git_sha": git_sha,
            "global_step": int(global_step),
            "wah_lora_snapshot_sha": snapshot_sha,
            "controller_state_sha": controller_state_sha(),
            "wah_model": str(config["wah_model"]),
            "source_checkpoint": str(config["source_checkpoint"]),
            "seed": int(exact.seed),
            "pyramid_steps": list(exact.pyramid_num_inference_steps_list),
            "history_sizes": list(exact.history_sizes),
            "memory_min_temporal_gap_frames": int(min_temporal_gap_frames),
            "history_corruption_sigma": float(training["history_corruption_sigma"]),
            "chunk_plan": {str(key): int(value) for key, value in sorted(chunk_plan.items())},
            "records": [
                [record["scene_id"], record["sequence_id"], int(record["chunk_count"]), record["sample_type"]]
                for record in selected
            ],
        }
        cache_digest = hashlib.sha256(
            json.dumps(cache_descriptor, sort_keys=True).encode("utf-8")
        ).hexdigest()
        persistent_cache_path = persistent_bank_dir / f"history_bank_{cache_digest}.pt"
        cache_hit = persistent_cache_path.exists()
        if cache_hit:
            payload = torch.load(persistent_cache_path, map_location="cpu", weights_only=False)
            if payload.get("descriptor") != cache_descriptor:
                raise RuntimeError("persistent History Bank descriptor mismatch")
            history_bank = payload["history_bank"]
            history_bank_cache = payload["history_bank_cache"]
            reset_runtime_wah_adapter()
        else:
            history_bank = {}
            history_bank_cache = {}
            try:
                for record in selected:
                    trajectory = record["sequence_id"]
                    entries = build_bank_entries(record, snapshot, snapshot_sha)
                    corruption_entries = (
                        build_bank_entries(record, snapshot, snapshot_sha, corrupt_generated_history=True)
                        if record["sample_type"] == "revisit" else {}
                    )
                    for current_chunk, entry in entries.items():
                        cache_key = history_bank_cache_key(
                            trajectory, current_chunk, record["scene_id"], record["sample_type"],
                        )
                        history_bank_cache[cache_key] = entry
                        history_bank[(str(trajectory), int(current_chunk), record["scene_id"], record["sample_type"])] = entry
                        if record["sample_type"] == "revisit":
                            corruption = corruption_entries[int(current_chunk)]
                            corruption_key = history_bank_cache_key(
                                trajectory, current_chunk, record["scene_id"], "corruption",
                            )
                            history_bank_cache[corruption_key] = corruption
                            history_bank[(str(trajectory), int(current_chunk), record["scene_id"], "corruption")] = corruption
                    maybe_cleanup_after_bank_entry()
            finally:
                reset_runtime_wah_adapter()
        expected_keys = {
            (str(record["sequence_id"]), int(current_chunk), record["scene_id"], kind)
            for record in selected
            for kind in (
                (record["sample_type"], "corruption")
                if record["sample_type"] == "revisit" else (record["sample_type"],)
            )
            for current_chunk in eligible_current_chunks(int(record["chunk_count"]))
        }
        if set(history_bank) != expected_keys:
            raise ValueError(
                f"History Bank selection is incomplete: expected {sorted(expected_keys)}, "
                f"got {sorted(history_bank)}"
            )
        if not cache_hit:
            _atomic_torch_save(persistent_cache_path, {
                "descriptor": cache_descriptor,
                "history_bank": history_bank,
                "history_bank_cache": history_bank_cache,
            })
        pipe.transformer.set_adapter(adapter_name)
        for name, parameter in pipe.transformer.named_parameters():
            parameter.requires_grad_(id(parameter) in trainable_ids)
        bank_step = global_step
        stat = {
            "global_step": global_step, "chunk_plan": dict(chunk_plan),
            "entries": len(history_bank),
            "eligible_current_chunks": {
                scene: list(eligible_current_chunks(int(chunk_plan[scene]))) for scene in scenes
            },
            "seconds": time.perf_counter() - refresh_started,
            "persistent_cache": {
                "hit": bool(cache_hit), "digest": cache_digest,
                "path": str(persistent_cache_path),
                "bytes": int(persistent_cache_path.stat().st_size),
            },
            "memory_cleanup": dict(memory_cleanup_stats),
        }
        bank_stats.append(stat)
        _atomic_json(run_dir / "history_bank_stats.json", bank_stats)
        _atomic_json(run_dir / "history_bank_index.json", {
            "global_step": global_step,
            "entries": [
                {
                    "scene_id": scene, "sample_type": sample_type,
                    "current_chunk_index": int(current_chunk),
                    "key": entry["key"], "key_payload": entry["key_payload"],
                    "metadata": entry["metadata"],
                    "history_shapes": {
                        key: None if value is None else list(value.shape)
                        for key, value in entry["histories"].items()
                        if key.startswith("latents_history_")
                    },
                }
                for (trajectory, current_chunk, scene, sample_type), entry in sorted(history_bank.items())
            ],
        })

    def phase_b_item(scene, sample_type, current_chunk, trajectory=None):
        if trajectory is None:
            candidates = [
                key for key in history_bank
                if key[1] == int(current_chunk) and key[2] == scene and key[3] == sample_type
            ]
            if len(candidates) != 1:
                raise KeyError(f"ambiguous Phase B trajectory/current chunk entry: {scene, sample_type, current_chunk}")
            trajectory = candidates[0][0]
        key = (str(trajectory), int(current_chunk), scene, sample_type)
        if key not in history_bank:
            raise KeyError(f"Phase B history bank has no trajectory/current chunk entry: {key}")
        entry = history_bank[key]
        histories = _tree_to(entry["histories"], device)
        return {
            "sample": entry["sample"], "target": _tree_to(entry["target"], device),
            "prompt": _tree_to(entry["prompt"], device), "histories": histories,
            "warp_latents": _tree_to(entry["warp_latents"], device),
            "memory_warp_latents": _tree_to(entry["memory_warp_latents"], device),
            "memory_visibility": entry["memory_visibility"],
            "memory_confidence": entry["memory_confidence"],
            "memory_report": entry["metadata"]["spatial_memory_warp"]["query"],
            "weights": entry["weights"], "record": entry["record"],
            "trajectory": str(entry["record"]["sequence_id"]),
            "current_chunk_index": int(entry["record"]["training_chunk_index"]),
            "supervised_latent_indices": list(entry.get("supervised_latent_indices", [])),
        }

    def benchmark_checkpointing():
        result = {}
        original = bool(pipe.transformer.gradient_checkpointing)
        for enabled in (True, False):
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(0)
            pipe.transformer.gradient_checkpointing = enabled
            begin = time.perf_counter()
            status = "ok"
            try:
                for _ in range(5):
                    optimizer.zero_grad(set_to_none=True)
                    loss = loss_for(fixed)
                    loss.backward()
                    controller.clear_context()
            except torch.cuda.OutOfMemoryError:
                status = "oom"
                torch.cuda.empty_cache()
            result[str(enabled).lower()] = {
                "status": status, "seconds": time.perf_counter() - begin,
                "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(0)),
            }
        off = result["false"]
        on = result["true"]
        selected = False if (
            off["status"] == "ok" and off["peak_reserved_bytes"] < 72 * 1024**3
            and off["seconds"] < on["seconds"]
        ) else True
        pipe.transformer.gradient_checkpointing = selected
        result["selected_gradient_checkpointing"] = selected
        _atomic_json(run_dir / "gradient_checkpointing_benchmark.json", result)
        optimizer.zero_grad(set_to_none=True)
        opt.seed_global_rng(exact.seed)
        return result

    benchmark = benchmark_checkpointing() if args.mode == "smoke" else None
    if args.mode == "train":
        benchmark_path = run_dir / "gradient_checkpointing_benchmark.json"
        if not benchmark_path.exists():
            raise FileNotFoundError("formal training requires the completed smoke benchmark report")
        benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
        pipe.transformer.gradient_checkpointing = bool(benchmark["selected_gradient_checkpointing"])

    phase_a_steps = int(training["phase_a_steps"])
    phase_b_steps = int(training["phase_b_steps"])
    phase_b_smoke = args.mode in ("phase-b-smoke", "phase-b-v2-smoke")
    max_steps = (
        20 if args.mode == "smoke" else
        global_step + int(args.phase_b_smoke_steps) if phase_b_smoke else
        phase_a_steps + phase_b_steps
    )
    accumulation = 1 if args.mode in ("smoke", "phase-b-smoke", "phase-b-v2-smoke") else int(training["gradient_accumulation_steps"])
    rng = np.random.default_rng(exact.seed)
    restored_phase_rng = None if restored is None else restored["metadata"].get("phase_rng_state")
    if restored_phase_rng is not None:
        rng.bit_generator.state = restored_phase_rng
    elif global_step:
        # Schema-v2 step600 predates phase_rng_state. Replay only the custom
        # NumPy generator calls made by Phase A so Phase B starts exactly where
        # an uninterrupted run would have started.
        for replay_step in range(min(global_step, phase_a_steps)):
            for replay_micro in range(int(training["gradient_accumulation_steps"])):
                replay_scene = phase_a_scenes[
                    (replay_step * int(training["gradient_accumulation_steps"]) + replay_micro)
                    % len(phase_a_scenes)
                ]
                replay_candidates = [item for item in phase_a if item["scene_id"] == replay_scene]
                rng.integers(len(replay_candidates))
                rng.random()
    ema_loss = None
    logged_utilization = {}
    diagnostics = []
    scene_counts = {scene: 0 for scene in scenes}
    chunk_counts = {}
    mix_counts = {"revisit": 0, "large_motion": 0, "corruption": 0}
    metrics_history_path = run_dir / "metrics_history.json"
    metrics_history = []
    if metrics_history_path.exists():
        try:
            loaded_metrics = json.loads(metrics_history_path.read_text(encoding="utf-8"))
            if isinstance(loaded_metrics, list):
                metrics_history = loaded_metrics
        except (OSError, ValueError):
            # A truncated metrics file must not make a valid checkpoint
            # unloadable; the next status interval atomically replaces it.
            metrics_history = []

    def supervision_counts_report():
        """Include zeroes for not-yet-selected chunks in every diagnostic."""

        return {
            str(record["sequence_id"]): round_robin.counts_for(
                record["sequence_id"], int(record["chunk_count"]),
            )
            for record in phase_b
        }
    checkpoint = run_dir / "checkpoint_latest.pt"
    pipe.transformer.train()
    status = None
    run_start_step = global_step
    smoke_key = None
    last_trajectory = None
    last_current_chunk = None
    smoke_gradient_checks = {
        "spatial_k_lora_nonzero": False, "spatial_v_lora_nonzero": False,
        "spatial_gate_finite": True, "frozen_parameters_have_no_grad": True,
    }
    smoke_losses = []
    if phase_b_smoke:
        smoke_record = next(
            item for item in phase_b
            if item["sample_type"] == "revisit" and int(item["chunk_count"]) == 8
        )
        reset_runtime_wah_adapter()
        smoke_snapshot = lora_snapshot()
        try:
            smoke_entries = build_bank_entries(
                smoke_record, smoke_snapshot, _sha256(smoke_snapshot),
            )
            smoke_chunk = int(smoke_record.get("training_chunk_index", 1))
            smoke_entry = smoke_entries[smoke_chunk]
        finally:
            reset_runtime_wah_adapter()
        smoke_key = (
            str(smoke_record["sequence_id"]), int(smoke_record["training_chunk_index"]),
            smoke_record["scene_id"], "revisit",
        )
        history_bank[smoke_key] = smoke_entry
        bank_step = global_step
        _atomic_json(run_dir / "phase_b_smoke_history.json", {
            "record": smoke_record,
            "key": smoke_entry["key"],
            "metadata": smoke_entry["metadata"],
            "history_shapes": {
                name: None if value is None else list(value.shape)
                for name, value in smoke_entry["histories"].items()
                if name.startswith("latents_history_")
            },
        })
    while global_step < max_steps:
        if args.mode == "smoke" or global_step < phase_a_steps:
            phase = "A"
            phase_step = global_step + 1
            chunk_count = 1
        else:
            phase = "B"
            phase_step = global_step - phase_a_steps + 1
            curriculum_target = 8 if phase_step <= 200 else (12 if phase_step <= 400 else 16)
            chunk_plan = {
                scene: curriculum_chunk(scene, curriculum_target) for scene in scenes
            }
            refresh_interval = (
                int(training["history_bank_refresh_first"])
                if phase_step <= 400 else int(training["history_bank_refresh_final"])
            )
            if not phase_b_smoke and (not history_bank or global_step - bank_step >= refresh_interval or any(
                int(item["record"]["chunk_count"]) != chunk_plan[item["record"]["scene_id"]]
                for item in history_bank.values()
            )):
                refresh_history_bank(chunk_plan)
        controller.collect_spatial_metrics = bool(
            phase_b_v2 and (
                phase_b_smoke or (global_step + 1) % int(training["status_every"]) == 0
            )
        )
        optimizer.zero_grad(set_to_none=True)
        micro_losses = []
        last_scene = None
        last_type = "single_chunk"
        for micro in range(accumulation):
            if phase == "A":
                scene = phase_a_scenes[
                    (global_step * accumulation + micro) % len(phase_a_scenes)
                ]
                candidates = [item for item in phase_a if item["scene_id"] == scene]
                item = phase_a_item(candidates[int(rng.integers(len(candidates)))])
                camera_only = bool(rng.random() < float(training["phase_a_camera_only_probability"]))
                loss = loss_for(
                    item, histories=item["empty_histories"] if camera_only else item["histories"],
                    anchor=not camera_only, spatial_warp=not camera_only,
                )
                sample_type = "camera_only" if camera_only else "warp_anchor"
            else:
                scene = scenes[(global_step * accumulation + micro) % len(scenes)]
                if phase_b_smoke:
                    scene = smoke_key[2]
                    sample_type = "revisit"
                    item = phase_b_item(scene, sample_type, smoke_key[1], smoke_key[0])
                    round_robin.record(
                        item["trajectory"], item["current_chunk_index"],
                        int(item["record"]["chunk_count"]),
                    )
                else:
                    draw = float(rng.random())
                    revisit_boundary = phase_b_mix["revisit"]
                    motion_boundary = revisit_boundary + phase_b_mix["large_motion"]
                    if draw < revisit_boundary:
                        sample_type = "revisit"
                    elif draw < motion_boundary:
                        sample_type = "large_motion"
                    else:
                        sample_type = "corruption"
                    trajectory = next(
                        record["sequence_id"] for record in phase_b
                        if record["scene_id"] == scene
                        and int(record["chunk_count"]) == int(chunk_plan[scene])
                        and (
                            record["sample_type"] == sample_type
                            or sample_type == "corruption" and record["sample_type"] == "revisit"
                        )
                    )
                    current_chunk = round_robin.next_chunk(
                        trajectory, int(chunk_plan[scene]),
                    )
                    item = phase_b_item(scene, sample_type, current_chunk, trajectory)
                loss = loss_for(item)
                mix_counts[sample_type] += 1
                chunk_count = int(item["record"]["chunk_count"])
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite spatial re-anchor loss")
            (loss / accumulation).backward()
            if phase_b_v2:
                for name, parameter in pipe.transformer.named_parameters():
                    gradient = parameter.grad
                    if name.startswith("spatial_reanchor.spatial_k_lora.") and gradient is not None:
                        smoke_gradient_checks["spatial_k_lora_nonzero"] |= bool((gradient != 0).any())
                    elif name.startswith("spatial_reanchor.spatial_v_lora.") and gradient is not None:
                        smoke_gradient_checks["spatial_v_lora_nonzero"] |= bool((gradient != 0).any())
                gate_gradient = controller.spatial_gate.grad
                smoke_gradient_checks["spatial_gate_finite"] &= bool(
                    gate_gradient is not None and torch.isfinite(gate_gradient).all()
                )
            controller.clear_context()
            micro_losses.append(float(loss.detach().cpu()))
            last_scene, last_type = scene, sample_type
            last_trajectory = None if phase == "A" else item.get("trajectory")
            last_current_chunk = None if phase == "A" else int(item.get("current_chunk_index"))
            scene_counts[scene] += 1
            chunk_counts[str(chunk_count)] = chunk_counts.get(str(chunk_count), 0) + 1
        assert_only_lora_gradients(pipe.transformer, trainable)
        grad = torch.nn.utils.clip_grad_norm_(trainable, float(training["max_grad_norm"]))
        optimizer.step()
        scheduler.step()
        global_step += 1
        value = float(np.mean(micro_losses))
        if phase_b_smoke:
            smoke_losses.append(value)
        ema_loss = value if ema_loss is None else 0.98 * ema_loss + 0.02 * value
        step_spatial_metrics = dict(controller.last_metrics)
        if global_step % int(training["diagnostic_every"]) == 0 or global_step == max_steps:
            entry = {"global_step": global_step, **diagnostic()}
            diagnostics.append(entry)
            _atomic_json(run_dir / "diagnostic_history.json", diagnostics)
            controller.last_metrics = step_spatial_metrics
        if global_step % int(training["checkpoint_every"]) == 0 or global_step == max_steps:
            checkpoint_path = (
                run_dir / f"checkpoint_step{global_step}_phaseB.pt"
                if phase == "B" else checkpoint
            )
            metadata = {
                "git_sha": git_sha, "manifest_sha": _sha256(manifest_path),
                "source_checkpoint": str(config["source_checkpoint"]),
                "scene_scale": {scene: 1.0 for scene in scenes},
                "config": config, "adapter_name": adapter_name,
                "phase_rng_state": rng.bit_generator.state,
                "round_robin": round_robin.snapshot(),
                "supervision_counts": supervision_counts_report(),
                "metrics_history_path": str(metrics_history_path),
            }
            if phase_b_v2:
                save_phase_b_v2_checkpoint(
                    checkpoint_path, pipe.transformer, optimizer, scheduler,
                    global_step=global_step, phase_step=phase_step, metadata=metadata,
                )
            else:
                save_spatial_training_checkpoint(
                    checkpoint_path, pipe.transformer, optimizer, scheduler,
                    global_step=global_step, phase=phase, phase_step=phase_step,
                    metadata=metadata,
                )
                opt.save_visible_lora_state(
                    pipe.transformer, run_dir, adapter_name, "spatial_reanchor_lora.pt"
                )
            checkpoint = checkpoint_path
        elapsed = time.perf_counter() - started
        remaining = max_steps - global_step
        completed_this_run = global_step - run_start_step
        eta = elapsed / max(completed_this_run, 1) * remaining
        status = {
            "status": "running" if global_step < max_steps else "completed",
            "global_step": global_step, "phase": phase, "phase_step": phase_step,
            "max_steps": max_steps, "scene": last_scene, "chunk_count": chunk_count,
            "sample_type": last_type, "trajectory": last_trajectory,
            "current_chunk_index": last_current_chunk,
            "source_prefix_fixed": True,
            "source_prefix_rope": 0,
            "memory_hit": None if phase == "A" else bool(item.get("memory_report", {}).get("memory_hit", False)),
            "memory_entry_id": None if phase == "A" else item.get("memory_report", {}).get("memory_entry_id"),
            "memory_translation_distance": None if phase == "A" else item.get("memory_report", {}).get("memory_translation_distance"),
            "memory_rotation_distance_degrees": None if phase == "A" else item.get("memory_report", {}).get("memory_rotation_distance_degrees"),
            "memory_temporal_gap_frames": None if phase == "A" else item.get("memory_report", {}).get("memory_temporal_gap_frames"),
            "excluded_recent_entry_count": None if phase == "A" else item.get("memory_report", {}).get("excluded_recent_entry_count"),
            "eligible_entry_count": None if phase == "A" else item.get("memory_report", {}).get("eligible_entry_count"),
            "W_visibility": None if phase == "A" else int(np.asarray(item["sample"]["visibility"]).sum()),
            "R_visibility": None if phase == "A" else int(np.asarray(item["memory_visibility"]).sum()),
            "W_token_slots": 540 if phase == "B" else None,
            "R_token_slots": 540 if phase == "B" else None,
            "spatial_kv_token_slots": 1080 if phase == "B" else None,
            "supervised_latent_indices": (
                None if phase == "A" else list(item.get("supervised_latent_indices", []))
            ),
            "supervised_latent_count": (
                None if phase == "A" else len(item.get("supervised_latent_indices", []))
            ),
            "loss": value, "ema_loss": ema_loss,
            "lr_lora": None if phase_b_v2 else optimizer.param_groups[0]["lr"],
            "lr_new": optimizer.param_groups[-1]["lr"], "grad_norm": float(grad),
            "anchor_gates": controller.anchor_gates.detach().cpu().tolist(),
            "camera_gate": float(controller.camera_gate.detach().cpu()),
            "spatial_gates": controller.spatial_gate.detach().cpu().tolist(),
            "anchor_ratio": max(
                [value for key, value in logged_utilization.items() if key.startswith("anchor_ratio")],
                default=0.0,
            ),
            "camera_ratio": logged_utilization.get("camera_ratio", 0.0),
            "history_bank_age": None if bank_step < 0 else global_step - bank_step,
            "latest_checkpoint": str(checkpoint) if checkpoint.exists() else None,
            "round_robin": round_robin.snapshot(),
            "supervision_counts": supervision_counts_report(),
            "gpu_memory": {
                "allocated": int(torch.cuda.memory_allocated(0)),
                "reserved": int(torch.cuda.memory_reserved(0)),
                "peak_reserved": int(torch.cuda.max_memory_reserved(0)),
                "peak_allocated": int(torch.cuda.max_memory_allocated(0)),
            },
            "elapsed": elapsed, "eta": eta, "git_sha": git_sha,
        }
        if global_step % int(training["status_every"]) == 0 or global_step == 1 or global_step == max_steps:
            logged_utilization = controller.metrics_snapshot()
            status["anchor_ratio"] = max(
                [value for key, value in logged_utilization.items() if key.startswith("anchor_ratio")],
                default=0.0,
            )
            status["camera_ratio"] = logged_utilization.get("camera_ratio", 0.0)
            status["W_valid_token_count"] = logged_utilization.get("warp_valid_token_count")
            status["R_valid_token_count"] = logged_utilization.get("memory_valid_token_count")
            status["W_attention_mass"] = {
                key: value for key, value in logged_utilization.items()
                if key.startswith("warp_attention_mass_block")
            }
            status["R_attention_mass"] = {
                key: value for key, value in logged_utilization.items()
                if key.startswith("memory_attention_mass_block")
            }
            status["spatial_delta_target_norm"] = {
                key: value for key, value in logged_utilization.items()
                if key.startswith("spatial_delta_ratio_block")
            }
            metrics_history.append({
                "global_step": int(global_step), "phase": phase,
                "phase_step": int(phase_step), "scene": last_scene,
                "sample_type": last_type, "trajectory": last_trajectory,
                "current_chunk_index": last_current_chunk, "loss": float(value),
                "ema_loss": None if ema_loss is None else float(ema_loss),
                "diagnostic": diagnostics[-1] if diagnostics and diagnostics[-1]["global_step"] == global_step else None,
                "supervision_counts": supervision_counts_report(),
                "eligible_current_chunks": {
                    str(record["sequence_id"]): list(eligible_current_chunks(int(record["chunk_count"])))
                    for record in phase_b
                },
                "supervised_latent_indices": status["supervised_latent_indices"],
                "supervised_latent_count": status["supervised_latent_count"],
                "memory_hit": status["memory_hit"],
                "memory_entry_id": status["memory_entry_id"],
                "memory_translation_distance": status["memory_translation_distance"],
                "memory_rotation_distance_degrees": status["memory_rotation_distance_degrees"],
                "W_visibility": status["W_visibility"], "R_visibility": status["R_visibility"],
                "W_valid_token_count": status.get("W_valid_token_count"),
                "R_valid_token_count": status.get("R_valid_token_count"),
                "W_attention_mass": status.get("W_attention_mass"),
                "R_attention_mass": status.get("R_attention_mass"),
                "spatial_gates": status["spatial_gates"],
                "spatial_delta_target_norm": status.get("spatial_delta_target_norm"),
            })
            _atomic_json(run_dir / "metrics_history.json", metrics_history)
            _atomic_json(status_path, status)
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(status) + "\n")

    if status is None:
        if not status_path.exists():
            raise RuntimeError("resume checkpoint is already complete but training status is missing")
        status = json.loads(status_path.read_text(encoding="utf-8"))
        status.update({
            "status": "completed", "global_step": global_step,
            "phase": phase, "phase_step": phase_step, "max_steps": max_steps,
        })
    final = {
        **status,
        "status": "completed",
        "phase_a_steps": min(global_step, phase_a_steps),
        "phase_b_steps": max(0, global_step - phase_a_steps),
        "scene_sample_counts": scene_counts,
        "chunk_length_distribution": chunk_counts,
        "phase_b_mix_counts": mix_counts,
        "round_robin": round_robin.snapshot(),
        "supervision_counts": supervision_counts_report(),
        "metrics_history": metrics_history,
        "history_bank_cache_keys": [list(key) for key in sorted(history_bank_cache)],
        "history_bank_stats": bank_stats,
        "memory_cleanup": memory_cleanup_stats,
        "diagnostics": diagnostics,
        "source_checkpoint_info": source_info,
        "lora_setup": lora_stats,
        "trainable_parameters": {
            "lora": int(sum(item.numel() for item in lora_params)),
            "new_modules": int(sum(item.numel() for item in new_params)),
            "total": int(sum(item.numel() for item in trainable)),
        },
        "gpu_processes": _gpu_snapshot(),
    }
    _atomic_json(run_dir / "training_status_final.json", final)
    if phase_b_smoke:
        if phase_b_v2 and not all(smoke_gradient_checks.values()):
            raise RuntimeError(f"Phase-B v2 smoke gradient assertions failed: {smoke_gradient_checks}")
        _atomic_json(run_dir / "phase_b_smoke_report.json", {
            "passed": status["status"] == "completed",
            "optimizer_steps": global_step - run_start_step,
            "start_global_step": run_start_step,
            "end_global_step": global_step,
            "uses_future_gt": False,
            "history_bank_key": history_bank[smoke_key]["key"],
            "history_metadata": history_bank[smoke_key]["metadata"],
            "finite_loss": bool(np.isfinite(status["loss"])),
            "losses": smoke_losses,
            "gpu_memory": status["gpu_memory"],
            "gradient_checks": smoke_gradient_checks,
            "source_prefix_fixed": True, "source_prefix_rope": 0,
            "W_tokens": 540, "R_tokens": 540, "spatial_kv_tokens": 1080,
            "R_miss_hard_mask": True,
        })
    print(json.dumps(final, indent=2))


if __name__ == "__main__":
    main()
