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
    parser.add_argument("--mode", choices=("smoke", "train"), required=True)
    parser.add_argument("--resume", default="")
    parser.add_argument("--set", action="append", default=[], dest="overrides")
    return parser.parse_args()


def _atomic_json(path, payload):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
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
    from long_video.geometry.point_renderer import render
    from long_video.memory.node_store import NodeStore
    from long_video.types import CameraBatch
    from long_video.oracle_training.wah_training import (
        assert_only_lora_gradients,
        load_source_trainable_state,
        load_spatial_training_checkpoint,
        masked_flow_matching_loss,
        save_spatial_training_checkpoint,
    )
    from long_video.oracle_training.history_bank import HistoryBankKey, validate_history_bank_entry
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
    scenes = sorted({item["scene_id"] for item in records})
    if len(scenes) != 2 or len(phase_a) < 8 or len(phase_a_diag) < 2 or not phase_b:
        raise ValueError("manifest must contain two scenes, 8 Phase A train, 2 diagnostics, and Phase B windows")
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
    ).to(device)
    new_params = list(controller.parameters())
    trainable = list(lora_params) + new_params
    trainable_ids = {id(item) for item in trainable}
    for name, parameter in pipe.transformer.named_parameters():
        parameter.requires_grad_(id(parameter) in trainable_ids)
    optimizer = torch.optim.AdamW([
        {"params": lora_params, "lr": float(training["lora_learning_rate"]), "name": "wah_lora"},
        {"params": new_params, "lr": float(training["new_module_learning_rate"]), "name": "spatial_new"},
    ], weight_decay=float(training["weight_decay"]))
    warmup = max(1, int(training["warmup_steps"]))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: min(1.0, (step + 1) / warmup)
    )
    global_step = phase_step = 0
    phase = "A"
    if args.resume:
        restored = load_spatial_training_checkpoint(
            args.resume, pipe.transformer, optimizer, scheduler
        )
        global_step = restored["global_step"]
        phase, phase_step = restored["phase"], restored["phase_step"]

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
            warp_provenance = causal.provenance
        elif chunk_index == 0:
            warp = _frames(root / "single_chunk_warp" / "warp_rgb")
            visibility = np.load(root / "single_chunk_warp" / "warp_visibility.npy")
            confidence = np.load(root / "single_chunk_warp" / "warp_confidence.npy")
        else:
            raise RuntimeError("Phase B warp must be rendered by the causal active-node renderer")
        metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
        return {
            "root": root, "target": target, "poses": poses, "intrinsics": intrinsics,
            "prompt": prompt, "warp": warp, "visibility": visibility,
            "confidence": confidence, "metadata": metadata, "start": start,
            "warp_provenance": warp_provenance,
        }

    spatial_geometry_cache = {}

    def spatial_conditioning(
        sample, warp_latents, stage_latents, *, anchor=True, camera=True, spatial_warp=True,
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
            )
            geometry = (visibility, rays)
            spatial_geometry_cache[geometry_key] = geometry
        visibility, rays = geometry
        return {
            "warp_latents": stage_warp,
            "visibility_tokens": visibility,
            "plucker_tokens": rays,
            "anchor_enabled": anchor,
            "camera_enabled": camera,
            "spatial_warp_enabled": spatial_warp,
        }

    def pyramid_spatial_conditioning(
        sample, warp_latents, stage_latents_list, *, anchor=True, camera=True, spatial_warp=True,
    ):
        contexts = [
            spatial_conditioning(
                sample, warp_latents, stage_latents, anchor=anchor,
                camera=camera, spatial_warp=spatial_warp,
            )
            for stage_latents in stage_latents_list
        ]
        return {
            "stage_contexts": [
                {
                    "warp_latents": context["warp_latents"],
                    "visibility_tokens": context["visibility_tokens"],
                    "plucker_tokens": context["plucker_tokens"],
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
        "phase_b_mix": phase_b_mix,
        "scene_scale": {scene: {"value": 1.0, "source": "Holo360D_dataset_calibrated_metric"} for scene in scenes},
        "trainable_parameters": {
            "lora": int(sum(item.numel() for item in lora_params)),
            "new_modules": int(sum(item.numel() for item in new_params)),
            "total": int(sum(item.numel() for item in trainable)),
        },
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
    _atomic_json(run_dir / "startup_report.json", startup_report)

    def loss_for(item, *, histories=None, anchor=True, camera=True, spatial_warp=True, fixed_items=None):
        def condition(stage_item):
            return spatial_conditioning(
                item["sample"], item["warp_latents"], stage_item["noisy_latents"],
                anchor=anchor, camera=camera, spatial_warp=spatial_warp,
            )
        return masked_flow_matching_loss(
            pipe, item["prompt"], item["target"], histories or item["histories"],
            exact, device, item["weights"], fixed_stage_items=fixed_items,
            spatial_conditioning=condition,
        )[0]

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

    history_bank = {}
    bank_step = -1
    bank_stats = []
    memory_cleanup_stats = {"calls": 0, "reasons": [], "entries": 0}

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

    def build_bank_entry(record, snapshot, snapshot_sha, *, corrupt_generated_history=False):
        if "training_chunk_index" not in record:
            raise ValueError("Phase B record is missing selector training_chunk_index")
        current_chunk = int(record["training_chunk_index"])
        if not 0 <= current_chunk < int(record["chunk_count"]):
            raise ValueError("training_chunk_index is outside the selected Phase B window")
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
        causal_renderer = CausalActiveNodeRenderer(
            NodeStore(root / "session"),
            renderer_kwargs={"device": "cuda:0", "near": 0.05, "far": 100.0,
                             "point_radius": 1, "chunk_points": 1000000},
        )
        warp_provenance = []
        corruption_generator = torch.Generator(device=device).manual_seed(
            exact.seed + global_step + 1009
        )
        for chunk in range(current_chunk):
            sample = sample_arrays(record, chunk, causal_renderer=causal_renderer)
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
            inference_pyramid = opt.training_exact_pyramid_latents(
                warp_latents, len(exact.pyramid_num_inference_steps_list),
            )
            controller.prepare_context(**pyramid_spatial_conditioning(
                sample, warp_latents, inference_pyramid,
            ))
            try:
                with torch.no_grad():
                    _, state = pipe.generate_next_chunk(
                        state, warp_video=np.stack([np.asarray(item) for item in sample["warp"]]),
                        warp_visibility_mask=sample["visibility"][None, None],
                        warp_confidence_mask=(sample["confidence"] * sample["visibility"])[None, None],
                        output_type="np",
                    )
            finally:
                controller.clear_context()
            del warp_latents
        sample = sample_arrays(record, current_chunk, causal_renderer=causal_renderer)
        warp_provenance.append(sample["warp_provenance"])
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
                pipe, sample["target"], exact, device, mean, std
            ).detach()
        warp_latents = encode_warp(sample["warp"])
        all_weights = np.load(root / "primary_loss_weight_latent.npy")
        latent_start = current_chunk * (exact.num_latent_frames_per_chunk - 1)
        weights = all_weights[latent_start:latent_start + exact.num_latent_frames_per_chunk]
        if len(weights) != exact.num_latent_frames_per_chunk:
            raise ValueError("current Phase B supervision weights are truncated")
        weights = weights.copy()
        weights[0] = 0.0
        bank_key = HistoryBankKey(
            checkpoint_sha=snapshot_sha, global_step=global_step,
            scene_id=record["scene_id"], source_id=record["sequence_id"],
            trajectory_id=record["sequence_id"], history_chunk_index=current_chunk,
            generation_config=(
                ("pyramid_steps", tuple(exact.pyramid_num_inference_steps_list)),
                ("history_sizes", tuple(exact.history_sizes)),
                ("visible_token_drop", bool(exact.history_visible_token_drop)),
                ("warp_downsample", str(exact.warp_history_downsample_mode)),
                ("spatial_reanchor", True),
                ("history_corruption", bool(corrupt_generated_history)),
                ("history_corruption_sigma", float(training["history_corruption_sigma"])),
            ),
            prompt=prompt, seed=exact.seed,
        )
        key_payload = dict(bank_key.__dict__)
        key = bank_key.digest()
        entry = {
            "key": key, "key_payload": key_payload, "record": record,
            "sample": sample, "target": _tree_to(target_latents, "cpu"),
            "prompt": _tree_to(prompt_embeds, "cpu"),
            "histories": _tree_to(histories, "cpu"),
            "warp_latents": _tree_to(warp_latents, "cpu"), "weights": weights,
            "metadata": {
                "uses_gt_future": False, "checkpoint_sha": snapshot_sha,
                "global_step": global_step, "history_chunk_index": current_chunk,
                "self_augmentation": bool(corrupt_generated_history),
                "restoration_steps_per_pyramid_stage": tuple(exact.pyramid_num_inference_steps_list),
                "node_mode": causal_renderer.node_mode,
                "warp_provenance": warp_provenance,
                "training_chunk_index": current_chunk,
            },
        }
        validate_history_bank_entry({
            "TEMP_LONG": entry["histories"].get("latents_history_long"),
            "TEMP_MID": entry["histories"].get("latents_history_mid"),
            "TEMP_SHORT": entry["histories"].get("latents_history_short"),
            "key": entry["key"], "metadata": entry["metadata"],
        })
        return entry

    def refresh_history_bank(chunk_plan):
        nonlocal history_bank, bank_step
        refresh_started = time.perf_counter()
        snapshot = lora_snapshot()
        snapshot_sha = _sha256(snapshot)
        selected = [
            item for item in phase_b
            if int(item["chunk_count"]) == int(chunk_plan[item["scene_id"]])
        ]
        history_bank = {}
        for record in selected:
            entry = build_bank_entry(record, snapshot, snapshot_sha)
            history_bank[(record["scene_id"], record["sample_type"])] = entry
            if record["sample_type"] == "revisit":
                corruption = build_bank_entry(
                    record, snapshot, snapshot_sha, corrupt_generated_history=True,
                )
                history_bank[(record["scene_id"], "corruption")] = corruption
            maybe_cleanup_after_bank_entry()
        expected_keys = {
            (scene, kind) for scene in scenes
            for kind in ("revisit", "large_motion", "corruption")
        }
        if set(history_bank) != expected_keys:
            raise ValueError(
                f"History Bank selection is incomplete: expected {sorted(expected_keys)}, "
                f"got {sorted(history_bank)}"
            )
        pipe.transformer.set_adapter(adapter_name)
        for name, parameter in pipe.transformer.named_parameters():
            parameter.requires_grad_(id(parameter) in trainable_ids)
        bank_step = global_step
        stat = {
            "global_step": global_step, "chunk_plan": dict(chunk_plan),
            "entries": len(history_bank), "seconds": time.perf_counter() - refresh_started,
            "memory_cleanup": dict(memory_cleanup_stats),
        }
        bank_stats.append(stat)
        _atomic_json(run_dir / "history_bank_stats.json", bank_stats)
        _atomic_json(run_dir / "history_bank_index.json", {
            "global_step": global_step,
            "entries": [
                {
                    "scene_id": scene, "sample_type": sample_type,
                    "key": entry["key"], "key_payload": entry["key_payload"],
                    "metadata": entry["metadata"],
                    "history_shapes": {
                        key: None if value is None else list(value.shape)
                        for key, value in entry["histories"].items()
                        if key.startswith("latents_history_")
                    },
                }
                for (scene, sample_type), entry in sorted(history_bank.items())
            ],
        })

    def phase_b_item(scene, sample_type):
        entry = history_bank[(scene, sample_type)]
        histories = _tree_to(entry["histories"], device)
        return {
            "sample": entry["sample"], "target": _tree_to(entry["target"], device),
            "prompt": _tree_to(entry["prompt"], device), "histories": histories,
            "warp_latents": _tree_to(entry["warp_latents"], device),
            "weights": entry["weights"], "record": entry["record"],
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
    max_steps = 20 if args.mode == "smoke" else phase_a_steps + phase_b_steps
    accumulation = 1 if args.mode == "smoke" else int(training["gradient_accumulation_steps"])
    rng = np.random.default_rng(exact.seed)
    ema_loss = None
    logged_utilization = {}
    diagnostics = []
    scene_counts = {scene: 0 for scene in scenes}
    chunk_counts = {}
    mix_counts = {"revisit": 0, "large_motion": 0, "corruption": 0}
    checkpoint = run_dir / "checkpoint_latest.pt"
    pipe.transformer.train()
    status = None
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
            if not history_bank or global_step - bank_step >= refresh_interval or any(
                int(item["record"]["chunk_count"]) != chunk_plan[item["record"]["scene_id"]]
                for item in history_bank.values()
            ):
                refresh_history_bank(chunk_plan)
        optimizer.zero_grad(set_to_none=True)
        micro_losses = []
        last_scene = None
        last_type = "single_chunk"
        for micro in range(accumulation):
            scene = scenes[(global_step * accumulation + micro) % len(scenes)]
            if phase == "A":
                candidates = [item for item in phase_a if item["scene_id"] == scene]
                item = phase_a_item(candidates[int(rng.integers(len(candidates)))])
                camera_only = bool(rng.random() < float(training["phase_a_camera_only_probability"]))
                loss = loss_for(
                    item, histories=item["empty_histories"] if camera_only else item["histories"],
                    anchor=not camera_only, spatial_warp=not camera_only,
                )
                sample_type = "camera_only" if camera_only else "warp_anchor"
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
                item = phase_b_item(scene, sample_type)
                loss = loss_for(item)
                mix_counts[sample_type] += 1
                chunk_count = int(item["record"]["chunk_count"])
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite spatial re-anchor loss")
            (loss / accumulation).backward()
            controller.clear_context()
            micro_losses.append(float(loss.detach().cpu()))
            last_scene, last_type = scene, sample_type
            scene_counts[scene] += 1
            chunk_counts[str(chunk_count)] = chunk_counts.get(str(chunk_count), 0) + 1
        assert_only_lora_gradients(pipe.transformer, trainable)
        grad = torch.nn.utils.clip_grad_norm_(trainable, float(training["max_grad_norm"]))
        optimizer.step()
        scheduler.step()
        global_step += 1
        value = float(np.mean(micro_losses))
        ema_loss = value if ema_loss is None else 0.98 * ema_loss + 0.02 * value
        if global_step % int(training["diagnostic_every"]) == 0 or global_step == max_steps:
            entry = {"global_step": global_step, **diagnostic()}
            diagnostics.append(entry)
            _atomic_json(run_dir / "diagnostic_history.json", diagnostics)
        if global_step % int(training["checkpoint_every"]) == 0 or global_step == max_steps:
            metadata = {
                "git_sha": git_sha, "manifest_sha": _sha256(manifest_path),
                "source_checkpoint": str(config["source_checkpoint"]),
                "scene_scale": {scene: 1.0 for scene in scenes},
                "config": config, "adapter_name": adapter_name,
            }
            save_spatial_training_checkpoint(
                checkpoint, pipe.transformer, optimizer, scheduler,
                global_step=global_step, phase=phase, phase_step=phase_step,
                metadata=metadata,
            )
            opt.save_visible_lora_state(
                pipe.transformer, run_dir, adapter_name, "spatial_reanchor_lora.pt"
            )
        elapsed = time.perf_counter() - started
        remaining = max_steps - global_step
        eta = elapsed / max(global_step, 1) * remaining
        status = {
            "status": "running" if global_step < max_steps else "completed",
            "global_step": global_step, "phase": phase, "phase_step": phase_step,
            "max_steps": max_steps, "scene": last_scene, "chunk_count": chunk_count,
            "sample_type": last_type, "loss": value, "ema_loss": ema_loss,
            "lr_lora": optimizer.param_groups[0]["lr"],
            "lr_new": optimizer.param_groups[1]["lr"], "grad_norm": float(grad),
            "anchor_gates": controller.anchor_gates.detach().cpu().tolist(),
            "camera_gate": float(controller.camera_gate.detach().cpu()),
            "anchor_ratio": max(
                [value for key, value in logged_utilization.items() if key.startswith("anchor_ratio")],
                default=0.0,
            ),
            "camera_ratio": logged_utilization.get("camera_ratio", 0.0),
            "history_bank_age": None if bank_step < 0 else global_step - bank_step,
            "latest_checkpoint": str(checkpoint) if checkpoint.exists() else None,
            "gpu_memory": {
                "allocated": int(torch.cuda.memory_allocated(0)),
                "reserved": int(torch.cuda.memory_reserved(0)),
                "peak_reserved": int(torch.cuda.max_memory_reserved(0)),
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
    print(json.dumps(final, indent=2))


if __name__ == "__main__":
    main()
