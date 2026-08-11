#!/usr/bin/env python3
"""Training-free Validated Causal World + World-Projected Flow inference."""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import sys
import time


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence", type=Path, required=True)
    parser.add_argument(
        "--source-image", type=Path,
        help="Optional source-prefix image; does not modify the sequence directory.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wah-root", type=Path, required=True)
    parser.add_argument("--wah-model", type=Path, required=True)
    initialization = parser.add_mutually_exclusive_group(required=True)
    initialization.add_argument(
        "--checkpoint", type=Path,
        help="Schema-v2 step600 Phase-A checkpoint.",
    )
    initialization.add_argument(
        "--phase-a-zero-source-checkpoint", type=Path,
        help=(
            "The source WAH LoRA checkpoint used before Phase-A step 1. "
            "Spatial Anchor/Camera stay at their untrained initialization."
        ),
    )
    parser.add_argument("--lora", type=Path, required=True)
    parser.add_argument("--pi3-repo", type=Path, required=True)
    parser.add_argument("--pi3-checkpoint", type=Path, required=True)
    parser.add_argument("--memory-config", type=Path, default=Path("configs/online_memory.yaml"))
    parser.add_argument(
        "--source-world-observed-erp-mask", type=Path,
        help=(
            "Inference-only filter: copy M0 with only points whose rays fall in "
            "the original perspective observation mask. The stored M0 is not modified."
        ),
    )
    parser.add_argument("--physical-gpu", type=int, default=1)
    parser.add_argument("--chunks", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--disable-anchor-adapter", action="store_true")
    parser.add_argument("--disable-camera-adapter", action="store_true")
    parser.add_argument("--disable-spatial-warp", action="store_true")
    parser.add_argument("--node-activation-delay-chunks", type=int, choices=(1, 2), default=2)
    parser.add_argument("--lambda-max-stage0", type=float, default=0.0)
    parser.add_argument("--lambda-max-stage1", type=float, default=0.15)
    parser.add_argument("--lambda-max-stage2", type=float, default=0.30)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--confidence-ramp-min", type=float, default=0.2)
    parser.add_argument("--confidence-ramp-max", type=float, default=0.5)
    parser.add_argument(
        "--confidence-threshold", type=float, default=None,
        help="Deprecated compatibility argument; WPF soft-ramp mode ignores it.",
    )
    parser.add_argument("--coverage-threshold", type=float)
    parser.add_argument(
        "--memory-set", action="append", default=[], metavar="KEY=VALUE",
        help="Override an existing MemoryManager threshold for this experiment.",
    )
    parser.add_argument("--video-name", default="school_world_projected.mp4")
    return parser.parse_args()


ARGS = parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(ARGS.physical_gpu)
os.environ.setdefault("XFORMERS_DISABLED", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(ARGS.wah_root))

from long_video.config import load_yaml
from long_video.initialization.geometry_backend import Pi3GeometryBackend
from long_video.memory.memory_manager import MemoryManager
from long_video.memory.node_filter import filter_node_to_observed_erp
from long_video.memory.node_store import NodeStore
from long_video.oracle_training.causal_warp import CausalActiveNodeRenderer
from long_video.oracle_training.contracts import GeneratedMemoryBatch
from long_video.types import CameraBatch
from long_video.wah.spatial_reanchor import (
    install_spatial_reanchor,
    plucker_camera_rays,
    resize_latents_spatial,
)
from long_video.wah.world_projected_pipeline import (
    DelayedNodeActivationQueue,
    WORLD_OWNERSHIP_COVERAGE_THRESHOLD,
    WorldProjectedWarpAsHistoryPipeline,
    WorldProjectionConfig,
    apply_previous_world_boundary,
    build_canonical_world_support,
    build_single_frame_world_support,
    build_world_projection_context,
    canonical_support_to_tokens,
    encode_canonical_video_latents,
    fill_invalid_warp_for_vae,
    mask_canonical_latent,
    posterior_mode_or_mean,
)
from warp_as_history.training.core import training_exact_pyramid_latents


HEIGHT, WIDTH = 384, 640
CHUNK_FRAMES, CHUNK_STRIDE = 33, 32
LATENT_FRAMES, VAE_TEMPORAL_SCALE = 9, 4
PYRAMID_STEPS = (2, 2, 2)


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def video_array(video):
    value = np.asarray(video)
    if value.ndim == 5 and value.shape[0] == 1:
        value = value[0]
    if value.ndim == 4 and value.shape[1] == 3 and value.shape[-1] != 3:
        value = np.moveaxis(value, 1, -1)
    if value.ndim != 4 or value.shape[-1] != 3:
        raise ValueError(f"WAH output must be [T,H,W,3], got {value.shape}")
    if value.dtype != np.uint8:
        value = np.rint(np.clip(value, 0.0, 1.0) * 255.0).astype(np.uint8)
    return value


def u8(value):
    array = np.asarray(value)
    if array.dtype == np.uint8:
        return array
    return np.rint(np.clip(array, 0.0, 1.0) * 255.0).astype(np.uint8)


def debug_panel(generated, warp, visibility, confidence, frame_index, chunk_index):
    vis = np.repeat(u8(np.asarray(visibility, np.float32))[..., None], 3, axis=-1)
    conf = np.repeat(u8(np.asarray(confidence, np.float32))[..., None], 3, axis=-1)
    top = np.concatenate((generated, warp), axis=1)
    bottom = np.concatenate((vis, conf), axis=1)
    panel = np.concatenate((top, bottom), axis=0)
    # Keep diagnostics machine-readable; avoid a font dependency in H100 runs.
    del frame_index, chunk_index
    return panel


def canonical_lora_name(name):
    parts = name.split(".")
    for marker in ("lora_A", "lora_B"):
        if marker in parts:
            index = parts.index(marker)
            if len(parts) == index + 3 and parts[-1] == "weight":
                return ".".join(parts[:index + 1] + ["*", "weight"])
    return None


def state_fingerprint(state):
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _map_trainable_state(transformer, source):
    named = dict(transformer.named_parameters())
    runtime_lora = {}
    for name in named:
        canonical = canonical_lora_name(name)
        if canonical:
            if canonical in runtime_lora:
                raise RuntimeError(f"ambiguous runtime LoRA key {canonical}")
            runtime_lora[canonical] = name
    mapping = {}
    for source_name in source:
        if source_name in named:
            mapping[source_name] = source_name
        else:
            canonical = canonical_lora_name(source_name)
            if canonical and canonical in runtime_lora:
                mapping[source_name] = runtime_lora[canonical]
    missing = sorted(set(source) - set(mapping))
    if missing:
        raise ValueError(f"checkpoint tensors missing from runtime: {missing[:10]}")
    with torch.no_grad():
        for source_name, target_name in mapping.items():
            parameter, value = named[target_name], source[source_name]
            if tuple(parameter.shape) != tuple(value.shape):
                raise ValueError(f"shape mismatch for {source_name}: {value.shape} vs {parameter.shape}")
            parameter.copy_(value.to(parameter.device, parameter.dtype))
    return mapping


def load_step600(transformer, checkpoint):
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if int(payload.get("schema_version", 0)) != 2:
        raise ValueError("World-Projected Flow requires the schema-v2 step600 Phase-A checkpoint")
    if int(payload.get("global_step", -1)) != 600 or str(payload.get("phase")) != "A":
        raise ValueError("checkpoint must be step600 Phase A")
    source = payload.get("trainable_state") or {}
    mapping = _map_trainable_state(transformer, source)
    lora_count = sum(canonical_lora_name(name) is not None for name in source)
    spatial_count = sum(name.startswith("spatial_reanchor.") for name in source)
    if len(mapping) != 337 or lora_count != 320 or spatial_count != 17:
        raise RuntimeError(
            f"step600 coverage mismatch total/lora/spatial={len(mapping)}/{lora_count}/{spatial_count}"
        )
    return {
        "global_step": 600,
        "phase": "A",
        "tensor_count": len(mapping),
        "lora_tensor_count": lora_count,
        "spatial_tensor_count": spatial_count,
        "fingerprint_sha256": state_fingerprint(source),
    }


def load_phase_a_zero(transformer, checkpoint):
    """Reproduce the trainable state immediately before Phase-A optimizer step 1."""
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    source = payload.get("trainable_state") or payload.get("lora_state") or {}
    if not source:
        raise ValueError("Phase-A zero-step source checkpoint contains no WAH LoRA state")
    mapping = _map_trainable_state(transformer, source)
    lora_count = sum(canonical_lora_name(name) is not None for name in source)
    spatial_count = sum(name.startswith("spatial_reanchor.") for name in source)
    if len(mapping) != 320 or lora_count != 320 or spatial_count != 0:
        raise RuntimeError(
            "Phase-A zero-step source must contain exactly 320 WAH LoRA tensors "
            f"and no Spatial Anchor/Camera tensors; got {len(mapping)}/{lora_count}/{spatial_count}"
        )
    initialized_spatial = {
        f"spatial_reanchor.{name}": parameter
        for name, parameter in transformer.spatial_reanchor.named_parameters()
        if not name.startswith(("spatial_k_lora.", "spatial_v_lora."))
        and name not in {"spatial_memory_role", "spatial_gate"}
    }
    if len(initialized_spatial) != 17:
        raise RuntimeError(
            f"expected 17 freshly initialized Spatial Anchor/Camera tensors, got {len(initialized_spatial)}"
        )
    return {
        "project_global_step": 0,
        "source_checkpoint_global_step": int(payload.get("global_step", -1)),
        "tensor_count": len(mapping),
        "lora_tensor_count": lora_count,
        "loaded_spatial_tensor_count": spatial_count,
        "initialized_spatial_tensor_count": len(initialized_spatial),
        "source_fingerprint_sha256": state_fingerprint(source),
        "initialized_spatial_fingerprint_sha256": state_fingerprint(initialized_spatial),
    }


def clean_warp_latents(
    pipe, state, warp, *, canonical_warp_rgb=None, return_first_frame=False,
):
    """Encode world-only latents without touching WAH state or generator RNG."""
    if canonical_warp_rgb is None:
        canonical_warp_rgb = fill_invalid_warp_for_vae(warp.rgb, warp.visibility)
    device = pipe._wah_execution_device()
    raw = pipe._coerce_warp_video_tensor(
        canonical_warp_rgb,
        height=int(state["height"]), width=int(state["width"]), device=device,
    )
    mean, std = pipe._latent_stats(device)
    first_frame_latent, clean = encode_canonical_video_latents(
        pipe,
        raw,
        latents_mean=mean,
        latents_std=std,
        num_latent_frames_per_chunk=LATENT_FRAMES,
        dtype=torch.float32,
        device=device,
    )
    clean = clean.detach()
    if return_first_frame:
        return first_frame_latent.detach(), clean
    return clean


def clean_boundary_frame_latent(pipe, state, frame):
    """Encode the one true stride-32 shared RGB frame with WAH VAE statistics."""
    device = pipe._wah_execution_device()
    tensor = pipe._coerce_warp_video_tensor(
        [np.asarray(frame)], height=HEIGHT, width=WIDTH, device=device,
    ).to(device=device, dtype=pipe.vae.dtype)
    if tuple(tensor.shape) != (1, 3, 1, HEIGHT, WIDTH):
        raise RuntimeError(f"single-frame boundary RGB tensor shape mismatch: {tuple(tensor.shape)}")
    mean, std = pipe._latent_stats(device)
    encoded_output = pipe.vae.encode(tensor)
    posterior = getattr(encoded_output, "latent_dist", encoded_output)
    latent = posterior_mode_or_mean(posterior)
    latent = (latent - mean) * std
    if tuple(latent.shape[2:]) != (1, 48, 80):
        raise RuntimeError(f"single-frame boundary VAE latent shape mismatch: {tuple(latent.shape)}")
    return latent.detach().to(dtype=torch.float32)


def prepare_spatial_context(
    controller, transformer, clean, canonical_support, poses, intrinsics, start,
    *, anchor_enabled=True, camera_enabled=True, spatial_warp_enabled=True,
):
    patch = tuple(int(value) for value in transformer.config.patch_size)
    stage_contexts, shapes = [], []
    for stage in training_exact_pyramid_latents(clean, len(PYRAMID_STEPS)):
        latent_height, latent_width = map(int, stage.shape[-2:])
        stage_warp = resize_latents_spatial(clean, height=latent_height, width=latent_width)
        token_height, token_width = latent_height // patch[-2], latent_width // patch[-1]
        visible_tokens = canonical_support_to_tokens(
            canonical_support.safe_support,
            latent_height=latent_height,
            latent_width=latent_width,
            patch_height=patch[-2],
            patch_width=patch[-1],
        )
        confidence_tokens = canonical_support_to_tokens(
            canonical_support.confidence,
            latent_height=latent_height,
            latent_width=latent_width,
            patch_height=patch[-2],
            patch_width=patch[-1],
        )
        plucker = plucker_camera_rays(
            poses,
            intrinsics,
            image_height=HEIGHT,
            image_width=WIDTH,
            token_height=token_height,
            token_width=token_width,
            latent_frames=LATENT_FRAMES,
            temporal_scale=VAE_TEMPORAL_SCALE,
            scene_scale=1.0,
            sequence_frame_start=int(start),
            validate_sequence_source_origin=False,
        )
        stage_contexts.append({
            "warp_latents": stage_warp,
            "visibility_tokens": visible_tokens,
            "warp_confidence_tokens": confidence_tokens,
            "plucker_tokens": plucker,
            "spatial_attention_enabled": False,
        })
        shapes.append({
            "stage_id": len(shapes),
            "latent": list(stage_warp.shape),
            "tokens": int(visible_tokens.shape[1]),
        })
    controller.prepare_context(
        stage_contexts=stage_contexts,
        anchor_enabled=bool(anchor_enabled),
        camera_enabled=bool(camera_enabled),
        spatial_warp_enabled=bool(spatial_warp_enabled),
    )
    return shapes


def slice_warp(warp, first):
    fields = {
        "rgb": warp.rgb[first:], "depth": warp.depth[first:],
        "visibility": warp.visibility[first:], "confidence": warp.confidence[first:],
        "source": warp.source[first:], "coverage_per_frame": warp.coverage_per_frame[first:],
    }
    for name in (
        "rgb_content_origin", "depth_content_origin", "evidence_role",
        "rgb_evidence_role", "depth_evidence_role",
    ):
        value = getattr(warp, name, None)
        if value is not None:
            fields[name] = value[first:]
    return replace(warp, **fields)


def clear_decoded_history(state):
    state["history_video"] = None
    state["last_video_delta"] = None
    state["returned_frame_count"] = 0
    keep = int(state["num_history_latent_frames"])
    state["history_latents"] = state["history_latents"][:, :, -keep:].detach()
    state["real_history_latents"] = None
    state["last_latents"] = None


def main():
    if torch.cuda.device_count() != 1:
        raise RuntimeError(f"expected exactly physical GPU {ARGS.physical_gpu}, got {torch.cuda.device_count()}")
    torch.cuda.set_device(0)
    torch.cuda.reset_peak_memory_stats(0)
    np.random.seed(ARGS.seed)
    torch.manual_seed(ARGS.seed)
    torch.cuda.manual_seed_all(ARGS.seed)
    ARGS.output.mkdir(parents=True, exist_ok=True)

    poses = np.load(ARGS.sequence / "target" / "target_c2w_local.npy")
    intrinsics = np.load(ARGS.sequence / "target" / "intrinsics.npy")
    required_frames = ARGS.chunks * CHUNK_STRIDE + 1
    if len(poses) < required_frames or len(intrinsics) < required_frames:
        raise ValueError(f"sequence has {len(poses)} frames, needs {required_frames}")
    source_path = ARGS.source_image or (ARGS.sequence / "source" / "source_perspective.png")
    source = Image.open(source_path).convert("RGB").resize((WIDTH, HEIGHT))
    prompt = (ARGS.sequence / "prompt.txt").read_text(encoding="utf-8").strip()

    rollout_store = NodeStore(ARGS.output / "validated_world_session")
    source_node = NodeStore(ARGS.sequence / "session").load("node_000")
    if ARGS.source_world_observed_erp_mask is not None:
        observed_mask = np.load(ARGS.source_world_observed_erp_mask)
        source_node, source_world_filter = filter_node_to_observed_erp(source_node, observed_mask)
        source_world_filter["observed_erp_mask"] = str(ARGS.source_world_observed_erp_mask)
    else:
        source_world_filter = {
            "mode": "full_completed_panorama_points",
            "original_point_count": int(len(source_node.points_xyz)),
            "kept_point_count": int(len(source_node.points_xyz)),
            "input_node_unchanged": True,
        }
    rollout_store.save(source_node)
    memory_config = load_yaml(ARGS.memory_config)
    if ARGS.coverage_threshold is not None:
        memory_config["coverage_threshold"] = float(ARGS.coverage_threshold)
    for override in ARGS.memory_set:
        key, separator, raw_value = override.partition("=")
        if not separator or key not in memory_config:
            raise ValueError(f"memory override must name an existing key, got {override!r}")
        original = memory_config[key]
        if isinstance(original, bool):
            memory_config[key] = raw_value.strip().lower() in {"1", "true", "yes", "on"}
        elif isinstance(original, int):
            memory_config[key] = int(raw_value)
        elif isinstance(original, float):
            memory_config[key] = float(raw_value)
        elif isinstance(original, str):
            memory_config[key] = raw_value.strip()
        else:
            raise TypeError(f"unsupported memory override type for {key}: {type(original).__name__}")
    geometry = Pi3GeometryBackend(
        ARGS.pi3_checkpoint, ARGS.pi3_repo, device="cuda:0", input_size=518,
    )
    manager = MemoryManager.from_config(
        memory_config, geometry_backend=geometry, node_store=rollout_store,
    )
    renderer = CausalActiveNodeRenderer(
        rollout_store, node_id="node_000", manager=manager,
        renderer_kwargs={"device": "cuda:0"},
    )

    pipe = WorldProjectedWarpAsHistoryPipeline.from_pretrained(
        str(ARGS.wah_model), torch_dtype=torch.bfloat16,
    ).to("cuda:0")
    if not hasattr(pipe.transformer.config, "image_dim"):
        pipe.transformer.register_to_config(image_dim=None)
    controller = install_spatial_reanchor(
        pipe.transformer, rank=64, refresh_blocks=(0, 10, 20, 30), gate_init=0.05,
    ).to("cuda:0")
    controller.enable_spatial_memory_attention = False
    state = pipe.init_autoregressive_state(
        prompt=prompt,
        image=source,
        conditioning_type="warp",
        lora_path=str(ARGS.lora),
        lora_prompt_trigger="camctl23x.",
        visible_token_drop=True,
        warp_history_downsample_mode="short",
        rope_alignment=True,
        height=HEIGHT,
        width=WIDTH,
        num_frames=CHUNK_FRAMES,
        output_type="np",
        generator=torch.Generator(device="cuda:0").manual_seed(ARGS.seed),
        add_noise_to_image_latents=False,
        add_noise_to_warp_latents=False,
        pyramid_num_inference_steps_list=list(PYRAMID_STEPS),
        is_amplify_first_chunk=False,
        prev_chunk_history_sizes=(16, 2, 1),
    )
    if int(state["indices_latents_history_short"][0, 0]) != 0:
        raise RuntimeError("permanent source prefix temporal RoPE must be zero")
    if ARGS.phase_a_zero_source_checkpoint is not None:
        initialization_mode = "phase_a_zero_step"
        initialization_path = ARGS.phase_a_zero_source_checkpoint
        checkpoint_report = load_phase_a_zero(pipe.transformer, initialization_path)
    else:
        initialization_mode = "phase_a_step600"
        initialization_path = ARGS.checkpoint
        checkpoint_report = load_step600(pipe.transformer, initialization_path)
    for parameter in pipe.transformer.parameters():
        parameter.requires_grad_(False)
    for parameter in pipe.vae.parameters():
        parameter.requires_grad_(False)
    if any(parameter.requires_grad for parameter in pipe.transformer.parameters()):
        raise RuntimeError("training-free inference left trainable transformer parameters")
    fixed_prefix = state["image_latents"].detach().clone()
    projection_config = WorldProjectionConfig(
        lambda_max_by_stage=(
            ARGS.lambda_max_stage0,
            ARGS.lambda_max_stage1,
            ARGS.lambda_max_stage2,
        ),
        gamma=ARGS.gamma,
        confidence_ramp_min=ARGS.confidence_ramp_min,
        confidence_ramp_max=ARGS.confidence_ramp_max,
    )
    startup = {
        "state": "running",
        "initialization_mode": initialization_mode,
        "project_global_step": int(checkpoint_report.get("project_global_step", 600)),
        "checkpoint": str(initialization_path),
        "checkpoint_report": checkpoint_report,
        "training_free": True,
        "optimizer_created": False,
        "requires_grad_parameter_count": 0,
        "source_prefix_fixed": True,
        "source_prefix_rope": 0,
        "source_image": str(source_path),
        "spatial_memory_attention_enabled": False,
        "spatial_memory_parameters_deleted": False,
        "generation_mode": "stage2_sparse_pixel_constraint",
        "canonical_residual_formula": None,
        "soft_wpf_enabled": False,
        "projection": {
            "mode": "stage2_clean_x0_sparse_pixel_constraint",
            "soft_wpf_enabled": False,
            "formula": "L_pixel=sum(V*abs(D(x0_opt)-I_warp))/(3*sum(V)+eps) + lambda_z*mean((x0_opt-x0_base)^2)",
            "stages": [2],
            "decodes_noisy_latent": False,
            "joint_33_frame_optimization": True,
            "vae_backward_activation_offload": "saved_tensors_cpu",
            "optimizer_created": False,
            "vae_encode_used": False,
            "step0": {"steps": 1, "lr": 0.005, "lambda_z": 1.0, "max_grad_norm": 1.0},
            "final": {"steps": 1, "lr": 0.002, "lambda_z": 2.0, "max_grad_norm": 1.0},
            "coverage_threshold_used": False,
            "nearest_fill_used": False,
        },
        "legacy_soft_wpf_parameters_ignored": vars(projection_config),
        "pyramid_num_inference_steps_list": list(PYRAMID_STEPS),
        "canonical_warp_vae_fill": "per_frame_nearest_visible_with_chunk_mean_fallback",
        "canonical_support_temporal_groups": [
            [0], [1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12],
            [13, 14, 15, 16], [17, 18, 19, 20], [21, 22, 23, 24],
            [25, 26, 27, 28], [29, 30, 31, 32],
        ],
        "canonical_support_shared_by": ["Spatial Anchor", "Boundary Bridge"],
        "canonical_safe_support_threshold": WORLD_OWNERSHIP_COVERAGE_THRESHOLD,
        "world_ownership_binary": True,
        "world_ownership_applies_to": ["Spatial Anchor", "Boundary Bridge"],
        "wah_conditioning_path": "pinned_original",
        "wah_warp_rgb_fill": True,
        "wah_safe_support_override": False,
        "wah_world_ownership_override": False,
        "sparse_constraint_visibility_source": "raw renderer visibility",
        "sparse_constraint_coverage_threshold": None,
        "shared_world_boundary_slot": 0,
        "source_world_filter": source_world_filter,
        "deprecated_confidence_threshold_ignored": ARGS.confidence_threshold,
        "memory_config": memory_config,
        "candidate_acceptance_policy": {
            "permanent": True,
            "minimum_history_frames": 12,
            "mode": "any",
            "translation": 2.5,
            "view_change_degrees": 25.0,
            "mean_new_area_ratio": 0.05,
            "maximum_mean_world_overlap_inclusive": 0.50,
            "secondary_rejection_gates": [],
        },
        "world_promotion_mode": "preserve_parent_append_eligible_novel_points",
        "parent_points_always_rendered": True,
        "generated_points_must_avoid_parent_projection": True,
        "anchor_adapter_enabled": not ARGS.disable_anchor_adapter,
        "camera_adapter_enabled": not ARGS.disable_camera_adapter,
        "spatial_warp_enabled": not ARGS.disable_spatial_warp,
        "new_node_render_delay_chunks": ARGS.node_activation_delay_chunks,
        "maximum_pending_accepted_nodes": 1,
        "boundary_projection": {
            "source": "previous_chunk_true_shared_generated_rgb_frame",
            "temporal_slots": [0],
            "beta_max_by_stage": list(projection_config.boundary_beta_max_by_stage),
            "scheduler_aligned": True,
            "wpf_uses_remaining_slot0_weight": False,
            "combined_with_wpf_from_same_z_raw": False,
            "combined_with_residual_scheduler_output": True,
        },
        "uses_future_gt": False,
        "chunks_total": ARGS.chunks,
        "chunks_complete": 0,
    }
    write_json(ARGS.output / "status.json", startup)
    writer = imageio.get_writer(ARGS.output / ARGS.video_name, fps=ARGS.fps, macro_block_size=1)
    debug_writer = imageio.get_writer(
        ARGS.output / "debug_4panel_world_projected.mp4", fps=ARGS.fps, macro_block_size=1,
    )
    records = []
    started = time.perf_counter()
    previous_generated = previous_warp = previous_projection_boundary = None
    generated_motion, warp_motion, rotation_motion = [], [], []
    visible_l1 = []
    boundary_generated_l1, boundary_warp_l1, boundary_projection_l1 = [], [], []
    activation_queue = DelayedNodeActivationQueue(
        delay_chunks=ARGS.node_activation_delay_chunks, max_pending=1,
    )
    try:
        with torch.no_grad():
            for chunk_index in range(ARGS.chunks):
                chunk_started = time.perf_counter()
                start = chunk_index * CHUNK_STRIDE
                scheduled_at_start = activation_queue.pending
                activated = activation_queue.activate_due(chunk_index)
                activated_node_id = None
                activated_shadow_hash = None
                activated_shadow_hash_at_creation = None
                activated_shadow_hash_equal = None
                if activated is not None:
                    # A shadow is immutable while pending.  Verify its point
                    # payload before the one and only parent->child commit,
                    # then render the newly active cumulative node this chunk.
                    activation_parent = renderer.active_node
                    activated_shadow_hash_at_creation = getattr(
                        activated.node, "shadow_hash_at_creation", None,
                    ) or activated.node.quality_metrics.get("shadow_hash_at_creation")
                    activated_shadow_hash = manager.verify_shadow(activated.node)
                    renderer.active_node = manager.commit_shadow(
                        activation_parent,
                        activated.node,
                        verified_hash=activated_shadow_hash,
                    )
                    activated_shadow_hash_equal = True
                    activated_node_id = activated.node_id
                scheduled_node_id = (
                    scheduled_at_start.node_id
                    if scheduled_at_start is not None else renderer.active_node.node_id
                )
                promotion_blocked_by_pending = activation_queue.pending is not None
                cameras = CameraBatch(
                    poses[start:start + CHUNK_FRAMES],
                    intrinsics[start:start + CHUNK_FRAMES],
                    HEIGHT, WIDTH,
                )
                rendered = renderer.render(
                    cameras, frame_start=start, allow_reactivation=False,
                )
                warp = rendered.warp
                node_at_start = renderer.active_node
                # Fill invalid renderer pixels exactly once.  The visibility,
                # confidence, depth, and source arrays remain the raw renderer
                # evidence; this RGB copy is used only as shared conditioning.
                canonical_warp_rgb = fill_invalid_warp_for_vae(
                    warp.rgb, warp.visibility,
                )
                canonical_first_frame_latent, canonical_latent = clean_warp_latents(
                    pipe, state, warp, canonical_warp_rgb=canonical_warp_rgb,
                    return_first_frame=True,
                )
                canonical_support = build_canonical_world_support(
                    np.asarray(warp.visibility, np.float32),
                    np.asarray(warp.confidence, np.float32),
                    latent_frames=LATENT_FRAMES,
                    latent_height=int(canonical_latent.shape[-2]),
                    latent_width=int(canonical_latent.shape[-1]),
                    temporal_scale=VAE_TEMPORAL_SCALE,
                )
                previous_world_boundary_latent = state.get("previous_world_boundary_latent")
                previous_world_boundary_visibility = state.get(
                    "previous_world_boundary_visibility"
                )
                previous_world_boundary_confidence = state.get(
                    "previous_world_boundary_confidence"
                )
                clean, canonical_first_frame_latent, canonical_support, boundary_world_shared = (
                    apply_previous_world_boundary(
                        canonical_latent,
                        canonical_first_frame_latent,
                        canonical_support,
                        previous_latent=previous_world_boundary_latent,
                        previous_visibility=previous_world_boundary_visibility,
                        previous_confidence=previous_world_boundary_confidence,
                    )
                )
                world_boundary_latent = clean_boundary_frame_latent(
                    pipe, state, canonical_warp_rgb[-1],
                )
                world_boundary_support = build_single_frame_world_support(
                    np.asarray(warp.visibility[-1], np.float32),
                    np.asarray(warp.confidence[-1], np.float32),
                    latent_height=int(world_boundary_latent.shape[-2]),
                    latent_width=int(world_boundary_latent.shape[-1]),
                )
                world_boundary_latent = mask_canonical_latent(
                    world_boundary_latent,
                    world_boundary_support.safe_support.to(world_boundary_latent.device),
                )
                stage_shapes = prepare_spatial_context(
                    controller, pipe.transformer, clean, canonical_support,
                    poses[start:start + CHUNK_FRAMES],
                    intrinsics[start:start + CHUNK_FRAMES], start,
                    anchor_enabled=not ARGS.disable_anchor_adapter,
                    camera_enabled=not ARGS.disable_camera_adapter,
                    spatial_warp_enabled=not ARGS.disable_spatial_warp,
                )
                projection = build_world_projection_context(
                    clean,
                    np.asarray(warp.visibility, np.float32),
                    np.asarray(warp.confidence * warp.visibility, np.float32),
                    stage_count=len(PYRAMID_STEPS),
                    temporal_scale=VAE_TEMPORAL_SCALE,
                    config=projection_config,
                    previous_clean_boundary_latent=state.get(
                        "previous_shared_frame_clean_latent"
                    ),
                    canonical_support=canonical_support,
                )
                boundary_source_global_frame = state.get("boundary_source_global_frame")
                pipe.set_world_projection_context(projection)
                pipe.set_sparse_pixel_constraint_context(
                    warp.rgb, warp.visibility, height=HEIGHT, width=WIDTH,
                )
                try:
                    output, state = pipe.generate_next_chunk(
                        state,
                        warp_video=canonical_warp_rgb,
                        warp_visibility_mask=np.asarray(warp.visibility, np.float32)[None, None],
                        warp_confidence_mask=np.asarray(
                            warp.confidence * warp.visibility, np.float32,
                        )[None, None],
                        output_type="np",
                    )
                finally:
                    pipe.clear_world_projection_context()
                    controller.clear_context()
                canonical_conditioning_cache_hit = None
                generated = video_array(output)
                if len(generated) != CHUNK_FRAMES:
                    raise RuntimeError(f"chunk returned {len(generated)} frames, expected {CHUNK_FRAMES}")
                if not torch.equal(state["image_latents"], fixed_prefix):
                    raise RuntimeError("AR generation replaced the permanent source prefix")
                state["previous_shared_frame_clean_latent"] = clean_boundary_frame_latent(
                    pipe, state, generated[-1],
                )
                state["boundary_source_global_frame"] = int(start + CHUNK_STRIDE)
                state["previous_world_boundary_latent"] = world_boundary_latent.detach()
                state["previous_world_boundary_visibility"] = (
                    world_boundary_support.safe_support.detach()
                )
                state["previous_world_boundary_confidence"] = (
                    world_boundary_support.confidence.detach()
                )

                first = 0 if chunk_index == 0 else 1
                memory_generated = generated[first:]
                memory_cameras = CameraBatch(
                    poses[start + first:start + CHUNK_FRAMES],
                    intrinsics[start + first:start + CHUNK_FRAMES],
                    HEIGHT, WIDTH,
                )
                memory_warp = slice_warp(warp, first)
                GeneratedMemoryBatch(memory_generated)
                next_active, event = manager.process_chunk(
                    node_at_start,
                    generated_rgb_for_memory=memory_generated,
                    cameras=memory_cameras,
                    warp=memory_warp,
                    frame_start=start + first,
                    allow_candidate_promotion=not promotion_blocked_by_pending,
                    defer_candidate_promotion=True,
                )
                candidate_schedule = None
                shadow_node = event.get("shadow_node")
                if event.get("accepted"):
                    if shadow_node is None:
                        raise RuntimeError("deferred candidate event did not expose its shadow node")
                    if shadow_node.status != "shadow" or next_active.node_id != node_at_start.node_id:
                        raise RuntimeError("candidate shadow changed the active node before activation")
                    candidate_schedule = activation_queue.schedule(
                        shadow_node, created_after_chunk=chunk_index,
                    )
                scheduled_node_id = (
                    activation_queue.pending.node_id
                    if activation_queue.pending is not None
                    else (activated.node_id if activated is not None else renderer.active_node.node_id)
                )
                activation_plan_for_record = activation_queue.pending or activated
                metrics = event.get("metrics") or {}
                verified_ratio = None
                promotion_metrics = {}
                if event.get("accepted"):
                    metrics_node = shadow_node or next_active
                    verified_ratio = float(metrics_node.quality_metrics.get("verified_point_ratio", 0.0))
                    promotion_metrics = {
                        key: metrics_node.quality_metrics.get(key)
                        for key in (
                            "parent_points_preserved",
                            "parent_point_count",
                            "eligible_candidate_point_count",
                            "appended_eligible_point_count",
                            "discarded_ineligible_candidate_point_count",
                            "discarded_duplicate_eligible_point_count",
                            "cumulative_point_count",
                        )
                    }

                warp_u8 = u8(warp.rgb)
                stage_diagnostics = projection.diagnostics
                if len(stage_diagnostics) != sum(PYRAMID_STEPS):
                    raise RuntimeError(
                        f"expected {sum(PYRAMID_STEPS)} scheduler projections, got {len(stage_diagnostics)}"
                    )
                for stage_id, expected_steps in enumerate(PYRAMID_STEPS):
                    stage_items = [item for item in stage_diagnostics if item["stage_id"] == stage_id]
                    if [item["step_id"] for item in stage_items] != list(range(expected_steps)):
                        raise RuntimeError(f"stage {stage_id} did not execute {expected_steps} ordered updates")
                    if stage_id == 0 and any(item["boundary_delta_ratio"] != 0.0 for item in stage_items):
                        raise RuntimeError("stage0 Boundary Bridge must remain exactly disabled")
                    expected_sparse = 1.0 if stage_id == 2 else 0.0
                    if any(item.get("sparse_pixel_constraint", 0.0) != expected_sparse for item in stage_items):
                        raise RuntimeError(
                            f"stage {stage_id} sparse constraint activation does not match stage-2-only policy"
                        )
                    if stage_id < 2 and any(
                        item["projection_strength_mean"] != 0.0 for item in stage_items
                    ):
                        raise RuntimeError(f"stage {stage_id} unexpectedly applied sparse constraint")
                    if stage_id == 2 and any(
                        item["sparse_optimizer_created"] != 0.0
                        or item["sparse_vae_encode_used"] != 0.0
                        or item["sparse_new_noise_sampled"] != 0.0
                        or item["sparse_clipped_grad_norm"] > 1.000001
                        for item in stage_items
                    ):
                        raise RuntimeError("stage2 sparse pixel constraint invariants failed")
                    if stage_id > 0:
                        next_sigmas = [item["next_sigma"] for item in stage_items]
                        if len(set(next_sigmas)) != expected_steps:
                            raise RuntimeError(f"stage {stage_id} did not expose distinct next_sigma values")
                        boundary_strengths = [item["boundary_strength"] for item in stage_items]
                        if projection.previous_boundary_latents is not None and any(
                            right < left for left, right in zip(
                                boundary_strengths, boundary_strengths[1:],
                            )
                        ):
                            raise RuntimeError(
                                f"stage {stage_id} Boundary Bridge strength did not increase"
                            )
                boundary_non_slot0_delta_max = max(
                    item["boundary_non_slot0_delta_max"] for item in stage_diagnostics
                )
                if boundary_non_slot0_delta_max != 0.0:
                    raise RuntimeError("Boundary Bridge changed temporal slots 1..8")
                boundary_delta_ratio_by_stage = {
                    str(stage_id): [
                        item["boundary_delta_ratio"] for item in stage_diagnostics
                        if item["stage_id"] == stage_id
                    ]
                    for stage_id in range(len(PYRAMID_STEPS))
                }
                projection_weight = projection.final_projection_weight
                if projection_weight is None or tuple(projection_weight.shape[2:]) != (33, HEIGHT, WIDTH):
                    raise RuntimeError("sparse constraint visibility must be raw [B,1,33,H,W] pixels")
                projection_weight = projection_weight.detach().float().cpu()
                chunk_boundary = None
                if chunk_index > 0:
                    chunk_boundary = {
                        "global_previous_frame": int(start),
                        "global_current_frame": int(start + 1),
                        "generation_l1": float(np.abs(
                            generated[1].astype(np.float32) - previous_generated.astype(np.float32)
                        ).mean() / 255.0),
                        "warp_l1": float(np.abs(
                            warp_u8[1].astype(np.float32) - previous_warp.astype(np.float32)
                        ).mean() / 255.0),
                        "projection_mask_l1": float(
                            (projection_weight[0, 0, 1] - previous_projection_boundary).abs().mean()
                        ),
                    }
                    boundary_generated_l1.append(chunk_boundary["generation_l1"])
                    boundary_warp_l1.append(chunk_boundary["warp_l1"])
                    boundary_projection_l1.append(chunk_boundary["projection_mask_l1"])
                for local in range(first, CHUNK_FRAMES):
                    global_index = start + local
                    generated_frame, warp_frame = generated[local], warp_u8[local]
                    writer.append_data(generated_frame)
                    debug_writer.append_data(debug_panel(
                        generated_frame, warp_frame, warp.visibility[local], warp.confidence[local],
                        global_index, chunk_index,
                    ))
                    mask = np.asarray(warp.visibility[local], bool)
                    if mask.any():
                        visible_l1.append(float(np.abs(
                            generated_frame.astype(np.float32) / 255.0
                            - warp_frame.astype(np.float32) / 255.0
                        )[mask].mean()))
                    if previous_generated is not None:
                        generated_motion.append(float(np.abs(
                            generated_frame.astype(np.float32) - previous_generated.astype(np.float32)
                        ).mean() / 255.0))
                        warp_motion.append(float(np.abs(
                            warp_frame.astype(np.float32) - previous_warp.astype(np.float32)
                        ).mean() / 255.0))
                        relative_rotation = poses[global_index - 1, :3, :3].T @ poses[global_index, :3, :3]
                        cosine = np.clip((np.trace(relative_rotation) - 1.0) / 2.0, -1.0, 1.0)
                        rotation_motion.append(float(np.degrees(np.arccos(cosine))))
                    previous_generated, previous_warp = generated_frame, warp_frame
                previous_projection_boundary = projection_weight[0, 0, -1].clone()

                visible = np.asarray(warp.visibility, bool)
                confidence = np.asarray(warp.confidence, np.float32)
                fill_only = canonical_support.safe_support.to(clean.device) <= 0
                fill_only_latent_abs_max = float(
                    clean.float().masked_select(fill_only.expand_as(clean)).abs().max().cpu()
                    if bool(fill_only.any()) else 0.0
                )
                if fill_only_latent_abs_max != 0.0:
                    raise RuntimeError("fill-only canonical latent was not hard masked")
                ownership = canonical_support.world_ownership_mask
                if not bool(((ownership == 0) | (ownership == 1)).all()):
                    raise RuntimeError("world ownership mask is not binary")
                fill_owned_latent_count = int((
                    (canonical_support.visibility < WORLD_OWNERSHIP_COVERAGE_THRESHOLD)
                    & (ownership > 0)
                ).sum().item())
                if fill_owned_latent_count != 0:
                    raise RuntimeError("fill-only latent cells acquired world ownership")
                world_owned_latent_ratio = float(ownership.float().mean().item())
                shared_boundary_latent_max_abs_diff = None
                shared_boundary_visibility_max_abs_diff = None
                shared_boundary_confidence_max_abs_diff = None
                if boundary_world_shared:
                    shared_boundary_latent_max_abs_diff = float((
                        clean[:, :, 0:1].float()
                        - previous_world_boundary_latent.to(clean.device).float()
                    ).abs().max().cpu())
                    shared_boundary_visibility_max_abs_diff = float((
                        canonical_support.safe_support[:, :, 0:1].float()
                        - previous_world_boundary_visibility.to(
                            canonical_support.safe_support.device
                        ).float()
                    ).abs().max().cpu())
                    shared_boundary_confidence_max_abs_diff = float((
                        canonical_support.confidence[:, :, 0:1].float()
                        - previous_world_boundary_confidence.to(
                            canonical_support.confidence.device
                        ).float()
                    ).abs().max().cpu())
                    if any(value != 0.0 for value in (
                        shared_boundary_latent_max_abs_diff,
                        shared_boundary_visibility_max_abs_diff,
                        shared_boundary_confidence_max_abs_diff,
                    )):
                        raise RuntimeError("shared canonical world boundary slot0 is not exact")
                shadow_metadata_node = (
                    shadow_node
                    if shadow_node is not None
                    else (activated.node if activated is not None else None)
                )
                record = {
                    "chunk_index": chunk_index,
                    "active_node_id": node_at_start.node_id,
                    "active_node_for_next_chunk": renderer.active_node.node_id,
                    "node_activated_at_chunk_start": activated_node_id,
                    "scheduled_node_id": scheduled_node_id,
                    "render_node_id": node_at_start.node_id,
                    "pending_node_id": (
                        activation_queue.pending.node_id if activation_queue.pending else None
                    ),
                    "created_after_chunk": (
                        activation_plan_for_record.created_after_chunk
                        if activation_plan_for_record else None
                    ),
                    "activate_at_chunk": (
                        activation_plan_for_record.activate_at_chunk
                        if activation_plan_for_record else None
                    ),
                    "candidate_created_after_chunk": (
                        candidate_schedule.created_after_chunk if candidate_schedule else None
                    ),
                    "candidate_activate_at_chunk": (
                        candidate_schedule.activate_at_chunk if candidate_schedule else None
                    ),
                    "candidate_renderable_from_chunk": (
                        candidate_schedule.activate_at_chunk if candidate_schedule else None
                    ),
                    "candidate_rendered_in_creation_chunk": False,
                    "candidate_promotion_blocked_by_pending": promotion_blocked_by_pending,
                    "world_coverage": float(np.asarray(warp.coverage_per_frame).mean()),
                    "world_confidence_mean": float(confidence[visible].mean() if visible.any() else 0.0),
                    "canonical_support_shape": list(canonical_support.safe_support.shape),
                    "canonical_conditioning_cache_hit": canonical_conditioning_cache_hit,
                    "wah_conditioning_path": "pinned_original",
                    "wah_safe_support_override": False,
                    "canonical_visibility_mean": float(canonical_support.visibility.mean()),
                    "canonical_safe_support_mean": float(canonical_support.safe_support.mean()),
                    "canonical_confidence_mean": float(canonical_support.confidence.mean()),
                    "world_ownership_binary": True,
                    "world_ownership_coverage_threshold": WORLD_OWNERSHIP_COVERAGE_THRESHOLD,
                    "world_ownership_applies_to": "Spatial Anchor / Boundary Bridge only",
                    "sparse_constraint_visibility_source": "raw renderer pixels",
                    "sparse_constraint_coverage_threshold": None,
                    "sparse_constraint_nearest_fill_used": False,
                    "fill_owned_latent_count": fill_owned_latent_count,
                    "world_owned_latent_ratio": world_owned_latent_ratio,
                    "fill_only_latent_abs_max": fill_only_latent_abs_max,
                    "shared_world_boundary_applied": boundary_world_shared,
                    "shared_boundary_latent_max_abs_diff": shared_boundary_latent_max_abs_diff,
                    "shared_boundary_visibility_max_abs_diff": shared_boundary_visibility_max_abs_diff,
                    "shared_boundary_confidence_max_abs_diff": shared_boundary_confidence_max_abs_diff,
                    "candidate_created": "candidate_id" in event,
                    "candidate_accepted": bool(event.get("accepted", False)),
                    "candidate_id": event.get("candidate_id"),
                    "candidate_rejection_reasons": event.get("rejection_reason", []),
                    "verified_point_ratio": verified_ratio,
                    "promotion_metrics": promotion_metrics,
                    "candidate_metrics": metrics,
                    "shadow_frozen": event.get("shadow_frozen", False),
                    "shadow_hash_at_creation": (
                        activated_shadow_hash_at_creation
                        if activated is not None else event.get("shadow_hash_at_creation")
                    ),
                    "shadow_hash_at_activation": (
                        activated_shadow_hash if activated is not None else None
                    ),
                    "shadow_hash_equal": (
                        activated_shadow_hash_equal if activated is not None else None
                    ),
                    "new_shadow_hash_at_creation": event.get("shadow_hash_at_creation"),
                    "shadow_boundary_frame": (
                        shadow_metadata_node.quality_metrics.get("shadow_boundary_frame")
                        if shadow_metadata_node is not None else None
                    ),
                    "shadow_mapping_frame_indices": (
                        shadow_metadata_node.quality_metrics.get("shadow_mapping_frame_indices")
                        if shadow_metadata_node is not None else None
                    ),
                    "delta_output_on_parent_visible": getattr(
                        warp, "delta_output_on_parent_visible", 0
                    ),
                    "delta_output_on_parent_protection_mask": getattr(
                        warp, "delta_output_on_parent_protection_mask", 0
                    ),
                    "transition_buffer": {
                        "frame_count": len(manager.buffer),
                        "translation_baseline": manager.buffer.translation_baseline,
                        "view_diversity": manager.buffer.view_diversity,
                        "mean_new_area_ratio": manager.buffer.mean_new_area_ratio,
                    },
                    "candidate_readiness": event["readiness"],
                    "projection_stages": stage_diagnostics,
                    "canonical_residual": projection.final_residual_diagnostics,
                    "sparse_pixel_constraint": projection.final_residual_diagnostics,
                    "boundary_projection_active": bool(
                        projection.previous_boundary_latents is not None
                    ),
                    "boundary_source_global_frame": boundary_source_global_frame,
                    "boundary_delta_ratio_by_stage": boundary_delta_ratio_by_stage,
                    "boundary_non_slot0_delta_max": boundary_non_slot0_delta_max,
                    "saved_clean_boundary_latent_shape": list(
                        state["previous_shared_frame_clean_latent"].shape
                    ),
                    "chunk_boundary": chunk_boundary,
                    "stage_shapes": stage_shapes,
                    "unknown_projection_exact_zero": all(
                        item["unknown_projection_delta_max"] == 0.0 for item in projection.diagnostics
                    ),
                    "uses_future_gt": False,
                    "seconds": time.perf_counter() - chunk_started,
                }
                records.append(record)
                startup.update({
                    "chunks_complete": chunk_index + 1,
                    "last_chunk": record,
                    "elapsed_seconds": time.perf_counter() - started,
                    "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(0)),
                    "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(0)),
                })
                write_json(ARGS.output / "status.json", startup)
                print(json.dumps({"event": "chunk_complete", **record}), flush=True)
                clear_decoded_history(state)
                del generated, clean, projection, output
    except Exception as error:
        startup.update({"state": "failed", "error": repr(error)})
        write_json(ARGS.output / "status.json", startup)
        raise
    finally:
        writer.close()
        debug_writer.close()

    def correlation(left, right):
        if len(left) < 2 or np.std(left) == 0 or np.std(right) == 0:
            return 0.0
        return float(np.corrcoef(left, right)[0, 1])

    result = {
        **startup,
        "state": "complete",
        "chunks": records,
        "final_active_node_id": renderer.active_node.node_id,
        "candidate_created_count": sum(item["candidate_created"] for item in records),
        "candidate_accepted_count": sum(item["candidate_accepted"] for item in records),
        "visible_generation_warp_l1_mean": float(np.mean(visible_l1)),
        "visible_generation_warp_l1_final": float(visible_l1[-1]),
        "generation_warp_motion_correlation": correlation(generated_motion, warp_motion),
        "generation_rotation_motion_correlation": correlation(generated_motion, rotation_motion),
        "warp_rotation_motion_correlation": correlation(warp_motion, rotation_motion),
        "chunk_boundary_generation_l1_mean": float(np.mean(boundary_generated_l1)) if boundary_generated_l1 else 0.0,
        "chunk_boundary_warp_l1_mean": float(np.mean(boundary_warp_l1)) if boundary_warp_l1 else 0.0,
        "chunk_boundary_projection_mask_l1_mean": (
            float(np.mean(boundary_projection_l1)) if boundary_projection_l1 else 0.0
        ),
        "chunk_boundaries": [item["chunk_boundary"] for item in records if item["chunk_boundary"]],
        "elapsed_seconds": time.perf_counter() - started,
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(0)),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(0)),
    }
    write_json(ARGS.output / "metrics.json", result)
    write_json(ARGS.output / "status.json", result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
