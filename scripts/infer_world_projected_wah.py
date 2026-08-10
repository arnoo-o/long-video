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
    visibility_to_target_tokens,
)
from long_video.wah.world_projected_pipeline import (
    WorldProjectedWarpAsHistoryPipeline,
    WorldProjectionConfig,
    build_world_projection_context,
    fill_invalid_warp_for_vae,
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


def clean_warp_latents(pipe, state, warp):
    generator = state["generator"]
    saved_rng = generator.get_state()
    filled_rgb = fill_invalid_warp_for_vae(warp.rgb, warp.visibility)
    pipe._prepare_autoregressive_warp_chunk(
        state,
        filled_rgb,
        np.asarray(warp.visibility, np.float32)[None, None],
        np.asarray(warp.confidence * warp.visibility, np.float32)[None, None],
    )
    raw = state["online_warp_video_tensor"]
    device = pipe._wah_execution_device()
    mean, std = pipe._latent_stats(device)
    try:
        _, clean = pipe.prepare_video_latents(
            raw,
            latents_mean=mean,
            latents_std=std,
            num_latent_frames_per_chunk=LATENT_FRAMES,
            dtype=torch.float32,
            device=device,
            generator=generator,
        )
    finally:
        generator.set_state(saved_rng)
    return clean.detach()


def prepare_spatial_context(controller, transformer, clean, warp, poses, intrinsics, start):
    patch = tuple(int(value) for value in transformer.config.patch_size)
    stage_contexts, shapes = [], []
    for stage in training_exact_pyramid_latents(clean, len(PYRAMID_STEPS)):
        latent_height, latent_width = map(int, stage.shape[-2:])
        stage_warp = resize_latents_spatial(clean, height=latent_height, width=latent_width)
        token_height, token_width = latent_height // patch[-2], latent_width // patch[-1]
        visible_tokens = visibility_to_target_tokens(
            np.asarray(warp.visibility, np.float32),
            latent_frames=LATENT_FRAMES,
            latent_height=latent_height,
            latent_width=latent_width,
            patch_height=patch[-2],
            patch_width=patch[-1],
            temporal_scale=VAE_TEMPORAL_SCALE,
        )
        confidence_tokens = visibility_to_target_tokens(
            np.asarray(warp.confidence * warp.visibility, np.float32),
            latent_frames=LATENT_FRAMES,
            latent_height=latent_height,
            latent_width=latent_width,
            patch_height=patch[-2],
            patch_width=patch[-1],
            temporal_scale=VAE_TEMPORAL_SCALE,
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
        anchor_enabled=True,
        camera_enabled=True,
        spatial_warp_enabled=True,
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
        "projection": vars(projection_config),
        "pyramid_num_inference_steps_list": list(PYRAMID_STEPS),
        "canonical_warp_vae_fill": "per_frame_nearest_visible_with_chunk_mean_fallback",
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
            "maximum_mean_world_overlap_exclusive": 0.20,
            "secondary_rejection_gates": [],
        },
        "world_promotion_mode": "preserve_parent_append_eligible_novel_points",
        "parent_points_always_rendered": True,
        "generated_points_must_avoid_parent_projection": True,
        "new_node_render_delay_chunks": 1,
        "boundary_projection": {
            "source": "previous_chunk_last_3_clean_temporal_latents",
            "beta": [0.6, 0.3, 0.1],
            "combined_with_wpf_from_same_z_raw": True,
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
    pending_active_node = None
    try:
        with torch.no_grad():
            for chunk_index in range(ARGS.chunks):
                chunk_started = time.perf_counter()
                start = chunk_index * CHUNK_STRIDE
                activated_node_id = None
                if pending_active_node is not None:
                    renderer.active_node = pending_active_node
                    activated_node_id = pending_active_node.node_id
                    pending_active_node = None
                cameras = CameraBatch(
                    poses[start:start + CHUNK_FRAMES],
                    intrinsics[start:start + CHUNK_FRAMES],
                    HEIGHT, WIDTH,
                )
                rendered = renderer.render(cameras, frame_start=start)
                warp = rendered.warp
                node_at_start = renderer.active_node
                clean = clean_warp_latents(pipe, state, warp)
                stage_shapes = prepare_spatial_context(
                    controller, pipe.transformer, clean, warp,
                    poses[start:start + CHUNK_FRAMES],
                    intrinsics[start:start + CHUNK_FRAMES], start,
                )
                projection = build_world_projection_context(
                    clean,
                    np.asarray(warp.visibility, np.float32),
                    np.asarray(warp.confidence * warp.visibility, np.float32),
                    stage_count=len(PYRAMID_STEPS),
                    temporal_scale=VAE_TEMPORAL_SCALE,
                    config=projection_config,
                    previous_clean_boundary_latent=state.get(
                        "previous_chunk_clean_boundary_latents"
                    ),
                    boundary_beta=(0.6, 0.3, 0.1),
                )
                pipe.set_world_projection_context(projection)
                try:
                    output, state = pipe.generate_next_chunk(
                        state,
                        warp_video=warp.rgb,
                        warp_visibility_mask=np.asarray(warp.visibility, np.float32)[None, None],
                        warp_confidence_mask=np.asarray(
                            warp.confidence * warp.visibility, np.float32,
                        )[None, None],
                        output_type="np",
                    )
                finally:
                    pipe.clear_world_projection_context()
                    controller.clear_context()
                generated = video_array(output)
                if len(generated) != CHUNK_FRAMES:
                    raise RuntimeError(f"chunk returned {len(generated)} frames, expected {CHUNK_FRAMES}")
                if not torch.equal(state["image_latents"], fixed_prefix):
                    raise RuntimeError("AR generation replaced the permanent source prefix")
                last_clean_latents = state.get("last_latents")
                if not isinstance(last_clean_latents, torch.Tensor) or last_clean_latents.shape[2] < 3:
                    raise RuntimeError("AR state did not expose three clean boundary latents")
                state["previous_chunk_clean_boundary_latents"] = (
                    last_clean_latents[:, :, -3:].detach().clone()
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
                )
                pending_active_node = next_active
                metrics = event.get("metrics") or {}
                verified_ratio = None
                promotion_metrics = {}
                if event.get("accepted"):
                    verified_ratio = float(next_active.quality_metrics.get("verified_point_ratio", 0.0))
                    promotion_metrics = {
                        key: next_active.quality_metrics.get(key)
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
                    if stage_id == 0 and any(item["projection_delta_ratio"] != 0.0 for item in stage_items):
                        raise RuntimeError("stage0 World Projection must remain exactly disabled")
                    if stage_id > 0:
                        next_sigmas = [item["next_sigma"] for item in stage_items]
                        if len(set(next_sigmas)) != expected_steps:
                            raise RuntimeError(f"stage {stage_id} did not expose distinct next_sigma values")
                        strengths = [item["projection_strength_mean"] for item in stage_items]
                        if any(right < left for left, right in zip(strengths, strengths[1:])):
                            raise RuntimeError(f"stage {stage_id} projection strength did not increase")
                projection_weight = projection.final_projection_weight
                if projection_weight is None or tuple(projection_weight.shape[2:]) != (9, 48, 80):
                    raise RuntimeError("final stage projection weight must be [B,1,9,48,80]")
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
                record = {
                    "chunk_index": chunk_index,
                    "active_node_id": node_at_start.node_id,
                    "active_node_for_next_chunk": next_active.node_id,
                    "node_activated_at_chunk_start": activated_node_id,
                    "candidate_renderable_from_chunk": (
                        chunk_index + 1 if event.get("accepted") else None
                    ),
                    "candidate_rendered_in_creation_chunk": False,
                    "world_coverage": float(np.asarray(warp.coverage_per_frame).mean()),
                    "world_confidence_mean": float(confidence[visible].mean() if visible.any() else 0.0),
                    "candidate_created": "candidate_id" in event,
                    "candidate_accepted": bool(event.get("accepted", False)),
                    "candidate_id": event.get("candidate_id"),
                    "candidate_rejection_reasons": event.get("rejection_reason", []),
                    "verified_point_ratio": verified_ratio,
                    "promotion_metrics": promotion_metrics,
                    "candidate_metrics": metrics,
                    "transition_buffer": {
                        "frame_count": len(manager.buffer),
                        "translation_baseline": manager.buffer.translation_baseline,
                        "view_diversity": manager.buffer.view_diversity,
                        "mean_new_area_ratio": manager.buffer.mean_new_area_ratio,
                    },
                    "candidate_readiness": event["readiness"],
                    "projection_stages": stage_diagnostics,
                    "boundary_projection_active": bool(
                        projection.previous_boundary_latents is not None
                    ),
                    "saved_clean_boundary_latent_shape": list(
                        state["previous_chunk_clean_boundary_latents"].shape
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
            if pending_active_node is not None:
                renderer.active_node = pending_active_node
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
