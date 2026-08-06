#!/usr/bin/env python3
"""Train a real masked single-chunk Oracle-M0 WAH LoRA on physical GPU 1."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import yaml


def _args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/oracle_wah_training.yaml")
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--set", action="append", default=[], dest="overrides")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--resume", default="")
    return parser.parse_args()


def _gpu_snapshot():
    command = ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_gpu_memory", "--format=csv,noheader,nounits"]
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def _frames(path):
    from PIL import Image
    return [Image.open(item).convert("RGB") for item in sorted(Path(path).glob("*.png"))]


def _mask_frames(values):
    import numpy as np
    from PIL import Image
    return [Image.fromarray(np.rint(np.clip(frame, 0, 1) * 255).astype(np.uint8), mode="L") for frame in values]


def _build_histories(opt, pipe, exact, device, mean, std, first_frame, prompt, warp_frames, visibility, confidence, seq):
    exact.history_visibility_extra_mask_frames = _mask_frames(visibility)
    exact.history_confidence_extra_mask_frames = _mask_frames(confidence)
    prompt_embeds, image_latents, fake_image_latents, video_latents = opt.prepare_condition(
        pipe, first_frame, prompt, exact, device, mean, std, history_frames=warp_frames
    )
    histories = opt.make_histories(
        pipe, image_latents, fake_image_latents, exact, device,
        video_latents=video_latents, seq=seq,
    )
    return prompt_embeds, histories


def _pixel_errors(target, warp, visibility):
    import numpy as np
    target = np.stack([np.asarray(frame, np.float32) / 255.0 for frame in target])
    warped = np.stack([np.asarray(frame, np.float32) / 255.0 for frame in warp])
    visible = np.asarray(visibility, bool)
    error = np.abs(target - warped).mean(-1)
    return {
        "visible_region_error": float(error[visible].mean()) if visible.any() else None,
        "new_region_error": float(error[~visible].mean()) if (~visible).any() else None,
    }


def main():
    args = _args()
    repo = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo))
    from long_video.config import load_yaml
    config = load_yaml(args.config, args.overrides)
    required = [key for key in ("wah_root", "wah_model", "checkpoint_root") if not config.get(key)]
    if required:
        raise ValueError(f"machine paths must be supplied by --set key=value: {required}")
    physical_gpu = int(config["physical_gpu"])
    gpu_processes_before = _gpu_snapshot()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(physical_gpu)
    os.environ.setdefault("XFORMERS_DISABLED", "1")
    # No torch or CUDA-backed import is allowed before the visibility mask above.
    import numpy as np
    import torch
    from PIL import Image
    if torch.cuda.device_count() != 1:
        raise RuntimeError(f"Expected one visible GPU, got {torch.cuda.device_count()}")
    torch.cuda.set_device(0)
    torch.cuda.reset_peak_memory_stats(0)
    sys.path.insert(0, str(Path(config["wah_root"])))
    from warp_as_history.training import core as opt
    from long_video.oracle_training.contracts import assert_history_frames_are_generated
    from long_video.oracle_training.wah_training import (
        assert_only_lora_gradients, load_training_checkpoint,
        masked_flow_matching_loss, save_training_checkpoint,
    )

    sequence = Path(args.sequence)
    metadata = json.loads((sequence / "metadata.json").read_text(encoding="utf-8"))
    target_frames = _frames(sequence / "target" / "target_rgb_for_loss")[: int(metadata["chunk_frames"])]
    warp_frames = _frames(sequence / "single_chunk_warp" / "warp_rgb")
    visibility = np.load(sequence / "single_chunk_warp" / "warp_visibility.npy")
    confidence = np.load(sequence / "single_chunk_warp" / "warp_confidence.npy")
    primary_mask = np.load(sequence / "primary_loss_mask_latent.npy")
    if not (len(target_frames) == len(warp_frames) == len(visibility) == len(confidence) == int(metadata["chunk_frames"])):
        raise ValueError("target/warp/visibility/confidence frame counts must match chunk_frames exactly")
    assert_history_frames_are_generated(warp_frames, target_frames)
    prompt = (sequence / "prompt.txt").read_text(encoding="utf-8")

    exact = opt.parse_args([])
    exact.base_model_path = str(config["wah_model"])
    exact.transformer_path = str(config["wah_model"])
    exact.height, exact.width = map(int, config["perspective_resolution"])
    exact.num_frames = int(metadata["chunk_frames"])
    exact.num_latent_frames_per_chunk = int(len(primary_mask))
    exact.history_sizes = [16, 2, 1]
    exact.history_temporal_layout = "long_mid_short"
    exact.pyramid_num_inference_steps_list = list(config["training"]["pyramid_num_inference_steps_list"])
    exact.attention_backend = "native"
    exact.use_warp_as_history = True
    exact.warp_history_downsample_mode = "short"
    exact.history_positioning = "last_n_same_order"
    exact.history_position_count = int(len(primary_mask))
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
    exact.flow_matching_stage_id = 0
    exact.flow_matching_train_exact_timestep_sampling = "training_density"
    exact.flow_matching_use_dynamic_shifting = "off"
    exact.weighting_scheme = "none"
    exact.seed = int(config["seed"])
    exact.lora_rank = int(config["training"]["lora_rank"])
    exact.lora_alpha = int(config["training"]["lora_alpha"])
    exact.lora_dropout = float(config["training"]["lora_dropout"])
    exact.lora_target_modules = str(config["training"]["lora_target_modules"])
    exact.lora_adapter_name = "oracle_wah"
    exact.iters = int(args.max_steps or config["training"]["max_steps"])
    exact.gradient_checkpointing = True
    opt.validate_args(exact)
    device = torch.device("cuda:0")
    started = time.perf_counter()
    opt.seed_global_rng(exact.seed)
    pipe = opt.load_pipeline(exact, device)
    mean, std = opt.latent_stats(pipe, device)
    with torch.no_grad():
        target_latents = opt.encode_video_latents(pipe, target_frames, exact, device, mean, std).detach()
    if target_latents.shape[2] != len(primary_mask):
        raise ValueError(f"VAE produced {target_latents.shape[2]} latent frames, mask has {len(primary_mask)}")
    prompt_embeds, correct_histories = _build_histories(
        opt, pipe, exact, device, mean, std, target_frames[0], prompt,
        warp_frames, visibility, confidence, metadata["sequence_id"],
    )
    opt.seed_global_rng(exact.seed)
    fixed_items = opt.flow_matching_train_exact_items(pipe, target_latents, exact, device)
    adapter_name, trainable_params, lora_stats = opt.setup_visible_lora(pipe.transformer, exact, metadata["sequence_id"])
    optimizer = torch.optim.AdamW(
        trainable_params, lr=float(config["training"]["learning_rate"]), weight_decay=0.01
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _step: 1.0)
    checkpoint_root = Path(config["checkpoint_root"])
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_root / "oracle_wah_training_checkpoint.pt"
    global_step = 0
    if args.resume:
        global_step, _ = load_training_checkpoint(
            args.resume, pipe.transformer, optimizer, scheduler, adapter_name
        )

    pipe.transformer.eval()
    with torch.no_grad():
        initial_loss, _, _ = masked_flow_matching_loss(
            pipe, prompt_embeds, target_latents, correct_histories, exact, device,
            primary_mask, fixed_stage_items=fixed_items,
        )
    initial_loss_value = float(initial_loss.cpu())
    losses = []
    max_steps = min(int(args.max_steps or config["training"]["max_steps"]), 20)
    pipe.transformer.train()
    resume_verified = False
    while global_step < max_steps:
        optimizer.zero_grad(set_to_none=True)
        loss, stats, _ = masked_flow_matching_loss(
            pipe, prompt_embeds, target_latents, correct_histories, exact, device,
            primary_mask, fixed_stage_items=fixed_items,
        )
        loss.backward()
        assert_only_lora_gradients(pipe.transformer, trainable_params)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            trainable_params, float(config["training"]["max_grad_norm"])
        )
        optimizer.step(); scheduler.step(); global_step += 1
        losses.append({"step": global_step, "loss": float(loss.detach().cpu()), "grad_norm": float(grad_norm)})
        if global_step == max(1, max_steps - 1):
            save_training_checkpoint(
                checkpoint_path, pipe.transformer, trainable_params, optimizer, scheduler,
                global_step, adapter_name, {"sequence_id": metadata["sequence_id"]},
            )
            restored_step, restored_metadata = load_training_checkpoint(
                checkpoint_path, pipe.transformer, optimizer, scheduler, adapter_name
            )
            if restored_step != global_step or restored_metadata.get("sequence_id") != metadata["sequence_id"]:
                raise RuntimeError("checkpoint global step or metadata did not restore")
            resume_verified = True
    save_training_checkpoint(
        checkpoint_path, pipe.transformer, trainable_params, optimizer, scheduler,
        global_step, adapter_name, {"sequence_id": metadata["sequence_id"]},
    )
    opt.save_visible_lora_state(pipe.transformer, checkpoint_root, adapter_name, "oracle_wah_lora.pt")
    pipe.transformer.eval()
    with torch.no_grad():
        final_loss, _, _ = masked_flow_matching_loss(
            pipe, prompt_embeds, target_latents, correct_histories, exact, device,
            primary_mask, fixed_stage_items=fixed_items,
        )

    permutation = np.random.default_rng(exact.seed).permutation(len(warp_frames))
    variants = {
        "correct": (warp_frames, visibility, confidence),
        "shuffled": ([warp_frames[index] for index in permutation], visibility[permutation], confidence[permutation]),
        "empty": ([Image.new("RGB", (exact.width, exact.height), (0, 0, 0)) for _ in warp_frames], np.zeros_like(visibility), np.zeros_like(confidence)),
    }
    diagnostics = {}
    for name, (variant_frames, variant_visibility, variant_confidence) in variants.items():
        variant_prompt, variant_histories = _build_histories(
            opt, pipe, exact, device, mean, std, target_frames[0], prompt,
            variant_frames, variant_visibility, variant_confidence,
            f"{metadata['sequence_id']}_{name}",
        )
        with torch.no_grad():
            value, _, _ = masked_flow_matching_loss(
                pipe, variant_prompt, target_latents, variant_histories, exact, device,
                primary_mask, fixed_stage_items=fixed_items,
            )
        diagnostics[name] = {"loss": float(value.cpu()), **_pixel_errors(target_frames, variant_frames, variant_visibility)}
    result = {
        "sequence_id": metadata["sequence_id"], "physical_gpu": physical_gpu,
        "visible_device_count": torch.cuda.device_count(),
        "visible_device_name": torch.cuda.get_device_name(0),
        "chunk_frames": int(metadata["chunk_frames"]),
        "vae_temporal_scale": int(pipe.vae_scale_factor_temporal),
        "source_prefix_length_rgb": int(metadata["source_prefix_length_rgb"]),
        "actual_loss_latent_count": int(np.asarray(primary_mask, bool).sum()),
        "initial_fixed_batch_loss": initial_loss_value,
        "final_fixed_batch_loss": float(final_loss.cpu()),
        "optimizer_steps": global_step, "losses": losses,
        "checkpoint": str(checkpoint_path), "checkpoint_resume_verified": resume_verified,
        "optimizer_state_restored": resume_verified, "scheduler_state_restored": resume_verified,
        "lora": lora_stats, "diagnostics": diagnostics,
        "warp_learned": diagnostics["correct"]["loss"] < min(diagnostics["shuffled"]["loss"], diagnostics["empty"]["loss"]),
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(0)),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(0)),
        "elapsed_seconds": time.perf_counter() - started,
        "gpu_processes_before": gpu_processes_before,
        "gpu_processes_after": _gpu_snapshot(),
    }
    (checkpoint_root / "training_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    if "--manifest" in sys.argv:
        from train_oracle_wah_lora_24fps import main as multiwindow_main
        multiwindow_main()
    else:
        main()
