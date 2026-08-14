#!/usr/bin/env python3
"""Train WPF-aware Stage1/2 LoRA on causal DL3DV rollouts."""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image
import torch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wah-root", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--pi3-repo", type=Path, required=True)
    parser.add_argument("--pi3-checkpoint", type=Path, required=True)
    parser.add_argument("--initial-world-cache-root", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--total-steps", type=int, default=1400)
    parser.add_argument("--seed", type=int, default=20260814)
    parser.add_argument("--record-count", type=int, default=100)
    parser.add_argument("--prompt", default="A stable realistic view of the same scene.")
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--resume-checkpoint", type=Path)
    return parser.parse_args()


def read_rgb(path):
    return np.asarray(Image.open(path).convert("RGB"), np.uint8)


def load_record_arrays(root, record):
    rgb_paths = sorted((root / record["rgb_dir"]).glob("*"))
    if len(rgb_paths) != 193:
        raise ValueError(f"six-chunk training requires 193 frames, got {len(rgb_paths)}")
    if bool(record.get("uses_future_gt", True)):
        raise ValueError("future-GT record is forbidden")
    return {
        "rgb_paths": rgb_paths,
        "c2w": np.load(root / record["target_c2w_local"]).astype(np.float32),
        "k": np.load(root / record["intrinsics"]).astype(np.float32),
    }


def deterministic_gt_pyramid(pipe, frames, device):
    from long_video.wah.world_projected_pipeline import build_world_pyramid, posterior_mode_or_mean

    tensor = pipe._coerce_warp_video_tensor(
        np.asarray(frames), height=384, width=640, device=torch.device(device),
    ).to(dtype=pipe.vae.dtype)
    mean, std = pipe._latent_stats(torch.device(device))
    with torch.no_grad():
        clean = (posterior_mode_or_mean(pipe.vae.encode(tensor)) - mean) * std
    if tuple(clean.shape[1:]) != (16, 9, 48, 80):
        raise RuntimeError(f"unexpected deterministic GT latent shape {tuple(clean.shape)}")
    return build_world_pyramid(clean, 3)


def adapter_state(transformer):
    from long_video.training.wpf_adaptation import adaptation_parameter_items
    return {name: value.detach().cpu().clone()
            for name, value in adaptation_parameter_items(transformer)}


def scheduler_lambda(step, total_steps):
    if step < 50:
        return float(step + 1) / 50.0
    return max(0.0, 1.0 - (step - 50) / max(1, total_steps - 50))


def save_checkpoint(path, *, pipe, optimizer, lr_scheduler, step, sampler):
    payload = {
        "wpf_adaptation": adapter_state(pipe.transformer),
        "optimizer": optimizer.state_dict(),
        "lr_scheduler": lr_scheduler.state_dict(),
        "global_step": int(step),
        "sampling_state": sampler.state_dict(),
        "trajectory_counters": copy.deepcopy(sampler.trajectory_counts),
        "rng_state": {
            "python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
    }
    torch.save(payload, path)


def restore_checkpoint(path, *, pipe, optimizer, lr_scheduler, sampler):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    required = {"wpf_adaptation", "optimizer", "lr_scheduler", "global_step",
                "sampling_state", "rng_state"}
    if not required.issubset(checkpoint):
        raise RuntimeError(f"incomplete WPF adaptation checkpoint: {sorted(checkpoint)}")
    current = dict(pipe.transformer.named_parameters())
    expected = set(adapter_state(pipe.transformer))
    if set(checkpoint["wpf_adaptation"]) != expected:
        raise RuntimeError("resume checkpoint does not match wpf_adaptation adapter")
    with torch.no_grad():
        for name, value in checkpoint["wpf_adaptation"].items():
            current[name].copy_(value.to(device=current[name].device, dtype=current[name].dtype))
    optimizer.load_state_dict(checkpoint["optimizer"])
    lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
    sampler.load_state_dict(checkpoint["sampling_state"])
    rng = checkpoint["rng_state"]
    random.setstate(rng["python"]); np.random.set_state(rng["numpy"])
    torch.set_rng_state(rng["torch"])
    if torch.cuda.is_available() and rng.get("cuda") is not None:
        torch.cuda.set_rng_state_all(rng["cuda"])
    return int(checkpoint["global_step"])


def build_online(args, pipe, geometry, record):
    from long_video.config import load_yaml
    from long_video.memory.memory_manager import MemoryManager
    from long_video.memory.node_store import NodeStore
    from long_video.online.pipeline import OnlineSpatialHistoryPipeline

    cache = args.initial_world_cache_root / record["trajectory_id"]
    metadata_path = cache / "cache_metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"initial Pi3 world cache is required: {metadata_path}")
    metadata = json.loads(metadata_path.read_text())
    if metadata.get("trajectory_id") != record["trajectory_id"] or metadata.get("uses_future_gt") is not False:
        raise RuntimeError(f"invalid causal initial-world cache metadata: {metadata_path}")
    node = NodeStore(cache).load("node_000")
    manager = MemoryManager.from_config(load_yaml("configs/online_memory.yaml"), geometry_backend=geometry)
    online = OnlineSpatialHistoryPipeline(
        wah_pipeline=pipe, active_node=node, memory_manager=manager, prompt=args.prompt,
        renderer_kwargs={"device": args.device},
        wah_state_kwargs={
            "height": 384, "width": 640, "num_frames": 33, "output_type": "np",
            "pyramid_num_inference_steps_list": [2, 2, 2],
        },
    )
    source = read_rgb(args.dataset_root / record["source"])
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


def run_trajectory(args, pipe, geometry, record, supervision_chunk, selected_position):
    from long_video.training.wpf_adaptation import WPFAdaptationObserver

    arrays = load_record_arrays(args.dataset_root, record)
    online = build_online(args, pipe, geometry, record)
    reports, observer = [], None
    try:
        for chunk_index in range(int(supervision_chunk) + 1):
            indices = slice(chunk_index * 32, chunk_index * 32 + 33)
            if chunk_index == supervision_chunk:
                gt = [read_rgb(path) for path in arrays["rgb_paths"][indices]]
                gt_pyramid = deterministic_gt_pyramid(pipe, gt, args.device)
                observer = WPFAdaptationObserver(gt_pyramid, *selected_position)
                pipe._pyramid_training_observer = observer
            else:
                pipe._pyramid_training_observer = None
            _, _, warp, report = online.generate_chunk_at_cameras(
                arrays["c2w"][indices], arrays["k"][indices], 384, 640,
            )
            reports.append(report)
            if chunk_index == supervision_chunk:
                observer.assert_complete()
                expected = [(stage, step) for stage in range(3) for step in range(2)]
                actual = [(item["stage_id"], item["step_id"])
                          for item in report["world_projection_diagnostics"]]
                if actual != expected:
                    raise RuntimeError(f"formal WPF schedule was not executed: {actual}")
    finally:
        pipe._pyramid_training_observer = None
    event = reports[-1].get("memory_event") or {}
    return observer, online, reports, {
        "warp_visibility_ratio": float(np.asarray(warp.visibility).mean()),
        "world_point_count": int(len(online.active_node.points_xyz)),
        "promotion_count": int(sum(bool((item.get("memory_event") or {}).get("accepted"))
                                   for item in reports)),
        "candidate_points": int(event.get("candidate_point_count", event.get("candidate_points", 0)) or 0),
        "appended_points": int(event.get("appended_point_count", event.get("appended_points", 0)) or 0),
        "rejected_points": int(event.get("rejected_point_count", event.get("rejected_points", 0)) or 0),
    }


def main():
    args = parse_args()
    sys.path.insert(0, str(args.wah_root))
    from long_video.initialization.geometry_backend import Pi3GeometryBackend
    from long_video.training.wpf_adaptation import (
        WPFTrainingSampler, configure_trainable_wpf_adapter,
        select_balanced_training_records,
    )
    from long_video.wah.world_projected_pipeline import (
        WPF_LAMBDAS, WorldProjectedWarpAsHistoryPipeline,
    )

    if args.run_dir.exists() and any(args.run_dir.iterdir()):
        raise FileExistsError(f"run directory must be new: {args.run_dir}")
    metrics_dir = args.run_dir / "metrics"
    checkpoints_dir = args.run_dir / "checkpoints"
    metrics_dir.mkdir(parents=True)
    checkpoints_dir.mkdir()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    manifest = json.loads((args.dataset_root / "dl3dv_24fps_manifest.json").read_text())
    records = select_balanced_training_records(manifest["records"], args.record_count)
    missing = [record["trajectory_id"] for record in records if not (
        args.initial_world_cache_root / record["trajectory_id"] / "cache_metadata.json"
    ).is_file()]
    if missing:
        raise RuntimeError(f"missing cached initial Pi3 worlds: {missing[:8]} (total {len(missing)})")

    pipe = WorldProjectedWarpAsHistoryPipeline.from_pretrained(
        args.model, torch_dtype=torch.bfloat16,
    ).to(args.device)
    if not hasattr(pipe.transformer.config, "image_dim"):
        pipe.transformer.register_to_config(image_dim=None)
    official = args.wah_root / "checkpoints/warp-as-history/visible_lora_state_step1000.safetensors"
    if not official.is_file():
        raise FileNotFoundError(official)
    pipe._configure_wah_lora(str(official))
    trainable = configure_trainable_wpf_adapter(pipe)
    for module in (pipe.vae, pipe.text_encoder):
        for parameter in module.parameters():
            parameter.requires_grad_(False)
    if not hasattr(pipe.transformer, "enable_gradient_checkpointing"):
        raise RuntimeError("WPF adaptation training requires transformer gradient checkpointing")
    pipe.transformer.enable_gradient_checkpointing()
    optimizer_parameters = [parameter for _, parameter in trainable]
    if {id(value) for value in optimizer_parameters} != {
        id(value) for value in pipe.transformer.parameters() if value.requires_grad
    }:
        raise RuntimeError("optimizer must contain exactly wpf_adaptation parameters")
    optimizer = torch.optim.AdamW(optimizer_parameters, lr=1e-5, weight_decay=0.01)
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: scheduler_lambda(step, args.total_steps),
    )
    sampler = WPFTrainingSampler(args.seed)
    start_step = 0
    if args.resume_checkpoint is not None:
        start_step = restore_checkpoint(
            args.resume_checkpoint, pipe=pipe, optimizer=optimizer,
            lr_scheduler=lr_scheduler, sampler=sampler,
        )
    geometry = Pi3GeometryBackend(args.pi3_checkpoint, args.pi3_repo, args.device)
    status = {
        "pid": os.getpid(), "global_step": start_step, "total_steps": args.total_steps,
        "run_dir": str(args.run_dir), "metrics_dir": str(metrics_dir),
        "checkpoint_dir": str(checkpoints_dir),
        "state": "smoke" if args.smoke_only else "training",
        "training_record_count": len(records), "official_wah_frozen": True,
        "gradient_checkpointing": True,
    }
    (args.run_dir / "status.json").write_text(json.dumps(status, indent=2))

    if args.smoke_only:
        selected = (1, 0)
        before = adapter_state(pipe.transformer)
        optimizer.zero_grad(set_to_none=True)
        observer, online, reports, diagnostics = run_trajectory(
            args, pipe, geometry, records[0], 0, selected,
        )
        grad_norm = float(torch.nn.utils.clip_grad_norm_(optimizer_parameters, 1.0))
        only_adapter_has_grad = any(
            value.grad is not None and bool(torch.count_nonzero(value.grad))
            for value in optimizer_parameters
        ) and all(
            value.grad is None or not bool(torch.count_nonzero(value.grad))
            for name, value in pipe.transformer.named_parameters()
            if ".wpf_adaptation." not in name
        )
        optimizer.step()
        after = adapter_state(pipe.transformer)
        loss = observer.losses[selected]
        result = {
            "passed": bool(
                set(observer.losses) == {selected} and math.isfinite(loss["total"])
                and only_adapter_has_grad
                and any(not torch.equal(before[name], after[name]) for name in before)
                and len(reports) == 1 and online.chunk_index == 1
                and all(report.get("uses_future_gt") is False for report in reports)
            ),
            "selected_stage": selected[0], "selected_step": selected[1],
            "loss": loss, "grad_norm": grad_norm,
            "only_wpf_adaptation_has_grad": only_adapter_has_grad,
            "initial_world_from_cache": True,
            "wpf_lambdas": WPF_LAMBDAS, **diagnostics,
        }
        (args.run_dir / "smoke_result.json").write_text(json.dumps(result, indent=2))
        if not result["passed"]:
            raise RuntimeError(f"minimal 1-step/1-chunk smoke failed: {result}")
        return

    started = time.time()
    recent_step_times = []
    for step in range(start_step + 1, args.total_steps + 1):
        step_started = time.time()
        rollout_length, supervision = sampler.choose_chunk(step)
        selected = sampler.choose_position()
        record = records[(step - 1) % len(records)]
        optimizer.zero_grad(set_to_none=True)
        observer, _, reports, diagnostics = run_trajectory(
            args, pipe, geometry, record, supervision, selected,
        )
        grad_norm = float(torch.nn.utils.clip_grad_norm_(optimizer_parameters, 1.0))
        optimizer.step(); lr_scheduler.step(); optimizer.zero_grad(set_to_none=True)
        elapsed = time.time() - started
        step_time = time.time() - step_started
        recent_step_times = (recent_step_times + [step_time])[-20:]
        loss = observer.losses[selected]
        if step % 10 == 0:
            values = adapter_state(pipe.transformer)
            parameter_norm = math.sqrt(sum(float((value.float() ** 2).sum()) for value in values.values()))
            metric = {
                "global_step": step, "trajectory_id": record["trajectory_id"],
                "rollout_chunk_count": rollout_length,
                "supervision_chunk_index": supervision,
                "selected_stage": selected[0], "selected_step": selected[1],
                "lambda_at_selected_step": WPF_LAMBDAS[selected[0]][selected[1]],
                "sigma_t": loss["sigma"], "world_mask_mean": loss["world_mask_mean"],
                "world_mask_nonzero_ratio": loss["world_mask_nonzero_ratio"],
                "L_total": loss["total"], "L_fill": loss["fill"], "L_keep": loss["keep"],
                "total_loss": loss["total"], "x0_pred_norm": loss["x0_pred_norm"],
                "x0_base_norm": loss["x0_base_norm"], "z_gt_norm": loss["z_gt_norm"],
                "lora_grad_norm": grad_norm, "grad_norm": grad_norm,
                "lora_parameter_norm": parameter_norm, "lr": optimizer.param_groups[0]["lr"],
                "optimizer_step_time_sec": step_time, "elapsed_training_time_sec": elapsed,
                **sampler.position_counts, **diagnostics,
            }
            (metrics_dir / f"step_{step:04d}.json").write_text(json.dumps(metric, indent=2))
        if step % 50 == 0:
            save_checkpoint(
                checkpoints_dir / f"checkpoint_step_{step:04d}.pt",
                pipe=pipe, optimizer=optimizer, lr_scheduler=lr_scheduler,
                step=step, sampler=sampler,
            )
        status.update(global_step=step, elapsed_training_time_sec=elapsed,
                      recent_step_times_sec=recent_step_times)
        (args.run_dir / "status.json").write_text(json.dumps(status, indent=2))
        print(json.dumps({
            "step": step, "loss": loss["total"], "selected_stage": selected[0],
            "selected_step": selected[1], "seconds": step_time,
            "rollout_chunks": rollout_length, "supervision_chunk": supervision,
        }), flush=True)
    status["state"] = "complete"
    (args.run_dir / "status.json").write_text(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
