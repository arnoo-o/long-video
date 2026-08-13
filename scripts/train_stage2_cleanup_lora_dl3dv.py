#!/usr/bin/env python3
"""Train a Stage2-only cleanup LoRA on causal DL3DV 24fps trajectories."""
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
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--total-steps", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument("--prompt", default="A stable realistic view of the same scene.")
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--skip-smoke", action="store_true")
    return parser.parse_args()


def read_rgb(path):
    return np.asarray(Image.open(path).convert("RGB"), np.uint8)


def load_record_arrays(root, record):
    rgb_paths = sorted((root / record["rgb_dir"]).glob("*"))
    init_paths = sorted((root / record["pi3_initial_rgb_dir"]).glob("*"))
    if len(rgb_paths) != 193 or len(init_paths) != 8:
        raise ValueError("training requires 193 dense frames and exactly 8 causal Pi3 views")
    if bool(record.get("uses_future_gt", True)):
        raise ValueError("future-GT record is forbidden")
    return {
        "rgb_paths": rgb_paths,
        "init_rgb": np.stack([read_rgb(path) for path in init_paths]),
        "init_c2w": np.load(root / record["pi3_initial_c2w_local"]).astype(np.float32),
        "init_k": np.load(root / record["pi3_initial_intrinsics"]).astype(np.float32),
        "c2w": np.load(root / record["target_c2w_local"]).astype(np.float32),
        "k": np.load(root / record["intrinsics"]).astype(np.float32),
    }


def deterministic_video_latent(pipe, frames, device):
    tensor = pipe._coerce_warp_video_tensor(
        np.asarray(frames), height=384, width=640, device=torch.device(device)
    ).to(dtype=pipe.vae.dtype)
    mean, std = pipe._latent_stats(torch.device(device))
    with torch.no_grad():
        posterior = pipe.vae.encode(tensor)
        posterior = getattr(posterior, "latent_dist", posterior)
        mode = getattr(posterior, "mode", None)
        clean = mode() if callable(mode) else posterior.mean
        clean = (clean - mean) * std
    if tuple(clean.shape[1:]) != (16, 9, 48, 80):
        raise RuntimeError(f"unexpected GT latent shape {tuple(clean.shape)}")
    return clean.detach()


def cleanup_state(transformer):
    from long_video.training.stage2_cleanup import cleanup_parameter_items
    return {name: value.detach().cpu().clone()
            for name, value in cleanup_parameter_items(transformer)}


def load_cleanup_state(transformer, state):
    named = dict(transformer.named_parameters())
    if set(named).intersection(state) != set(state):
        raise RuntimeError("cleanup checkpoint parameter names do not match model")
    with torch.no_grad():
        for name, value in state.items():
            named[name].copy_(value.to(device=named[name].device, dtype=named[name].dtype))


def state_changed(before, after):
    return any(not torch.equal(before[name], after[name]) for name in before)


def scheduler_lambda(step):
    if step < 50:
        return float(step + 1) / 50.0
    progress = (step - 50) / max(1, 1500 - 50)
    return max(0.0, 1.0 - progress)


def save_checkpoint(path, *, pipe, optimizer, lr_scheduler, step, sampler):
    payload = {
        "stage2_cleanup": cleanup_state(pipe.transformer),
        "optimizer": optimizer.state_dict(), "lr_scheduler": lr_scheduler.state_dict(),
        "global_step": int(step), "sampling_state": sampler.state_dict(),
        "trajectory_counters": copy.deepcopy(sampler.trajectory_counts),
        "rng_state": {"python": random.getstate(), "numpy": np.random.get_state(),
                      "torch": torch.get_rng_state(),
                      "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None},
    }
    torch.save(payload, path)


def build_online(args, pipe, geometry, record, arrays):
    from long_video.config import load_yaml
    from long_video.memory.memory_manager import MemoryManager
    from long_video.online.pipeline import OnlineSpatialHistoryPipeline
    from long_video.types import ViewSet

    views = ViewSet(
        rgb=arrays["init_rgb"],
        depth=np.full(arrays["init_rgb"].shape[:3], np.nan, np.float32),
        depth_confidence=np.zeros(arrays["init_rgb"].shape[:3], np.float32),
        c2w=arrays["init_c2w"], intrinsics=arrays["init_k"],
        source=np.zeros(arrays["init_rgb"].shape[:3], np.int8),
        image_confidence=np.ones(arrays["init_rgb"].shape[:3], np.float32),
    )
    manager = MemoryManager.from_config(load_yaml("configs/online_memory.yaml"), geometry_backend=geometry)
    online = OnlineSpatialHistoryPipeline(
        wah_pipeline=pipe, memory_manager=manager, prompt=args.prompt,
        renderer_kwargs={"device": args.device},
        wah_state_kwargs={"height": 384, "width": 640, "num_frames": 33,
                          "output_type": "np", "pyramid_num_inference_steps_list": [2, 2, 4]},
    )
    source = read_rgb(args.dataset_root / record["source"])
    online.initialize(views, args.prompt, geometry, {"voxel_size": 0.02}, first_image=source)
    online.autoregressive_state["is_amplify_first_chunk"] = False
    return online


def run_trajectory(args, pipe, geometry, record, supervision_chunk, *, train):
    from long_video.training.stage2_cleanup import Stage2FlowObserver

    arrays = load_record_arrays(args.dataset_root, record)
    online = build_online(args, pipe, geometry, record, arrays)
    reports = []
    observer = None
    for chunk_index in range(int(supervision_chunk) + 1):
        indices = slice(chunk_index * 32, chunk_index * 32 + 33)
        if chunk_index == supervision_chunk:
            gt = [read_rgb(path) for path in arrays["rgb_paths"][indices]]
            z_gt = deterministic_video_latent(pipe, gt, args.device)
            observer = Stage2FlowObserver(z_gt, pipe.scheduler)
            pipe._stage2_training_observer = observer
        else:
            pipe._stage2_training_observer = None
        video, _, warp, report = online.generate_chunk_at_cameras(
            arrays["c2w"][indices], arrays["k"][indices], 384, 640
        )
        reports.append(report)
        if chunk_index == supervision_chunk:
            observer.assert_complete()
            diagnostics = report["rgb_clamp_diagnostics"]
            clamps = [item["rgb_clamp"] for item in diagnostics if item["stage_id"] == 2]
            if clamps != [1, 1, 0, 0]:
                raise RuntimeError(f"formal clamp schedule not executed: {clamps}")
    pipe._stage2_training_observer = None
    event = reports[-1].get("memory_event") or {}
    return observer, online, reports, {
        "warp_visibility_ratio": float(np.asarray(warp.visibility).mean()),
        "world_point_count": int(len(online.active_node.points_xyz)),
        "promotion_count": int(sum(bool((x.get("memory_event") or {}).get("accepted")) for x in reports)),
        "candidate_points": int(event.get("candidate_point_count", event.get("candidate_points", 0)) or 0),
        "appended_points": int(event.get("appended_point_count", event.get("appended_points", 0)) or 0),
        "rejected_points": int(event.get("rejected_point_count", event.get("rejected_points", 0)) or 0),
    }


def smoke_test(args, pipe, geometry, records, trainable, optimizer, lr_scheduler, sampler):
    smoke_dir = args.run_dir / "smoke"
    smoke_dir.mkdir(parents=True, exist_ok=True)
    # PyTorch increments a Parameter's version on in-place optimizer updates.
    # This proves frozen weights did not change without cloning the full Helios.
    before_versions = {name: value._version for name, value in pipe.transformer.named_parameters()
                       if ".stage2_cleanup." not in name}
    initial_cleanup = cleanup_state(pipe.transformer)
    optimizer.zero_grad(set_to_none=True)
    observer, online, reports, diagnostics = run_trajectory(
        args, pipe, geometry, records[0], 1, train=True,
    )
    official_grad_zero = all(
        value.grad is None or not bool(torch.count_nonzero(value.grad))
        for name, value in pipe.transformer.named_parameters() if ".wah." in name
    )
    cleanup_nonzero = any(value.grad is not None and bool(torch.count_nonzero(value.grad))
                          for _, value in trainable)
    other_grad_zero = all(
        value.grad is None or not bool(torch.count_nonzero(value.grad))
        for name, value in pipe.transformer.named_parameters()
        if ".stage2_cleanup." not in name
    )
    grad_norm = float(torch.nn.utils.clip_grad_norm_([value for _, value in trainable], 1.0))
    optimizer.step(); lr_scheduler.step()
    after_cleanup = cleanup_state(pipe.transformer)
    unchanged_non_cleanup = all(
        before_versions[name] == value._version
        for name, value in pipe.transformer.named_parameters() if ".stage2_cleanup." not in name
    )
    checkpoint = smoke_dir / "checkpoint_smoke.pt"
    save_checkpoint(checkpoint, pipe=pipe, optimizer=optimizer, lr_scheduler=lr_scheduler,
                    step=1, sampler=sampler)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    load_cleanup_state(pipe.transformer, initial_cleanup)
    load_cleanup_state(pipe.transformer, saved["stage2_cleanup"])
    checkpoint_roundtrip = all(torch.equal(after_cleanup[name], cleanup_state(pipe.transformer)[name])
                               for name in after_cleanup)
    checks = {
        "official_wah_loaded": bool(getattr(pipe, "_wah_loaded_lora_path", None)),
        "official_wah_grad_zero": official_grad_zero,
        "helios_base_grad_zero": other_grad_zero,
        "only_cleanup_nonzero_gradient": cleanup_nonzero and other_grad_zero,
        "four_finite_losses": len(observer.losses) == 4 and all(math.isfinite(x) for x in observer.losses),
        "clamp_step0_step1": [x["rgb_clamp"] for x in reports[-1]["rgb_clamp_diagnostics"] if x["stage_id"] == 2] == [1, 1, 0, 0],
        "prior_chunk_updated_world": len(reports) == 2 and online.chunk_index == 2,
        "no_future_gt": all(x.get("uses_future_gt") is False for x in reports),
        "only_cleanup_changed": state_changed(initial_cleanup, after_cleanup) and unchanged_non_cleanup,
        "checkpoint_roundtrip": checkpoint_roundtrip,
    }
    result = {"passed": all(checks.values()), "checks": checks, "losses": observer.losses,
              "grad_norm": grad_norm, **diagnostics}
    (smoke_dir / "smoke_result.json").write_text(json.dumps(result, indent=2))
    load_cleanup_state(pipe.transformer, initial_cleanup)
    optimizer.zero_grad(set_to_none=True)
    if not result["passed"]:
        raise RuntimeError(f"mandatory smoke test failed: {result}")
    return result


def main():
    args = parse_args()
    sys.path.insert(0, str(args.wah_root))
    from long_video.config import load_yaml
    from long_video.initialization.geometry_backend import Pi3GeometryBackend
    from long_video.training.stage2_cleanup import (
        BalancedChunkSampler, configure_trainable_cleanup_adapter,
    )
    from long_video.wah.rgb_clamp_pipeline import RGBClampWarpAsHistoryPipeline

    if args.run_dir.exists() and any(args.run_dir.iterdir()):
        raise FileExistsError(f"run directory must be new: {args.run_dir}")
    metrics_dir = args.run_dir / "metrics"; checkpoints_dir = args.run_dir / "checkpoints"
    metrics_dir.mkdir(parents=True); checkpoints_dir.mkdir()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    manifest = json.loads((args.dataset_root / "dl3dv_24fps_manifest.json").read_text())
    records = [item for item in manifest["records"] if item["split"] == "train"
               and len(list((args.dataset_root / item["pi3_initial_rgb_dir"]).glob("*"))) == 8]
    if not records:
        raise RuntimeError("no train records with 8 causal Pi3 initialization views")
    pipe = RGBClampWarpAsHistoryPipeline.from_pretrained(args.model, torch_dtype=torch.bfloat16).to(args.device)
    if not hasattr(pipe.transformer.config, "image_dim"):
        pipe.transformer.register_to_config(image_dim=None)
    official = args.wah_root / "checkpoints/warp-as-history/visible_lora_state_step1000.safetensors"
    if not official.is_file():
        raise FileNotFoundError(official)
    pipe._configure_wah_lora(str(official))
    trainable = configure_trainable_cleanup_adapter(pipe)
    for module in (pipe.vae, pipe.text_encoder):
        for parameter in module.parameters(): parameter.requires_grad_(False)
    if hasattr(pipe.transformer, "enable_gradient_checkpointing"):
        pipe.transformer.enable_gradient_checkpointing()
    optimizer = torch.optim.AdamW([value for _, value in trainable], lr=5e-5, weight_decay=0.01)
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, scheduler_lambda)
    sampler = BalancedChunkSampler(args.seed)
    geometry = Pi3GeometryBackend(args.pi3_checkpoint, args.pi3_repo, args.device)
    status = {"pid": os.getpid(), "global_step": 0, "total_steps": args.total_steps,
              "run_dir": str(args.run_dir), "metrics_dir": str(metrics_dir),
              "checkpoint_dir": str(checkpoints_dir), "state": "smoke"}
    (args.run_dir / "status.json").write_text(json.dumps(status, indent=2))
    if not args.skip_smoke:
        smoke_test(args, pipe, geometry, records, trainable, optimizer, lr_scheduler, sampler)
    if args.smoke_only:
        return
    # Smoke is a gate, not training step zero: rebuild all optimization state.
    optimizer = torch.optim.AdamW([value for _, value in trainable], lr=5e-5, weight_decay=0.01)
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, scheduler_lambda)
    sampler = BalancedChunkSampler(args.seed)
    started = time.time(); status["state"] = "training"; recent_step_times = []
    (args.run_dir / "status.json").write_text(json.dumps(status, indent=2))
    for step in range(1, args.total_steps + 1):
        step_started = time.time()
        trajectory_length, supervision = sampler.choose(step)
        record = records[(step - 1) % len(records)]
        optimizer.zero_grad(set_to_none=True)
        observer, online, reports, diagnostics = run_trajectory(
            args, pipe, geometry, record, supervision, train=True,
        )
        grad_norm = float(torch.nn.utils.clip_grad_norm_([value for _, value in trainable], 1.0))
        optimizer.step(); lr_scheduler.step(); optimizer.zero_grad(set_to_none=True)
        elapsed = time.time() - started; step_time = time.time() - step_started
        recent_step_times = (recent_step_times + [step_time])[-20:]
        if step % 10 == 0:
            values = cleanup_state(pipe.transformer)
            norm = math.sqrt(sum(float((value.float() ** 2).sum()) for value in values.values()))
            event = {
                "global_step": step, "total_loss": float(sum(observer.losses) / 4),
                **{f"stage2_step{i}_loss": observer.losses[i] for i in range(4)},
                "lr": optimizer.param_groups[0]["lr"], "grad_norm": grad_norm,
                "stage2_lora_parameter_norm": norm, "trajectory_length": trajectory_length,
                "supervision_chunk_index": supervision, **diagnostics,
                "clamp_step0_executed": True, "clamp_step1_executed": True,
                **{f"{i}_chunk_cumulative_count": sampler.trajectory_counts.get(i, 0) for i in range(1, 7)},
                "optimizer_step_time_sec": step_time, "elapsed_training_time_sec": elapsed,
            }
            (metrics_dir / f"step_{step:04d}.json").write_text(json.dumps(event, indent=2))
        if step % 100 == 0:
            save_checkpoint(checkpoints_dir / f"checkpoint_step_{step:04d}.pt", pipe=pipe,
                            optimizer=optimizer, lr_scheduler=lr_scheduler, step=step, sampler=sampler)
        status.update(global_step=step, elapsed_training_time_sec=elapsed,
                      recent_step_times_sec=recent_step_times)
        (args.run_dir / "status.json").write_text(json.dumps(status, indent=2))
        print(json.dumps({"step": step, "loss": sum(observer.losses) / 4,
                          "seconds": step_time, "trajectory_length": trajectory_length,
                          "supervision_chunk": supervision}), flush=True)
    status["state"] = "complete"
    (args.run_dir / "status.json").write_text(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
