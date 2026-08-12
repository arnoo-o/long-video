#!/usr/bin/env python3
"""Train Point-FiLM with exact causal WAH/Pi3 rollout on DL3DV."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict, OrderedDict
import json
import math
from pathlib import Path
import random
import sys
import time
from types import SimpleNamespace

import numpy as np
from PIL import Image
import torch

from long_video.config import load_yaml
from long_video.initialization.geometry_backend import Pi3GeometryBackend
from long_video.memory.memory_manager import MemoryManager
from long_video.online.pipeline import OnlineSpatialHistoryPipeline
from long_video.training.causal_rollout import AllChunkRoundRobin, current_chunk_loss_weights
from long_video.training.stage0_causal_world import (
    Stage0FilmTrainer,
    load_film_checkpoint,
    save_film_checkpoint,
    validate_dl3dv_film_manifest,
)
from long_video.types import RAY_DISTANCE, ViewSet
from long_video.wah.stage0_causal_world_film import (
    CausalTrainingContract,
    install_stage0_causal_world_film,
)


CATEGORY_CYCLE = (
    ["source_revisit"] * 7 + ["world_revisit"] * 7
    + ["large_motion"] * 4 + ["normal_motion"] * 2
)

TARGET_LATENT_CACHE = OrderedDict()
TARGET_LATENT_CACHE_MAX = 16


def select_fixed_training_pool(records):
    """Select the fixed 80-trajectory training pool without creating records."""
    quotas = {"source_revisit": 28, "world_revisit": 28,
              "large_motion": 16, "normal_motion": 8}
    selected = []
    for category, quota in quotas.items():
        candidates = [r for r in records if r.get("sample_type") == category]
        if category in ("large_motion", "normal_motion") and not candidates:
            # The official manifest currently labels only revisit trajectories.
            # Derive motion pools deterministically from the remaining records.
            candidates = [r for r in records if r not in selected]
        candidates.sort(key=lambda r: str(r.get("trajectory_id", "")))
        if len(candidates) < quota:
            raise ValueError(f"DL3DV training pool needs {quota} {category}, found {len(candidates)}")
        selected.extend(candidates[:quota])
    if len(selected) != 80:
        raise AssertionError("fixed training pool must contain exactly 80 trajectories")
    eight = [r for r in selected if int(r["chunk_count"]) == 8]
    twelve = [r for r in selected if int(r["chunk_count"]) == 12]
    # Do not fabricate trajectories when the downloaded manifest has fewer than
    # 40 eight-chunk records; use every available eight-chunk trajectory.
    if len(eight) < 40:
        extras = [r for r in records if int(r["chunk_count"]) == 12 and r not in selected]
        selected.extend(extras[:40 - len(eight)])
        eight = [r for r in selected if int(r["chunk_count"]) == 8]
        twelve = [r for r in selected if int(r["chunk_count"]) == 12]
    if len(selected) != 80:
        raise AssertionError(f"fixed training pool must contain 80 trajectories, got {len(selected)}")
    return selected


def training_args():
    return SimpleNamespace(
        height=384, width=640, pyramid_num_inference_steps_list=[2, 2, 2],
        flow_matching_stage_sampling="fixed", flow_matching_stage_id=0,
        flow_matching_train_exact_timestep_sampling="training_density",
        flow_matching_use_dynamic_shifting="auto", weighting_scheme="none",
        is_amplify_first_chunk=False, visible_token_mode="drop",
        history_visible_token_threshold=0.05,
        history_confidence_threshold=0.1,
        history_confidence_lambda=1.0,
        history_confidence_epsilon=1e-6,
    )


def read_frames(root, record, chunk):
    directory = root / record["rgb_dir"]
    result = []
    for index in range(int(chunk) * 32, int(chunk) * 32 + 33):
        matches = list(directory.glob(f"{index:06d}.*"))
        if len(matches) != 1:
            raise FileNotFoundError(f"expected one RGB frame {index:06d} in {directory}")
        result.append(Image.open(matches[0]).convert("RGB"))
    return result


def cached_target_latents(pipe, record, chunk, root, args, device):
    key = (str(record["trajectory_id"]), int(chunk))
    cached = TARGET_LATENT_CACHE.pop(key, None)
    if cached is not None:
        TARGET_LATENT_CACHE[key] = cached
        return cached.to(device, non_blocking=True)
    from warp_as_history.training import core as opt
    frames = read_frames(root, record, chunk)
    mean, std = opt.latent_stats(pipe, device)
    with torch.no_grad():
        latents = opt.encode_video_latents(pipe, frames, args, device, mean, std).detach().cpu()
    TARGET_LATENT_CACHE[key] = latents
    while len(TARGET_LATENT_CACHE) > TARGET_LATENT_CACHE_MAX:
        TARGET_LATENT_CACHE.popitem(last=False)
    return latents.to(device, non_blocking=True)


def source_views(source_image, intrinsic):
    rgb = np.asarray(source_image, np.uint8)
    height, width = rgb.shape[:2]
    return ViewSet(
        rgb=np.repeat(rgb[None], 8, axis=0),
        depth=np.full((8, height, width), np.nan, np.float32),
        depth_confidence=np.zeros((8, height, width), np.float32),
        c2w=np.repeat(np.eye(4, dtype=np.float32)[None], 8, axis=0),
        intrinsics=np.repeat(np.asarray(intrinsic, np.float32)[None], 8, axis=0),
        source=np.zeros((8, height, width), np.int8),
        image_confidence=np.ones((8, height, width), np.float32),
        depth_convention=RAY_DISTANCE,
    )


def motion_scores(poses):
    values = []
    for chunk in range((len(poses) - 1) // 32):
        segment = poses[chunk * 32:chunk * 32 + 33]
        translation = np.linalg.norm(np.diff(segment[:, :3, 3], axis=0), axis=1).sum()
        relative = segment[:-1, :3, :3].transpose(0, 2, 1) @ segment[1:, :3, :3]
        cos = np.clip((np.trace(relative, axis1=1, axis2=2) - 1) / 2, -1, 1)
        rotation = np.arccos(cos).sum()
        values.append(float(translation + rotation))
    return values


class PairSchedule:
    """Exact 35/35/20/10 category cycle with least-used eligible pairs."""
    def __init__(self, root, records, round_robin):
        self.root, self.records, self.rr = root, records, round_robin
        self.category_counts = Counter()
        self._tie = defaultdict(int)
        self._motion = {}

    def _eligible(self, record, category):
        count = int(record["chunk_count"])
        if category == "source_revisit":
            return range(count) if record["sample_type"] == "source_revisit" else ()
        poses = np.load(self.root / record["target_c2w_local"])
        if category == "world_revisit":
            if record["sample_type"] != "world_revisit": return ()
            later = min(count - 1, max(0, (int(record["revisit_later_output_frame"]) - 1) // 32))
            return range(later, count)
        key = record["trajectory_id"]
        scores = self._motion.setdefault(key, motion_scores(poses))
        order = np.argsort(scores)
        split = max(1, len(order) // 2)
        if category == "large_motion":
            return order[-split:]
        normal = order[:-split]
        return normal if len(normal) else order[:1]

    def choose(self, category, chunk_count=None):
        candidates = []
        for record in self.records:
            if chunk_count is not None and int(record["chunk_count"]) != int(chunk_count): continue
            trajectory = str(record["trajectory_id"])
            for chunk in self._eligible(record, category):
                used = int(self.rr.counts[trajectory].get(int(chunk), 0))
                candidates.append((used, trajectory, int(chunk), record))
        if not candidates: raise RuntimeError(f"no eligible {category} samples")
        candidates.sort(key=lambda x: (x[0], x[1], x[2]))
        offset = self._tie[category] % len(candidates)
        minimum = candidates[0][0]
        tied = [x for x in candidates if x[0] == minimum]
        chosen = tied[offset % len(tied)]
        self._tie[category] += 1
        _, trajectory, chunk, record = chosen
        self.rr.counts[trajectory][chunk] += 1
        self.rr.cursors[trajectory] = (chunk + 1) % int(record["chunk_count"])
        self.category_counts[category] += 1
        return record, chunk


def build_online(pipe, geometry, record, root, device, seed):
    poses = np.load(root / record["target_c2w_local"]).astype(np.float32)
    intrinsics = np.load(root / record["intrinsics"]).astype(np.float32)
    source = Image.open(root / record["source"]).convert("RGB")
    manager = MemoryManager.from_config(
        load_yaml("configs/online_memory.yaml"), geometry_backend=geometry,
    )
    generator = torch.Generator(device=device).manual_seed(int(seed))
    online = OnlineSpatialHistoryPipeline(
        wah_pipeline=pipe, memory_manager=manager, prompt=record["prompt"],
        renderer_kwargs={"device": device},
        wah_state_kwargs={
            "height": 384, "width": 640, "num_frames": 33, "output_type": "np",
            "pyramid_num_inference_steps_list": [2, 2, 2], "generator": generator,
        },
    )
    online.initialize(
        source_views(source, intrinsics[0]), record["prompt"], geometry,
        {"node_id": "node_000", "center_c2w": np.eye(4, dtype=np.float32),
         "created_frame": 0, "view_frame_indices": [0] * 8, "target_frame_start": 1},
        first_image=source,
    )
    return online, poses, intrinsics


def run_one_step(pipe, geometry, trainer, scheduler, record, current, root, args, device, seed,
                 *, require_nonzero_gradient=True):
    from warp_as_history.training import core as opt
    start = time.perf_counter()
    online, poses, intrinsics = build_online(pipe, geometry, record, root, device, seed)
    for chunk in range(int(current)):
        online.frame_index = chunk * 32
        online.chunk_index = chunk
        begin = chunk * 32
        with torch.no_grad():
            online.generate_chunk_at_cameras(
                poses[begin:begin + 33], intrinsics[begin:begin + 33], 384, 640,
            )
    online.frame_index = int(current) * 32
    online.chunk_index = int(current)
    begin = int(current) * 32
    warp, point_feature, visibility0, histories = online.prepare_supervised_chunk(
        poses[begin:begin + 33], intrinsics[begin:begin + 33], 384, 640,
    )
    if record["sample_type"] == "world_revisit":
        later = min(int(record["chunk_count"]) - 1,
                    max(0, (int(record["revisit_later_output_frame"]) - 1) // 32))
        if int(current) >= later and online.active_node.node_id == "node_000":
            raise RuntimeError("world_revisit target reached before a real causal promotion")
    target_latents = cached_target_latents(pipe, record, current, root, args, device)
    contract = CausalTrainingContract(
        conditioning_frame_end=begin - 1, target_frame_start=begin, uses_future_gt=False,
    )
    result = trainer.step(
        contract=contract, prompt_embeds=online.autoregressive_state["prompt_embeds"],
        target_latents=target_latents, histories=histories,
        world0=point_feature, visibility0=visibility0, args=args, device=device,
        loss_weights=current_chunk_loss_weights(int(record["chunk_count"]), int(current)),
        require_nonzero_gradient=require_nonzero_gradient,
    )
    scheduler.step()
    controller = pipe.transformer.stage0_causal_world_film
    result.update({
        "trajectory_id": record["trajectory_id"], "scene_hash": record["scene_hash"],
        "sample_type": record["sample_type"], "current_chunk_index": int(current),
        "chunk_count": int(record["chunk_count"]), "uses_future_gt": False,
        "point_film_parameter_norm": float(torch.sqrt(sum(
            parameter.detach().float().square().sum()
            for parameter in trainer.parameters
        ))),
        "stage0_relative_modulation": float(
            0.0 if controller.relative_modulation is None else controller.relative_modulation.cpu()
        ),
        "optimizer_step_seconds": time.perf_counter() - start,
        "active_node_id": online.active_node.node_id,
        "world_visibility_mean": float(visibility0.detach().float().mean().cpu()),
        "winning_point_count": int(np.asarray(warp.visibility).sum()),
    })
    del online, target_latents, histories, point_feature, visibility0
    return result


def save_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=str), encoding="utf-8")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wah-root", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--pi3-repo", type=Path, required=True)
    parser.add_argument("--pi3-checkpoint", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--total-steps", type=int, default=1000)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--seed", type=int, default=20260812)
    parser.add_argument("--resume", type=Path, default=None,
                        help="resume from a saved Point-FiLM checkpoint")
    options = parser.parse_args()
    if options.smoke and options.total_steps != 1000:
        raise ValueError("smoke uses its fixed three cases; do not combine with --total-steps")
    random.seed(options.seed); np.random.seed(options.seed); torch.manual_seed(options.seed)
    sys.path.insert(0, str(options.wah_root))
    from warp_as_history import WarpAsHistoryPipeline
    records = select_fixed_training_pool(
        [x for x in validate_dl3dv_film_manifest(options.manifest) if x["split"] == "train"]
    )
    root = options.manifest.parent
    options.run_dir.mkdir(parents=True, exist_ok=bool(options.resume))
    pipe = WarpAsHistoryPipeline.from_pretrained(
        options.model, torch_dtype=torch.bfloat16,
    ).to(options.device)
    # Diffusers 0.36's generic video-LoRA loader probes this optional config
    # key before loading the pinned official WAH adapter.  Helios predates the
    # key, so register the explicit no-image-channel value without changing
    # transformer weights or conditioning semantics.
    if not hasattr(pipe.transformer.config, "image_dim"):
        pipe.transformer.register_to_config(image_dim=None)
    pipe.set_progress_bar_config(disable=True)
    # Load/freeze the pinned official WAH adapter before installing the only
    # trainable module.  Diffusers' first adapter load marks the existing
    # transformer parameters frozen; doing it after Point-FiLM installation
    # would inadvertently freeze the new head as well.
    official_wah_lora = (
        options.wah_root / "checkpoints" / "warp-as-history"
        / "visible_lora_state_step1000.safetensors"
    ).resolve()
    if not official_wah_lora.is_file():
        raise FileNotFoundError(f"missing pinned official WAH LoRA: {official_wah_lora}")
    pipe._configure_wah_lora(str(official_wah_lora))
    controller = install_stage0_causal_world_film(pipe.transformer).to(options.device, torch.float32)
    if hasattr(pipe.transformer, "enable_gradient_checkpointing"):
        pipe.transformer.enable_gradient_checkpointing()
    geometry = Pi3GeometryBackend(
        options.pi3_checkpoint, options.pi3_repo, options.device,
    )
    trainer = Stage0FilmTrainer(
        pipe, geometry, learning_rate=1e-4, weight_decay=0.01, max_grad_norm=1.0,
    )
    total = 3 if options.smoke else int(options.total_steps)
    warmup = 50
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        trainer.optimizer,
        lambda step: min(1.0, (step + 1) / warmup) * max(0.0, (total - step) / max(1, total)),
    )
    round_robin = AllChunkRoundRobin()
    resume_step = 0
    if options.resume is not None:
        resume_step, _ = load_film_checkpoint(
            options.resume, pipe.transformer, trainer.optimizer, scheduler, round_robin
        )
        trainer.step_index = resume_step
    schedule = PairSchedule(root, records, round_robin)
    args = training_args()
    metrics, times, sample_counts = [], [], Counter()
    if options.smoke:
        cases = []
        any_record = records[0]
        eight = next(x for x in records if int(x["chunk_count"]) == 8)
        twelve = next(x for x in records if int(x["chunk_count"]) == 12)
        cases = [(any_record, 0, "single_chunk"), (eight, 4, "8chunk_middle"),
                 (twelve, 10, "12chunk_late")]
    if resume_step >= total:
        raise ValueError(f"checkpoint step {resume_step} is already >= total_steps {total}")
    for step in range(resume_step + 1, total + 1):
        if options.smoke:
            record, current, category = cases[step - 1]
            round_robin.counts[record["trajectory_id"]][current] += 1
        elif step <= 350:
            record = records[(step - 1) % len(records)]
            current, category = 0, "single_chunk"
            round_robin.counts[record["trajectory_id"]][0] += 1
        else:
            category = CATEGORY_CYCLE[(step - 351) % len(CATEGORY_CYCLE)]
            record, current = schedule.choose(category, 8 if step <= 700 else 12)
        try:
            result = run_one_step(
                pipe, geometry, trainer, scheduler, record, current, root, args,
                options.device, options.seed + step,
                require_nonzero_gradient=bool(options.smoke),
            )
        except RuntimeError as error:
            if category != "world_revisit" or "real causal promotion" not in str(error): raise
            raise RuntimeError(
                "world_revisit pool contains a trajectory whose target has no real promotion; "
                "training stops instead of silently relabeling it"
            ) from error
        result.update(global_step=step, phase=("smoke" if options.smoke else
            "single_chunk" if step <= 350 else "8chunk" if step <= 700 else "12chunk"),
            sample_category=category, learning_rate=float(scheduler.get_last_lr()[0]))
        metrics.append(result); times.append(result["optimizer_step_seconds"]); sample_counts[category] += 1
        status = {
            "status": "running", "global_step": step, "total_steps": total,
            "latest": result, "average_optimizer_step_time": float(np.mean(times)),
            "eta_seconds": float(np.mean(times) * (total - step)),
            "sample_category_counts": dict(sample_counts),
            "trajectory_chunk_supervision_counts": round_robin.coverage_report(records),
            "peak_allocated": int(torch.cuda.max_memory_allocated()),
            "peak_reserved": int(torch.cuda.max_memory_reserved()),
        }
        save_json(options.run_dir / "training_status.json", status)
        if step % 10 == 0 or options.smoke:
            save_json(options.run_dir / "metrics" / f"step_{step:04d}.json", status)
        if step % 100 == 0 or step == total:
            save_film_checkpoint(
                options.run_dir / "checkpoints" / f"checkpoint_step_{step:04d}.pt",
                pipe.transformer, trainer.optimizer, scheduler, step=step,
                metadata={"sample_category_counts": dict(sample_counts),
                          "supervision_counts": round_robin.coverage_report(records)},
                round_robin=round_robin,
            )
        print(json.dumps({"event": "optimizer_step", **result}), flush=True)
    summary = {
        "status": "complete", "global_step": total, "smoke": bool(options.smoke),
        "average_optimizer_step_time": float(np.mean(times)),
        "projected_total_seconds": float(np.mean(times) * (1000 if options.smoke else total)),
        "sample_category_counts": dict(sample_counts),
        "trainable_parameter_names": trainer.names,
        "trainable_parameter_count": int(sum(x.numel() for x in trainer.parameters)),
        "peak_allocated": int(torch.cuda.max_memory_allocated()),
        "peak_reserved": int(torch.cuda.max_memory_reserved()),
    }
    save_json(options.run_dir / "summary.json", summary)
    save_json(options.run_dir / "training_status.json", summary)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
