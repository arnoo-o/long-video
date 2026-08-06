#!/usr/bin/env python3
"""Build real Indoor_013 Oracle-M0 training/diagnostic windows."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from zipfile import ZipFile

import yaml


def _parse_bootstrap():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/oracle_wah_training.yaml")
    parser.add_argument("--set", action="append", default=[], dest="overrides")
    return parser.parse_args()


def _gpu_inventory(physical_gpu: int):
    query = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,uuid,name,memory.used", "--format=csv,noheader,nounits"],
        text=True,
    ).strip().splitlines()
    rows = [line.split(", ") for line in query]
    selected = next(row for row in rows if int(row[0]) == int(physical_gpu))
    return {"physical_gpu": int(selected[0]), "uuid": selected[1], "name": selected[2], "memory_used_mib": int(selected[3])}


def _resolve_contract(config, first_image):
    import torch
    if torch.cuda.device_count() != 1:
        raise RuntimeError(f"Expected exactly one visible GPU, got {torch.cuda.device_count()}")
    sys.path.insert(0, str(Path(config["wah_root"])))
    from warp_as_history import WarpAsHistoryPipeline
    pipe = WarpAsHistoryPipeline.from_pretrained(
        config["wah_model"], torch_dtype=torch.bfloat16
    ).to("cuda:0")
    state = pipe.init_autoregressive_state(
        prompt=config["prompt"], image=first_image, conditioning_type="warp",
        lora_path=None, visible_token_drop=True, warp_history_downsample_mode="short",
        rope_alignment=True, height=384, width=640, num_frames=1,
        output_type="np", add_noise_to_image_latents=False,
        pyramid_num_inference_steps_list=[1, 1, 1], is_amplify_first_chunk=False,
    )
    result = {
        "window_num_frames": int(state["window_num_frames"]),
        "vae_temporal_scale": int(pipe.vae_scale_factor_temporal),
        "state_initializations": 1,
    }
    del state, pipe
    torch.cuda.empty_cache()
    return result


def _zip_rgb_members(handle):
    names = [name for name in handle.namelist() if "/rgb/" in name and name.endswith(".jpg")]
    return sorted(names, key=lambda name: float(Path(name).stem))


def _extract_window(archive, destination, source_start, frame_stride, count):
    destination = Path(destination)
    with ZipFile(archive) as handle:
        rgb_members = _zip_rgb_members(handle)
        indices = [int(source_start) + i * int(frame_stride) for i in range(int(count))]
        if indices[-1] >= len(rgb_members):
            raise IndexError(f"window needs source index {indices[-1]}, archive has {len(rgb_members)} RGB frames")
        root = rgb_members[0].split("/")[0]
        for index in indices:
            stem = Path(rgb_members[index]).stem
            for relative in (
                f"rgb/{stem}.jpg", f"depth/mesh_depth/{stem}.exr",
                f"mask/{stem}.jpg", f"poses/{stem}.txt",
            ):
                handle.extract(f"{root}/{relative}", destination)
    return destination / root


def _git_metadata(repo):
    def run(*args):
        return subprocess.check_output(["git", *args], cwd=repo, text=True).strip()
    sha = run("rev-parse", "HEAD")
    dirty = bool(run("status", "--porcelain"))
    diff = subprocess.check_output(["git", "diff", "--binary", "HEAD"], cwd=repo)
    return sha, dirty, hashlib.sha256(diff).hexdigest() if diff else "clean"


def main():
    args = _parse_bootstrap()
    repo = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo))
    from long_video.config import load_yaml
    config = load_yaml(args.config, args.overrides)
    missing = [key for key in ("holo_root", "wah_root", "wah_model", "output_root") if not config.get(key)]
    if missing:
        raise ValueError(f"machine paths must be supplied by --set key=value: {missing}")
    physical_gpu = int(config["physical_gpu"])
    gpu_before = _gpu_inventory(physical_gpu)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(physical_gpu)
    os.environ.setdefault("XFORMERS_DISABLED", "1")
    # torch/CUDA imports occur only after the physical visibility mask is fixed.
    from PIL import Image
    contract_info = _resolve_contract(config, Image.new("RGB", (640, 384), color=(0, 0, 0)))
    from long_video.oracle_training.dataset import build_oracle_sequence
    from long_video.oracle_training.temporal import ChunkContract
    contract = ChunkContract(**{key: contract_info[key] for key in ("window_num_frames", "vae_temporal_scale")})
    total_frames = 1 + int(config["num_chunks"]) * (contract.window_num_frames - 1)
    output_root = Path(config["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    base_sha, dirty, diff_sha = _git_metadata(repo)
    results = []
    holo_root = Path(config["holo_root"])
    for split, starts in config["source_starts"].items():
        for ordinal, source_start in enumerate(starts):
            sequence_id = f"{config['scene_id']}_{split}_{ordinal:03d}"
            if holo_root.suffix.lower() == ".zip":
                cache = output_root / "_extracted" / sequence_id
                scene_root = _extract_window(
                    holo_root, cache, int(source_start), int(config["frame_stride"]), total_frames
                )
                local_source_index, local_stride = 0, 1
            else:
                scene_root = holo_root
                local_source_index, local_stride = int(source_start), int(config["frame_stride"])
            sequence, metadata = build_oracle_sequence(
                scene_root, output_root, sequence_id=sequence_id, split=str(split),
                source_index=local_source_index, frame_stride=local_stride,
                num_chunks=int(config["num_chunks"]), contract=contract,
                erp_resolution=config["erp_resolution"],
                perspective_resolution=config["perspective_resolution"],
                fov_degrees=float(config["fov_degrees"]), pixel_center=float(config["pixel_center"]),
                prompt=config["prompt"], assumed_fps=float(config.get("assumed_fps", 16.0)),
                voxel_size=float(config.get("oracle_voxel_size", 0.01)),
                renderer_kwargs={"device": "cuda:0", **dict(config.get("renderer") or {})},
                base_commit_sha=base_sha, worktree_dirty=dirty, source_diff_sha256=diff_sha,
            )
            metadata["frame_stride"] = int(config["frame_stride"])
            (sequence / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
            results.append({"sequence": str(sequence), "metadata": metadata})
    summary = {
        "resolved_config": {key: value for key, value in config.items() if key not in {"holo_root", "wah_root", "wah_model", "output_root", "checkpoint_root", "pi3_checkpoint"}},
        "path_fields_supplied": {key: bool(config.get(key)) for key in ("holo_root", "wah_root", "wah_model", "output_root", "checkpoint_root", "pi3_checkpoint")},
        "wah_contract": contract_info, "gpu_before": gpu_before,
        "gpu_after": _gpu_inventory(physical_gpu), "sequences": results,
    }
    (output_root / "build_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
