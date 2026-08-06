#!/usr/bin/env python3
"""Four-chunk no-grad rollout with active-node online warp and generated-only M1."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def _args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/oracle_wah_training.yaml")
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--set", action="append", default=[], dest="overrides")
    return parser.parse_args()


def _gpu_processes():
    result = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_gpu_memory", "--format=csv,noheader,nounits"],
        text=True, capture_output=True, check=False,
    )
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def _tensor_bytes(value):
    import torch
    if torch.is_tensor(value):
        return value.numel() * value.element_size()
    if isinstance(value, dict):
        return sum(_tensor_bytes(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_tensor_bytes(item) for item in value)
    return 0


def _clear_decoded_history(state):
    import torch
    decoded_bytes = _tensor_bytes(state.get("history_video")) + _tensor_bytes(state.get("last_video_delta"))
    state["history_video"] = None
    state["last_video_delta"] = None
    state["returned_frame_count"] = 0
    # prev_chunk_last_frame is the single bounded RGB boundary required by
    # official WAH warp history construction; it is replaced each chunk.
    history_count = int(state["num_history_latent_frames"])
    state["history_latents"] = state["history_latents"][:, :, -history_count:].detach()
    state["real_history_latents"] = None
    state["last_latents"] = None
    return decoded_bytes


def _video_array(video):
    import numpy as np
    value = np.asarray(video)
    if value.ndim == 5 and value.shape[0] == 1:
        value = value[0]
    if value.ndim == 4 and value.shape[1] == 3 and value.shape[-1] != 3:
        value = np.moveaxis(value, 1, -1)
    if value.ndim != 4 or value.shape[-1] != 3:
        raise ValueError(f"WAH output must be [T,H,W,3], got {value.shape}")
    return value


def main():
    args = _args()
    repo = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo))
    from long_video.config import load_yaml
    config = load_yaml(args.config, args.overrides)
    required = [key for key in ("wah_root", "wah_model", "pi3_checkpoint", "output_root") if not config.get(key)]
    pi3_repo = config.get("pi3_repo") or os.environ.get("LONG_VIDEO_PI3_REPO")
    if required or not pi3_repo:
        raise ValueError(f"missing machine paths: {required + ([] if pi3_repo else ['pi3_repo'])}")
    physical_gpu = int(config["physical_gpu"])
    gpu0_and_gpu1_before = _gpu_processes()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(physical_gpu)
    os.environ.setdefault("XFORMERS_DISABLED", "1")
    import imageio.v2 as imageio
    import numpy as np
    import torch
    from PIL import Image
    if torch.cuda.device_count() != 1:
        raise RuntimeError(f"Expected exactly one visible GPU, got {torch.cuda.device_count()}")
    torch.cuda.set_device(0); torch.cuda.reset_peak_memory_stats(0)
    np.random.seed(int(config["seed"]))
    torch.manual_seed(int(config["seed"]))
    torch.cuda.manual_seed_all(int(config["seed"]))
    sys.path.insert(0, str(Path(config["wah_root"])))
    from warp_as_history import WarpAsHistoryPipeline
    from long_video.geometry.point_renderer import render
    from long_video.initialization.geometry_backend import Pi3GeometryBackend
    from long_video.memory.memory_manager import MemoryManager
    from long_video.memory.node_store import NodeStore
    from long_video.oracle_training.contracts import GeneratedMemoryBatch
    from long_video.oracle_training.dataset import attach_warp_provenance
    from long_video.types import CameraBatch

    sequence = Path(args.sequence)
    metadata = json.loads((sequence / "metadata.json").read_text(encoding="utf-8"))
    chunk_frames = int(metadata["chunk_frames"])
    chunk_stride = int(metadata["chunk_stride"])
    poses = np.load(sequence / "target" / "target_c2w_local.npy")
    intrinsics = np.load(sequence / "target" / "intrinsics.npy")
    expected_global = 1 + int(config["num_chunks"]) * chunk_stride
    if not (len(poses) == len(intrinsics) == expected_global):
        raise ValueError("trajectory/intrinsics length does not match four-chunk boundary contract")
    source_image = Image.open(sequence / "source" / "source_perspective.png").convert("RGB")
    source_store = NodeStore(sequence / "session")
    active_node = source_store.load("node_000")
    rollout_root = Path(config["output_root"]) / f"{metadata['sequence_id']}_rollout"
    rollout_store = NodeStore(rollout_root / "session")
    rollout_store.save(active_node)
    memory_config = load_yaml(repo / config["production_threshold_config"])
    geometry = Pi3GeometryBackend(
        config["pi3_checkpoint"], pi3_repo, device="cuda:0", input_size=518
    )
    manager = MemoryManager.from_config(memory_config, geometry_backend=geometry, node_store=rollout_store)
    manager.register(active_node)
    lora_path = Path(config["checkpoint_root"]) / "oracle_wah_lora.pt"
    if not lora_path.exists():
        raise FileNotFoundError(f"trained LoRA not found: {lora_path}")
    started = time.perf_counter()
    pipe = WarpAsHistoryPipeline.from_pretrained(
        config["wah_model"], torch_dtype=torch.bfloat16
    ).to("cuda:0")
    # Diffusers' generic video LoRA loader checks image_dim even though
    # Helios/WAH does not use a second image-conditioning projection.
    if not hasattr(pipe.transformer.config, "image_dim"):
        pipe.transformer.register_to_config(image_dim=None)
    state = pipe.init_autoregressive_state(
        prompt=config["prompt"], image=source_image, conditioning_type="warp",
        lora_path=lora_path, visible_token_drop=True,
        warp_history_downsample_mode="short", rope_alignment=True,
        height=384, width=640, num_frames=chunk_frames, output_type="np",
        add_noise_to_image_latents=False,
        pyramid_num_inference_steps_list=config["training"]["pyramid_num_inference_steps_list"],
        is_amplify_first_chunk=False,
    )
    state_initializations = 1
    if int(state["window_num_frames"]) != chunk_frames:
        raise ValueError("dataset chunk_frames differs from official WAH state.window_num_frames")
    rollout_root.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(rollout_root / "rollout.mp4", fps=float(config.get("assumed_fps", 16.0)), macro_block_size=1)
    chunk_records = []
    try:
        with torch.no_grad():
            for chunk_index in range(int(config["num_chunks"])):
                global_start = chunk_index * chunk_stride
                global_end = global_start + chunk_frames
                chunk_poses = poses[global_start:global_end]
                chunk_intrinsics = intrinsics[global_start:global_end]
                if len(chunk_poses) != chunk_frames:
                    raise ValueError("chunk trajectory length mismatch")
                node_at_start = active_node
                cameras = CameraBatch(chunk_poses, chunk_intrinsics, 384, 640)
                warp = attach_warp_provenance(render(
                    node_at_start, cameras, device="cuda:0", **dict(config["renderer"])
                ), node_at_start)
                generated_video, state = pipe.generate_next_chunk(
                    state,
                    warp_video=warp.rgb,
                    warp_visibility_mask=warp.visibility[None, None].astype(np.float32),
                    warp_confidence_mask=(warp.confidence * warp.visibility)[None, None].astype(np.float32),
                    output_type="np",
                )
                generated_full = _video_array(generated_video)
                if len(generated_full) != chunk_frames:
                    raise ValueError(f"unbounded-history-cleared WAH chunk returned {len(generated_full)}, expected {chunk_frames}")
                drop_boundary = chunk_index > 0
                generated_rgb_for_memory = generated_full[1:] if drop_boundary else generated_full
                memory_poses = chunk_poses[1:] if drop_boundary else chunk_poses
                memory_intrinsics = chunk_intrinsics[1:] if drop_boundary else chunk_intrinsics
                memory_warp = warp
                if drop_boundary:
                    from dataclasses import replace
                    memory_warp = replace(
                        warp, rgb=warp.rgb[1:], depth=warp.depth[1:], visibility=warp.visibility[1:],
                        confidence=warp.confidence[1:], source=warp.source[1:],
                        coverage_per_frame=warp.coverage_per_frame[1:],
                        rgb_content_origin=warp.rgb_content_origin[1:],
                        depth_content_origin=warp.depth_content_origin[1:],
                        evidence_role=warp.evidence_role[1:],
                        rgb_evidence_role=warp.rgb_evidence_role[1:],
                        depth_evidence_role=warp.depth_evidence_role[1:],
                    )
                if not (len(generated_rgb_for_memory) == len(memory_poses) == len(memory_warp.rgb)):
                    raise ValueError("generated/memory camera/warp frame counts differ")
                GeneratedMemoryBatch(generated_rgb_for_memory)
                memory_cameras = CameraBatch(memory_poses, memory_intrinsics, 384, 640)
                active_node, event = manager.process_chunk(
                    node_at_start, generated_rgb_for_memory=generated_rgb_for_memory,
                    cameras=memory_cameras, warp=memory_warp,
                    frame_start=global_start + (1 if drop_boundary else 0),
                )
                output_uint8 = generated_rgb_for_memory if generated_rgb_for_memory.dtype == np.uint8 else np.rint(np.clip(generated_rgb_for_memory, 0, 1) * 255).astype(np.uint8)
                for frame in output_uint8:
                    writer.append_data(frame)
                del output_uint8, generated_full, generated_video
                decoded_before_clear = _clear_decoded_history(state)
                torch.cuda.empty_cache()
                decoded_after_clear = _tensor_bytes(state.get("history_video")) + _tensor_bytes(state.get("last_video_delta"))
                if decoded_after_clear != 0:
                    raise RuntimeError("decoded GPU history was not released")
                chunk_records.append({
                    "chunk_index": chunk_index, "global_frame_start": global_start,
                    "global_frame_end": global_end - 1,
                    "active_node_at_start": node_at_start.node_id,
                    "warp_renderer_node_id": node_at_start.node_id,
                    "warp_frame_count": int(len(warp.rgb)),
                    "generated_frame_count": int(len(generated_rgb_for_memory)),
                    "candidate_event": event,
                    "active_node_for_next_chunk": active_node.node_id,
                    "cuda_allocated": int(torch.cuda.memory_allocated(0)),
                    "cuda_reserved": int(torch.cuda.memory_reserved(0)),
                    "wah_latent_state_bytes": int(_tensor_bytes(state)),
                    "decoded_history_tensor_bytes_before_release": int(decoded_before_clear),
                    "gpu_decoded_history_tensor_bytes": int(decoded_after_clear),
                    "boundary_frame_state_bytes": int(_tensor_bytes(state.get("prev_chunk_last_frame"))),
                })
    finally:
        writer.close()
    promotion_chunks = [record["chunk_index"] for record in chunk_records if record["candidate_event"].get("accepted")]
    switched_after_promotion = all(
        index + 1 < len(chunk_records) and chunk_records[index + 1]["warp_renderer_node_id"] == chunk_records[index]["active_node_for_next_chunk"]
        for index in promotion_chunks
    )
    result = {
        "sequence_id": metadata["sequence_id"], "state_initializations": state_initializations,
        "chunks": chunk_records, "promotion_chunks": promotion_chunks,
        "m1_switched_on_next_chunk": switched_after_promotion if promotion_chunks else None,
        "candidate_builder_executed": any("candidate_id" in record["candidate_event"] for record in chunk_records),
        "validator_executed": any("metrics" in record["candidate_event"] for record in chunk_records),
        "final_active_node": active_node.node_id,
        "state_bounded": all(record["gpu_decoded_history_tensor_bytes"] == 0 for record in chunk_records),
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(0)),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(0)),
        "elapsed_seconds": time.perf_counter() - started,
        "gpu_processes_before": gpu0_and_gpu1_before, "gpu_processes_after": _gpu_processes(),
    }
    (rollout_root / "rollout_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
