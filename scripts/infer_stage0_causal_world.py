#!/usr/bin/env python3
"""Run original WAH with Pi3 causal world and Stage0-only FiLM."""
import argparse
import json
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image

from long_video.config import load_yaml
from long_video.data.camera import resize_intrinsics
from long_video.initialization.geometry_backend import Pi3GeometryBackend
from long_video.memory.memory_manager import MemoryManager
from long_video.memory.node_store import NodeStore
from long_video.online.pipeline import OnlineSpatialHistoryPipeline
from long_video.training.stage0_causal_world import load_film_checkpoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wah-root", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--session", type=Path, required=True)
    parser.add_argument("--controls", type=Path, required=True, help="JSON list of per-chunk control lists")
    parser.add_argument("--pi3-repo", type=Path, required=True)
    parser.add_argument("--pi3-checkpoint", type=Path, required=True)
    parser.add_argument("--film-checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--node-id", default="node_000")
    parser.add_argument("--height", type=int, default=384)
    parser.add_argument("--width", type=int, default=640)
    args = parser.parse_args()

    sys.path.insert(0, str(args.wah_root))
    from warp_as_history import WarpAsHistoryPipeline
    node = NodeStore(args.session).load(args.node_id)
    pipe = WarpAsHistoryPipeline.from_pretrained(args.model, torch_dtype=torch.bfloat16).to(args.device)
    geometry = Pi3GeometryBackend(args.pi3_checkpoint, args.pi3_repo, args.device)
    manager = MemoryManager.from_config(load_yaml("configs/online_memory.yaml"), geometry_backend=geometry)
    online = OnlineSpatialHistoryPipeline(
        wah_pipeline=pipe, active_node=node, memory_manager=manager,
        prompt="Continue the scene consistently.", renderer_kwargs={"device": args.device},
        wah_state_kwargs={"height": args.height, "width": args.width, "num_frames": 33,
                          "pyramid_num_inference_steps_list": [2, 2, 2]},
    )
    online.autoregressive_state = pipe.init_autoregressive_state(
        prompt=online.prompt, image=Image.fromarray(node.view_rgb[0]),
        conditioning_type="warp", warp_history_downsample_mode="short",
        rope_alignment=True, height=args.height, width=args.width, num_frames=33,
        output_type="np", pyramid_num_inference_steps_list=[2, 2, 2],
    )
    online.wah_adapter.configure_state(online.autoregressive_state)
    if args.film_checkpoint:
        load_film_checkpoint(args.film_checkpoint, pipe.transformer)
    controls = json.loads(args.controls.read_text(encoding="utf-8"))
    intrinsics = resize_intrinsics(
        node.view_intrinsics[0], node.view_rgb.shape[1:3], (args.height, args.width),
    )
    frames, diagnostics = [], []
    for chunk_controls in controls:
        generated, _poses, _warp, report = online.generate_chunk(
            chunk_controls, intrinsics, args.height, args.width,
        )
        frames.append(generated if not frames else generated[1:])
        diagnostics.append(report)
    video = np.concatenate(frames)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if video.dtype != np.uint8:
        video = np.rint(np.clip(video, 0, 1) * 255).astype(np.uint8)
    imageio.mimwrite(args.output, video, fps=24, macro_block_size=1)
    args.output.with_suffix(".json").write_text(json.dumps(diagnostics, indent=2, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
