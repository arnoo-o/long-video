#!/usr/bin/env python3
"""Run one real WAH chunk from an existing active node and controls JSON."""
import argparse
import json
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image

from long_video.memory.node_store import NodeStore
from long_video.data.camera import resize_intrinsics
from long_video.online.pipeline import OnlineSpatialHistoryPipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wah-root", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--session", required=True)
    parser.add_argument("--node-id", default="node_000")
    parser.add_argument("--controls", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--height", type=int, default=384)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--num-frames", type=int, default=17)
    parser.add_argument("--device",required=True,help="Explicit device, e.g. cuda:0 after CUDA_VISIBLE_DEVICES=1")
    args = parser.parse_args()
    sys.path.insert(0, str(args.wah_root))
    from warp_as_history import WarpAsHistoryPipeline

    node = NodeStore(args.session).load(args.node_id)
    controls = json.loads(Path(args.controls).read_text(encoding="utf-8"))
    wah = WarpAsHistoryPipeline.from_pretrained(
        args.model, torch_dtype=torch.bfloat16
    ).to(args.device)
    online = OnlineSpatialHistoryPipeline(
        wah_pipeline=wah, active_node=node, prompt="Continue the indoor scene.",
        renderer_kwargs={"device":args.device},
        wah_state_kwargs={"height": args.height, "width": args.width,
                          "num_frames": args.num_frames, "output_type": "np",
                          "lora_path": None,
                          "pyramid_num_inference_steps_list": [1, 1, 1],
                          "is_amplify_first_chunk": False},
    )
    online.autoregressive_state = wah.init_autoregressive_state(
        prompt=online.prompt, image=Image.fromarray(node.view_rgb[0]),
        conditioning_type="warp", warp_history_downsample_mode="short",
        rope_alignment=True, height=args.height, width=args.width,
        num_frames=args.num_frames, output_type="np", lora_path=None,
        pyramid_num_inference_steps_list=[1, 1, 1], is_amplify_first_chunk=False,
    )
    intrinsics = resize_intrinsics(node.view_intrinsics[0],
                                   node.view_rgb.shape[1:3],(args.height,args.width))
    generated, poses, warp, stats = online.generate_chunk(
        controls, intrinsics, args.height, args.width
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    video = generated if generated.dtype == np.uint8 else (
        np.clip(generated, 0, 1) * 255
    ).round().astype(np.uint8)
    imageio.mimwrite(args.output, video, fps=15, macro_block_size=1)
    np.save(args.output.with_suffix(".poses.npy"), poses)
    args.output.with_suffix(".json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
