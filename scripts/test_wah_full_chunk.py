#!/usr/bin/env python3
"""Run one real confidence-aware WAH chunk on a pre-rendered warp sample."""
import argparse
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image


def frames(path, count):
    reader = imageio.get_reader(str(path))
    try:
        return [np.asarray(frame) for index, frame in enumerate(reader) if index < count]
    finally:
        reader.close()


def unwrap(result):
    value = result.frames if hasattr(result, "frames") else result[0]
    array = np.asarray(value)
    if array.ndim == 5:
        array = array[0]
    if array.dtype != np.uint8:
        array = (np.clip(array, 0, 1) * 255).round().astype(np.uint8)
    return array


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wah-root", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--first-frame", type=Path, required=True)
    parser.add_argument("--warp-video", type=Path, required=True)
    parser.add_argument("--visibility-video", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-frames", type=int, default=17)
    args = parser.parse_args()
    sys.path.insert(0, str(args.wah_root))
    from warp_as_history import WarpAsHistoryPipeline

    warp = frames(args.warp_video, 1000000)
    visibility_rgb = frames(args.visibility_video, 1000000)
    visibility = np.stack([
        frame[..., 0] if frame.ndim == 3 else frame for frame in visibility_rgb
    ]).astype(np.float32) / 255.0
    confidence = visibility.copy()
    confidence[:, :, confidence.shape[2] // 2:] *= 0.25
    pipe = WarpAsHistoryPipeline.from_pretrained(
        args.model, torch_dtype=torch.bfloat16
    ).to("cuda")
    result = pipe(
        prompt="A cyclist continues through a stable wooded scene.",
        image=Image.open(args.first_frame).convert("RGB"),
        warp_video=warp,
        warp_visibility_mask=visibility,
        warp_confidence_mask=confidence,
        lora_path=None,
        height=384,
        width=640,
        num_frames=args.num_frames,
        output_type="np",
        warp_history_downsample_mode="short",
        rope_alignment=True,
        visible_token_drop=True,
        pyramid_num_inference_steps_list=[1, 1, 1],
        is_amplify_first_chunk=False,
    )
    output = unwrap(result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(args.output, output, fps=15, macro_block_size=1)
    print({"frames": len(output), "shape": output.shape, "output": str(args.output)})


if __name__ == "__main__":
    main()
