#!/usr/bin/env python3
"""Numerically compare original WAH with patched confidence=1 behavior."""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image


def read_frames(path: Path):
    reader = imageio.get_reader(str(path))
    try:
        return [np.asarray(frame) for frame in reader]
    finally:
        reader.close()


def unwrap(result):
    value = result.frames if hasattr(result, "frames") else result[0]
    array = np.asarray(value)
    return array[0] if array.ndim == 5 else array


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wah-root", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--first-frame", type=Path, required=True)
    parser.add_argument("--warp-video", type=Path, required=True)
    parser.add_argument("--visibility-video", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("original", "confidence_one"), required=True)
    parser.add_argument("--seed", type=int, default=20260806)
    parser.add_argument("--num-frames", type=int, default=17)
    args = parser.parse_args()

    sys.path.insert(0, str(args.wah_root))
    from warp_as_history import WarpAsHistoryPipeline

    warp = read_frames(args.warp_video)
    visibility_rgb = read_frames(args.visibility_video)
    visibility = np.stack([
        frame[..., 0] if frame.ndim == 3 else frame for frame in visibility_rgb
    ]).astype(np.float32) / 255.0

    torch.manual_seed(args.seed)
    generator = torch.Generator(device="cuda").manual_seed(args.seed)
    torch.cuda.reset_peak_memory_stats()
    pipe = WarpAsHistoryPipeline.from_pretrained(
        args.model, torch_dtype=torch.bfloat16
    ).to("cuda")
    kwargs = dict(
        prompt="A cyclist continues through a stable wooded scene.",
        image=Image.open(args.first_frame).convert("RGB"),
        warp_video=warp,
        warp_visibility_mask=visibility,
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
        generator=generator,
    )
    if args.mode == "confidence_one":
        kwargs["warp_confidence_mask"] = np.ones_like(visibility, dtype=np.float32)
    start = time.time()
    output = unwrap(pipe(**kwargs))
    elapsed = time.time() - start
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, output)
    metadata = {
        "mode": args.mode,
        "seed": args.seed,
        "shape": list(output.shape),
        "dtype": str(output.dtype),
        "elapsed_seconds": elapsed,
        "peak_gpu_bytes": int(torch.cuda.max_memory_allocated()),
    }
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()