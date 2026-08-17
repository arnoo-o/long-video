#!/usr/bin/env python3
"""Generate a source frame from text on the inference host."""
from __future__ import annotations

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--negative-prompt", default="low quality, blurry, distorted")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--height", type=int, default=384)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--steps", type=int, default=28)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    import torch
    from diffusers import AutoPipelineForText2Image

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    pipe = AutoPipelineForText2Image.from_pretrained(args.model, torch_dtype=dtype).to(args.device)
    generator = torch.Generator(device=args.device).manual_seed(args.seed)
    image = pipe(
        prompt=args.prompt, negative_prompt=args.negative_prompt,
        height=args.height, width=args.width, num_inference_steps=args.steps,
        generator=generator,
    ).images[0]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    image.save(args.output)
    print(str(args.output), flush=True)


if __name__ == "__main__":
    main()
