#!/usr/bin/env python3
"""Cache frozen deterministic VAE targets for all six GeoToken windows."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
import torch


def deterministic_latent(pipe, frames, device):
    pixels = torch.from_numpy(np.stack(frames)).permute(3, 0, 1, 2).unsqueeze(0).float() / 127.5 - 1
    pixels = pixels.to(device=device, dtype=pipe.vae.dtype)
    clean = pipe.vae.encode(pixels).latent_dist.mode()
    mean = torch.tensor(pipe.vae.config.latents_mean, device=device, dtype=clean.dtype).view(1, -1, 1, 1, 1)
    std = 1 / torch.tensor(pipe.vae.config.latents_std, device=device, dtype=clean.dtype).view(1, -1, 1, 1, 1)
    return ((clean - mean) * std).detach().cpu()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wah-root", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--trajectory-ids-json", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    sys.path.insert(0, str(args.wah_root))
    from warp_as_history import WarpAsHistoryPipeline

    pipe = WarpAsHistoryPipeline.from_pretrained(args.model, torch_dtype=torch.bfloat16).to(args.device)
    pipe.vae.eval()
    identity = hashlib.sha256(json.dumps(dict(pipe.vae.config), sort_keys=True, default=str).encode()).hexdigest()
    manifest = json.loads((args.dataset_root / "dl3dv_24fps_manifest.json").read_text())
    wanted = set(json.loads(args.trajectory_ids_json.read_text()))
    for record in manifest["records"]:
        trajectory_id = record["trajectory_id"]
        if trajectory_id not in wanted:
            continue
        target = args.output_root / trajectory_id
        target.mkdir(parents=True, exist_ok=True)
        rgb_paths = [args.dataset_root / path for path in record["rgb_paths"]]
        for chunk_index in range(6):
            output = target / f"chunk_{chunk_index:02d}.pt"
            if output.exists():
                continue
            start = chunk_index * 32
            frames = [np.asarray(Image.open(path).convert("RGB"), np.uint8) for path in rgb_paths[start:start + 33]]
            if len(frames) != 33:
                raise RuntimeError(f"{trajectory_id} lacks chunk {chunk_index}")
            with torch.no_grad():
                latent = deterministic_latent(pipe, frames, args.device)
            torch.save({"latent": latent, "vae_identity": identity, "shape": tuple(latent.shape)}, output)
        (target / "metadata.json").write_text(json.dumps({"vae_identity": identity, "model": str(args.model)}))


if __name__ == "__main__":
    main()
