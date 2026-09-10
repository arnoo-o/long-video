#!/usr/bin/env python3
"""Audit the temporal anchors of the exact Helios Wan VAE used for training.

This audit intentionally does not import or use the repository's temporal-group
constants.  The static pass symbolically executes the actual diffusers
``WanEncoder3d`` with the temporal behaviour of every ``WanCausalConv3d`` and
``downsample3d`` cache.  The runtime pass then perturbs one RGB frame at a time
and calls the real ``AutoencoderKLWan.encode(...).latent_dist.mode()``.

Run this with the pinned Helios environment (diffusers==0.36.0), for example:

  CUDA_VISIBLE_DEVICES=0 /ephemeral/arnoo/sightline-parallel/rgbd-env/bin/python \
      scripts/audit_vae_temporal_anchors.py \
      --model /ephemeral/arnoo/sightline-parallel/model \
      --helios-root /ephemeral/arnoo/sightline-parallel/helios-root \
      --repo-root . --device cuda --output /tmp/vae_temporal_anchor_audit.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_commit(path: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _jsonable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    return str(value)


class TemporalDeps:
    """A shape-carrying temporal list whose entries are source-frame sets."""

    def __init__(
        self,
        frames: Iterable[Iterable[int]],
        channels: int = 1,
        height: int = 1,
        width: int = 1,
    ) -> None:
        self.frames = [frozenset(frame) for frame in frames]
        self.channels = int(channels)
        self.height = int(height)
        self.width = int(width)

    @property
    def shape(self) -> tuple[int, int, int, int, int]:
        return (1, self.channels, len(self.frames), self.height, self.width)

    @property
    def device(self) -> str:
        return "shadow"

    def size(self, dim: int | None = None):
        shape = self.shape
        return shape if dim is None else shape[dim]

    def clone(self) -> "TemporalDeps":
        return TemporalDeps(self.frames, self.channels, self.height, self.width)

    def to(self, *_args, **_kwargs) -> "TemporalDeps":
        return self

    def unsqueeze(self, dim: int) -> "TemporalDeps":
        if dim != 2:
            raise ValueError(f"temporal shadow only supports unsqueeze(dim=2), got {dim}")
        return self

    def __getitem__(self, index):
        if not isinstance(index, tuple):
            raise TypeError("TemporalDeps expects a 5D index")
        if len(index) != 5:
            raise IndexError("TemporalDeps expects a 5D index")
        temporal = index[2]
        if isinstance(temporal, int):
            selected = [self.frames[temporal]]
        elif isinstance(temporal, slice):
            selected = self.frames[temporal]
        else:
            raise TypeError(f"unsupported temporal index {temporal!r}")
        return TemporalDeps(selected, self.channels, self.height, self.width)

    def __add__(self, other):
        if not isinstance(other, TemporalDeps):
            return self
        if len(self.frames) != len(other.frames):
            raise ValueError("cannot add temporal tensors with different lengths")
        return TemporalDeps(
            [left | right for left, right in zip(self.frames, other.frames)],
            max(self.channels, other.channels),
            self.height,
            self.width,
        )

    __radd__ = __add__

    def __repr__(self) -> str:
        return f"TemporalDeps(T={len(self.frames)}, channels={self.channels})"


def _concat_temporal(values: Iterable[TemporalDeps], dim: int) -> TemporalDeps:
    values = list(values)
    if not values or not all(isinstance(value, TemporalDeps) for value in values):
        raise TypeError("temporal shadow concat received a non-temporal value")
    if dim != 2:
        raise ValueError(f"temporal shadow only supports dim=2, got {dim}")
    first = values[0]
    frames = [frame for value in values for frame in value.frames]
    return TemporalDeps(frames, first.channels, first.height, first.width)


def _shadow_temporal_conv(module, x: TemporalDeps, cache_x=None) -> TemporalDeps:
    """Exact temporal index arithmetic of WanCausalConv3d.forward."""
    kernel = int(module.kernel_size[0])
    stride = int(module.stride[0])
    original_left = int(module._padding[4])
    sequence = x
    left = original_left
    if cache_x is not None and original_left > 0:
        sequence = _concat_temporal([cache_x, x], dim=2)
        left = max(0, original_left - cache_x.shape[2])
    length = len(sequence.frames)
    output_length = max(0, (length + left - kernel) // stride + 1)
    output = []
    for output_index in range(output_length):
        first = output_index * stride - left
        support = set()
        for input_index in range(first, first + kernel):
            if 0 <= input_index < length:
                support.update(sequence.frames[input_index])
        output.append(frozenset(support))
    return TemporalDeps(output, module.out_channels, x.height, x.width)


def _install_shadow_runtime(wan_module):
    """Patch only the temporal shadow objects, returning an undo callback."""
    import torch

    originals = {
        wan_module.WanCausalConv3d.forward: wan_module.WanCausalConv3d.forward,
        wan_module.WanRMS_norm.forward: wan_module.WanRMS_norm.forward,
        wan_module.WanAttentionBlock.forward: wan_module.WanAttentionBlock.forward,
        wan_module.WanResample.forward: wan_module.WanResample.forward,
    }
    # Keep a separate class-keyed map; the first dict above is only a compact
    # guard against accidental duplicate patch targets.
    original_methods = {
        "conv": wan_module.WanCausalConv3d.forward,
        "rms": wan_module.WanRMS_norm.forward,
        "attn": wan_module.WanAttentionBlock.forward,
        "resample": wan_module.WanResample.forward,
    }

    def conv_forward(self, x, cache_x=None):
        return _shadow_temporal_conv(self, x, cache_x)

    def identity_forward(self, x):
        return x

    def resample_forward(self, x, feat_cache=None, feat_idx=[0]):
        if self.mode != "downsample3d":
            return x
        index = feat_idx[0]
        if feat_cache[index] is None:
            feat_cache[index] = x.clone()
            feat_idx[0] += 1
            return x
        previous = feat_cache[index][:, :, -1:, :, :].clone()
        feat_cache[index] = x[:, :, -1:, :, :].clone()
        feat_idx[0] += 1
        return _shadow_temporal_conv(self.time_conv, _concat_temporal([previous, x], 2), None)

    wan_module.WanCausalConv3d.forward = conv_forward
    wan_module.WanRMS_norm.forward = identity_forward
    wan_module.WanAttentionBlock.forward = identity_forward
    wan_module.WanResample.forward = resample_forward

    import torch.nn as nn

    original_silu = nn.SiLU.forward
    original_dropout = nn.Dropout.forward
    original_torch_cat = torch.cat
    nn.SiLU.forward = identity_forward
    nn.Dropout.forward = identity_forward

    def cat(values, dim=0, *args, **kwargs):
        if any(isinstance(value, TemporalDeps) for value in values):
            return _concat_temporal(values, dim)
        return original_torch_cat(values, dim=dim, *args, **kwargs)

    torch.cat = cat

    def restore() -> None:
        wan_module.WanCausalConv3d.forward = original_methods["conv"]
        wan_module.WanRMS_norm.forward = original_methods["rms"]
        wan_module.WanAttentionBlock.forward = original_methods["attn"]
        wan_module.WanResample.forward = original_methods["resample"]
        nn.SiLU.forward = original_silu
        nn.Dropout.forward = original_dropout
        torch.cat = original_torch_cat

    return restore


def _static_trace(vae, wan_module, frame_count: int) -> dict[str, Any]:
    temporal_layers = []
    for name, module in vae.encoder.named_modules():
        if isinstance(module, wan_module.WanCausalConv3d):
            temporal_layers.append(
                {
                    "name": name,
                    "kind": "WanCausalConv3d",
                    "kernel_t": int(module.kernel_size[0]),
                    "stride_t": int(module.stride[0]),
                    "left_padding_t": int(module._padding[4]),
                    "padding_argument_t": int(module._padding[4] // 2),
                    "cache_window_t": 2,
                    "cache_used_by_encoder": True,
                }
            )
        elif isinstance(module, wan_module.WanResample) and module.mode == "downsample3d":
            conv = module.time_conv
            temporal_layers.append(
                {
                    "name": name + ".time_conv",
                    "kind": "WanResample.downsample3d.time_conv",
                    "kernel_t": int(conv.kernel_size[0]),
                    "stride_t": int(conv.stride[0]),
                    "left_padding_t": int(conv._padding[4]),
                    "padding_argument_t": int(conv._padding[4] // 2),
                    "cache_window_t": 1,
                    "cache_used_by_encoder": True,
                }
            )

    source = TemporalDeps([{frame} for frame in range(frame_count)], channels=3)
    restore = _install_shadow_runtime(wan_module)
    try:
        shadow = vae._encode(source)
    finally:
        restore()
    ranges = []
    for index, support in enumerate(shadow.frames):
        ranges.append(
            {
                "latent_index": index,
                "min_rgb_frame": min(support) if support else None,
                "max_rgb_frame": max(support) if support else None,
                "rgb_frames": sorted(support),
            }
        )
    return {
        "frame_count": frame_count,
        "temporal_layers": temporal_layers,
        "latent_shape_temporal": len(shadow.frames),
        "latent_ranges": ranges,
        "right_edge_anchors": [item["max_rgb_frame"] for item in ranges],
    }


def _load_mode(vae, x):
    encoded = vae.encode(x)
    distribution = getattr(encoded, "latent_dist", None)
    if distribution is None:
        raise RuntimeError("AutoencoderKLWan.encode returned no latent_dist")
    return distribution.mode()


def _dependency_test(vae, torch, frame_count: int, height: int, width: int, perturbation: float):
    device = next(vae.parameters()).device
    generator = torch.Generator(device=device).manual_seed(20260910)
    delta_generator = torch.Generator(device=device).manual_seed(20260911)
    x = torch.randn((1, 3, frame_count, height, width), generator=generator, device=device, dtype=torch.float32)
    delta = torch.randn((1, 3, 1, height, width), generator=delta_generator, device=device, dtype=torch.float32)
    delta = delta / delta.square().mean().sqrt() * perturbation
    with torch.inference_mode():
        base = _load_mode(vae, x)
        dependency = torch.zeros((base.shape[2], frame_count), device=device, dtype=torch.float32)
        for frame in range(frame_count):
            perturbed = x.clone()
            perturbed[:, :, frame : frame + 1] += delta
            changed = _load_mode(vae, perturbed)
            dependency[:, frame] = (changed - base).float().square().mean(dim=(0, 1, 3, 4)).sqrt()
    return {
        "input_shape": list(x.shape),
        "latent_shape": list(base.shape),
        "dependency": dependency.cpu().tolist(),
    }


def _validate(static_result, runtime_result, zero_atol: float, nonzero_atol: float) -> dict[str, Any]:
    dependency = runtime_result["dependency"]
    latent_count = len(dependency)
    expected = [4 * index for index in range(latent_count)]
    actual = [
        max((frame for frame, value in enumerate(row) if abs(value) > zero_atol), default=None)
        for row in dependency
    ]
    future_max = []
    anchor_values = []
    for latent, row in enumerate(dependency):
        future = [abs(value) for frame, value in enumerate(row) if frame > 4 * latent]
        future_max.append(max(future, default=0.0))
        anchor_values.append(row[4 * latent] if 4 * latent < len(row) else None)
    static_anchors = static_result["right_edge_anchors"]
    checks = {
        "static_anchor_exact": static_anchors == expected,
        "runtime_anchor_exact": actual == expected,
        "future_dependency_zero": all(value <= zero_atol for value in future_max),
        "right_edge_nonzero": all(value is not None and value > nonzero_atol for value in anchor_values),
    }
    return {
        "expected_right_edge_anchors": expected,
        "static_right_edge_anchors": static_anchors,
        "runtime_right_edge_anchors": actual,
        "future_dependency_max": future_max,
        "right_edge_dependency": anchor_values,
        "checks": checks,
        "passed": all(checks.values()),
        "thresholds": {"zero_atol": zero_atol, "nonzero_atol": nonzero_atol},
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--helios-root", required=True)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--frames", type=int, default=33)
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--perturbation", type=float, default=0.125)
    parser.add_argument("--zero-atol", type=float, default=0.0)
    parser.add_argument("--nonzero-atol", type=float, default=1e-7)
    parser.add_argument("--revision", default="local")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    if args.frames != 33:
        raise SystemExit("the blocking contract is defined for exactly 33 frames")
    if not args.device.startswith("cuda"):
        raise SystemExit("this audit refuses CPU execution; pass --device cuda")

    import torch
    import diffusers
    from diffusers import AutoencoderKLWan
    import diffusers.models.autoencoders.autoencoder_kl_wan as wan_module

    if diffusers.__version__ != "0.36.0":
        raise SystemExit(f"expected diffusers==0.36.0, got {diffusers.__version__}")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    vae = AutoencoderKLWan.from_pretrained(args.model, subfolder="vae", torch_dtype=torch.float32)
    vae = vae.to(args.device).eval()
    vae.requires_grad_(False)

    static_result = _static_trace(vae, wan_module, args.frames)
    runtime_result = _dependency_test(vae, torch, args.frames, args.height, args.width, args.perturbation)
    validation = _validate(static_result, runtime_result, args.zero_atol, args.nonzero_atol)

    model_dir = Path(args.model)
    vae_dir = model_dir / "vae"
    config_path = vae_dir / "config.json"
    weight_files = sorted(path for path in vae_dir.iterdir() if path.is_file() and path.name != "config.json")
    helios_root = Path(args.helios_root)
    helios_files = [
        helios_root / "helios" / "diffusers_version" / "pipeline_helios_diffusers.py",
        helios_root / "helios" / "diffusers_version" / "transformer_helios_diffusers.py",
    ]
    provenance = {
        "diffusers_version": diffusers.__version__,
        "diffusers_module": str(Path(diffusers.__file__).resolve()),
        "python": sys.executable,
        "torch_version": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "device": str(next(vae.parameters()).device),
        "vae_class": f"{AutoencoderKLWan.__module__}.{AutoencoderKLWan.__name__}",
        "vae_snapshot": str(model_dir.resolve()),
        "vae_revision": args.revision,
        "vae_config_hash": _sha256(config_path),
        "vae_config": _jsonable(dict(vae.config)),
        "vae_files": [{"name": path.name, "size": path.stat().st_size, "sha256": _sha256(path)} for path in weight_files],
        "helios_source_commit": _git_commit(helios_root),
        "helios_source_tree": str(helios_root.resolve()),
        "helios_source_hashes": {str(path.relative_to(helios_root)): _sha256(path) for path in helios_files if path.exists()},
        "sightline_repo_commit": _git_commit(Path(args.repo_root)),
    }
    result = {
        "provenance": provenance,
        "static_derivation": static_result,
        "runtime_dependency": runtime_result,
        "validation": validation,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(output), "validation": validation}, indent=2))
    if not validation["passed"]:
        print("VAE TEMPORAL ANCHOR CONTRACT FAILED; no downstream modifications are permitted", file=sys.stderr)
        return 2
    print("VAE TEMPORAL ANCHOR CONTRACT PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

