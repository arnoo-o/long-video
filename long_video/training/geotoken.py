"""Curriculum, geometry rendering, and checkpoint helpers for GeoToken."""
from __future__ import annotations

import copy
import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from ..geometry.geotoken import GeometryTokenBatch


TOTAL_STEPS = 2000
PHASE_BOUNDARIES = {500: "phase_a_final", 1100: "phase_b_final", 2000: "phase_c_final"}


def phase_for_step(step: int) -> str:
    step = int(step)
    if not 1 <= step <= TOTAL_STEPS:
        raise ValueError(f"global step must be in [1,{TOTAL_STEPS}]")
    return "A" if step <= 500 else "B" if step <= 1100 else "C"


def max_chunks_for_step(step: int) -> int:
    step = int(step)
    schedule = (
        (160, 1), (320, 2), (500, 3),
        (700, 1), (900, 2), (1100, 3),
        (1300, 1), (1500, 2), (1700, 3), (1860, 4), (2000, 6),
    )
    if not 1 <= step <= TOTAL_STEPS:
        raise ValueError(f"global step must be in [1,{TOTAL_STEPS}]")
    return next(value for upper, value in schedule if step <= upper)


def checkpoint_names(step: int) -> tuple[str, ...]:
    names = []
    if int(step) % 80 == 0:
        names.append(f"checkpoint_step_{int(step):04d}.pt")
    phase = PHASE_BOUNDARIES.get(int(step))
    if phase:
        names.append(f"{phase}_step_{int(step):04d}.pt")
    return tuple(names)


@dataclass
class BalancedRolloutSampler:
    seed: int = 0
    counts: dict[int, list[int]] = field(default_factory=dict)
    trajectory_counts: dict[str, int] = field(default_factory=dict)

    def choose_length(self, step: int) -> int:
        maximum = max_chunks_for_step(step)
        values = self.counts.setdefault(maximum, [0] * maximum)
        floor = min(values)
        candidates = [index for index, value in enumerate(values) if value == floor]
        choice = random.Random((int(self.seed) << 24) ^ (int(step) * 131) ^ maximum).choice(candidates)
        values[choice] += 1
        return choice + 1

    def choose_record(self, records, step: int):
        if not records:
            raise ValueError("training record list is empty")
        rng = random.Random((int(self.seed) << 16) ^ int(step))
        record = records[rng.randrange(len(records))]
        key = str(record["trajectory_id"])
        self.trajectory_counts[key] = self.trajectory_counts.get(key, 0) + 1
        return record

    def state_dict(self):
        return {
            "seed": int(self.seed),
            "counts": copy.deepcopy(self.counts),
            "trajectory_counts": copy.deepcopy(self.trajectory_counts),
        }

    def load_state_dict(self, state):
        self.seed = int(state["seed"])
        self.counts = {int(key): list(value) for key, value in state["counts"].items()}
        self.trajectory_counts = {str(key): int(value) for key, value in state["trajectory_counts"].items()}


def apply_partial_geometry_augmentation(
    geometry: torch.Tensor,
    *,
    point_dropout: float,
    confidence_dropout: float,
    depth_noise: float,
    xyz_jitter: float,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Apply Phase-B noise without changing coordinates of retained observations."""
    result = geometry.clone()
    valid = result[:, 10:11] > 0
    if point_dropout > 0:
        keep = torch.rand(valid.shape, device=result.device, generator=generator) >= point_dropout
        valid &= keep
    if confidence_dropout > 0:
        keep_conf = torch.rand(valid.shape, device=result.device, generator=generator) >= confidence_dropout
        result[:, 11:12] *= keep_conf
    if xyz_jitter > 0:
        result[:, :3] += torch.randn(
            result[:, :3].shape, device=result.device, generator=generator,
        ) * float(xyz_jitter) * valid
    if depth_noise > 0:
        result[:, 9:10] += torch.randn(
            result[:, 9:10].shape, device=result.device, generator=generator,
        ) * float(depth_noise) * valid
    result[:, 10:11] = valid
    result[:, 11:12] *= valid
    return result * valid


def concatenate_conditioning(parts: list[GeometryTokenBatch]) -> GeometryTokenBatch:
    if not parts:
        raise ValueError("at least one geometry token part is required")
    return GeometryTokenBatch(
        tokens=torch.cat([part.tokens for part in parts], dim=1),
        support=torch.cat([part.support for part in parts], dim=1),
    )


def save_geotoken_checkpoint(path: Path, *, transformer, optimizer, lr_scheduler, step, sampler):
    state = {
        name: value.detach().cpu()
        for name, value in transformer.named_parameters() if "geotoken." in name
    }
    if not state:
        raise RuntimeError("GeoToken checkpoint has no trainable module state")
    payload = {
        "geotoken": state,
        "optimizer": optimizer.state_dict(),
        "lr_scheduler": lr_scheduler.state_dict(),
        "global_step": int(step),
        "sampling_state": sampler.state_dict(),
        "rng_state": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
    }
    torch.save(payload, path)


def load_geotoken_checkpoint(path: Path, *, transformer, optimizer, lr_scheduler, sampler) -> int:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    named = dict(transformer.named_parameters())
    if set(payload["geotoken"]) != {name for name in named if "geotoken." in name}:
        raise RuntimeError("GeoToken checkpoint parameter set does not match the model")
    with torch.no_grad():
        for name, value in payload["geotoken"].items():
            named[name].copy_(value.to(device=named[name].device, dtype=named[name].dtype))
    optimizer.load_state_dict(payload["optimizer"])
    lr_scheduler.load_state_dict(payload["lr_scheduler"])
    sampler.load_state_dict(payload["sampling_state"])
    random.setstate(payload["rng_state"]["python"])
    np.random.set_state(payload["rng_state"]["numpy"])
    torch.set_rng_state(payload["rng_state"]["torch"])
    if torch.cuda.is_available() and payload["rng_state"].get("cuda") is not None:
        torch.cuda.set_rng_state_all(payload["rng_state"]["cuda"])
    return int(payload["global_step"])
