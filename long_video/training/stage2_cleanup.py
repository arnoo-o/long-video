"""Utilities for point-masked Stage2 completion LoRA training."""
from __future__ import annotations

import copy
import random
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

STAGE2_ADAPTER_NAME = "stage2_completion"
STAGE2_ADAPTER_SCHEDULE = (0, 0, 1, 1)
STAGE2_TARGET_MODULES = (
    "attn1.to_q", "attn1.to_k", "attn1.to_v", "attn1.to_out.0",
)


def select_balanced_training_records(records, count: int = 100):
    """Select the stable 50/50 indoor/outdoor subset shared by cache and training."""
    count = int(count)
    if count <= 0 or count % 2:
        raise ValueError("training record count must be a positive even number")
    selected = []
    for environment in ("indoor", "outdoor"):
        pool = sorted(
            (item for item in records if item.get("split") == "train"
             and item.get("environment") == environment),
            key=lambda item: item["trajectory_id"],
        )
        if len(pool) < count // 2:
            raise RuntimeError(f"insufficient {environment} train trajectories")
        selected.extend(pool[:count // 2])
    return sorted(selected, key=lambda item: item["trajectory_id"])


def curriculum_max_chunks(step: int) -> int:
    for upper, count in ((400, 1), (800, 2), (1000, 3), (1200, 4), (1400, 6)):
        if int(step) <= upper:
            return count
    return 6


@dataclass
class BalancedChunkSampler:
    """Balance supervision positions and randomly sample Stage2 step 2 or 3."""

    seed: int = 0
    counts: dict[int, list[int]] = field(default_factory=dict)
    trajectory_counts: dict[int, int] = field(default_factory=lambda: {i: 0 for i in range(1, 7)})
    stage2_step_counts: dict[int, int] = field(default_factory=lambda: {2: 0, 3: 0})
    _step_rng: random.Random = field(init=False, repr=False)

    def __post_init__(self):
        self._step_rng = random.Random(int(self.seed) ^ 0x5A17E2)

    def choose(self, step: int, *, forced_max_chunks: int | None = None) -> tuple[int, int]:
        maximum = int(forced_max_chunks or curriculum_max_chunks(step))
        values = self.counts.setdefault(maximum, [0] * maximum)
        floor = min(values)
        candidates = [i for i, value in enumerate(values) if value == floor]
        rng = random.Random((self.seed << 20) + int(step) * 17 + maximum)
        selected = rng.choice(candidates)
        values[selected] += 1
        actual_length = selected + 1
        self.trajectory_counts[actual_length] = self.trajectory_counts.get(actual_length, 0) + 1
        return actual_length, selected

    def choose_stage2_step(self) -> int:
        selected = self._step_rng.choice((2, 3))
        self.stage2_step_counts[selected] += 1
        return selected

    def state_dict(self):
        return {
            "seed": self.seed, "counts": copy.deepcopy(self.counts),
            "trajectory_counts": copy.deepcopy(self.trajectory_counts),
            "stage2_step_counts": copy.deepcopy(self.stage2_step_counts),
            "stage2_step_rng_state": self._step_rng.getstate(),
        }

    def load_state_dict(self, state):
        self.seed = int(state["seed"])
        self.counts = {int(k): list(v) for k, v in state["counts"].items()}
        self.trajectory_counts = {int(k): int(v) for k, v in state["trajectory_counts"].items()}
        self.stage2_step_counts = {int(k): int(v) for k, v in state["stage2_step_counts"].items()}
        self._step_rng = random.Random()
        self._step_rng.setstate(state["stage2_step_rng_state"])


def renderer_visibility_to_latent_mask(visibility, latent_shape, *, device=None):
    """Area-map raw 33-frame visibility to the VAE's exact 9 temporal groups."""
    value = torch.as_tensor(visibility, dtype=torch.float32, device=device)
    if value.ndim == 3:
        value = value[None, None]
    elif value.ndim == 4:
        value = value[:, None]
    if value.ndim != 5 or value.shape[1] != 1 or value.shape[2] != 33:
        raise ValueError(f"visibility must be [B,1,33,H,W], got {tuple(value.shape)}")
    if not bool(((value == 0) | (value == 1)).all()):
        raise ValueError("renderer visibility must be binary before latent mapping")
    latent_t, latent_h, latent_w = map(int, latent_shape)
    if latent_t != 9:
        raise ValueError(f"33-frame VAE visibility requires 9 latent frames, got {latent_t}")
    groups = [value[:, :, :1]] + [value[:, :, start:start + 4] for start in range(1, 33, 4)]
    temporal = torch.cat([group.mean(dim=2, keepdim=True) for group in groups], dim=2)
    batch = temporal.shape[0]
    spatial = F.adaptive_avg_pool2d(
        temporal.permute(0, 2, 1, 3, 4).reshape(batch * 9, 1, *temporal.shape[-2:]),
        (latent_h, latent_w),
    )
    return spatial.reshape(batch, 9, 1, latent_h, latent_w).permute(0, 2, 1, 3, 4)


def weighted_mse(prediction, target, weight):
    weight = weight.to(device=prediction.device, dtype=torch.float32)
    error = (prediction.float() - target.float()).square()
    channels = prediction.shape[1]
    return (error * weight).sum() / (channels * weight.sum()).clamp_min(1e-8)


class Stage2CompletionObserver:
    """Train one real Stage2 step with generation and point-preservation losses."""

    def __init__(self, z_gt, scheduler, selected_step: int, lambda_keep: float = 0.1):
        self.z_gt = z_gt.detach()
        self.scheduler = scheduler
        self.selected_step = int(selected_step)
        if self.selected_step not in (2, 3):
            raise ValueError("completion training step must be 2 or 3")
        self.m_lat = None
        self.lambda_keep = float(lambda_keep)
        self.losses = {}
        self.records = []

    def should_train_step(self, step_id: int) -> bool:
        return int(step_id) == self.selected_step

    @staticmethod
    def _sigma(timestep, timesteps, sigmas, *, device, dtype):
        ts = torch.as_tensor(timesteps, device=device).flatten()
        ss = torch.as_tensor(sigmas, device=device, dtype=dtype).flatten()
        value = torch.as_tensor(timestep, device=device).flatten()[0].to(ts.dtype)
        sigma = ss[int(torch.argmin((ts - value).abs()).item())]
        if not bool(torch.isfinite(sigma)) or float(sigma) <= 0:
            raise RuntimeError(f"Stage2 training requires positive finite sigma_t, got {sigma}")
        return sigma

    def __call__(self, event):
        if int(event["stage_id"]) != 2:
            return None
        prediction, x_t = event["model_output"], event["sample"]
        step_id = int(event["step_id"])
        adapter_active = bool(event.get("completion_adapter_active", False))
        self.records.append({"step_id": step_id, "completion_adapter_active": adapter_active})
        if step_id != self.selected_step:
            return prediction.detach(), x_t.detach()
        if self.m_lat is None:
            self.m_lat = renderer_visibility_to_latent_mask(
                event["point_visibility"], self.z_gt.shape[2:], device=self.z_gt.device,
            ).detach()
        base = event.get("base_model_output")
        if base is None or base.requires_grad:
            raise RuntimeError("selected completion step requires detached frozen-Helios baseline")
        sigma = self._sigma(event["timestep"], event["dmd_timesteps"], event["dmd_sigmas"],
                            device=x_t.device, dtype=x_t.dtype)
        target = (x_t.detach() - self.z_gt.to(device=x_t.device, dtype=x_t.dtype)) / sigma
        mask = self.m_lat.to(device=x_t.device)
        loss_gen = weighted_mse(prediction, target, 1.0 - mask)
        loss_keep = weighted_mse(prediction, base.detach(), mask)
        loss = loss_gen + self.lambda_keep * loss_keep
        if not all(bool(torch.isfinite(item)) for item in (loss, loss_gen, loss_keep)):
            raise RuntimeError(f"non-finite Stage2 completion loss at step {step_id}")
        loss.backward()
        self.losses[step_id] = {
            "total": float(loss.detach()), "gen": float(loss_gen.detach()),
            "keep": float(loss_keep.detach()), "sigma": float(sigma.detach()),
            "point_mask_ratio": float(mask.mean()),
        }
        return prediction.detach(), x_t.detach()

    def assert_complete(self):
        if [item["step_id"] for item in self.records] != [0, 1, 2, 3]:
            raise RuntimeError(f"expected all four Stage2 scheduler steps, got {self.records}")
        active = [int(item["completion_adapter_active"]) for item in self.records]
        if active != list(STAGE2_ADAPTER_SCHEDULE):
            raise RuntimeError(f"completion adapter schedule must be [0,0,1,1], got {active}")
        if set(self.losses) != {self.selected_step}:
            raise RuntimeError(f"expected only Stage2 step {self.selected_step} loss, got {self.losses}")


def completion_parameter_items(transformer):
    return [(name, value) for name, value in transformer.named_parameters()
            if f".{STAGE2_ADAPTER_NAME}." in name]


def configure_trainable_completion_adapter(pipe):
    from peft import LoraConfig

    pipe.transformer.add_adapter(
        LoraConfig(r=16, lora_alpha=16, lora_dropout=0.0,
                   target_modules=list(STAGE2_TARGET_MODULES), bias="none"),
        adapter_name=STAGE2_ADAPTER_NAME,
    )
    for parameter in pipe.transformer.parameters():
        parameter.requires_grad_(False)
    items = completion_parameter_items(pipe.transformer)
    if not items:
        raise RuntimeError("stage2_completion adapter created no parameters")
    for _, parameter in items:
        parameter.requires_grad_(True)
    pipe._pyramid_adapter_names = {0: pipe._wah_adapter_name, 1: None, 2: None}
    pipe._stage2_completion_adapter_name = STAGE2_ADAPTER_NAME
    pipe._stage2_completion_schedule = STAGE2_ADAPTER_SCHEDULE
    pipe._trainable_pyramid_adapter_name = STAGE2_ADAPTER_NAME
    trainable = [(name, parameter) for name, parameter in pipe.transformer.named_parameters()
                 if parameter.requires_grad]
    if [name for name, _ in trainable] != [name for name, _ in items]:
        raise RuntimeError("only stage2_completion parameters may be trainable")
    return trainable
