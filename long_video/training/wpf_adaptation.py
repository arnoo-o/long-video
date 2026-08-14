"""Utilities for training one LoRA on real Stage1/2 WPF rollout states."""
from __future__ import annotations

import copy
import random
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F


WPF_ADAPTER_NAME = "wpf_adaptation"
WPF_TARGET_MODULES = (
    "attn1.to_q", "attn1.to_k", "attn1.to_v", "attn1.to_out.0",
)
WPF_TRAINING_POSITIONS = ((1, 0), (1, 1), (2, 0), (2, 1))


def select_balanced_training_records(records, count: int = 100):
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
    for upper, count in ((400, 1), (800, 2), (1000, 3), (1200, 4)):
        if int(step) <= upper:
            return count
    return 6


@dataclass
class WPFTrainingSampler:
    """Balance rollout positions and independently sample one of four WPF steps."""

    seed: int = 0
    chunk_counts: dict[int, list[int]] = field(default_factory=dict)
    trajectory_counts: dict[int, int] = field(default_factory=lambda: {i: 0 for i in range(1, 7)})
    position_counts: dict[str, int] = field(default_factory=lambda: {
        f"stage{stage}_step{step}": 0 for stage, step in WPF_TRAINING_POSITIONS
    })
    _position_rng: random.Random = field(init=False, repr=False)

    def __post_init__(self):
        self._position_rng = random.Random(int(self.seed) ^ 0x57FADA)

    def choose_chunk(self, step: int, *, forced_max_chunks: int | None = None) -> tuple[int, int]:
        maximum = int(forced_max_chunks or curriculum_max_chunks(step))
        values = self.chunk_counts.setdefault(maximum, [0] * maximum)
        floor = min(values)
        candidates = [index for index, value in enumerate(values) if value == floor]
        rng = random.Random((self.seed << 20) + int(step) * 17 + maximum)
        selected = rng.choice(candidates)
        values[selected] += 1
        length = selected + 1
        self.trajectory_counts[length] = self.trajectory_counts.get(length, 0) + 1
        return length, selected

    def choose_position(self) -> tuple[int, int]:
        position = self._position_rng.choice(WPF_TRAINING_POSITIONS)
        self.position_counts[f"stage{position[0]}_step{position[1]}"] += 1
        return position

    def state_dict(self):
        return {
            "seed": self.seed,
            "chunk_counts": copy.deepcopy(self.chunk_counts),
            "trajectory_counts": copy.deepcopy(self.trajectory_counts),
            "position_counts": copy.deepcopy(self.position_counts),
            "position_rng_state": self._position_rng.getstate(),
        }

    def load_state_dict(self, state):
        self.seed = int(state["seed"])
        self.chunk_counts = {int(key): list(value) for key, value in state["chunk_counts"].items()}
        self.trajectory_counts = {
            int(key): int(value) for key, value in state["trajectory_counts"].items()
        }
        self.position_counts = {str(key): int(value) for key, value in state["position_counts"].items()}
        self._position_rng = random.Random()
        self._position_rng.setstate(state["position_rng_state"])


def renderer_visibility_to_latent_mask(visibility, latent_shape, *, device=None):
    """Map raw visibility with exact [0],[1..4],... VAE grouping and area pooling."""
    value = torch.as_tensor(visibility, dtype=torch.float32, device=device)
    if value.ndim == 3:
        value = value[None, None]
    elif value.ndim == 4:
        value = value[:, None]
    if value.ndim != 5 or value.shape[1] != 1 or value.shape[2] != 33:
        raise ValueError(f"visibility must be [B,1,33,H,W], got {tuple(value.shape)}")
    if not bool(((value == 0) | (value == 1)).all()):
        raise ValueError("raw renderer visibility must be binary")
    latent_t, latent_h, latent_w = map(int, latent_shape)
    if latent_t != 9:
        raise ValueError(f"33 RGB frames require 9 latent frames, got {latent_t}")
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
    return (error * weight).sum() / (prediction.shape[1] * weight.sum()).clamp_min(1e-8)


class WPFAdaptationObserver:
    """Backprop one selected real WPF step in clean-latent coordinates."""

    def __init__(self, gt_pyramid, selected_stage: int, selected_step: int, lambda_keep: float = 0.1):
        self.gt_pyramid = [value.detach() for value in gt_pyramid]
        self.selected_stage = int(selected_stage)
        self.selected_step = int(selected_step)
        if (self.selected_stage, self.selected_step) not in WPF_TRAINING_POSITIONS:
            raise ValueError("selected WPF position must be Stage1/2 step0/1")
        self.lambda_keep = float(lambda_keep)
        self.losses = {}
        self.records = []

    def should_train_step(self, stage_id: int, step_id: int) -> bool:
        return (int(stage_id), int(step_id)) == (self.selected_stage, self.selected_step)

    @staticmethod
    def _sigma(step_id, sigmas, *, device, dtype):
        values = torch.as_tensor(sigmas, device=device, dtype=dtype).flatten()
        if int(step_id) >= len(values):
            raise RuntimeError("scheduler did not expose sigma_t for selected WPF step")
        sigma = values[int(step_id)]
        if not bool(torch.isfinite(sigma)) or float(sigma) <= 0:
            raise RuntimeError(f"selected WPF step requires positive finite sigma_t, got {sigma}")
        return sigma

    def __call__(self, event):
        stage_id, step_id = int(event["stage_id"]), int(event["step_id"])
        self.records.append((stage_id, step_id))
        prediction, x_t = event["model_output"], event["sample"]
        if not self.should_train_step(stage_id, step_id):
            return prediction.detach(), x_t.detach()
        base = event.get("base_model_output")
        if base is None or base.requires_grad:
            raise RuntimeError("selected WPF step requires a detached frozen-base prediction")
        z_gt = self.gt_pyramid[stage_id].to(device=x_t.device, dtype=x_t.dtype)
        if z_gt.shape != x_t.shape:
            raise RuntimeError(f"Stage{stage_id} GT/state shape mismatch: {z_gt.shape} != {x_t.shape}")
        sigma = self._sigma(step_id, event["dmd_sigmas"], device=x_t.device, dtype=x_t.dtype)
        x0_pred = x_t.detach().float() - sigma.float() * prediction.float()
        x0_base = x_t.detach().float() - sigma.float() * base.detach().float()
        mask = renderer_visibility_to_latent_mask(
            event["point_visibility"], z_gt.shape[2:], device=x_t.device,
        ).detach()
        loss_fill = weighted_mse(x0_pred, z_gt.float(), 1.0 - mask)
        loss_keep = weighted_mse(x0_pred, x0_base, mask)
        loss = loss_fill + self.lambda_keep * loss_keep
        if not all(bool(torch.isfinite(value)) for value in (loss, loss_fill, loss_keep)):
            raise RuntimeError("non-finite WPF adaptation loss")
        loss.backward()
        self.losses[(stage_id, step_id)] = {
            "total": float(loss.detach()),
            "fill": float(loss_fill.detach()),
            "keep": float(loss_keep.detach()),
            "sigma": float(sigma.detach()),
            "world_mask_mean": float(mask.mean()),
            "world_mask_nonzero_ratio": float((mask > 0).float().mean()),
            "x0_pred_norm": float(x0_pred.norm().detach()),
            "x0_base_norm": float(x0_base.norm().detach()),
            "z_gt_norm": float(z_gt.float().norm().detach()),
        }
        return prediction.detach(), x_t.detach()

    def assert_complete(self):
        selected = (self.selected_stage, self.selected_step)
        if set(self.losses) != {selected}:
            raise RuntimeError(f"selected WPF step was not supervised: {self.losses.keys()}")
        if self.records != [(stage, step) for stage in range(3) for step in range(2)]:
            raise RuntimeError(f"formal six-step WPF rollout was not observed: {self.records}")


def adaptation_parameter_items(transformer):
    marker = f".{WPF_ADAPTER_NAME}."
    return [(name, value) for name, value in transformer.named_parameters() if marker in name]


def configure_trainable_wpf_adapter(pipe):
    from peft import LoraConfig

    pipe._unfuse_wah_lora()
    pipe.transformer.add_adapter(
        LoraConfig(
            r=16, lora_alpha=16, lora_dropout=0.0,
            target_modules=list(WPF_TARGET_MODULES), bias="none",
        ),
        adapter_name=WPF_ADAPTER_NAME,
    )
    for parameter in pipe.transformer.parameters():
        parameter.requires_grad_(False)
    items = adaptation_parameter_items(pipe.transformer)
    if not items:
        raise RuntimeError("wpf_adaptation adapter created no parameters")
    for _, parameter in items:
        parameter.requires_grad_(True)
    pipe._pyramid_adapter_names = {
        0: pipe._wah_adapter_name,
        1: WPF_ADAPTER_NAME,
        2: WPF_ADAPTER_NAME,
    }
    pipe._pyramid_training_adapter_name = WPF_ADAPTER_NAME
    # The pinned training patch uses this name when restoring requires_grad
    # after the autoregressive sampler exits.
    pipe._trainable_pyramid_adapter_name = WPF_ADAPTER_NAME
    trainable = [(name, parameter) for name, parameter in pipe.transformer.named_parameters()
                 if parameter.requires_grad]
    if [name for name, _ in trainable] != [name for name, _ in items]:
        raise RuntimeError("optimizer eligibility must contain only wpf_adaptation parameters")
    return trainable
