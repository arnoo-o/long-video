"""Utilities for the Stage2-only cleanup LoRA experiment."""
from __future__ import annotations

import copy
import random
from dataclasses import dataclass, field

import torch

STAGE2_ADAPTER_NAME = "stage2_cleanup"
STAGE2_TARGET_MODULES = (
    "attn1.to_q", "attn1.to_k", "attn1.to_v", "attn1.to_out.0",
)


def select_balanced_training_records(records, count: int = 100):
    """Select a stable 50/50 indoor/outdoor subset shared by cache and training."""
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
    for upper, count in ((400, 1), (800, 2), (1000, 3), (1200, 4), (1400, 5)):
        if int(step) <= upper:
            return count
    return 6


@dataclass
class BalancedChunkSampler:
    """Balance supervision positions within each curriculum horizon."""

    seed: int = 0
    counts: dict[int, list[int]] = field(default_factory=dict)
    trajectory_counts: dict[int, int] = field(default_factory=lambda: {i: 0 for i in range(1, 7)})

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

    def state_dict(self):
        return {"seed": self.seed, "counts": copy.deepcopy(self.counts),
                "trajectory_counts": copy.deepcopy(self.trajectory_counts)}

    def load_state_dict(self, state):
        self.seed = int(state["seed"])
        self.counts = {int(k): list(v) for k, v in state["counts"].items()}
        self.trajectory_counts = {int(k): int(v) for k, v in state["trajectory_counts"].items()}


class Stage2FlowObserver:
    """Backpropagate each real Stage2 state independently, then detach it."""

    def __init__(self, z_gt: torch.Tensor, scheduler, selected_step: int):
        self.z_gt = z_gt.detach()
        self.scheduler = scheduler
        self.selected_step = int(selected_step)
        if self.selected_step not in range(4):
            raise ValueError("selected Stage2 step must be in [0,3]")
        self.losses: dict[int, float] = {}
        self.records: list[dict] = []

    def should_train_step(self, step_id: int) -> bool:
        return int(step_id) == self.selected_step

    @staticmethod
    def _sigma(timestep, timesteps, sigmas, *, device, dtype):
        ts = torch.as_tensor(timesteps, device=device).flatten()
        ss = torch.as_tensor(sigmas, device=device, dtype=dtype).flatten()
        value = torch.as_tensor(timestep, device=device).flatten()[0].to(ts.dtype)
        index = int(torch.argmin((ts - value).abs()).item())
        sigma = ss[index]
        if not bool(torch.isfinite(sigma)) or float(sigma) <= 0:
            raise RuntimeError(f"Stage2 training requires positive finite sigma_t, got {sigma}")
        return sigma

    def __call__(self, event):
        if int(event["stage_id"]) != 2:
            return None
        prediction = event["model_output"]
        x_t = event["sample"]
        step_id = int(event["step_id"])
        self.records.append({"step_id": step_id})
        if step_id != self.selected_step:
            return prediction.detach(), x_t.detach()
        sigma = self._sigma(event["timestep"], event["dmd_timesteps"], event["dmd_sigmas"],
                            device=x_t.device, dtype=x_t.dtype)
        target = (x_t.detach() - self.z_gt.to(device=x_t.device, dtype=x_t.dtype)) / sigma
        loss = torch.nn.functional.mse_loss(prediction.float(), target.float())
        if not bool(torch.isfinite(loss)):
            raise RuntimeError(f"non-finite Stage2 loss at step {event['step_id']}")
        loss.backward()
        self.losses[step_id] = float(loss.detach())
        self.records[-1]["sigma"] = float(sigma.detach())
        return prediction.detach(), x_t.detach()

    def assert_complete(self):
        if [x["step_id"] for x in self.records] != [0, 1, 2, 3]:
            raise RuntimeError(f"expected all four Stage2 scheduler steps, got {self.records}")
        if set(self.losses) != {self.selected_step}:
            raise RuntimeError(f"expected only Stage2 step {self.selected_step} loss, got {self.losses}")


def cleanup_parameter_items(transformer):
    return [(name, value) for name, value in transformer.named_parameters()
            if f".{STAGE2_ADAPTER_NAME}." in name]


def configure_trainable_cleanup_adapter(pipe):
    from peft import LoraConfig

    pipe.transformer.add_adapter(
        LoraConfig(r=16, lora_alpha=16, lora_dropout=0.0,
                   target_modules=list(STAGE2_TARGET_MODULES), bias="none"),
        adapter_name=STAGE2_ADAPTER_NAME,
    )
    for parameter in pipe.transformer.parameters():
        parameter.requires_grad_(False)
    items = cleanup_parameter_items(pipe.transformer)
    if not items:
        raise RuntimeError("stage2_cleanup adapter created no parameters")
    for _, parameter in items:
        parameter.requires_grad_(True)
    pipe._pyramid_adapter_names = {0: pipe._wah_adapter_name, 1: None, 2: STAGE2_ADAPTER_NAME}
    pipe._trainable_pyramid_adapter_name = STAGE2_ADAPTER_NAME
    trainable = [(name, p) for name, p in pipe.transformer.named_parameters() if p.requires_grad]
    if [name for name, _ in trainable] != [name for name, _ in items]:
        raise RuntimeError("only stage2_cleanup parameters may be trainable")
    return trainable
