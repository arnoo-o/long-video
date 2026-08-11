"""Training utilities for the only trainable Stage0 causal-world FiLM."""
from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset

from ..wah.stage0_causal_world_film import (
    CausalTrainingContract,
    freeze_for_stage0_film_training,
    install_stage0_causal_world_film,
)


def validate_generic_rgb_manifest(path):
    """Validate ordinary RGB-video records without dataset-specific semantics."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    records = payload.get("records", payload) if isinstance(payload, dict) else payload
    if not isinstance(records, list) or not records:
        raise ValueError("manifest must contain a non-empty records list")
    for record in records:
        required = {"video", "camera_poses", "intrinsics", "prompt", "target_frame_start"}
        missing = required.difference(record)
        if missing:
            raise ValueError(f"generic RGB record is missing {sorted(missing)}")
        contract = CausalTrainingContract(
            conditioning_frame_end=int(record.get("conditioning_frame_end", -1)),
            target_frame_start=int(record["target_frame_start"]),
            uses_future_gt=bool(record.get("uses_future_gt", False)),
        )
        contract.validate()
    return records


def validate_dl3dv_film_manifest(path):
    """Validate the standalone DL3DV trajectory schema and causal contract."""
    from ..data.dl3dv import validate_trajectory_record
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("dataset") != "DL3DV-10K official 480P images+poses":
        raise ValueError("not a DL3DV FiLM manifest")
    if payload.get("uses_future_gt") is not False:
        raise ValueError("DL3DV manifest must declare uses_future_gt=false")
    root = Path(path).parent
    records = payload.get("records") or []
    if not records: raise ValueError("DL3DV manifest has no trajectories")
    for record in records: validate_trajectory_record(record, root)
    return records


class GenericRGBVideoDataset(Dataset):
    """Plain video/camera records; no scene or dataset-specific fields."""

    def __init__(self, manifest):
        self.records = validate_generic_rgb_manifest(manifest)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        import imageio.v2 as imageio
        import numpy as np

        record = dict(self.records[index])
        frames = np.asarray(imageio.mimread(record["video"]))
        cameras = np.load(record["camera_poses"]).astype(np.float32)
        intrinsics = np.load(record["intrinsics"]).astype(np.float32)
        if len(frames) != len(cameras):
            raise ValueError("RGB frames and camera poses must have equal length")
        if intrinsics.ndim == 2:
            intrinsics = np.repeat(intrinsics[None], len(frames), axis=0)
        if len(intrinsics) != len(frames):
            raise ValueError("RGB frames and intrinsics must have equal length")
        record.update(rgb=frames, c2w=cameras, intrinsics_array=intrinsics)
        return record


class Stage0FilmTrainer:
    """Small optimizer wrapper; all model computation remains native train_exact."""

    def __init__(self, pipe, pi3_backend, *, learning_rate=1e-4, max_grad_norm=1.0):
        self.names, self.parameters = freeze_causal_world_training_stack(pipe, pi3_backend)
        self.pipe = pipe
        self.max_grad_norm = float(max_grad_norm)
        self.optimizer = torch.optim.AdamW(self.parameters, lr=float(learning_rate), weight_decay=0.0)
        self.step_index = 0

    def step(self, *, contract, prompt_embeds, target_latents, histories, world0, visibility0, args, device):
        contract.validate()
        self.optimizer.zero_grad(set_to_none=True)
        loss = stage0_flow_matching_loss(
            self.pipe, prompt_embeds, target_latents, histories,
            world0, visibility0, args, device,
        )
        loss.backward()
        illegal = [name for name, parameter in self.pipe.transformer.named_parameters()
                   if not parameter.requires_grad and parameter.grad is not None]
        if illegal:
            raise RuntimeError(f"frozen model parameters received gradients: {illegal[:5]}")
        grad_norm = torch.nn.utils.clip_grad_norm_(self.parameters, self.max_grad_norm)
        self.optimizer.step()
        self.step_index += 1
        return {"step": self.step_index, "loss": float(loss.detach()), "grad_norm": float(grad_norm)}


def freeze_causal_world_training_stack(pipe, pi3_backend):
    """Freeze Helios, original WAH, VAE and Pi3; return only FiLM parameters."""
    controller = install_stage0_causal_world_film(pipe.transformer)
    names = freeze_for_stage0_film_training(pipe.transformer)
    for parameter in pipe.vae.parameters():
        parameter.requires_grad_(False)
    model = getattr(pi3_backend, "_model", None)
    if model is not None:
        for parameter in model.parameters():
            parameter.requires_grad_(False)
    trainable = [parameter for parameter in controller.film.parameters() if parameter.requires_grad]
    if {name for name, parameter in pipe.transformer.named_parameters() if parameter.requires_grad} != set(names):
        raise RuntimeError("Stage0 FiLM must be the only trainable transformer module")
    return names, trainable


def stage0_flow_matching_loss(
    pipe, prompt_embeds, target_latents, histories, world0, visibility0, args, device,
):
    """Run the pinned train-exact Stage0 objective with causal FiLM context.

    ``target_latents`` are supervision only.  ``world0`` and ``histories`` must
    already have been constructed from frames ending before the target; this
    separation is enforced by :class:`CausalTrainingContract` at the data edge.
    """
    from warp_as_history.training import core as opt

    items = opt.flow_matching_train_exact_items(pipe, target_latents, args, device)
    stage0 = [item for item in items if int(item["stage_id"]) == 0]
    if len(stage0) != 1:
        raise RuntimeError("Stage0 FiLM training requires exactly one native Stage0 flow item")
    item = stage0[0]
    expected = tuple(item["noisy_latents"].shape)
    if tuple(world0.shape) != expected:
        raise ValueError(f"causal world Stage0 shape {tuple(world0.shape)} != {expected}")
    controller = pipe.transformer.stage0_causal_world_film
    controller.set_context(world0, visibility0)
    dtype = opt.transformer_compute_dtype(pipe.transformer)
    prediction = opt.transformer_model_forward(
        pipe,
        [item["noisy_latents"].to(dtype=dtype)],
        [item["timesteps"]],
        prompt_embeds,
        histories,
        attention_kwargs={
            "history_visible_token_mode": str(getattr(args, "visible_token_mode", "drop")),
            "history_visible_token_threshold": float(getattr(args, "history_visible_token_threshold", 0.05)),
            "history_confidence_threshold": float(getattr(args, "history_confidence_threshold", 0.1)),
            "history_confidence_lambda": float(getattr(args, "history_confidence_lambda", 1.0)),
            "history_confidence_epsilon": float(getattr(args, "history_confidence_epsilon", 1e-6)),
        },
        target_channel_fusion_latents=None,
        is_first_denoising_step=False,
    )
    if not isinstance(prediction, list) or len(prediction) != 1:
        raise TypeError("native Helios Stage0 forward must return one prediction")
    return (prediction[0].float() - item["target"].float()).square().mean()


def save_film_checkpoint(path, transformer, optimizer=None, scheduler=None, *, step=0, metadata=None,
                         round_robin=None):
    controller = transformer.stage0_causal_world_film
    payload = {
        "schema_version": 1,
        "architecture": "stage0_causal_world_film",
        "global_step": int(step),
        "film": controller.film.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "metadata": {"uses_future_gt": False, **dict(metadata or {}),
                     "round_robin": None if round_robin is None else round_robin.state_dict()},
        "torch_rng_state": torch.get_rng_state(),
    }
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(target)
    return target


def load_film_checkpoint(path, transformer, optimizer=None, scheduler=None, round_robin=None):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("architecture") != "stage0_causal_world_film":
        raise ValueError("checkpoint is not a Stage0 causal-world FiLM checkpoint")
    transformer.stage0_causal_world_film.film.load_state_dict(payload["film"], strict=True)
    if optimizer is not None and payload.get("optimizer") is not None:
        optimizer.load_state_dict(payload["optimizer"])
    if scheduler is not None and payload.get("scheduler") is not None:
        scheduler.load_state_dict(payload["scheduler"])
    metadata = dict(payload.get("metadata") or {})
    if round_robin is not None and metadata.get("round_robin") is not None:
        round_robin.load_state_dict(metadata["round_robin"])
    return int(payload["global_step"]), metadata
