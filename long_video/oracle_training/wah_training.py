"""Masked official WAH/Helios flow-matching LoRA training helpers."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def masked_flow_matching_loss(
    pipe,
    prompt_embeds,
    target_latents,
    histories,
    args,
    device,
    primary_loss_mask_latent,
    *,
    fixed_stage_items=None,
):
    """Official train-exact flow matching with a temporal primary-loss mask."""
    import torch
    from diffusers.training_utils import compute_loss_weighting_for_sd3
    from warp_as_history.training import core as opt

    raw_mask = np.asarray(primary_loss_mask_latent)
    if raw_mask.dtype == np.bool_:
        raw_mask = raw_mask.astype(np.float32)
    weights = torch.as_tensor(raw_mask, device=device, dtype=torch.float32)
    if weights.ndim != 1 or weights.numel() != int(target_latents.shape[2]):
        raise ValueError(
            f"latent loss weights {tuple(weights.shape)} does not match target T={target_latents.shape[2]}"
        )
    if not bool(torch.isfinite(weights).all()) or bool((weights < 0).any()):
        raise ValueError("latent temporal weights must be finite and non-negative")
    if not bool((weights > 0).any()):
        raise ValueError("primary_loss_mask_latent selects no latent frames")
    stage_items = fixed_stage_items or opt.flow_matching_train_exact_items(pipe, target_latents, args, device)
    transformer_dtype = opt.transformer_compute_dtype(pipe.transformer)
    predictions = opt.transformer_model_forward(
        pipe,
        [item["noisy_latents"].to(dtype=transformer_dtype) for item in stage_items],
        [item["timesteps"] for item in stage_items],
        prompt_embeds,
        histories,
        attention_kwargs={
            "history_visible_token_mode": str(getattr(args, "visible_token_mode", "drop")),
            "history_visible_token_threshold": float(getattr(args, "history_visible_token_threshold", 0.1)),
            "history_confidence_threshold": float(getattr(args, "history_confidence_threshold", 0.0)),
            "history_confidence_lambda": float(getattr(args, "history_confidence_lambda", 1.0)),
            "history_confidence_epsilon": float(getattr(args, "history_confidence_epsilon", 1e-6)),
        },
        target_channel_fusion_latents=None,
        is_first_denoising_step=False,
    )
    if not isinstance(predictions, list) or len(predictions) != len(stage_items):
        raise TypeError("official train-exact Helios forward must return one NaViT prediction per stage")
    stage_losses, stats = [], {}
    for item, prediction in zip(stage_items, predictions):
        target = item["target"].float()
        error = (prediction.float() - target).square()
        temporal = weights.view(1, 1, -1, 1, 1).to(error)
        weighting = compute_loss_weighting_for_sd3(
            weighting_scheme=args.weighting_scheme, sigmas=item["sigmas"]
        ).float()
        weighted = error * weighting * temporal
        denominator = temporal.expand_as(error).sum().clamp_min(1.0)
        stage_loss = weighted.sum() / denominator
        stage_losses.append(stage_loss)
        stage = int(item["stage_id"])
        stats[f"flow_mse_stage{stage}"] = stage_loss.detach()
        stats[f"sigma_stage{stage}"] = item["sigmas"].detach().float().mean()
        stats[f"timestep_stage{stage}"] = item["timesteps"].detach().float().mean()
    total = torch.stack(stage_losses).mean()
    stats["flow_mse"] = total.detach()
    stats["actual_loss_latent_count"] = int((weights > 0).sum().item())
    stats["actual_loss_latent_weight"] = weights.sum().detach()
    return total, stats, stage_items


def save_training_checkpoint(path, transformer, trainable_params, optimizer, scheduler, global_step, adapter_name, metadata):
    import random
    import torch
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    trainable_names = {id(parameter): name for name, parameter in transformer.named_parameters() if parameter.requires_grad}
    lora_state = {
        trainable_names[id(parameter)]: parameter.detach().cpu()
        for parameter in trainable_params
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save({
        "schema_version": 1,
        "adapter_name": adapter_name,
        "global_step": int(global_step),
        "lora_state": lora_state,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "metadata": dict(metadata),
        "rng_state": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        },
    }, temporary)
    temporary.replace(path)
    return path


def load_training_checkpoint(path, transformer, optimizer, scheduler, expected_adapter_name):
    import random
    import torch
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("adapter_name") != expected_adapter_name:
        raise ValueError("checkpoint adapter name does not match current LoRA adapter")
    named = dict(transformer.named_parameters())
    missing = [name for name in payload["lora_state"] if name not in named]
    if missing:
        raise ValueError(f"checkpoint LoRA parameters are missing from transformer: {missing[:5]}")
    with torch.no_grad():
        for name, value in payload["lora_state"].items():
            named[name].copy_(value.to(device=named[name].device, dtype=named[name].dtype))
    optimizer.load_state_dict(payload["optimizer"])
    scheduler.load_state_dict(payload["scheduler"])
    rng = payload.get("rng_state")
    if rng:
        random.setstate(rng["python"]); np.random.set_state(rng["numpy"]); torch.set_rng_state(rng["torch_cpu"])
        if torch.cuda.is_available() and rng.get("torch_cuda"):
            torch.cuda.set_rng_state_all(rng["torch_cuda"])
    return int(payload["global_step"]), dict(payload.get("metadata") or {})


def assert_only_lora_gradients(transformer, trainable_params):
    trainable_ids = {id(parameter) for parameter in trainable_params}
    illegal = [name for name, parameter in transformer.named_parameters()
               if id(parameter) not in trainable_ids and parameter.grad is not None]
    if illegal:
        raise RuntimeError(f"non-LoRA parameters received gradients: {illegal[:10]}")
    if not any(parameter.grad is not None for parameter in trainable_params):
        raise RuntimeError("no LoRA parameter received a gradient")
