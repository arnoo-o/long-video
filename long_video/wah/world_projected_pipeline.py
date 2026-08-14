"""World-Projected Flow for the pinned Warp-as-History pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as F

try:
    from warp_as_history import WarpAsHistoryPipeline
except ImportError:
    WarpAsHistoryPipeline = object


PYRAMID_INFERENCE_STEPS = (2, 2, 2)
WPF_LAMBDAS = ((0.00, 0.05), (0.15, 0.20), (0.25, 0.30))


def posterior_mode_or_mean(posterior: Any) -> torch.Tensor:
    posterior = getattr(posterior, "latent_dist", posterior)
    mode = getattr(posterior, "mode", None)
    if callable(mode):
        try:
            value = mode()
        except (AttributeError, NotImplementedError):
            value = None
        if value is not None:
            return value
    value = getattr(posterior, "mean", None)
    if value is None:
        raise TypeError("VAE posterior must expose mode() or mean")
    return value


def masked_bilinear_fill_per_frame(rgb: torch.Tensor, visibility: torch.Tensor) -> torch.Tensor:
    """Fill each frame spatially without changing any renderer-valid pixel."""
    if rgb.ndim != 5 or rgb.shape[1] != 3:
        raise ValueError(f"RGB must be [B,3,T,H,W], got {tuple(rgb.shape)}")
    expected = (rgb.shape[0], 1, *rgb.shape[2:])
    if tuple(visibility.shape) != expected:
        raise ValueError(f"visibility must be {expected}, got {tuple(visibility.shape)}")
    valid = (visibility > 0).to(device=rgb.device, dtype=rgb.dtype)
    b, _, t, h, w = rgb.shape
    weighted = (rgb * valid).permute(0, 2, 1, 3, 4).reshape(b * t, 3, h, w)
    weights = valid.permute(0, 2, 1, 3, 4).reshape(b * t, 1, h, w)

    pyramid = [(weighted, weights)]
    while pyramid[-1][0].shape[-2:] != (1, 1):
        values, support = pyramid[-1]
        next_size = (max(1, (values.shape[-2] + 1) // 2), max(1, (values.shape[-1] + 1) // 2))
        pyramid.append((
            F.interpolate(values, size=next_size, mode="area"),
            F.interpolate(support, size=next_size, mode="area"),
        ))

    values, support = pyramid[-1]
    filled = values / support.clamp_min(1e-6)
    propagated = (support > 0).to(filled.dtype)
    for values, support in reversed(pyramid[:-1]):
        filled = F.interpolate(filled, size=values.shape[-2:], mode="bilinear", align_corners=False)
        propagated = F.interpolate(
            propagated, size=values.shape[-2:], mode="bilinear", align_corners=False,
        )
        local_valid = (support > 0).to(filled.dtype)
        local = values / support.clamp_min(1e-6)
        filled = local_valid * local + (1 - local_valid) * filled
        propagated = torch.maximum(local_valid, propagated)

    # Empty frames use a stable zero fallback. Their visibility remains zero,
    # so this placeholder can never create a WPF update.
    filled = torch.where(propagated > 0, filled, torch.zeros_like(filled))
    filled = filled.reshape(b, t, 3, h, w).permute(0, 2, 1, 3, 4)
    return torch.where(valid.bool(), rgb, filled)


def _resize_video(value: torch.Tensor, height: int, width: int, mode: str) -> torch.Tensor:
    b, c, t, h, w = value.shape
    if (h, w) == (height, width):
        return value
    flat = value.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
    kwargs = {"size": (height, width), "mode": mode}
    if mode in {"bilinear", "bicubic"}:
        kwargs["align_corners"] = False
    flat = F.interpolate(flat.float(), **kwargs).to(value.dtype)
    return flat.reshape(b, t, c, height, width).permute(0, 2, 1, 3, 4)


def build_world_pyramid(clean: torch.Tensor, stage_count: int = 3) -> list[torch.Tensor]:
    levels = [clean.detach()]
    current = clean.detach()
    for _ in range(stage_count - 1):
        current = _resize_video(current, current.shape[-2] // 2, current.shape[-1] // 2, "bilinear") * 2
        levels.append(current)
    return list(reversed(levels))


def pixel_support_to_latent(value: Any, clean: torch.Tensor) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=torch.float32)
    if tensor.ndim == 3:
        tensor = tensor[None, None]
    elif tensor.ndim == 4:
        tensor = tensor[:, None]
    if tensor.shape[2] != 33 or clean.shape[2] != 9:
        raise ValueError("WPF expects the Helios 33-frame / 9-latent layout")
    groups = [(0, 1)] + [(1 + 4 * index, 5 + 4 * index) for index in range(8)]
    grouped = torch.stack([tensor[:, :, start:end].mean(2) for start, end in groups], dim=2)
    return _resize_video(grouped, clean.shape[-2], clean.shape[-1], "area").clamp(0, 1)


@dataclass
class WorldProjectionContext:
    endpoints: list[torch.Tensor]
    visibility: list[torch.Tensor]
    confidence: list[torch.Tensor]
    raw_visibility: torch.Tensor | None = None
    diagnostics: list[dict[str, Any]] = field(default_factory=list)


class WorldProjectedWarpAsHistoryPipeline(WarpAsHistoryPipeline):
    """Apply fixed-strength WPF after every original Helios scheduler step."""

    def _wpf_device(self):
        helper = getattr(self, "_wah_execution_device", None)
        return helper() if callable(helper) else self._execution_device

    def _latent_stats(self, device):
        dtype = self.vae.dtype
        mean = torch.tensor(self.vae.config.latents_mean, device=device, dtype=dtype).view(1, -1, 1, 1, 1)
        std = 1 / torch.tensor(self.vae.config.latents_std, device=device, dtype=dtype).view(1, -1, 1, 1, 1)
        return mean, std

    def set_world_projection_from_renderer(
        self, warp_rgb: Any, visibility: Any, confidence: Any, *, height: int, width: int,
    ) -> None:
        device = self._wpf_device()
        rgb = self._coerce_warp_video_tensor(
            warp_rgb, height=height, width=width, device=device,
        ).to(device=device, dtype=self.vae.dtype)
        mask = torch.as_tensor(visibility, device=device)
        if mask.ndim == 3:
            mask = mask[None, None]
        elif mask.ndim == 4:
            mask = mask[:, None]
        filled = masked_bilinear_fill_per_frame(rgb, mask)
        mean, std = self._latent_stats(device)
        with torch.no_grad():
            clean = (posterior_mode_or_mean(self.vae.encode(filled)) - mean) * std
        endpoints = build_world_pyramid(clean, len(PYRAMID_INFERENCE_STEPS))
        full_visibility = pixel_support_to_latent(visibility, clean)
        full_confidence = pixel_support_to_latent(confidence, clean)
        visibility_pyramid = [
            _resize_video(full_visibility, endpoint.shape[-2], endpoint.shape[-1], "area")
            for endpoint in endpoints
        ]
        confidence_pyramid = [
            _resize_video(full_confidence, endpoint.shape[-2], endpoint.shape[-1], "area")
            for endpoint in endpoints
        ]
        self._world_projection_context = WorldProjectionContext(
            endpoints=endpoints, visibility=visibility_pyramid, confidence=confidence_pyramid,
            raw_visibility=mask.detach(),
        )

    def clear_world_projection_context(self) -> None:
        self._world_projection_context = None

    @staticmethod
    def _confidence_weight(visibility: torch.Tensor, confidence: torch.Tensor) -> torch.Tensor:
        ramp = ((confidence.float() - 0.2) / 0.3).clamp(0, 1)
        return visibility.float().clamp(0, 1) * ramp

    def stage2_sample(self, *args, **kwargs):
        context = getattr(self, "_world_projection_context", None)
        if context is None:
            return super().stage2_sample(*args, **kwargs)
        scheduler, original_step = self.scheduler, self.scheduler.step
        stage_id, stage_start = -1, None

        def projected_step(model_output, timestep, sample, *step_args, **step_kwargs):
            nonlocal stage_id, stage_start
            step_id = int(step_kwargs.get("cur_sampling_step", 0))
            if step_id == 0:
                stage_id += 1
                stage_start = sample.detach().clone()
            observer = getattr(self, "_pyramid_training_observer", None)
            if observer is not None:
                observed = observer({
                    "stage_id": stage_id,
                    "step_id": step_id,
                    "model_output": model_output,
                    "base_model_output": getattr(self, "_pyramid_base_model_output", None),
                    "sample": sample,
                    "timestep": timestep,
                    "dmd_sigmas": step_kwargs.get("dmd_sigmas", getattr(scheduler, "sigmas", None)),
                    "dmd_timesteps": step_kwargs.get(
                        "dmd_timesteps", getattr(scheduler, "timesteps", None),
                    ),
                    "point_visibility": context.raw_visibility,
                })
                if observed is not None:
                    model_output, sample = observed
            result = original_step(model_output, timestep, sample, *step_args, **step_kwargs)
            z_raw = result[0]
            sigmas = step_kwargs.get("dmd_sigmas", getattr(scheduler, "sigmas", None))
            if sigmas is None or step_id + 1 >= len(sigmas):
                raise RuntimeError("Helios scheduler did not expose next_sigma")
            next_sigma = torch.as_tensor(sigmas[step_id + 1], device=z_raw.device, dtype=torch.float32)
            endpoint = context.endpoints[stage_id].to(device=z_raw.device, dtype=z_raw.dtype)
            if endpoint.shape != stage_start.shape:
                raise RuntimeError(f"Stage{stage_id} endpoint/start shape mismatch")
            # This follows the requested WPF trajectory coordinate exactly.
            z_world = next_sigma * stage_start.float() - (1 - next_sigma) * endpoint.float()
            visible = context.visibility[stage_id].to(z_raw.device)
            confidence = context.confidence[stage_id].to(z_raw.device)
            strength = float(WPF_LAMBDAS[stage_id][step_id]) * self._confidence_weight(visible, confidence)
            projected = z_raw.float() + strength * (z_world - z_raw.float())
            delta = projected - z_raw.float()
            unknown = visible == 0
            unknown_delta = delta.masked_select(unknown.expand_as(delta))
            unknown_delta_max = unknown_delta.abs().max() if unknown_delta.numel() else delta.new_zeros(())
            if float(unknown_delta_max.detach().cpu()) != 0.0:
                raise RuntimeError("WPF changed a visibility=0 latent element")
            context.diagnostics.append({
                "stage_id": stage_id,
                "step_id": step_id,
                "lambda": float(WPF_LAMBDAS[stage_id][step_id]),
                "next_sigma": float(next_sigma.detach().cpu()),
                "strength_mean": float(strength.mean().detach().cpu()),
                "unknown_delta_max": float(unknown_delta_max.detach().cpu()),
            })
            return (projected.to(z_raw.dtype), *result[1:])

        scheduler.step = projected_step
        try:
            sampled = super().stage2_sample(*args, **kwargs)
            expected = [(stage, step) for stage in range(3) for step in range(2)]
            actual = [(item["stage_id"], item["step_id"]) for item in context.diagnostics]
            if actual != expected:
                raise RuntimeError(f"unexpected pyramid scheduler steps: {actual}")
            return sampled
        finally:
            scheduler.step = original_step
