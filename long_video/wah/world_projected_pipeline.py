"""Training-free world projection for Warp-as-History pyramid flow sampling.

This module is intentionally isolated from the default WAH pipeline.  It wraps
the scheduler update only while a :class:`WorldProjectionContext` is active and
therefore leaves training and ordinary inference unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as F

try:  # The pinned WAH checkout is added to sys.path by inference entrypoints.
    from warp_as_history import WarpAsHistoryPipeline
except ImportError:  # Keep CPU helper tests importable without the submodule.
    class WarpAsHistoryPipeline:  # type: ignore[no-redef]
        pass


@dataclass(frozen=True)
class WorldProjectionConfig:
    lambda_max: float = 0.5
    gamma: float = 1.0
    confidence_power: float = 1.0
    confidence_threshold: float = 0.3

    def __post_init__(self):
        if not 0.0 <= self.lambda_max <= 1.0:
            raise ValueError("lambda_max must be in [0, 1]")
        if self.gamma < 0.0:
            raise ValueError("gamma must be non-negative")
        if self.confidence_power < 0.0:
            raise ValueError("confidence_power must be non-negative")
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be in [0, 1]")


@dataclass
class WorldProjectionContext:
    canonical_latents: list[torch.Tensor]
    visibility: list[torch.Tensor]
    confidence: list[torch.Tensor]
    config: WorldProjectionConfig = field(default_factory=WorldProjectionConfig)
    diagnostics: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self):
        count = len(self.canonical_latents)
        if count == 0 or len(self.visibility) != count or len(self.confidence) != count:
            raise ValueError("world projection pyramids must have equal non-zero stage counts")
        for stage_id, (latent, visible, confidence) in enumerate(
            zip(self.canonical_latents, self.visibility, self.confidence)
        ):
            if latent.ndim != 5 or visible.ndim != 5 or confidence.ndim != 5:
                raise ValueError(f"stage {stage_id} projection tensors must be 5D")
            expected = (latent.shape[0], 1, latent.shape[2], latent.shape[3], latent.shape[4])
            if tuple(visible.shape) != expected or tuple(confidence.shape) != expected:
                raise ValueError(
                    f"stage {stage_id} support shape mismatch: latent={tuple(latent.shape)} "
                    f"visibility={tuple(visible.shape)} confidence={tuple(confidence.shape)}"
                )


def _resize_video_latents(value: torch.Tensor, height: int, width: int, *, mode: str) -> torch.Tensor:
    batch, channels, frames, source_height, source_width = value.shape
    if (source_height, source_width) == (height, width):
        return value
    flat = value.permute(0, 2, 1, 3, 4).reshape(batch * frames, channels, source_height, source_width)
    kwargs = {"size": (int(height), int(width)), "mode": mode}
    if mode in {"bilinear", "bicubic"}:
        kwargs["align_corners"] = False
    resized = F.interpolate(flat.float(), **kwargs).to(value.dtype)
    return resized.reshape(batch, frames, channels, height, width).permute(0, 2, 1, 3, 4)


def build_canonical_world_pyramid(clean_latent: torch.Tensor, stage_count: int = 3) -> list[torch.Tensor]:
    """Apply the exact Helios pyramid spatial construction to a clean endpoint.

    Helios downsamples with bilinear interpolation and multiplies the latent by
    two at every level.  Returning the list coarse-to-fine makes its indices
    match ``stage2_sample`` stage ids.
    """
    if clean_latent.ndim != 5:
        raise ValueError(f"clean_latent must be [B,C,T,H,W], got {tuple(clean_latent.shape)}")
    if stage_count < 1:
        raise ValueError("stage_count must be positive")
    levels = [clean_latent.detach()]
    current = clean_latent.detach()
    for _ in range(stage_count - 1):
        height, width = current.shape[-2] // 2, current.shape[-1] // 2
        if height < 1 or width < 1:
            raise ValueError("world latent is too small for the requested pyramid")
        current = _resize_video_latents(current, height, width, mode="bilinear") * 2.0
        levels.append(current)
    return list(reversed(levels))


def pixel_support_to_latent(
    value: Any,
    *,
    latent_frames: int,
    latent_height: int,
    latent_width: int,
    temporal_scale: int = 4,
) -> torch.Tensor:
    """Map 33 RGB support frames to the real ``[0],[1..4],...`` VAE groups."""
    tensor = torch.as_tensor(value, dtype=torch.float32)
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0).unsqueeze(0)
    elif tensor.ndim == 4:
        tensor = tensor.unsqueeze(1)
    if tensor.ndim != 5 or tensor.shape[1] != 1:
        raise ValueError(f"pixel support must become [B,1,T,H,W], got {tuple(tensor.shape)}")
    expected_frames = 1 + (int(latent_frames) - 1) * int(temporal_scale)
    if tensor.shape[2] != expected_frames:
        raise ValueError(f"expected {expected_frames} RGB support frames, got {tensor.shape[2]}")
    groups = [(0, 1)] + [
        (1 + index * temporal_scale, 1 + (index + 1) * temporal_scale)
        for index in range(latent_frames - 1)
    ]
    grouped = torch.stack([tensor[:, :, start:end].mean(dim=2) for start, end in groups], dim=2)
    return _resize_video_latents(grouped, latent_height, latent_width, mode="area").clamp(0.0, 1.0)


def build_world_projection_context(
    clean_latent: torch.Tensor,
    visibility: Any,
    confidence: Any,
    *,
    stage_count: int = 3,
    temporal_scale: int = 4,
    config: WorldProjectionConfig | None = None,
) -> WorldProjectionContext:
    canonical = build_canonical_world_pyramid(clean_latent, stage_count)
    full_visibility = pixel_support_to_latent(
        visibility,
        latent_frames=clean_latent.shape[2],
        latent_height=clean_latent.shape[-2],
        latent_width=clean_latent.shape[-1],
        temporal_scale=temporal_scale,
    )
    full_confidence = pixel_support_to_latent(
        confidence,
        latent_frames=clean_latent.shape[2],
        latent_height=clean_latent.shape[-2],
        latent_width=clean_latent.shape[-1],
        temporal_scale=temporal_scale,
    )
    visible_pyramid, confidence_pyramid = [], []
    for endpoint in canonical:
        visible_pyramid.append(_resize_video_latents(
            full_visibility, endpoint.shape[-2], endpoint.shape[-1], mode="area",
        ).clamp(0.0, 1.0))
        confidence_pyramid.append(_resize_video_latents(
            full_confidence, endpoint.shape[-2], endpoint.shape[-1], mode="area",
        ).clamp(0.0, 1.0))
    return WorldProjectionContext(
        canonical_latents=canonical,
        visibility=visible_pyramid,
        confidence=confidence_pyramid,
        config=config or WorldProjectionConfig(),
    )


def build_world_state_at_sigma(
    *,
    stage_id: int,
    current_sigma: float | torch.Tensor,
    next_sigma: float | torch.Tensor,
    canonical_endpoint: torch.Tensor,
    stage_start_state: torch.Tensor,
) -> torch.Tensor:
    """Place the committed world endpoint in the real Helios stage coordinate.

    ``stage_start_state`` is captured from the actual scheduler input.  For
    stages 1/2 it therefore already contains the previous pyramid result after
    nearest upsampling plus Helios' alpha/beta block-noise transition.
    """
    del current_sigma  # Kept in the API so callers cannot confuse the two coordinates.
    if tuple(canonical_endpoint.shape) != tuple(stage_start_state.shape):
        raise ValueError(
            f"stage {stage_id} endpoint/start mismatch: "
            f"{tuple(canonical_endpoint.shape)} vs {tuple(stage_start_state.shape)}"
        )
    sigma = torch.as_tensor(next_sigma, device=stage_start_state.device, dtype=torch.float32).clamp(0.0, 1.0)
    while sigma.ndim < stage_start_state.ndim:
        sigma = sigma.unsqueeze(-1)
    return (
        sigma * stage_start_state.float()
        + (1.0 - sigma) * canonical_endpoint.to(stage_start_state.device).float()
    ).to(stage_start_state.dtype)


def apply_world_projection(
    z_raw: torch.Tensor,
    z_world: torch.Tensor,
    visibility: torch.Tensor,
    confidence: torch.Tensor,
    *,
    sigma: float | torch.Tensor,
    lambda_max: float = 0.5,
    gamma: float = 1.0,
    confidence_power: float = 1.0,
    confidence_threshold: float = 0.3,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Blend a scheduler result toward verified world evidence."""
    if tuple(z_world.shape) != tuple(z_raw.shape):
        raise ValueError(f"z_world shape {tuple(z_world.shape)} != z_raw {tuple(z_raw.shape)}")
    support_shape = (z_raw.shape[0], 1, z_raw.shape[2], z_raw.shape[3], z_raw.shape[4])
    if tuple(visibility.shape) != support_shape or tuple(confidence.shape) != support_shape:
        raise ValueError(
            f"support must be {support_shape}, got {tuple(visibility.shape)}/{tuple(confidence.shape)}"
        )
    sigma_value = torch.as_tensor(sigma, device=z_raw.device, dtype=torch.float32).clamp(0.0, 1.0)
    schedule = float(lambda_max) * (1.0 - sigma_value).pow(float(gamma))
    visible = visibility.to(device=z_raw.device, dtype=torch.float32).clamp(0.0, 1.0)
    confidence_value = confidence.to(device=z_raw.device, dtype=torch.float32).clamp(0.0, 1.0)
    confidence_weight = torch.where(
        confidence_value >= float(confidence_threshold),
        confidence_value.pow(float(confidence_power)),
        torch.zeros_like(confidence_value),
    )
    strength = visible * confidence_weight * schedule
    projected = z_raw.float() + strength * (z_world.float() - z_raw.float())
    projected = projected.to(z_raw.dtype)
    delta = projected.float() - z_raw.float()
    unknown = visible == 0
    unknown_delta_max = (
        delta.masked_select(unknown.expand_as(delta)).abs().max()
        if bool(unknown.any()) else torch.zeros((), device=z_raw.device)
    )
    if float(unknown_delta_max.detach().cpu()) != 0.0:
        raise RuntimeError("world projection changed a V=0 latent element")
    diagnostics = {
        "lambda": schedule.detach(),
        "projection_mask_ratio": (strength > 0).float().mean().detach(),
        "projection_strength_mean": strength.mean().detach(),
        "projection_delta_ratio": (
            delta.norm() / z_raw.float().norm().clamp_min(1e-8)
        ).detach(),
        "unknown_projection_delta_max": unknown_delta_max.detach(),
    }
    return projected, diagnostics


class WorldProjectedWarpAsHistoryPipeline(WarpAsHistoryPipeline):
    """WAH subclass that projects only scheduler outputs, never predictions."""

    def set_world_projection_context(self, context: WorldProjectionContext | None) -> None:
        self._world_projection_context = context

    def clear_world_projection_context(self) -> None:
        self._world_projection_context = None

    def stage2_sample(self, *args, **kwargs):  # noqa: C901 - preserves pinned pipeline semantics.
        context = getattr(self, "_world_projection_context", None)
        if context is None:
            return super().stage2_sample(*args, **kwargs)

        scheduler = self.scheduler
        original_step = scheduler.step
        stage_id = -1
        stage_start = None

        def projected_step(model_output, timestep, sample, *step_args, **step_kwargs):
            nonlocal stage_id, stage_start
            step_index = int(step_kwargs.get("cur_sampling_step", 0))
            if step_index == 0:
                stage_id += 1
                stage_start = sample.detach().clone()
            if stage_id >= len(context.canonical_latents) or stage_start is None:
                raise RuntimeError(f"unexpected Helios pyramid stage {stage_id}")
            result = original_step(model_output, timestep, sample, *step_args, **step_kwargs)
            z_raw = result[0]
            sigmas = step_kwargs.get("dmd_sigmas", getattr(scheduler, "sigmas", None))
            if sigmas is None or len(sigmas) <= step_index:
                raise RuntimeError("Helios scheduler did not expose the active stage sigma coordinate")
            current_sigma = sigmas[step_index]
            next_sigma = sigmas[min(step_index + 1, len(sigmas) - 1)]
            endpoint = context.canonical_latents[stage_id].to(device=z_raw.device, dtype=z_raw.dtype)
            visible = context.visibility[stage_id].to(device=z_raw.device)
            confidence = context.confidence[stage_id].to(device=z_raw.device)
            z_world = build_world_state_at_sigma(
                stage_id=stage_id,
                current_sigma=current_sigma,
                next_sigma=next_sigma,
                canonical_endpoint=endpoint,
                stage_start_state=stage_start,
            )
            config = context.config
            projected, diagnostic = apply_world_projection(
                z_raw, z_world, visible, confidence,
                sigma=next_sigma,
                lambda_max=config.lambda_max,
                gamma=config.gamma,
                confidence_power=config.confidence_power,
                confidence_threshold=config.confidence_threshold,
            )
            context.diagnostics.append({
                "stage_id": int(stage_id),
                "sigma": float(torch.as_tensor(next_sigma).detach().cpu()),
                **{key: float(value.detach().cpu()) for key, value in diagnostic.items()},
            })
            return (projected, *result[1:])

        scheduler.step = projected_step
        try:
            return super().stage2_sample(*args, **kwargs)
        finally:
            scheduler.step = original_step

