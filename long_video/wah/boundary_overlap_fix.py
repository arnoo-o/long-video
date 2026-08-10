"""Runtime fix for chunk-boundary continuity in World-Projected Flow.

The AR chunks overlap by one RGB frame.  The previous implementation copied the
last three clean temporal latents onto the first three latents of the next
chunk at every scheduler step.  Those latent groups are not temporally aligned,
and clean endpoints must not be injected directly into noisy scheduler states.

This fix keeps native WAH history conditioning and adds only a weak, overlap-only
lock at sigma=0.  It uses the previous chunk's final latent group as evidence for
the next chunk's first temporal slot; future slots remain unconstrained.
"""
from __future__ import annotations

import torch

from . import world_projected_pipeline as _wpf


def overlap_only_boundary_projection(
    z_raw: torch.Tensor,
    z_world: torch.Tensor,
    visibility: torch.Tensor,
    confidence: torch.Tensor,
    *,
    previous_boundary: torch.Tensor | None,
    boundary_beta: tuple[float, ...] = (0.6, 0.3, 0.1),
    sigma: float | torch.Tensor,
    lambda_max: float = 0.5,
    gamma: float = 1.0,
    confidence_ramp_min: float = 0.2,
    confidence_ramp_max: float = 0.5,
):
    """Apply WPF plus a clean-coordinate lock on the single overlap slot."""
    if tuple(z_world.shape) != tuple(z_raw.shape):
        raise ValueError("world/raw shape mismatch")

    world_strength, schedule = _wpf.world_projection_weight(
        visibility,
        confidence,
        sigma=sigma,
        lambda_max=lambda_max,
        gamma=gamma,
        confidence_ramp_min=confidence_ramp_min,
        confidence_ramp_max=confidence_ramp_max,
        device=z_raw.device,
    )
    raw = z_raw.float()
    world_delta = world_strength * (z_world.float() - raw)

    boundary_delta = torch.zeros_like(raw)
    beta_tensor = torch.zeros(
        (1, 1, z_raw.shape[2], 1, 1),
        device=z_raw.device,
        dtype=torch.float32,
    )
    sigma_value = float(torch.as_tensor(sigma).detach().cpu())

    # Only the RGB overlap frame is shared by adjacent 33-frame/stride-32 chunks.
    # The final previous temporal group contains that frame; earlier previous
    # groups must never be copied into future temporal slots.  Apply the lock only
    # at sigma=0 so clean endpoint evidence is never mixed into a noisy coordinate.
    if previous_boundary is not None and sigma_value <= 1e-8:
        if previous_boundary.ndim != 5 or previous_boundary.shape[2] < 1:
            raise ValueError("previous boundary must be [B,C,T,H,W] with T>=1")
        expected_spatial = (z_raw.shape[0], z_raw.shape[1], z_raw.shape[3], z_raw.shape[4])
        actual_spatial = (
            previous_boundary.shape[0],
            previous_boundary.shape[1],
            previous_boundary.shape[3],
            previous_boundary.shape[4],
        )
        if actual_spatial != expected_spatial:
            raise ValueError(
                f"previous boundary spatial shape {actual_spatial} != {expected_spatial}"
            )
        overlap_beta = float(boundary_beta[-1]) if boundary_beta else 0.0
        overlap_beta = min(max(overlap_beta, 0.0), 1.0)
        previous_overlap = previous_boundary[:, :, -1:].to(
            device=z_raw.device, dtype=torch.float32,
        )
        beta_tensor[:, :, :1] = overlap_beta
        boundary_delta[:, :, :1] = overlap_beta * (
            previous_overlap - raw[:, :, :1]
        )

    # Boundary evidence has priority only on the shared temporal slot.  WPF uses
    # the remaining interpolation budget there and is unchanged everywhere else.
    combined = (
        raw + boundary_delta + (1.0 - beta_tensor) * world_delta
    ).to(z_raw.dtype)

    unknown = visibility.to(device=z_raw.device) == 0
    unknown_world_delta_max = (
        world_delta.masked_select(unknown.expand_as(world_delta)).abs().max()
        if bool(unknown.any())
        else torch.zeros((), device=z_raw.device)
    )
    if float(unknown_world_delta_max.detach().cpu()) != 0.0:
        raise RuntimeError("world projection changed a V=0 latent element")

    return combined, {
        "lambda": schedule.detach(),
        "projection_mask_ratio": (world_strength > 0).float().mean().detach(),
        "projection_strength_mean": world_strength.mean().detach(),
        "projection_delta_ratio": (
            world_delta.norm() / raw.norm().clamp_min(1e-8)
        ).detach(),
        "unknown_projection_delta_max": unknown_world_delta_max.detach(),
        "boundary_active": torch.as_tensor(
            previous_boundary is not None and sigma_value <= 1e-8,
            device=z_raw.device,
            dtype=torch.float32,
        ),
        "boundary_strength_mean": beta_tensor.mean().detach(),
        "boundary_delta_ratio": (
            boundary_delta.norm() / raw.norm().clamp_min(1e-8)
        ).detach(),
        "combined_delta_ratio": (
            (boundary_delta + (1.0 - beta_tensor) * world_delta).norm()
            / raw.norm().clamp_min(1e-8)
        ).detach(),
    }


def install_boundary_overlap_fix() -> None:
    """Patch only WPF sampling; keep the helper API unchanged for existing tests."""
    cls = _wpf.WorldProjectedWarpAsHistoryPipeline
    if getattr(cls, "_overlap_boundary_fix_installed", False):
        return

    original_stage2_sample = cls.stage2_sample

    def stage2_sample(self, *args, **kwargs):
        original_projection = _wpf.apply_world_and_boundary_projection
        _wpf.apply_world_and_boundary_projection = overlap_only_boundary_projection
        try:
            return original_stage2_sample(self, *args, **kwargs)
        finally:
            _wpf.apply_world_and_boundary_projection = original_projection

    cls.stage2_sample = stage2_sample
    cls._overlap_boundary_fix_installed = True
