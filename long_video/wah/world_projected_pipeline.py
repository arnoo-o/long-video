"""Training-free world projection for Warp-as-History pyramid flow sampling.

This module is intentionally isolated from the default WAH pipeline.  It wraps
the scheduler update only while a :class:`WorldProjectionContext` is active and
therefore leaves training and ordinary inference unchanged.
"""
from __future__ import annotations

from collections import deque
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F

try:  # The pinned WAH checkout is added to sys.path by inference entrypoints.
    from warp_as_history import WarpAsHistoryPipeline
except ImportError:  # Keep CPU helper tests importable without the submodule.
    class WarpAsHistoryPipeline:  # type: ignore[no-redef]
        pass


# Helios uses nine temporal latent slots for the 33-frame WAH chunks.  The
# first three latent slots are deliberately warmed up at chunk start; later
# slots are trusted immediately.  Keep this schedule separate from the
# renderer support masks: it is applied only to the final WPF strength.
DEFAULT_TEMPORAL_WARMUP: tuple[float, ...] = (
    0.0, 0.25, 0.60, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
)
WAH_VISIBLE_TOKEN_THRESHOLD = 0.1
WORLD_OWNERSHIP_COVERAGE_THRESHOLD = 0.9


@dataclass(frozen=True)
class WorldProjectionConfig:
    lambda_max_by_stage: tuple[float, ...] = (0.0, 0.15, 0.30)
    boundary_beta_max_by_stage: tuple[float, ...] = (0.0, 0.10, 0.20)
    gamma: float = 1.0
    confidence_ramp_min: float = 0.2
    confidence_ramp_max: float = 0.5
    temporal_warmup: tuple[float, ...] = DEFAULT_TEMPORAL_WARMUP

    def __post_init__(self):
        if not self.lambda_max_by_stage:
            raise ValueError("lambda_max_by_stage must not be empty")
        if any(not 0.0 <= value <= 1.0 for value in self.lambda_max_by_stage):
            raise ValueError("every stage lambda_max must be in [0, 1]")
        if len(self.boundary_beta_max_by_stage) != len(self.lambda_max_by_stage):
            raise ValueError("boundary beta stages must match world projection stages")
        if any(not 0.0 <= value <= 1.0 for value in self.boundary_beta_max_by_stage):
            raise ValueError("every stage boundary beta_max must be in [0, 1]")
        if self.gamma < 0.0:
            raise ValueError("gamma must be non-negative")
        if not 0.0 <= self.confidence_ramp_min < self.confidence_ramp_max <= 1.0:
            raise ValueError("confidence ramp must satisfy 0 <= min < max <= 1")
        if not self.temporal_warmup:
            raise ValueError("temporal_warmup must not be empty")
        if any(not 0.0 <= value <= 1.0 for value in self.temporal_warmup):
            raise ValueError("every temporal warmup value must be in [0, 1]")

    def lambda_max(self, stage_id: int) -> float:
        if not 0 <= int(stage_id) < len(self.lambda_max_by_stage):
            raise ValueError(f"no projection lambda configured for stage {stage_id}")
        return float(self.lambda_max_by_stage[int(stage_id)])

    def boundary_beta_max(self, stage_id: int) -> float:
        if not 0 <= int(stage_id) < len(self.boundary_beta_max_by_stage):
            raise ValueError(f"no boundary beta configured for stage {stage_id}")
        return float(self.boundary_beta_max_by_stage[int(stage_id)])

    @property
    def temporal_warmup_by_slot(self) -> tuple[float, ...]:
        """Compatibility alias for callers that describe the schedule by slots."""
        return tuple(float(value) for value in self.temporal_warmup)


@dataclass(frozen=True)
class CanonicalWorldSupport:
    """Canonical coverage plus the binary ownership mask shared by WAH/clamp."""

    visibility: torch.Tensor
    confidence: torch.Tensor
    safe_support: torch.Tensor

    def __post_init__(self):
        expected = tuple(self.visibility.shape)
        if len(expected) != 5 or expected[1] != 1:
            raise ValueError(f"canonical visibility must be [B,1,T,H,W], got {expected}")
        if tuple(self.confidence.shape) != expected or tuple(self.safe_support.shape) != expected:
            raise ValueError("canonical visibility/confidence/safe support shapes must match")
        for name, value in (
            ("visibility", self.visibility),
            ("confidence", self.confidence),
            ("safe_support", self.safe_support),
        ):
            if not bool(torch.isfinite(value).all()) or bool((value < 0).any()) or bool((value > 1).any()):
                raise ValueError(f"canonical {name} must be finite and in [0,1]")
        if not bool(((self.safe_support == 0) | (self.safe_support == 1)).all()):
            raise ValueError("canonical world ownership must contain only binary 0/1 values")

    @property
    def world_ownership_mask(self) -> torch.Tensor:
        return self.safe_support


@dataclass(frozen=True)
class ScheduledNodeActivation:
    node: Any
    created_after_chunk: int
    activate_at_chunk: int

    @property
    def node_id(self) -> str:
        return str(self.node.node_id)


class DelayedNodeActivationQueue:
    """A one-entry FIFO whose schedule cannot be overwritten by newer nodes."""

    def __init__(self, delay_chunks: int = 2, max_pending: int = 1):
        if int(delay_chunks) not in (1, 2):
            raise ValueError("Validated Causal World activation delay must be one or two chunks")
        if int(max_pending) != 1:
            raise ValueError("this experiment permits exactly one pending accepted node")
        self.delay_chunks = int(delay_chunks)
        self.max_pending = int(max_pending)
        self._queue: deque[ScheduledNodeActivation] = deque()

    def __len__(self) -> int:
        return len(self._queue)

    @property
    def pending(self) -> ScheduledNodeActivation | None:
        return self._queue[0] if self._queue else None

    def schedule(self, node: Any, *, created_after_chunk: int) -> ScheduledNodeActivation:
        if self._queue:
            raise RuntimeError(
                f"cannot replace pending activation {self._queue[0].node_id} with {node.node_id}"
            )
        entry = ScheduledNodeActivation(
            node=node,
            created_after_chunk=int(created_after_chunk),
            activate_at_chunk=int(created_after_chunk) + self.delay_chunks,
        )
        self._queue.append(entry)
        return entry

    def activate_due(self, chunk_index: int) -> ScheduledNodeActivation | None:
        if not self._queue:
            return None
        entry = self._queue[0]
        if int(chunk_index) < entry.activate_at_chunk:
            return None
        if int(chunk_index) > entry.activate_at_chunk:
            raise RuntimeError(
                f"missed activation of {entry.node_id} at chunk {entry.activate_at_chunk}"
            )
        return self._queue.popleft()


@dataclass
class WorldProjectionContext:
    canonical_latents: list[torch.Tensor]
    visibility: list[torch.Tensor]
    confidence: list[torch.Tensor]
    config: WorldProjectionConfig = field(default_factory=WorldProjectionConfig)
    previous_boundary_latents: list[torch.Tensor] | None = None
    diagnostics: list[dict[str, Any]] = field(default_factory=list)
    final_projection_weight: torch.Tensor | None = None
    final_residual_diagnostics: dict[str, Any] | None = None
    pixel_warp_rgb: torch.Tensor | None = None
    pixel_visibility: torch.Tensor | None = None

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
            if not bool(((visible == 0) | (visible == 1)).all()):
                raise ValueError(f"stage {stage_id} world ownership must be binary")
        if self.previous_boundary_latents is not None:
            if len(self.previous_boundary_latents) != count:
                raise ValueError("boundary pyramid must match world stage count")
            for stage_id, (world, boundary) in enumerate(zip(
                self.canonical_latents, self.previous_boundary_latents,
            )):
                expected = (
                    world.shape[0], world.shape[1], 1,
                    world.shape[3], world.shape[4],
                )
                if tuple(boundary.shape) != expected:
                    raise ValueError(
                        f"stage {stage_id} boundary shape {tuple(boundary.shape)} != {expected}"
                    )
        if (self.pixel_warp_rgb is None) != (self.pixel_visibility is None):
            raise ValueError("sparse pixel constraint requires both warp RGB and visibility")
        if self.pixel_warp_rgb is not None:
            expected = (
                self.pixel_warp_rgb.shape[0], 1, self.pixel_warp_rgb.shape[2],
                self.pixel_warp_rgb.shape[3], self.pixel_warp_rgb.shape[4],
            )
            if self.pixel_warp_rgb.ndim != 5 or tuple(self.pixel_visibility.shape) != expected:
                raise ValueError("sparse constraint tensors must be aligned [B,C/T,H,W]")
            if not bool(torch.isfinite(self.pixel_warp_rgb).all()):
                raise ValueError("pixel warp RGB must be finite")
            if not bool(((self.pixel_visibility == 0) | (self.pixel_visibility == 1)).all()):
                raise ValueError("renderer pixel visibility must be binary")


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


def fill_invalid_warp_for_vae(rgb: Any, visibility: Any) -> np.ndarray:
    """Nearest-fill invalid renderer pixels only for canonical VAE encoding."""
    from scipy.ndimage import distance_transform_edt

    value = np.asarray(rgb)
    visible = np.asarray(visibility, bool)
    if value.ndim != 4 or value.shape[-1] != 3 or visible.shape != value.shape[:3]:
        raise ValueError(f"expected RGB [T,H,W,3] and visibility [T,H,W], got {value.shape}/{visible.shape}")
    result = value.copy()
    if visible.any():
        fallback = value[visible].reshape(-1, 3).mean(axis=0)
    else:
        fallback = np.full((3,), 0.5 if np.issubdtype(value.dtype, np.floating) else 128.0)
    for frame_index, mask in enumerate(visible):
        if mask.all():
            continue
        if not mask.any():
            result[frame_index] = fallback
            continue
        indices = distance_transform_edt(
            ~mask, return_distances=False, return_indices=True,
        )
        filled = result[frame_index]
        filled[~mask] = value[frame_index][indices[0, ~mask], indices[1, ~mask]]
    if np.issubdtype(value.dtype, np.integer):
        result = np.rint(result).astype(value.dtype)
    return result


def posterior_mode_or_mean(posterior: Any) -> torch.Tensor:
    """Return a deterministic VAE posterior representative.

    The normal Helios path intentionally samples from VAE posteriors.  World
    projection conditioning is different: its canonical endpoint must be
    repeatable and must not advance the generation ``torch.Generator``.  A
    few VAE implementations expose ``mode()`` while light-weight test doubles
    only provide ``mean``; support both without ever calling ``sample``.
    """
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
        raise TypeError("VAE posterior must expose mode() or mean for deterministic encoding")
    return value


def _encode_vae_deterministic(
    pipe: Any,
    video: torch.Tensor,
    *,
    latents_mean: torch.Tensor,
    latents_std: torch.Tensor,
) -> torch.Tensor:
    """Encode one video tensor with posterior mode/mean and WAH statistics."""
    encoded_output = pipe.vae.encode(video)
    posterior = getattr(encoded_output, "latent_dist", encoded_output)
    encoded = posterior_mode_or_mean(posterior)
    return (encoded - latents_mean) * latents_std


def encode_canonical_video_latents(
    pipe: Any,
    video: torch.Tensor,
    *,
    latents_mean: torch.Tensor,
    latents_std: torch.Tensor,
    num_latent_frames_per_chunk: int,
    dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode canonical WAH video latents deterministically.

    This mirrors Helios' ``prepare_video_latents`` temporal grouping exactly,
    but intentionally omits its posterior ``sample(generator=...)`` calls.
    It is used only for the shared canonical conditioning path; ordinary
    Helios generation continues to use the pinned stochastic implementation.
    """
    device = device or getattr(pipe, "_execution_device", None)
    if device is None:
        device = video.device
    vae = pipe.vae
    video = video.to(device=device, dtype=getattr(vae, "dtype", video.dtype))
    num_frames = int(video.shape[2])
    temporal_scale = int(getattr(pipe, "vae_scale_factor_temporal", 4))
    min_frames = (int(num_latent_frames_per_chunk) - 1) * temporal_scale + 1
    num_chunks = num_frames // min_frames
    if num_chunks == 0:
        raise ValueError(
            f"Video must have at least {min_frames} frames (got {num_frames} frames)."
        )
    total_valid_frames = num_chunks * min_frames
    start_frame = num_frames - total_valid_frames

    first_frame = video[:, :, 0:1]
    first_frame_latent = _encode_vae_deterministic(
        pipe, first_frame, latents_mean=latents_mean, latents_std=latents_std,
    )
    latents_chunks = []
    for index in range(num_chunks):
        chunk_start = start_frame + index * min_frames
        chunk_end = chunk_start + min_frames
        video_chunk = video[:, :, chunk_start:chunk_end]
        latents_chunks.append(_encode_vae_deterministic(
            pipe, video_chunk, latents_mean=latents_mean, latents_std=latents_std,
        ))
    latents = torch.cat(latents_chunks, dim=2)
    if dtype is not None:
        first_frame_latent = first_frame_latent.to(device=device, dtype=dtype)
        latents = latents.to(device=device, dtype=dtype)
    else:
        first_frame_latent = first_frame_latent.to(device=device)
        latents = latents.to(device=device)
    return first_frame_latent, latents


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


def smooth_latent_visibility(value: torch.Tensor) -> torch.Tensor:
    """Erode and soften spatial support without leaking into true unknowns."""
    if value.ndim != 5 or value.shape[1] != 1:
        raise ValueError(f"visibility must be [B,1,T,H,W], got {tuple(value.shape)}")
    batch, _, frames, height, width = value.shape
    flat = value.float().permute(0, 2, 1, 3, 4).reshape(batch * frames, 1, height, width)
    original = flat.clamp(0.0, 1.0)
    binary = (original > 0.0).float()
    inverse = F.pad(1.0 - binary, (1, 1, 1, 1), mode="constant", value=1.0)
    eroded = 1.0 - F.max_pool2d(inverse, kernel_size=3, stride=1)
    softened = F.avg_pool2d(eroded, kernel_size=3, stride=1, padding=1)
    # Multiplying by original support makes exact unknowns stay exact zero and
    # retains fractional temporal/spatial coverage produced by VAE grouping.
    softened = (softened * original).clamp(0.0, 1.0)
    return softened.reshape(batch, frames, 1, height, width).permute(0, 2, 1, 3, 4)


def build_canonical_world_support(
    visibility: Any,
    confidence: Any,
    *,
    latent_frames: int = 9,
    latent_height: int = 48,
    latent_width: int = 80,
    temporal_scale: int = 4,
    visible_threshold: float = WORLD_OWNERSHIP_COVERAGE_THRESHOLD,
) -> CanonicalWorldSupport:
    """Build raw coverage and strict binary ownership from renderer visibility."""
    visibility_value = torch.as_tensor(visibility, dtype=torch.float32)
    confidence_value = torch.as_tensor(confidence, dtype=torch.float32)
    if tuple(confidence_value.shape) != tuple(visibility_value.shape):
        raise ValueError("pixel visibility and confidence shapes must match")
    visibility_value = visibility_value.clamp(0.0, 1.0)
    confidence_value = confidence_value.clamp(0.0, 1.0)
    canonical_visibility = pixel_support_to_latent(
        visibility_value,
        latent_frames=latent_frames,
        latent_height=latent_height,
        latent_width=latent_width,
        temporal_scale=temporal_scale,
    )
    weighted_confidence = pixel_support_to_latent(
        confidence_value * visibility_value,
        latent_frames=latent_frames,
        latent_height=latent_height,
        latent_width=latent_width,
        temporal_scale=temporal_scale,
    )
    canonical_confidence = torch.where(
        canonical_visibility > 0,
        weighted_confidence / canonical_visibility.clamp_min(1e-6),
        torch.zeros_like(weighted_confidence),
    ).clamp(0.0, 1.0)
    canonical_safe_support = (
        canonical_visibility >= float(visible_threshold)
    ).to(canonical_visibility.dtype)
    canonical_confidence = canonical_confidence * canonical_safe_support.to(
        canonical_confidence.dtype
    )
    return CanonicalWorldSupport(
        visibility=canonical_visibility,
        confidence=canonical_confidence,
        safe_support=canonical_safe_support,
    )


def build_single_frame_world_support(
    visibility: Any,
    confidence: Any,
    *,
    latent_height: int = 48,
    latent_width: int = 80,
    visible_threshold: float = WORLD_OWNERSHIP_COVERAGE_THRESHOLD,
) -> CanonicalWorldSupport:
    """Build latent support for one explicitly encoded shared boundary RGB frame."""
    visibility_value = torch.as_tensor(visibility, dtype=torch.float32)
    confidence_value = torch.as_tensor(confidence, dtype=torch.float32)
    if visibility_value.ndim == 2:
        visibility_value = visibility_value[None, None, None]
        confidence_value = confidence_value[None, None, None]
    elif visibility_value.ndim == 3:
        visibility_value = visibility_value[:, None, None]
        confidence_value = confidence_value[:, None, None]
    if visibility_value.ndim != 5 or tuple(confidence_value.shape) != tuple(visibility_value.shape):
        raise ValueError("single-frame support must become matching [B,1,1,H,W] tensors")
    # Preserve the original tensors so confidence can be averaged only over
    # actually visible pixels before spatial downsampling.
    original_visibility = torch.as_tensor(visibility, dtype=torch.float32)
    original_confidence = torch.as_tensor(confidence, dtype=torch.float32)
    if original_visibility.ndim == 2:
        original_visibility = original_visibility[None, None, None]
        original_confidence = original_confidence[None, None, None]
    elif original_visibility.ndim == 3:
        original_visibility = original_visibility[:, None, None]
        original_confidence = original_confidence[:, None, None]
    visibility_value = _resize_video_latents(
        original_visibility.clamp(0.0, 1.0), latent_height, latent_width, mode="area",
    )
    weighted = _resize_video_latents(
        original_visibility.clamp(0.0, 1.0) * original_confidence.clamp(0.0, 1.0),
        latent_height, latent_width, mode="area",
    )
    canonical_confidence = torch.where(
        visibility_value > 0,
        weighted / visibility_value.clamp_min(1e-6),
        torch.zeros_like(weighted),
    ).clamp(0.0, 1.0)
    safe = (visibility_value >= float(visible_threshold)).to(visibility_value.dtype)
    canonical_confidence = canonical_confidence * safe.to(canonical_confidence.dtype)
    return CanonicalWorldSupport(visibility_value, canonical_confidence, safe)


def mask_canonical_latent(
    latent: torch.Tensor,
    safe_support: torch.Tensor,
) -> torch.Tensor:
    """Hard-zero every VAE placeholder latent outside binary world ownership."""
    expected = (latent.shape[0], 1, latent.shape[2], latent.shape[3], latent.shape[4])
    if tuple(safe_support.shape) != expected:
        raise ValueError(f"safe support {tuple(safe_support.shape)} != {expected}")
    return latent * (safe_support.to(device=latent.device) > 0).to(latent.dtype)


def apply_previous_world_boundary(
    canonical_latent: torch.Tensor,
    first_frame_latent: torch.Tensor,
    canonical_support: CanonicalWorldSupport,
    *,
    previous_latent: torch.Tensor | None,
    previous_visibility: torch.Tensor | None,
    previous_confidence: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, CanonicalWorldSupport, bool]:
    """Make the current slot0 exactly equal to the previous rendered world boundary."""
    applied = previous_latent is not None
    latent = canonical_latent.clone()
    first = first_frame_latent.clone()
    visibility = canonical_support.visibility.clone()
    safe_support = canonical_support.safe_support.clone()
    confidence = canonical_support.confidence.clone()
    if applied:
        if previous_visibility is None or previous_confidence is None:
            raise ValueError("previous world boundary latent/support state is incomplete")
        expected_latent = tuple(latent[:, :, 0:1].shape)
        expected_support = tuple(safe_support[:, :, 0:1].shape)
        if tuple(previous_latent.shape) != expected_latent:
            raise ValueError(f"previous boundary latent {tuple(previous_latent.shape)} != {expected_latent}")
        if tuple(previous_visibility.shape) != expected_support:
            raise ValueError("previous boundary visibility shape mismatch")
        if tuple(previous_confidence.shape) != expected_support:
            raise ValueError("previous boundary confidence shape mismatch")
        previous_latent = previous_latent.to(device=latent.device, dtype=latent.dtype)
        previous_visibility = previous_visibility.to(
            device=safe_support.device, dtype=safe_support.dtype,
        )
        previous_confidence = previous_confidence.to(
            device=confidence.device, dtype=confidence.dtype,
        )
        latent[:, :, 0:1] = previous_latent
        first = previous_latent.clone()
        visibility[:, :, 0:1] = previous_visibility
        safe_support[:, :, 0:1] = previous_visibility
        confidence[:, :, 0:1] = previous_confidence
    support = CanonicalWorldSupport(visibility, confidence, safe_support)
    latent = mask_canonical_latent(latent, safe_support.to(latent.device))
    first = mask_canonical_latent(first, safe_support[:, :, 0:1].to(first.device))
    return latent, first, support, applied


def canonical_support_to_tokens(
    support: torch.Tensor,
    *,
    latent_height: int,
    latent_width: int,
    patch_height: int,
    patch_width: int,
) -> torch.Tensor:
    """Resize an already-grouped canonical support and pool only spatial patches."""
    if support.ndim != 5 or support.shape[1] != 1:
        raise ValueError(f"canonical support must be [B,1,T,H,W], got {tuple(support.shape)}")
    stage = _resize_video_latents(
        support, int(latent_height), int(latent_width), mode="area",
    ).clamp(0.0, 1.0)
    batch, _, frames, height, width = stage.shape
    if height % int(patch_height) or width % int(patch_width):
        raise ValueError("stage support must be divisible by the spatial patch size")
    flat = stage.permute(0, 2, 1, 3, 4).reshape(batch * frames, 1, height, width)
    pooled = F.avg_pool2d(
        flat,
        kernel_size=(int(patch_height), int(patch_width)),
        stride=(int(patch_height), int(patch_width)),
    )
    pooled = pooled.reshape(batch, frames, pooled.shape[-2], pooled.shape[-1], 1)
    return pooled.reshape(batch, -1, 1).clamp(0.0, 1.0)


def build_world_projection_context(
    clean_latent: torch.Tensor,
    visibility: Any,
    confidence: Any,
    *,
    stage_count: int = 3,
    temporal_scale: int = 4,
    config: WorldProjectionConfig | None = None,
    previous_clean_boundary_latent: torch.Tensor | None = None,
    canonical_support: CanonicalWorldSupport | None = None,
) -> WorldProjectionContext:
    canonical = build_canonical_world_pyramid(clean_latent, stage_count)
    if canonical_support is None:
        canonical_support = build_canonical_world_support(
            visibility,
            confidence,
            latent_frames=clean_latent.shape[2],
            latent_height=clean_latent.shape[-2],
            latent_width=clean_latent.shape[-1],
            temporal_scale=temporal_scale,
        )
    full_visibility_coverage = canonical_support.visibility
    full_confidence = canonical_support.confidence
    expected_support = (
        clean_latent.shape[0], 1, clean_latent.shape[2],
        clean_latent.shape[3], clean_latent.shape[4],
    )
    if tuple(full_visibility_coverage.shape) != expected_support or tuple(full_confidence.shape) != expected_support:
        raise ValueError(
            f"canonical support must align to clean latent {expected_support}, got "
            f"{tuple(full_visibility_coverage.shape)}/{tuple(full_confidence.shape)}"
        )
    visible_pyramid, confidence_pyramid = [], []
    for endpoint in canonical:
        stage_coverage = _resize_video_latents(
            full_visibility_coverage, endpoint.shape[-2], endpoint.shape[-1], mode="area",
        ).clamp(0.0, 1.0)
        stage_ownership = (
            stage_coverage >= WORLD_OWNERSHIP_COVERAGE_THRESHOLD
        ).to(stage_coverage.dtype)
        visible_pyramid.append(stage_ownership)
        stage_confidence = _resize_video_latents(
            full_confidence, endpoint.shape[-2], endpoint.shape[-1], mode="area",
        ).clamp(0.0, 1.0)
        confidence_pyramid.append(stage_confidence * stage_ownership)
    boundary = None
    if previous_clean_boundary_latent is not None:
        previous = previous_clean_boundary_latent.detach()
        if previous.ndim != 5 or previous.shape[2] != 1:
            raise ValueError(
                "previous clean boundary latent must be [B,C,1,H,W], "
                f"got {tuple(previous.shape)}"
            )
        boundary = build_canonical_world_pyramid(previous, stage_count)
    return WorldProjectionContext(
        canonical_latents=canonical,
        visibility=visible_pyramid,
        confidence=confidence_pyramid,
        config=config or WorldProjectionConfig(),
        previous_boundary_latents=boundary,
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


def build_boundary_state_at_sigma(
    *,
    next_sigma: float | torch.Tensor,
    clean_boundary_endpoint: torch.Tensor,
    stage_start_state: torch.Tensor,
) -> torch.Tensor:
    """Put the shared-frame endpoint in the active stage scheduler coordinate."""
    expected = (
        stage_start_state.shape[0], stage_start_state.shape[1], 1,
        stage_start_state.shape[3], stage_start_state.shape[4],
    )
    if tuple(clean_boundary_endpoint.shape) != expected:
        raise ValueError(
            f"boundary endpoint {tuple(clean_boundary_endpoint.shape)} != {expected}"
        )
    sigma = torch.as_tensor(
        next_sigma, device=stage_start_state.device, dtype=torch.float32,
    ).clamp(0.0, 1.0)
    while sigma.ndim < stage_start_state.ndim:
        sigma = sigma.unsqueeze(-1)
    start_slot = stage_start_state[:, :, 0:1].float()
    endpoint = clean_boundary_endpoint.to(stage_start_state.device).float()
    return (sigma * start_slot + (1.0 - sigma) * endpoint).to(stage_start_state.dtype)


def apply_world_projection(
    z_raw: torch.Tensor,
    z_world: torch.Tensor,
    visibility: torch.Tensor,
    confidence: torch.Tensor,
    *,
    sigma: float | torch.Tensor,
    lambda_max: float = 0.5,
    gamma: float = 1.0,
    confidence_ramp_min: float = 0.2,
    confidence_ramp_max: float = 0.5,
    temporal_warmup: Sequence[float] | torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Blend a scheduler result toward verified world evidence."""
    if tuple(z_world.shape) != tuple(z_raw.shape):
        raise ValueError(f"z_world shape {tuple(z_world.shape)} != z_raw {tuple(z_raw.shape)}")
    support_shape = (z_raw.shape[0], 1, z_raw.shape[2], z_raw.shape[3], z_raw.shape[4])
    if tuple(visibility.shape) != support_shape or tuple(confidence.shape) != support_shape:
        raise ValueError(
            f"support must be {support_shape}, got {tuple(visibility.shape)}/{tuple(confidence.shape)}"
        )
    raw_strength, schedule = world_projection_weight(
        visibility, confidence, sigma=sigma, lambda_max=lambda_max, gamma=gamma,
        confidence_ramp_min=confidence_ramp_min,
        confidence_ramp_max=confidence_ramp_max,
        device=z_raw.device,
    )
    strength = _final_wpf_strength(
        raw_strength,
        temporal_warmup=temporal_warmup,
        boundary_active=False,
    )
    projected = z_raw.float() + strength * (z_world.float() - z_raw.float())
    projected = projected.to(z_raw.dtype)
    delta = projected.float() - z_raw.float()
    unknown = visibility.to(device=z_raw.device) == 0
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
        "wpf_slot0_strength_max": strength[:, :, 0:1].max().detach(),
        "projection_delta_ratio": (
            delta.norm() / z_raw.float().norm().clamp_min(1e-8)
        ).detach(),
        "unknown_projection_delta_max": unknown_delta_max.detach(),
    }
    return projected, diagnostics


def apply_world_and_boundary_projection(
    z_raw: torch.Tensor,
    z_world: torch.Tensor,
    visibility: torch.Tensor,
    confidence: torch.Tensor,
    *,
    boundary_state: torch.Tensor | None,
    boundary_beta_max: float = 0.0,
    sigma: float | torch.Tensor,
    lambda_max: float = 0.5,
    gamma: float = 1.0,
    confidence_ramp_min: float = 0.2,
    confidence_ramp_max: float = 0.5,
    temporal_warmup: Sequence[float] | torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Apply Boundary Projection and WPF from the same scheduler output."""
    if tuple(z_world.shape) != tuple(z_raw.shape):
        raise ValueError("world/raw shape mismatch")
    raw_world_strength, schedule = world_projection_weight(
        visibility, confidence,
        sigma=sigma, lambda_max=lambda_max, gamma=gamma,
        confidence_ramp_min=confidence_ramp_min,
        confidence_ramp_max=confidence_ramp_max,
        device=z_raw.device,
    )
    world_strength = _final_wpf_strength(
        raw_world_strength,
        temporal_warmup=temporal_warmup,
        boundary_active=boundary_state is not None,
    )
    raw = z_raw.float()
    effective_world_strength = world_strength.clone()
    boundary_delta = torch.zeros_like(raw)
    boundary_schedule = torch.zeros((), device=z_raw.device, dtype=torch.float32)
    if boundary_state is not None:
        expected = (
            z_raw.shape[0], z_raw.shape[1], 1,
            z_raw.shape[3], z_raw.shape[4],
        )
        if tuple(boundary_state.shape) != expected:
            raise ValueError(
                f"boundary state shape {tuple(boundary_state.shape)} != {expected}"
            )
        sigma_value = torch.as_tensor(
            sigma, device=z_raw.device, dtype=torch.float32,
        ).clamp(0.0, 1.0)
        boundary_schedule = float(boundary_beta_max) * (1.0 - sigma_value)
        boundary_delta[:, :, 0:1] = boundary_schedule * (
            boundary_state.to(device=z_raw.device, dtype=torch.float32)
            - raw[:, :, 0:1]
        )
        # Boundary owns slot0 completely; WPF strength is strictly zero there.
        effective_world_strength[:, :, 0:1] = 0.0
    world_delta = effective_world_strength * (z_world.float() - raw)
    combined = (raw + boundary_delta + world_delta).to(z_raw.dtype)
    unknown = visibility.to(device=z_raw.device) == 0
    unknown_world_delta_max = (
        world_delta.masked_select(unknown.expand_as(world_delta)).abs().max()
        if bool(unknown.any()) else torch.zeros((), device=z_raw.device)
    )
    if float(unknown_world_delta_max.detach().cpu()) != 0.0:
        raise RuntimeError("world projection changed a V=0 latent element")
    non_slot0_delta_max = (
        boundary_delta[:, :, 1:].abs().max()
        if z_raw.shape[2] > 1 else torch.zeros((), device=z_raw.device)
    )
    if float(non_slot0_delta_max.detach().cpu()) != 0.0:
        raise RuntimeError("Boundary Bridge changed a non-slot0 temporal latent")
    return combined, {
        "lambda": schedule.detach(),
        "projection_mask_ratio": (effective_world_strength > 0).float().mean().detach(),
        "projection_strength_mean": effective_world_strength.mean().detach(),
        "wpf_slot0_strength_max": effective_world_strength[:, :, 0:1].max().detach(),
        "projection_delta_ratio": (
            world_delta.norm() / raw.norm().clamp_min(1e-8)
        ).detach(),
        "unknown_projection_delta_max": unknown_world_delta_max.detach(),
        "boundary_active": torch.as_tensor(
            boundary_state is not None, device=z_raw.device, dtype=torch.float32,
        ),
        "boundary_strength": boundary_schedule.detach(),
        "boundary_delta_ratio": (
            boundary_delta.norm() / raw.norm().clamp_min(1e-8)
        ).detach(),
        "boundary_non_slot0_delta_max": non_slot0_delta_max.detach(),
        "combined_delta_ratio": (
            (boundary_delta + world_delta).norm() / raw.norm().clamp_min(1e-8)
        ).detach(),
    }


def compose_canonical_residual(
    canonical_endpoint: torch.Tensor,
    support: torch.Tensor,
    residual: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compose ``L = B + (1-M)R`` with ``B = M E(W)`` exactly once."""
    if tuple(canonical_endpoint.shape) != tuple(residual.shape):
        raise ValueError("canonical endpoint/residual shape mismatch")
    expected = (
        residual.shape[0], 1, residual.shape[2], residual.shape[3], residual.shape[4],
    )
    if tuple(support.shape) != expected:
        raise ValueError(f"canonical residual support must be {expected}, got {tuple(support.shape)}")
    mask = support.to(device=residual.device, dtype=torch.float32).clamp(0.0, 1.0)
    endpoint = canonical_endpoint.to(device=residual.device, dtype=torch.float32)
    residual_float = residual.float()
    base = mask * endpoint
    composed = base + (1.0 - mask) * residual_float
    return composed.to(residual.dtype), base.to(residual.dtype)


def apply_residual_boundary_bridge(
    residual_raw: torch.Tensor,
    boundary_state: torch.Tensor | None,
    support: torch.Tensor,
    *,
    sigma: float | torch.Tensor,
    boundary_beta_max: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Apply the existing slot0 Boundary Bridge only in residual/unknown space."""
    expected_support = (
        residual_raw.shape[0], 1, residual_raw.shape[2],
        residual_raw.shape[3], residual_raw.shape[4],
    )
    if tuple(support.shape) != expected_support:
        raise ValueError("residual boundary support shape mismatch")
    raw = residual_raw.float()
    delta = torch.zeros_like(raw)
    schedule = torch.zeros((), device=raw.device, dtype=torch.float32)
    if boundary_state is not None:
        expected_boundary = (
            raw.shape[0], raw.shape[1], 1, raw.shape[3], raw.shape[4],
        )
        if tuple(boundary_state.shape) != expected_boundary:
            raise ValueError("residual boundary state shape mismatch")
        sigma_value = torch.as_tensor(sigma, device=raw.device, dtype=torch.float32).clamp(0.0, 1.0)
        schedule = float(boundary_beta_max) * (1.0 - sigma_value)
        unknown_slot0 = 1.0 - support[:, :, 0:1].to(device=raw.device, dtype=torch.float32).clamp(0.0, 1.0)
        delta[:, :, 0:1] = schedule * unknown_slot0 * (
            boundary_state.to(device=raw.device, dtype=torch.float32) - raw[:, :, 0:1]
        )
    result = (raw + delta).to(residual_raw.dtype)
    non_slot0 = (
        delta[:, :, 1:].abs().max()
        if raw.shape[2] > 1 else torch.zeros((), device=raw.device)
    )
    if float(non_slot0.detach().cpu()) != 0.0:
        raise RuntimeError("Residual Boundary Bridge changed temporal slots 1..8")
    return result, {
        "lambda": torch.zeros((), device=raw.device),
        "projection_mask_ratio": (support > 0).float().mean().detach(),
        "projection_strength_mean": support.float().mean().detach(),
        "wpf_slot0_strength_max": torch.zeros((), device=raw.device),
        "projection_delta_ratio": torch.zeros((), device=raw.device),
        "unknown_projection_delta_max": torch.zeros((), device=raw.device),
        "boundary_active": torch.as_tensor(boundary_state is not None, device=raw.device, dtype=torch.float32),
        "boundary_strength": schedule.detach(),
        "boundary_delta_ratio": (delta.norm() / raw.norm().clamp_min(1e-8)).detach(),
        "boundary_non_slot0_delta_max": non_slot0.detach(),
        "combined_delta_ratio": (delta.norm() / raw.norm().clamp_min(1e-8)).detach(),
        "residual_coordinate": torch.ones((), device=raw.device),
    }


def apply_boundary_then_world_clamp(
    z_raw: torch.Tensor,
    z_world: torch.Tensor,
    support: torch.Tensor,
    boundary_state: torch.Tensor | None,
    *,
    sigma: float | torch.Tensor,
    boundary_beta_max: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Apply unknown-only Boundary Bridge, then hard-clamp world every step."""
    if tuple(z_world.shape) != tuple(z_raw.shape):
        raise ValueError("world/raw shape mismatch")
    expected_support = (
        z_raw.shape[0], 1, z_raw.shape[2], z_raw.shape[3], z_raw.shape[4],
    )
    if tuple(support.shape) != expected_support:
        raise ValueError(f"world clamp support must be {expected_support}")
    mask = support.to(device=z_raw.device, dtype=torch.float32).clamp(0.0, 1.0)
    if not bool(((mask == 0) | (mask == 1)).all()):
        raise ValueError("per-step world ownership mask must be binary")
    unknown = 1.0 - mask
    raw = z_raw.float()
    boundary_delta = torch.zeros_like(raw)
    boundary_schedule = torch.zeros((), device=raw.device, dtype=torch.float32)
    if boundary_state is not None:
        expected_boundary = (
            raw.shape[0], raw.shape[1], 1, raw.shape[3], raw.shape[4],
        )
        if tuple(boundary_state.shape) != expected_boundary:
            raise ValueError("world-clamp boundary state shape mismatch")
        sigma_value = torch.as_tensor(sigma, device=raw.device, dtype=torch.float32).clamp(0.0, 1.0)
        boundary_schedule = float(boundary_beta_max) * (1.0 - sigma_value)
        boundary_delta[:, :, 0:1] = boundary_schedule * unknown[:, :, 0:1] * (
            boundary_state.to(device=raw.device, dtype=torch.float32) - raw[:, :, 0:1]
        )
    candidate = raw + boundary_delta
    world = z_world.to(device=raw.device, dtype=torch.float32)
    clamped = mask * world + unknown * candidate
    world_delta = clamped - candidate
    zero_support = mask == 0
    unknown_world_delta_max = (
        world_delta.masked_select(zero_support.expand_as(world_delta)).abs().max()
        if bool(zero_support.any()) else torch.zeros((), device=raw.device)
    )
    if float(unknown_world_delta_max.detach().cpu()) != 0.0:
        raise RuntimeError("per-step world clamp changed an M=0 latent element")
    non_slot0 = (
        boundary_delta[:, :, 1:].abs().max()
        if raw.shape[2] > 1 else torch.zeros((), device=raw.device)
    )
    if float(non_slot0.detach().cpu()) != 0.0:
        raise RuntimeError("Boundary Bridge changed temporal slots 1..8")
    formula_error = (
        clamped - (mask * world + unknown * candidate)
    ).abs().max()
    return clamped.to(z_raw.dtype), {
        "lambda": torch.ones((), device=raw.device),
        "projection_mask_ratio": (mask > 0).float().mean().detach(),
        "projection_strength_mean": mask.mean().detach(),
        "wpf_slot0_strength_max": torch.zeros((), device=raw.device),
        "projection_delta_ratio": (world_delta.norm() / candidate.norm().clamp_min(1e-8)).detach(),
        "unknown_projection_delta_max": unknown_world_delta_max.detach(),
        "boundary_active": torch.as_tensor(boundary_state is not None, device=raw.device, dtype=torch.float32),
        "boundary_strength": boundary_schedule.detach(),
        "boundary_delta_ratio": (boundary_delta.norm() / raw.norm().clamp_min(1e-8)).detach(),
        "boundary_non_slot0_delta_max": non_slot0.detach(),
        "combined_delta_ratio": ((clamped - raw).norm() / raw.norm().clamp_min(1e-8)).detach(),
        "per_step_world_clamp": torch.ones((), device=raw.device),
        "world_clamp_formula_max_error": formula_error.detach(),
    }


def scheduler_clean_prediction(
    scheduler: Any,
    model_output: torch.Tensor,
    timestep: float | torch.Tensor,
    sample: torch.Tensor,
    *,
    dmd_sigmas: torch.Tensor,
    dmd_timesteps: torch.Tensor,
) -> torch.Tensor:
    """Use Helios' native flow conversion; never decode a noisy latent."""
    timestep_value = torch.as_tensor(timestep, device=model_output.device).item()
    timestep_batch = torch.full(
        (model_output.shape[0],), timestep_value,
        dtype=torch.long, device=model_output.device,
    )
    return scheduler.convert_flow_pred_to_x0(
        flow_pred=model_output,
        xt=sample,
        timestep=timestep_batch,
        sigmas=dmd_sigmas,
        timesteps=dmd_timesteps,
    )


def scheduler_align_clean_prediction(
    scheduler: Any,
    clean_prediction: torch.Tensor,
    *,
    step_index: int,
    dmd_noisy_tensor: torch.Tensor,
    dmd_sigmas: torch.Tensor,
    dmd_timesteps: torch.Tensor,
    all_timesteps: torch.Tensor,
) -> torch.Tensor:
    """Return a clean endpoint to Helios' native next coordinate.

    ``dmd_noisy_tensor`` is the exact tensor passed by the pinned Helios
    pipeline to the current scheduler call.  It must not be replaced by an
    inferred stage start or newly sampled noise.
    """
    if int(step_index) >= len(all_timesteps) - 1:
        return clean_prediction
    next_timestep = torch.full(
        (clean_prediction.shape[0],), all_timesteps[int(step_index) + 1],
        dtype=torch.long, device=clean_prediction.device,
    )
    return scheduler.add_noise(
        clean_prediction,
        dmd_noisy_tensor.to(clean_prediction.device, clean_prediction.dtype),
        next_timestep,
        sigmas=dmd_sigmas,
        timesteps=dmd_timesteps,
    )


def sparse_pixel_constraint(
    x0_base: torch.Tensor,
    *,
    decode_fn: Any,
    warp_rgb: torch.Tensor,
    visibility: torch.Tensor,
    steps: int,
    lr: float,
    lambda_z: float,
    max_grad_norm: float,
    activation_offload_budget_bytes: int = 32 * 1024**3,
    activation_offload_min_tensor_bytes: int = 8 * 1024**2,
    activation_offload_min_spatial_area: int = 96 * 160,
    epsilon: float = 1e-8,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Jointly optimize one clean 33-frame latent against sparse renderer pixels.

    Only ``x0_opt`` receives gradients.  The renderer visibility is used
    exactly as supplied: it is neither dilated nor converted to latent-cell
    ownership, and invisible RGB pixels contribute no pixel loss.
    """
    if int(steps) < 0:
        raise ValueError("sparse constraint steps must be non-negative")
    if float(lr) < 0 or float(lambda_z) < 0 or float(max_grad_norm) <= 0:
        raise ValueError("invalid sparse constraint optimization hyperparameters")
    if x0_base.ndim != 5:
        raise ValueError(f"x0_base must be [B,C,T,H,W], got {tuple(x0_base.shape)}")
    warp = warp_rgb.detach().to(device=x0_base.device)
    visible = visibility.detach().to(device=x0_base.device, dtype=torch.float32)
    if warp.ndim != 5 or visible.ndim != 5:
        raise ValueError("sparse renderer tensors must be five-dimensional")
    if tuple(visible.shape) != (warp.shape[0], 1, warp.shape[2], warp.shape[3], warp.shape[4]):
        raise ValueError("sparse visibility must align exactly with renderer RGB")
    if not bool(((visible == 0) | (visible == 1)).all()):
        raise ValueError("sparse constraint requires raw binary renderer visibility")

    base = x0_base.detach()
    x0_opt = base
    last_pixel = torch.zeros((), device=base.device)
    last_latent = torch.zeros((), device=base.device)
    last_grad_norm = torch.zeros((), device=base.device)
    last_clipped_norm = torch.zeros((), device=base.device)
    visible_count = visible.sum().detach()
    offloaded_bytes = 0

    def pack_saved_tensor(tensor: torch.Tensor):
        nonlocal offloaded_bytes
        tensor_bytes = int(tensor.numel() * tensor.element_size())
        should_offload = (
            tensor.is_cuda
            and not tensor.is_leaf
            and tensor.ndim >= 4
            and int(tensor.shape[-2] * tensor.shape[-1]) >= int(activation_offload_min_spatial_area)
            and tensor_bytes >= int(activation_offload_min_tensor_bytes)
            and offloaded_bytes + tensor_bytes <= int(activation_offload_budget_bytes)
        )
        if not should_offload:
            return False, tensor
        offloaded_bytes += tensor_bytes
        return True, tensor.detach().to(device="cpu", non_blocking=False)

    def unpack_saved_tensor(packed):
        was_offloaded, tensor = packed
        if not was_offloaded:
            return tensor
        return tensor.to(device=base.device, non_blocking=False)

    with torch.enable_grad():
        for _ in range(int(steps)):
            variable = x0_opt.detach().requires_grad_(True)
            # Wan's decoder does not implement gradient checkpointing.  Save
            # a bounded subset of its largest non-parameter activations on CPU
            # so the joint 33-frame graph fits beside resident Helios state.
            # This changes storage only: decode, loss, and gradient are exact.
            saved_tensor_context = (
                torch.autograd.graph.saved_tensors_hooks(
                    pack_saved_tensor, unpack_saved_tensor,
                )
                if variable.is_cuda else nullcontext()
            )
            with saved_tensor_context:
                decoded = decode_fn(variable)
                if tuple(decoded.shape) != tuple(warp.shape):
                    raise RuntimeError(
                        f"decoded 33-frame RGB {tuple(decoded.shape)} != renderer {tuple(warp.shape)}"
                    )
                warp_value = warp.to(device=decoded.device, dtype=decoded.dtype)
                visible_value = visible.to(device=decoded.device, dtype=decoded.dtype)
                pixel_loss = (
                    visible_value * (decoded - warp_value).abs()
                ).sum() / (3.0 * visible_value.sum() + float(epsilon))
                latent_loss = (variable.float() - base.float()).square().mean()
                loss = pixel_loss.float() + float(lambda_z) * latent_loss
                gradient, = torch.autograd.grad(
                    loss, variable, create_graph=False, retain_graph=False,
                )
            grad_float = gradient.float()
            grad_norm = grad_float.norm()
            clip_scale = (float(max_grad_norm) / grad_norm.clamp_min(float(epsilon))).clamp(max=1.0)
            clipped = gradient * clip_scale.to(dtype=gradient.dtype)
            x0_opt = (variable - float(lr) * clipped).detach()
            last_pixel = pixel_loss.detach()
            last_latent = latent_loss.detach()
            last_grad_norm = grad_norm.detach()
            last_clipped_norm = clipped.float().norm().detach()
    return x0_opt.detach(), {
        "sparse_pixel_loss": last_pixel,
        "sparse_latent_loss": last_latent,
        "sparse_grad_norm": last_grad_norm,
        "sparse_clipped_grad_norm": last_clipped_norm,
        "sparse_visible_pixel_count": visible_count,
        "sparse_activation_offload_bytes": torch.as_tensor(
            offloaded_bytes, device=base.device, dtype=torch.float64,
        ),
        "sparse_latent_delta_ratio": (
            (x0_opt.float() - base.float()).norm()
            / base.float().norm().clamp_min(float(epsilon))
        ).detach(),
    }


def sparse_pixel_constraint_enabled(stage_id: int, stage_count: int) -> bool:
    """Sparse pixel optimization is restricted to the final pyramid stage."""
    if int(stage_count) <= 0:
        raise ValueError("stage_count must be positive")
    return int(stage_id) == int(stage_count) - 1


def world_projection_weight(
    visibility: torch.Tensor,
    confidence: torch.Tensor,
    *,
    sigma: float | torch.Tensor,
    lambda_max: float,
    gamma: float,
    confidence_ramp_min: float,
    confidence_ramp_max: float,
    device: torch.device | str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the exact soft projection weight and its scalar sigma schedule."""
    if confidence_ramp_max <= confidence_ramp_min:
        raise ValueError("confidence_ramp_max must be greater than confidence_ramp_min")
    device = device or visibility.device
    sigma_value = torch.as_tensor(sigma, device=device, dtype=torch.float32).clamp(0.0, 1.0)
    schedule = float(lambda_max) * (1.0 - sigma_value).pow(float(gamma))
    visible = visibility.to(device=device, dtype=torch.float32).clamp(0.0, 1.0)
    confidence_value = confidence.to(device=device, dtype=torch.float32).clamp(0.0, 1.0)
    confidence_weight = (
        (confidence_value - float(confidence_ramp_min))
        / (float(confidence_ramp_max) - float(confidence_ramp_min))
    ).clamp(0.0, 1.0)
    return visible * confidence_weight * schedule, schedule


def _temporal_warmup_tensor(
    warmup: Sequence[float] | torch.Tensor | None,
    frames: int,
    *,
    device: torch.device,
) -> torch.Tensor:
    """Broadcast a WPF temporal warmup vector over latent spatial dimensions."""
    if warmup is None:
        # Keep small synthetic helper tensors backwards-compatible while the
        # real Helios 9-slot path receives the validated default schedule.
        values = (1.0,) * int(frames) if int(frames) != len(DEFAULT_TEMPORAL_WARMUP) else DEFAULT_TEMPORAL_WARMUP
    else:
        values = warmup.detach().flatten().tolist() if torch.is_tensor(warmup) else list(warmup)
    if len(values) != int(frames):
        raise ValueError(
            f"temporal warmup must have one value per latent slot ({frames}), got {len(values)}"
        )
    tensor = torch.as_tensor(values, device=device, dtype=torch.float32)
    if bool((tensor < 0).any()) or bool((tensor > 1).any()):
        raise ValueError("temporal warmup values must be in [0, 1]")
    return tensor.view(1, 1, int(frames), 1, 1)


def _final_wpf_strength(
    strength: torch.Tensor,
    *,
    temporal_warmup: Sequence[float] | torch.Tensor | None,
    boundary_active: bool,
) -> torch.Tensor:
    """Apply warmup and slot-0 policy to the final WPF strength only."""
    if strength.ndim != 5:
        raise ValueError(f"WPF strength must be 5D, got {tuple(strength.shape)}")
    warmup = _temporal_warmup_tensor(
        temporal_warmup, int(strength.shape[2]), device=strength.device,
    )
    result = strength * warmup
    if boundary_active:
        # Boundary owns slot 0 completely; WPF must not retain even a tiny
        # residual weight when its beta schedule is still warming up.
        result[:, :, 0:1] = 0.0
    return result


class WorldProjectedWarpAsHistoryPipeline(WarpAsHistoryPipeline):
    """WAH subclass that projects only scheduler outputs, never predictions."""

    def set_world_projection_context(self, context: WorldProjectionContext | None) -> None:
        self._world_projection_context = context

    def set_sparse_pixel_constraint_context(
        self,
        raw_warp_rgb: Any,
        raw_renderer_visibility: Any,
        *,
        height: int,
        width: int,
    ) -> None:
        """Attach raw renderer evidence used only by the stage-2 sparse loss."""
        context = getattr(self, "_world_projection_context", None)
        if context is None:
            raise RuntimeError("set world projection context before sparse constraint pixels")
        device = self._wah_execution_device()
        rgb = self._coerce_warp_video_tensor(
            raw_warp_rgb, height=int(height), width=int(width), device=device,
        ).to(device=device, dtype=self.vae.dtype)
        visibility = torch.as_tensor(
            raw_renderer_visibility, device=device, dtype=torch.float32,
        )
        if visibility.ndim == 3:
            visibility = visibility.unsqueeze(0).unsqueeze(0)
        elif visibility.ndim == 4:
            visibility = visibility.unsqueeze(1)
        expected = (rgb.shape[0], 1, rgb.shape[2], rgb.shape[3], rgb.shape[4])
        if tuple(visibility.shape) != expected:
            raise ValueError(
                f"raw renderer visibility must align exactly to RGB {expected}, got {tuple(visibility.shape)}"
            )
        visibility = (visibility > 0).to(torch.float32)
        context.pixel_warp_rgb = rgb.detach()
        context.pixel_visibility = visibility.detach()

    def clear_world_projection_context(self) -> None:
        self._world_projection_context = None

    def set_canonical_warp_conditioning(
        self,
        canonical_warp_rgb: Any,
        canonical_warp_latents: torch.Tensor,
        *,
        first_frame_latent: torch.Tensor | None = None,
        canonical_support: CanonicalWorldSupport | None = None,
        pixel_visibility: Any | None = None,
        pixel_confidence: Any | None = None,
        height: int | None = None,
        width: int | None = None,
    ) -> None:
        """Cache the shared canonical RGB/latent pair for this chunk.

        WAH normally re-encodes its warp history inside ``generate_next_chunk``
        with a stochastic VAE posterior.  The world-projected path has already
        encoded this exact filled RGB once, so reuse that latent and leave the
        normal Helios generation noise path untouched.
        """
        device = self._wah_execution_device()
        height = int(height if height is not None else getattr(self, "_canonical_conditioning_height", 384))
        width = int(width if width is not None else getattr(self, "_canonical_conditioning_width", 640))
        object.__setattr__(self, "_canonical_warp_video_tensor", self._coerce_warp_video_tensor(
            canonical_warp_rgb, height=height, width=width, device=device,
        ).detach())
        object.__setattr__(self, "_canonical_warp_latents", canonical_warp_latents.detach().to(
            device=device, dtype=torch.float32,
        ))
        object.__setattr__(self, "_canonical_warp_first_frame_latent", (
            self._canonical_warp_latents[:, :, :1]
            if first_frame_latent is None
            else first_frame_latent.detach().to(device=device, dtype=torch.float32)
        ))
        cached_support = None
        if canonical_support is not None:
            cached_support = CanonicalWorldSupport(
                canonical_support.visibility.to(device=device, dtype=torch.float32),
                canonical_support.confidence.to(device=device, dtype=torch.float32),
                canonical_support.safe_support.to(device=device, dtype=torch.float32),
            )
        object.__setattr__(self, "_canonical_world_support", cached_support)
        object.__setattr__(self, "_canonical_pixel_visibility", (
            None if pixel_visibility is None
            else self._coerce_visibility_mask(pixel_visibility).to(device=device, dtype=torch.float32)
        ))
        object.__setattr__(self, "_canonical_pixel_confidence", (
            None if pixel_confidence is None
            else self._coerce_visibility_mask(pixel_confidence).to(device=device, dtype=torch.float32)
        ))
        object.__setattr__(self, "_canonical_conditioning_cache_hit", False)

    def clear_canonical_warp_conditioning(self) -> None:
        for name in (
            "_canonical_warp_video_tensor", "_canonical_warp_latents",
            "_canonical_warp_first_frame_latent", "_canonical_world_support",
            "_canonical_pixel_visibility", "_canonical_pixel_confidence",
        ):
            object.__setattr__(self, name, None)

    def _visibility_mask_to_history_latents(
        self, visibility_mask, *, latent_frames, latent_height, latent_width, temporal_scale,
    ):
        """Reuse the one canonical 33-to-9 support instead of regrouping pixels."""
        support = getattr(self, "_canonical_world_support", None)
        if support is not None and int(latent_frames) == int(support.safe_support.shape[2]):
            value = visibility_mask.to(
                device=support.safe_support.device, dtype=torch.float32,
            )
            pixel_visibility = getattr(self, "_canonical_pixel_visibility", None)
            pixel_confidence = getattr(self, "_canonical_pixel_confidence", None)
            canonical = None
            if pixel_visibility is not None and tuple(value.shape) == tuple(pixel_visibility.shape):
                if torch.equal(value, pixel_visibility):
                    canonical = support.safe_support
            if canonical is None and pixel_confidence is not None and tuple(value.shape) == tuple(pixel_confidence.shape):
                if torch.equal(value, pixel_confidence * pixel_visibility):
                    # The pinned WAH path divides this weighted value by its
                    # visibility latent, yielding canonical confidence.
                    canonical = support.safe_support * support.confidence
            if canonical is not None:
                return _resize_video_latents(
                    canonical, int(latent_height), int(latent_width), mode="area",
                ).to(device=value.device, dtype=torch.float32)
        return super()._visibility_mask_to_history_latents(
            visibility_mask,
            latent_frames=latent_frames,
            latent_height=latent_height,
            latent_width=latent_width,
            temporal_scale=temporal_scale,
        )

    def prepare_video_latents(self, video, *args, **kwargs):
        """Reuse a matching canonical encode; defer all other calls to WAH."""
        cached_video = getattr(self, "_canonical_warp_video_tensor", None)
        cached_latents = getattr(self, "_canonical_warp_latents", None)
        if cached_video is not None and cached_latents is not None:
            candidate = video.detach() if torch.is_tensor(video) else None
            if candidate is not None:
                candidate = candidate.to(device=cached_video.device, dtype=cached_video.dtype)
                if tuple(candidate.shape) == tuple(cached_video.shape) and torch.equal(candidate, cached_video):
                    object.__setattr__(self, "_canonical_conditioning_cache_hit", True)
                    first = getattr(
                        self, "_canonical_warp_first_frame_latent", None,
                    )
                    if first is None:
                        first = cached_latents[:, :, :1]
                    dtype = kwargs.get("dtype")
                    device = kwargs.get("device") or cached_latents.device
                    if dtype is not None:
                        first = first.to(device=device, dtype=dtype)
                        latents = cached_latents.to(device=device, dtype=dtype)
                    else:
                        first = first.to(device=device)
                        latents = cached_latents.to(device=device)
                    return first, latents
        return super().prepare_video_latents(video, *args, **kwargs)

    def stage2_sample(self, *args, **kwargs):  # noqa: C901 - preserves pinned pipeline semantics.
        context = getattr(self, "_world_projection_context", None)
        if context is None:
            return super().stage2_sample(*args, **kwargs)

        scheduler = self.scheduler
        original_step = scheduler.step
        stage_id = -1
        stage_start = None

        def world_clamped_step(model_output, timestep, sample, *step_args, **step_kwargs):
            nonlocal stage_id, stage_start
            step_index = int(step_kwargs.get("cur_sampling_step", 0))
            if step_index == 0:
                stage_id += 1
                stage_start = sample.detach().clone()
            if stage_id >= len(context.canonical_latents) or stage_start is None:
                raise RuntimeError(f"unexpected Helios pyramid stage {stage_id}")
            # Helios always performs its native scheduler update first.
            result = original_step(model_output, timestep, sample, *step_args, **step_kwargs)
            z_raw = result[0]
            sigmas = step_kwargs.get("dmd_sigmas", getattr(scheduler, "sigmas", None))
            dmd_timesteps = step_kwargs.get("dmd_timesteps", getattr(scheduler, "timesteps", None))
            all_timesteps = step_kwargs.get("all_timesteps")
            if sigmas is None or len(sigmas) <= step_index:
                raise RuntimeError("Helios scheduler did not expose the active stage sigma coordinate")
            if dmd_timesteps is None or all_timesteps is None:
                raise RuntimeError("Helios scheduler did not expose native DMD timestep coordinates")
            current_sigma = sigmas[step_index]
            next_sigma = sigmas[min(step_index + 1, len(sigmas) - 1)]
            visible = context.visibility[stage_id].to(device=z_raw.device)
            config = context.config
            boundary_endpoint = (
                None if context.previous_boundary_latents is None
                else context.previous_boundary_latents[stage_id].to(
                    device=z_raw.device, dtype=z_raw.dtype,
                )
            )
            boundary_state = (
                None if boundary_endpoint is None else build_boundary_state_at_sigma(
                    next_sigma=next_sigma,
                    clean_boundary_endpoint=boundary_endpoint,
                    stage_start_state=stage_start,
                )
            )
            boundary_beta_max = config.boundary_beta_max(stage_id)
            if not sparse_pixel_constraint_enabled(stage_id, len(context.canonical_latents)):
                # Stage 0/1 retain only the existing Boundary Bridge.  They do
                # not decode, composite, encode, or apply any world clamp.
                z_next, diagnostic = apply_residual_boundary_bridge(
                    z_raw, boundary_state, visible,
                    boundary_beta_max=boundary_beta_max, sigma=next_sigma,
                )
                diagnostic.update({
                    "projection_mask_ratio": torch.zeros((), device=z_raw.device),
                    "projection_strength_mean": torch.zeros((), device=z_raw.device),
                    "per_step_world_clamp": torch.zeros((), device=z_raw.device),
                    "world_clamp_formula_max_error": torch.zeros((), device=z_raw.device),
                    "sparse_pixel_constraint": torch.zeros((), device=z_raw.device),
                    "rgb_visibility_mean": torch.zeros((), device=z_raw.device),
                    "rgb_visible_exact_warp_max_error": torch.zeros((), device=z_raw.device),
                    "rgb_unknown_exact_model_max_error": torch.zeros((), device=z_raw.device),
                    "rgb_composite_formula_max_error": torch.zeros((), device=z_raw.device),
                    "sparse_clean_delta_ratio": torch.zeros((), device=z_raw.device),
                })
            else:
                if context.pixel_warp_rgb is None or context.pixel_visibility is None:
                    raise RuntimeError("stage-2 sparse constraint is missing raw renderer pixels")
                # Obtain the true clean/x0 prediction from the scheduler's
                # native flow coordinate.  Optimize all 33 decoded frames as
                # one latent variable; never composite/re-encode RGB.
                clean_model = scheduler_clean_prediction(
                    scheduler, model_output, timestep, sample,
                    dmd_sigmas=sigmas, dmd_timesteps=dmd_timesteps,
                )
                # Keep Boundary Bridge in clean coordinates and only in its
                # existing unknown slot0 region before RGB composition.
                clean_candidate, diagnostic = apply_residual_boundary_bridge(
                    clean_model, boundary_endpoint, visible,
                    boundary_beta_max=boundary_beta_max, sigma=next_sigma,
                )
                latents_mean, latents_std = self._latent_stats(z_raw.device)
                vae_dtype = self.vae.dtype
                warp_rgb = context.pixel_warp_rgb.to(
                    device=z_raw.device, dtype=vae_dtype,
                )
                pixel_visibility = context.pixel_visibility.to(z_raw.device)
                is_final_step = int(step_index) >= len(all_timesteps) - 1
                sparse_steps = 1
                sparse_lr = 0.002 if is_final_step else 0.005
                sparse_lambda_z = 2.0 if is_final_step else 1.0

                def decode_clean(value: torch.Tensor) -> torch.Tensor:
                    vae_latents = value.to(dtype=vae_dtype) / latents_std + latents_mean
                    return self.vae.decode(vae_latents, return_dict=False)[0]

                optimized_clean, sparse_metrics = sparse_pixel_constraint(
                    clean_candidate,
                    decode_fn=decode_clean,
                    warp_rgb=warp_rgb,
                    visibility=pixel_visibility,
                    steps=sparse_steps,
                    lr=sparse_lr,
                    lambda_z=sparse_lambda_z,
                    max_grad_norm=1.0,
                )
                if is_final_step:
                    z_next = optimized_clean
                else:
                    dmd_noisy_tensor = step_kwargs.get("dmd_noisy_tensor")
                    if dmd_noisy_tensor is None:
                        raise RuntimeError("Helios scheduler call omitted native dmd_noisy_tensor")
                    z_next = scheduler_align_clean_prediction(
                        scheduler, optimized_clean,
                        step_index=step_index,
                        dmd_noisy_tensor=dmd_noisy_tensor,
                        dmd_sigmas=sigmas,
                        dmd_timesteps=dmd_timesteps,
                        all_timesteps=all_timesteps,
                    )
                diagnostic.update(sparse_metrics)
                diagnostic.update({
                    "projection_mask_ratio": pixel_visibility.mean().detach(),
                    "projection_strength_mean": pixel_visibility.mean().detach(),
                    "projection_delta_ratio": (
                        (z_next.float() - z_raw.float()).norm()
                        / z_raw.float().norm().clamp_min(1e-8)
                    ).detach(),
                    "unknown_projection_delta_max": torch.zeros((), device=z_raw.device),
                    "combined_delta_ratio": (
                        (z_next.float() - z_raw.float()).norm()
                        / z_raw.float().norm().clamp_min(1e-8)
                    ).detach(),
                    "per_step_world_clamp": torch.ones((), device=z_raw.device),
                    "world_clamp_formula_max_error": torch.zeros((), device=z_raw.device),
                    "sparse_pixel_constraint": torch.ones((), device=z_raw.device),
                    "sparse_optimizer_created": torch.zeros((), device=z_raw.device),
                    "sparse_vae_encode_used": torch.zeros((), device=z_raw.device),
                    "sparse_new_noise_sampled": torch.zeros((), device=z_raw.device),
                    "sparse_final_sigma_zero": torch.as_tensor(
                        float(is_final_step), device=z_raw.device,
                    ),
                    "sparse_lr": torch.as_tensor(sparse_lr, device=z_raw.device),
                    "sparse_lambda_z": torch.as_tensor(sparse_lambda_z, device=z_raw.device),
                    "sparse_clean_delta_ratio": (
                        (optimized_clean.float() - clean_model.float()).norm()
                        / clean_model.float().norm().clamp_min(1e-8)
                    ).detach(),
                })
            context.diagnostics.append({
                "stage_id": int(stage_id),
                "step_id": int(step_index),
                "sigma": float(torch.as_tensor(current_sigma).detach().cpu()),
                "next_sigma": float(torch.as_tensor(next_sigma).detach().cpu()),
                **{key: float(value.detach().cpu()) for key, value in diagnostic.items()},
            })
            return (z_next, *result[1:])

        scheduler.step = world_clamped_step
        try:
            sampled = super().stage2_sample(*args, **kwargs)
            if context.pixel_visibility is None:
                raise RuntimeError("stage-2 sparse constraint did not retain renderer visibility")
            context.final_projection_weight = context.pixel_visibility.detach()
            final_step = context.diagnostics[-1] if context.diagnostics else None
            if final_step is None or final_step["stage_id"] != len(context.canonical_latents) - 1:
                raise RuntimeError("final scheduler step did not record the sparse constraint")
            if (
                final_step["next_sigma"] != 0.0
                or final_step["sparse_pixel_constraint"] != 1.0
                or final_step["sparse_final_sigma_zero"] != 1.0
            ):
                raise RuntimeError("final scheduler output is not a verified sigma-zero sparse constraint")
            context.final_residual_diagnostics = {
                "sampled_norm": float(sampled.float().norm().detach().cpu()),
                "composed_norm": float(sampled.float().norm().detach().cpu()),
                "stage2_sparse_constraint_step_count": int(sum(
                    item.get("sparse_pixel_constraint", 0.0) == 1.0
                    for item in context.diagnostics
                )),
                "final_step_next_sigma": float(final_step["next_sigma"]),
                "final_sparse_pixel_loss": float(final_step["sparse_pixel_loss"]),
                "final_sparse_latent_loss": float(final_step["sparse_latent_loss"]),
                "final_sparse_grad_norm": float(final_step["sparse_grad_norm"]),
                "final_output_posthoc_clamp_applied": False,
                "noisy_latent_decoded": False,
                "rgb_hard_composite_used": False,
                "composite_rgb_vae_encode_used": False,
                "optimizer_created": False,
                "dmd_noisy_tensor_reused": True,
                "clamp_coverage_threshold_used": False,
                "clamp_nearest_fill_used": False,
                "soft_wpf_used": False,
                "residual_coordinates_used": False,
            }
            return sampled
        finally:
            scheduler.step = original_step
