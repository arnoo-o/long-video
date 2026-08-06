"""WAH RGB/latent chunk indexing and primary loss masks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ChunkContract:
    window_num_frames: int
    vae_temporal_scale: int
    source_prefix_length_rgb: int = 1
    shared_boundary_rule: str = "reuse_previous_boundary_as_next_chunk_frame_zero"

    def __post_init__(self):
        if self.window_num_frames <= 0 or self.vae_temporal_scale <= 0:
            raise ValueError("chunk and VAE temporal sizes must be positive")
        expected = (self.latent_frames - 1) * self.vae_temporal_scale + 1
        if expected != self.window_num_frames:
            raise ValueError(
                f"window_num_frames={self.window_num_frames} is incompatible with "
                f"temporal scale {self.vae_temporal_scale}; reconstructed {expected}"
            )

    @property
    def latent_frames(self) -> int:
        return (self.window_num_frames - 1) // self.vae_temporal_scale + 1

    def rgb_group_for_latent(self, latent_index: int) -> np.ndarray:
        index = int(latent_index)
        if not 0 <= index < self.latent_frames:
            raise IndexError(index)
        if index == 0:
            return np.array([0], np.int64)
        start = 1 + (index - 1) * self.vae_temporal_scale
        return np.arange(start, min(start + self.vae_temporal_scale, self.window_num_frames), dtype=np.int64)


def build_primary_loss_masks(
    contract: ChunkContract,
    *,
    valid_target_frames: np.ndarray | None = None,
    padding_frames: np.ndarray | None = None,
    shared_boundary_frames: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    count = contract.window_num_frames
    valid = np.ones(count, bool) if valid_target_frames is None else np.asarray(valid_target_frames, bool).copy()
    if valid.shape != (count,):
        raise ValueError("valid_target_frames must match window_num_frames")
    rgb_mask = valid
    rgb_mask[: contract.source_prefix_length_rgb] = False
    for excluded in (padding_frames, shared_boundary_frames):
        if excluded is not None:
            value = np.asarray(excluded, bool)
            if value.shape != (count,):
                raise ValueError("excluded RGB masks must match window_num_frames")
            rgb_mask &= ~value
    latent_mask = np.zeros(contract.latent_frames, bool)
    for index in range(contract.latent_frames):
        group = contract.rgb_group_for_latent(index)
        # A partial/invalid temporal group is excluded rather than leaking padding.
        latent_mask[index] = bool(len(group) and rgb_mask[group].all())
    return rgb_mask, latent_mask
