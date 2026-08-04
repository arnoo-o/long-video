"""Reference confidence mapping matching the patched WAH short-history path."""
import numpy as np


def _as_bcthw(value, name):
    import torch
    tensor = torch.as_tensor(value, dtype=torch.float32)
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0).unsqueeze(0)
    elif tensor.ndim == 4:
        tensor = tensor.unsqueeze(1)
    elif tensor.ndim == 5 and tensor.shape[1] != 1 and tensor.shape[-1] == 1:
        tensor = tensor.permute(0, 4, 1, 2, 3)
    if tensor.ndim != 5 or tensor.shape[1] != 1:
        raise ValueError(f"{name} must become [B,1,T,H,W], got {tuple(tensor.shape)}")
    return tensor.clamp(0, 1)


def confidence_to_history_latents(pixel_confidence, pixel_visibility, latent_shape, temporal_scale):
    """Mirror WAH's sampled temporal indices and trilinear latent interpolation."""
    import torch
    import torch.nn.functional as functional
    confidence = _as_bcthw(pixel_confidence, "pixel_confidence")
    visibility = _as_bcthw(pixel_visibility, "pixel_visibility").to(confidence)
    if confidence.shape != visibility.shape:
        raise ValueError(f"confidence/visibility mismatch: {confidence.shape} vs {visibility.shape}")
    latent_t, latent_h, latent_w = map(int, latent_shape)
    sample_ids = torch.arange(latent_t, device=confidence.device) * max(1, int(temporal_scale))
    sample_ids = sample_ids.clamp(max=confidence.shape[2] - 1)
    weighted = (confidence * visibility).index_select(2, sample_ids)
    sampled_visibility = visibility.index_select(2, sample_ids)
    weighted = functional.interpolate(
        weighted, size=(latent_t, latent_h, latent_w), mode="trilinear", align_corners=False
    )
    sampled_visibility = functional.interpolate(
        sampled_visibility, size=(latent_t, latent_h, latent_w), mode="trilinear", align_corners=False
    )
    return (weighted / sampled_visibility.clamp_min(1e-6)).clamp(0, 1), sampled_visibility.clamp(0, 1)


def _pad_replicate(value, patch_size):
    import torch.nn.functional as functional
    _, _, t, h, w = value.shape
    pt, ph, pw = map(int, patch_size)
    return functional.pad(
        value,
        (0, (pw-w%pw)%pw, 0, (ph-h%ph)%ph, 0, (pt-t%pt)%pt),
        mode="replicate",
    )


def build_token_confidence(
    pixel_confidence,
    pixel_visibility,
    actual_vae_layout,
    actual_patch_layout,
    temporal_scale=1,
):
    """Map pixels using the actual VAE layout and patch_short kernel/stride."""
    import torch.nn.functional as functional
    confidence, visibility = confidence_to_history_latents(
        pixel_confidence, pixel_visibility, actual_vae_layout, temporal_scale
    )
    patch = tuple(map(int, actual_patch_layout))
    visible_ratio = functional.avg_pool3d(_pad_replicate(visibility, patch), patch, stride=patch)
    weighted = functional.avg_pool3d(_pad_replicate(confidence * visibility, patch), patch, stride=patch)
    token_confidence = weighted / visible_ratio.clamp_min(1e-6)
    return token_confidence.clamp(0, 1), visible_ratio.clamp(0, 1)
