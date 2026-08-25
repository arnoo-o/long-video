"""One padded coordinate system shared by RGB, VAE latents, rays and tokens."""
from __future__ import annotations
import math
import torch
from PIL import Image, ImageOps

MODEL_SPATIAL_FACTOR = 64
VAE_SPATIAL_FACTOR = 8


def padded_size(height: int, width: int, factor: int = MODEL_SPATIAL_FACTOR) -> tuple[int, int]:
    if min(height, width, factor) < 1:
        raise ValueError("image dimensions and factor must be positive")
    return math.ceil(height / factor) * factor, math.ceil(width / factor) * factor


def pad_image_bottom_right(image: Image.Image, height: int, width: int) -> Image.Image:
    """Pad without resizing; the top-left pixel coordinate system and K stay unchanged."""
    image = image.convert("RGB")
    target_h, target_w = padded_size(height, width)
    if image.size != (width, height):
        raise ValueError(f"expected preprocessed source {(width, height)}, got {image.size}")
    return ImageOps.expand(image, border=(0, 0, target_w - width, target_h - height), fill=0)


def assert_latent_geometry(latents: torch.Tensor, *, height: int, width: int,
                           pyramid_stages: int = 3, patch_size=(1, 2, 2)) -> None:
    padded_h, padded_w = padded_size(height, width)
    expected = (padded_h // VAE_SPATIAL_FACTOR, padded_w // VAE_SPATIAL_FACTOR)
    if tuple(latents.shape[-2:]) != expected:
        raise ValueError(f"latent grid {tuple(latents.shape[-2:])} != padded grid {expected}")
    h, w = expected
    divisor = 2 ** (pyramid_stages - 1)
    if h % divisor or w % divisor:
        raise ValueError("latent grid is not divisible by the Helios pyramid")
    h //= divisor; w //= divisor
    for stage in range(pyramid_stages):
        if h % int(patch_size[1]) or w % int(patch_size[2]):
            raise ValueError(f"stage {stage} grid {(h, w)} is not divisible by patch {tuple(patch_size)}")
        h *= 2; w *= 2


def crop_video(value, height: int, width: int):
    """Crop tensor/ndarray videos on their two spatial axes."""
    if value.ndim == 5 and value.shape[-1] in (1, 3, 4):
        return value[..., :height, :width, :]
    if value.ndim >= 4:
        return value[..., :height, :width]
    raise ValueError("video output has no spatial axes")
