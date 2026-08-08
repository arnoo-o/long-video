"""Spatial re-anchoring and source-relative Plucker camera conditioning."""
from __future__ import annotations

from dataclasses import dataclass
import weakref

import numpy as np


def _as_batched(value, trailing_shape, name):
    import torch

    tensor = torch.as_tensor(value, dtype=torch.float32)
    if tensor.ndim == len(trailing_shape) + 1:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != len(trailing_shape) + 2 or tuple(tensor.shape[-len(trailing_shape):]) != tuple(trailing_shape):
        raise ValueError(f"{name} must end in {tuple(trailing_shape)}, got {tuple(tensor.shape)}")
    return tensor


def _temporal_groups(frame_count: int, latent_frames: int, temporal_scale: int):
    frame_count = int(frame_count)
    latent_frames = int(latent_frames)
    temporal_scale = int(temporal_scale)
    expected = 1 + (latent_frames - 1) * temporal_scale
    if frame_count != expected:
        raise ValueError(
            f"VAE temporal contract requires {expected} RGB frames for {latent_frames} latents, "
            f"got {frame_count}"
        )
    return [(0, 1)] + [
        (1 + index * temporal_scale, 1 + (index + 1) * temporal_scale)
        for index in range(latent_frames - 1)
    ]


def plucker_camera_rays(
    source_relative_c2w,
    intrinsics,
    *,
    image_height: int,
    image_width: int,
    token_height: int,
    token_width: int,
    latent_frames: int,
    temporal_scale: int,
    scene_scale: float,
):
    """Build target-patch-center Plucker rays in the frozen source frame."""
    import torch

    c2w = _as_batched(source_relative_c2w, (4, 4), "source_relative_c2w")
    k = _as_batched(intrinsics, (3, 3), "intrinsics")
    if c2w.shape[:2] != k.shape[:2]:
        raise ValueError(f"camera/intrinsics mismatch: {tuple(c2w.shape)} vs {tuple(k.shape)}")
    if not np.isfinite(float(scene_scale)) or float(scene_scale) <= 0:
        raise ValueError("scene_scale must be finite and positive")
    identity = torch.eye(4, dtype=c2w.dtype, device=c2w.device)
    if not torch.allclose(c2w[:, 0], identity.expand_as(c2w[:, 0]), atol=1e-5, rtol=0):
        raise ValueError("the first source-relative c2w must be identity")

    yy = (torch.arange(int(token_height), dtype=torch.float32, device=c2w.device) + 0.5) * (
        float(image_height) / float(token_height)
    )
    xx = (torch.arange(int(token_width), dtype=torch.float32, device=c2w.device) + 0.5) * (
        float(image_width) / float(token_width)
    )
    grid_y, grid_x = torch.meshgrid(yy, xx, indexing="ij")
    pixels = torch.stack((grid_x, grid_y, torch.ones_like(grid_x)), dim=-1)
    pixels = pixels.view(1, 1, token_height, token_width, 3, 1)
    inverse_k = torch.linalg.inv(k).view(*k.shape[:2], 1, 1, 3, 3)
    d_camera = torch.matmul(inverse_k, pixels).squeeze(-1)
    d_camera = torch.nn.functional.normalize(d_camera, dim=-1)
    rotation = c2w[:, :, :3, :3].view(*c2w.shape[:2], 1, 1, 3, 3)
    direction = torch.matmul(rotation, d_camera.unsqueeze(-1)).squeeze(-1)
    direction = torch.nn.functional.normalize(direction, dim=-1)
    origin = c2w[:, :, :3, 3] / float(scene_scale)
    origin = origin[:, :, None, None].expand_as(direction)
    moment = torch.cross(origin, direction, dim=-1)
    per_frame = torch.cat((direction, moment), dim=-1)
    groups = _temporal_groups(c2w.shape[1], latent_frames, temporal_scale)
    result = torch.stack(
        [per_frame[:, start:end].mean(dim=1) for start, end in groups], dim=1
    )
    expected = (c2w.shape[0], latent_frames, token_height, token_width, 6)
    if tuple(result.shape) != expected or not bool(torch.isfinite(result).all()):
        raise RuntimeError(f"invalid Plucker result {tuple(result.shape)}, expected {expected}")
    return result


def visibility_to_target_tokens(
    visibility,
    *,
    latent_frames: int,
    latent_height: int,
    latent_width: int,
    patch_height: int,
    patch_width: int,
    temporal_scale: int,
):
    """Apply the project's sampled 33-to-9 mapping and exact area pooling."""
    import torch
    import torch.nn.functional as functional

    value = torch.as_tensor(visibility, dtype=torch.float32)
    if value.ndim == 3:
        value = value.unsqueeze(0).unsqueeze(0)
    elif value.ndim == 4:
        value = value.unsqueeze(1)
    if value.ndim != 5 or value.shape[1] != 1:
        raise ValueError(f"visibility must become [B,1,T,H,W], got {tuple(value.shape)}")
    groups = _temporal_groups(value.shape[2], latent_frames, temporal_scale)
    sampled = torch.stack(
        [value[:, :, start:end].mean(dim=2) for start, end in groups], dim=2
    )
    height_factor = sampled.shape[-2] // int(latent_height)
    width_factor = sampled.shape[-1] // int(latent_width)
    if sampled.shape[-2] != latent_height * height_factor or sampled.shape[-1] != latent_width * width_factor:
        raise ValueError("visibility resolution must be an integer multiple of the latent resolution")
    batch, _, frames, height, width = sampled.shape
    sampled = sampled.reshape(batch * frames, 1, height, width)
    latent = functional.avg_pool2d(sampled, (height_factor, width_factor), (height_factor, width_factor))
    token = functional.avg_pool2d(latent, (patch_height, patch_width), (patch_height, patch_width))
    token = token.reshape(batch, frames, token.shape[-2], token.shape[-1], 1)
    return token.reshape(batch, -1, 1).clamp(0, 1)


def _module_classes():
    import torch
    from torch import nn

    class Adapter(nn.Module):
        def __init__(self, hidden_size: int, rank: int):
            super().__init__()
            self.target_norm = nn.LayerNorm(hidden_size)
            self.warp_norm = nn.LayerNorm(hidden_size)
            self.target_down = nn.Linear(hidden_size, rank)
            self.warp_down = nn.Linear(hidden_size, rank)
            self.up = nn.Linear(rank, hidden_size)

        def forward(self, target, warp):
            target32 = target.float()
            warp32 = warp.float()
            hidden = torch.nn.functional.silu(
                self.target_down(self.target_norm(target32)) + self.warp_down(self.warp_norm(warp32))
            )
            return self.up(hidden)

    class Camera(nn.Module):
        def __init__(self, hidden_size: int, rank: int):
            super().__init__()
            self.down = nn.Linear(6, rank)
            self.up = nn.Linear(rank, hidden_size)

        def forward(self, rays):
            return self.up(torch.nn.functional.silu(self.down(rays.float())))

    return nn, Adapter, Camera


@dataclass
class ReanchorContext:
    warp_tokens: object
    visibility_tokens: object
    plucker_tokens: object
    target_token_count: int
    warp_latent_frames: int
    anchor_enabled: bool
    camera_enabled: bool
    spatial_warp_enabled: bool


def build_spatial_reanchor_controller(hidden_size: int, *, rank: int = 64, refresh_blocks=(0, 10, 20, 30), gate_init=0.05):
    """Create the torch module lazily so importing geometry helpers stays CPU-light."""
    import torch
    nn, Adapter, Camera = _module_classes()

    class Controller(nn.Module):
        def __init__(self):
            super().__init__()
            self.hidden_size = int(hidden_size)
            self.rank = int(rank)
            self.refresh_blocks = tuple(int(item) for item in refresh_blocks)
            self.anchor_adapter = Adapter(self.hidden_size, self.rank)
            self.camera_adapter = Camera(self.hidden_size, self.rank)
            self.anchor_gates = nn.Parameter(torch.full((len(self.refresh_blocks),), float(gate_init)))
            self.camera_gate = nn.Parameter(torch.tensor(float(gate_init)))
            self.spatial_warp_role = nn.Parameter(torch.zeros(self.hidden_size))
            self._context = None
            self._handles = []
            self.last_metrics = {}

        def install(self, transformer):
            if self._handles:
                raise RuntimeError("spatial re-anchor hooks are already installed")
            if len(transformer.blocks) <= max(self.refresh_blocks):
                raise ValueError("refresh block exceeds transformer depth")
            object.__setattr__(self, "_transformer_ref", weakref.ref(transformer))
            self._handles.append(transformer.patch_short.register_forward_hook(self._patch_short_hook))
            for gate_index, block_index in enumerate(self.refresh_blocks):
                handle = transformer.blocks[block_index].register_forward_pre_hook(
                    self._block_hook(gate_index, block_index), with_kwargs=True
                )
                self._handles.append(handle)
            return self

        def prepare_context(
            self,
            warp_latents,
            visibility_tokens,
            plucker_tokens,
            *,
            anchor_enabled=True,
            camera_enabled=True,
            spatial_warp_enabled=True,
        ):
            transformer = self._transformer_ref()
            if transformer is None:
                raise RuntimeError("the attached transformer no longer exists")
            patch = transformer.patch_embedding
            if any(parameter.requires_grad for parameter in patch.parameters()):
                raise RuntimeError("target/anchor patch_embedding must remain frozen")
            warp = warp_latents.to(device=patch.weight.device, dtype=patch.weight.dtype)
            warp_tokens = patch(warp).flatten(2).transpose(1, 2).detach()
            visibility_tokens = visibility_tokens.to(device=warp_tokens.device, dtype=torch.float32)
            plucker_tokens = plucker_tokens.reshape(plucker_tokens.shape[0], -1, 6).to(warp_tokens.device)
            if warp_tokens.shape[:2] != visibility_tokens.shape[:2] or warp_tokens.shape[:2] != plucker_tokens.shape[:2]:
                raise ValueError(
                    f"target/warp/visibility/Plucker token mismatch: {tuple(warp_tokens.shape)}, "
                    f"{tuple(visibility_tokens.shape)}, {tuple(plucker_tokens.shape)}"
                )
            self._context = ReanchorContext(
                warp_tokens=warp_tokens,
                visibility_tokens=visibility_tokens,
                plucker_tokens=plucker_tokens,
                target_token_count=int(warp_tokens.shape[1]),
                warp_latent_frames=int(warp_latents.shape[2]),
                anchor_enabled=bool(anchor_enabled),
                camera_enabled=bool(camera_enabled),
                spatial_warp_enabled=bool(spatial_warp_enabled),
            )
            self.last_metrics = {}

        def clear_context(self):
            self._context = None

        def _patch_short_hook(self, _module, _args, output):
            context = self._context
            if context is None or not context.spatial_warp_enabled:
                return output
            if output.ndim != 5 or output.shape[1] != self.hidden_size:
                raise RuntimeError(f"unexpected patch_short output {tuple(output.shape)}")
            if output.shape[2] < context.warp_latent_frames:
                raise RuntimeError("patch_short output is shorter than current SPATIAL_WARP")
            result = output.clone()
            role = self.spatial_warp_role.to(device=result.device, dtype=result.dtype).view(1, -1, 1, 1, 1)
            result[:, :, -context.warp_latent_frames:] = result[:, :, -context.warp_latent_frames:] + role
            return result

        def _block_hook(self, gate_index, block_index):
            def hook(_module, args, kwargs):
                context = self._context
                if context is None:
                    return args, kwargs
                hidden = args[0]
                count = context.target_token_count
                if hidden.shape[1] < count:
                    raise RuntimeError("transformer hidden state is shorter than target token count")
                target = hidden[:, -count:]
                base_norm = target.float().norm(dim=-1).mean().clamp_min(1e-8)
                if block_index == 0 and context.camera_enabled:
                    camera_delta = self.camera_adapter(context.plucker_tokens)
                    target = target + (self.camera_gate.float() * camera_delta).to(target.dtype)
                    self.last_metrics["camera_ratio"] = float(
                        ((self.camera_gate.float() * camera_delta).norm(dim=-1).mean() / base_norm).detach().cpu()
                    )
                if context.anchor_enabled:
                    anchor_delta = self.anchor_adapter(target, context.warp_tokens)
                    contribution = self.anchor_gates[gate_index].float() * context.visibility_tokens * anchor_delta
                    if not bool(torch.equal(contribution[context.visibility_tokens.expand_as(contribution) == 0],
                                            torch.zeros_like(contribution[context.visibility_tokens.expand_as(contribution) == 0]))):
                        raise RuntimeError("invisible Spatial Anchor contribution must be exactly zero")
                    target = target + contribution.to(target.dtype)
                    self.last_metrics[f"anchor_ratio_block{block_index}"] = float(
                        (contribution.norm(dim=-1).mean() / base_norm).detach().cpu()
                    )
                updated = torch.cat((hidden[:, :-count], target), dim=1)
                return (updated, *args[1:]), kwargs
            return hook

    return Controller()


def install_spatial_reanchor(transformer, *, rank=64, refresh_blocks=(0, 10, 20, 30), gate_init=0.05):
    hidden_size = int(transformer.patch_embedding.out_channels)
    for parameter in transformer.patch_embedding.parameters():
        parameter.requires_grad_(False)
    controller = build_spatial_reanchor_controller(
        hidden_size, rank=rank, refresh_blocks=refresh_blocks, gate_init=gate_init
    )
    transformer.add_module("spatial_reanchor", controller)
    controller.install(transformer)
    return controller
