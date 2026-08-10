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


def resize_latents_spatial(latents, *, height: int, width: int):
    """Match WAH train_exact's bilinear spatial pyramid without changing T."""
    import torch
    import torch.nn.functional as functional

    value = torch.as_tensor(latents)
    if value.ndim != 5:
        raise ValueError(f"latents must be [B,C,T,H,W], got {tuple(value.shape)}")
    height, width = int(height), int(width)
    if height <= 0 or width <= 0:
        raise ValueError("target latent height and width must be positive")
    if tuple(value.shape[-2:]) == (height, width):
        return value
    batch, channels, frames, source_height, source_width = value.shape
    flattened = value.permute(0, 2, 1, 3, 4).reshape(
        batch * frames, channels, source_height, source_width
    )
    resized = functional.interpolate(flattened.float(), size=(height, width), mode="bilinear")
    resized = resized.to(dtype=value.dtype)
    return resized.reshape(batch, frames, channels, height, width).permute(0, 2, 1, 3, 4)


def _group_representative_cameras(c2w, intrinsics, groups):
    """Represent each VAE temporal group in pose space, never Plucker space."""
    import torch
    from scipy.spatial.transform import Rotation, Slerp

    representative_poses = []
    representative_intrinsics = []
    for batch_index in range(c2w.shape[0]):
        batch_poses = []
        batch_intrinsics = []
        for start, end in groups:
            group = c2w[batch_index, start:end]
            translation = group[:, :3, 3].mean(dim=0)
            if end - start == 1:
                rotation = group[0, :3, :3]
            else:
                endpoint_matrices = torch.stack(
                    (group[0, :3, :3], group[-1, :3, :3]), dim=0
                ).detach().cpu().double().numpy()
                endpoint_rotations = Rotation.from_matrix(endpoint_matrices)
                rotation_matrix = Slerp([0.0, 1.0], endpoint_rotations)([0.5]).as_matrix()[0]
                rotation = torch.as_tensor(rotation_matrix, dtype=c2w.dtype, device=c2w.device)
            pose = torch.eye(4, dtype=c2w.dtype, device=c2w.device)
            pose[:3, :3] = rotation
            pose[:3, 3] = translation
            batch_poses.append(pose)
            batch_intrinsics.append(intrinsics[batch_index, start:end].mean(dim=0))
        representative_poses.append(torch.stack(batch_poses, dim=0))
        representative_intrinsics.append(torch.stack(batch_intrinsics, dim=0))
    return torch.stack(representative_poses, dim=0), torch.stack(representative_intrinsics, dim=0)


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
    sequence_frame_start: int = 0,
    validate_sequence_source_origin: bool = True,
):
    """Build target-patch-center Plucker rays in the frozen source frame."""
    import torch

    c2w = _as_batched(source_relative_c2w, (4, 4), "source_relative_c2w")
    k = _as_batched(intrinsics, (3, 3), "intrinsics")
    if c2w.shape[:2] != k.shape[:2]:
        raise ValueError(f"camera/intrinsics mismatch: {tuple(c2w.shape)} vs {tuple(k.shape)}")
    if not np.isfinite(float(scene_scale)) or float(scene_scale) <= 0:
        raise ValueError("scene_scale must be finite and positive")
    sequence_frame_start = int(sequence_frame_start)
    if sequence_frame_start < 0:
        raise ValueError("sequence_frame_start must be non-negative")
    if validate_sequence_source_origin and sequence_frame_start == 0:
        identity = torch.eye(4, dtype=c2w.dtype, device=c2w.device)
        if not torch.allclose(c2w[:, 0], identity.expand_as(c2w[:, 0]), atol=1e-5, rtol=0):
            raise ValueError("the full sequence frame0 source-relative c2w must be identity")

    groups = _temporal_groups(c2w.shape[1], latent_frames, temporal_scale)
    representative_c2w, representative_k = _group_representative_cameras(c2w, k, groups)
    if validate_sequence_source_origin and sequence_frame_start == 0 and not torch.allclose(
        representative_c2w[:, 0, :3, 3], torch.zeros_like(representative_c2w[:, 0, :3, 3]),
        atol=1e-6, rtol=0,
    ):
        raise RuntimeError("the first latent group must retain the source canonical origin")

    yy = (torch.arange(int(token_height), dtype=torch.float32, device=c2w.device) + 0.5) * (
        float(image_height) / float(token_height)
    )
    xx = (torch.arange(int(token_width), dtype=torch.float32, device=c2w.device) + 0.5) * (
        float(image_width) / float(token_width)
    )
    grid_y, grid_x = torch.meshgrid(yy, xx, indexing="ij")
    pixels = torch.stack((grid_x, grid_y, torch.ones_like(grid_x)), dim=-1)
    pixels = pixels.view(1, 1, token_height, token_width, 3, 1)
    inverse_k = torch.linalg.inv(representative_k).view(
        *representative_k.shape[:2], 1, 1, 3, 3
    )
    d_camera = torch.matmul(inverse_k, pixels).squeeze(-1)
    d_camera = torch.nn.functional.normalize(d_camera, dim=-1)
    rotation = representative_c2w[:, :, :3, :3].view(
        *representative_c2w.shape[:2], 1, 1, 3, 3
    )
    direction = torch.matmul(rotation, d_camera.unsqueeze(-1)).squeeze(-1)
    direction = torch.nn.functional.normalize(direction, dim=-1)
    origin = representative_c2w[:, :, :3, 3] / float(scene_scale)
    origin = origin[:, :, None, None].expand_as(direction)
    moment = torch.cross(origin, direction, dim=-1)
    result = torch.cat((direction, moment), dim=-1)
    expected = (c2w.shape[0], latent_frames, token_height, token_width, 6)
    if tuple(result.shape) != expected or not bool(torch.isfinite(result).all()):
        raise RuntimeError(f"invalid Plucker result {tuple(result.shape)}, expected {expected}")
    direction_norm = result[..., :3].norm(dim=-1)
    orthogonality = (result[..., :3] * result[..., 3:]).sum(dim=-1).abs()
    if not torch.allclose(direction_norm, torch.ones_like(direction_norm), atol=1e-5, rtol=0):
        raise RuntimeError("Plucker ray directions must have unit norm")
    if bool((orthogonality > 1e-5).any()):
        raise RuntimeError("Plucker direction and moment must be orthogonal")
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

    class LowRankDelta(nn.Module):
        def __init__(self, hidden_size: int, output_size: int, rank: int):
            super().__init__()
            self.down = nn.Linear(hidden_size, rank, bias=False)
            self.up = nn.Linear(rank, output_size, bias=False)
            nn.init.kaiming_uniform_(self.down.weight, a=5 ** 0.5)
            nn.init.zeros_(self.up.weight)

        def forward(self, hidden):
            return self.up(self.down(hidden.float()))

    return nn, Adapter, Camera, LowRankDelta


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
    warp_confidence_tokens: object = None
    memory_tokens: object = None
    memory_visibility_tokens: object = None
    memory_confidence_tokens: object = None
    spatial_attention_enabled: bool = False


def build_spatial_reanchor_controller(
    hidden_size: int, *, rank: int = 64, refresh_blocks=(0, 10, 20, 30),
    gate_init=0.05, spatial_rank: int = 8, block_count: int = 40,
):
    """Create the torch module lazily so importing geometry helpers stays CPU-light."""
    import torch
    nn, Adapter, Camera, LowRankDelta = _module_classes()

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
            self.spatial_memory_role = nn.Parameter(torch.zeros(self.hidden_size))
            self.spatial_k_lora = nn.ModuleList([
                LowRankDelta(self.hidden_size, self.hidden_size, int(spatial_rank))
                for _ in range(int(block_count))
            ])
            self.spatial_v_lora = nn.ModuleList([
                LowRankDelta(self.hidden_size, self.hidden_size, int(spatial_rank))
                for _ in range(int(block_count))
            ])
            self.spatial_gate = nn.Parameter(torch.zeros(int(block_count)))
            self._context = None
            self._contexts = {}
            self._handles = []
            self.last_metrics = {}
            self.collect_spatial_metrics = False

        def install(self, transformer):
            if self._handles:
                raise RuntimeError("spatial re-anchor hooks are already installed")
            if len(transformer.blocks) <= max(self.refresh_blocks):
                raise ValueError("refresh block exceeds transformer depth")
            if len(transformer.blocks) != len(self.spatial_gate):
                raise ValueError(
                    f"spatial attention expects {len(self.spatial_gate)} blocks, got {len(transformer.blocks)}"
                )
            object.__setattr__(self, "_transformer_ref", weakref.ref(transformer))
            self._handles.append(transformer.patch_short.register_forward_hook(self._patch_short_hook))
            for gate_index, block_index in enumerate(self.refresh_blocks):
                handle = transformer.blocks[block_index].register_forward_pre_hook(
                    self._block_hook(gate_index, block_index), with_kwargs=True
                )
                self._handles.append(handle)
            for block_index, block in enumerate(transformer.blocks):
                object.__setattr__(block, "_spatial_controller_ref", weakref.ref(self))
                block._spatial_block_index = int(block_index)
            return self

        def prepare_context(
            self,
            warp_latents=None,
            visibility_tokens=None,
            plucker_tokens=None,
            warp_confidence_tokens=None,
            memory_warp_latents=None,
            memory_visibility_tokens=None,
            memory_confidence_tokens=None,
            spatial_attention_enabled=False,
            *,
            anchor_enabled=True,
            camera_enabled=True,
            spatial_warp_enabled=True,
            stage_contexts=None,
        ):
            transformer = self._transformer_ref()
            if transformer is None:
                raise RuntimeError("the attached transformer no longer exists")
            patch = transformer.patch_embedding
            if any(parameter.requires_grad for parameter in patch.parameters()):
                raise RuntimeError("target/anchor patch_embedding must remain frozen")
            raw_contexts = stage_contexts or [{
                "warp_latents": warp_latents,
                "visibility_tokens": visibility_tokens,
                "plucker_tokens": plucker_tokens,
                "warp_confidence_tokens": warp_confidence_tokens,
                "memory_warp_latents": memory_warp_latents,
                "memory_visibility_tokens": memory_visibility_tokens,
                "memory_confidence_tokens": memory_confidence_tokens,
                "spatial_attention_enabled": spatial_attention_enabled,
            }]
            contexts = {}
            for raw in raw_contexts:
                stage_warp = raw["warp_latents"]
                warp = stage_warp.to(device=patch.weight.device, dtype=patch.weight.dtype)
                warp_tokens = patch(warp).flatten(2).transpose(1, 2).detach()
                stage_visibility = raw["visibility_tokens"].to(
                    device=warp_tokens.device, dtype=torch.float32
                )
                stage_plucker = raw["plucker_tokens"].reshape(
                    raw["plucker_tokens"].shape[0], -1, 6
                ).to(warp_tokens.device)
                if (
                    warp_tokens.shape[:2] != stage_visibility.shape[:2]
                    or warp_tokens.shape[:2] != stage_plucker.shape[:2]
                ):
                    raise ValueError(
                        f"target/warp/visibility/Plucker token mismatch: {tuple(warp_tokens.shape)}, "
                        f"{tuple(stage_visibility.shape)}, {tuple(stage_plucker.shape)}"
                    )
                count = int(warp_tokens.shape[1])
                if count in contexts:
                    raise ValueError(f"duplicate spatial stage token count: {count}")
                raw_warp_confidence = raw.get("warp_confidence_tokens")
                warp_confidence = (
                    stage_visibility if raw_warp_confidence is None else raw_warp_confidence
                ).to(
                    device=warp_tokens.device, dtype=torch.float32,
                )
                memory_latents = raw.get("memory_warp_latents")
                memory_tokens = None
                memory_visibility = None
                memory_confidence = None
                if memory_latents is not None:
                    memory = memory_latents.to(device=patch.weight.device, dtype=patch.weight.dtype)
                    memory_tokens = patch(memory).flatten(2).transpose(1, 2).detach()
                    memory_visibility = raw["memory_visibility_tokens"].to(
                        device=memory_tokens.device, dtype=torch.float32,
                    )
                    raw_memory_confidence = raw.get("memory_confidence_tokens")
                    memory_confidence = (
                        memory_visibility if raw_memory_confidence is None else raw_memory_confidence
                    ).to(device=memory_tokens.device, dtype=torch.float32)
                    expected_memory = (warp_tokens.shape[0], 540, warp_tokens.shape[2])
                    if tuple(memory_tokens.shape) != expected_memory:
                        raise ValueError(
                            f"memory warp must provide fixed 540 stage0 tokens, got {tuple(memory_tokens.shape)}"
                        )
                    if warp_tokens.shape[1] != 540:
                        raise ValueError("W/R spatial attention is restricted to stage0 target count 540")
                    for label, value in (
                        ("warp confidence", warp_confidence),
                        ("memory visibility", memory_visibility),
                        ("memory confidence", memory_confidence),
                    ):
                        if value.shape[:2] != warp_tokens.shape[:2] or value.shape[-1] != 1:
                            raise ValueError(f"{label} token mismatch: {tuple(value.shape)}")
                contexts[count] = ReanchorContext(
                    warp_tokens=warp_tokens,
                    visibility_tokens=stage_visibility,
                    plucker_tokens=stage_plucker,
                    target_token_count=count,
                    warp_latent_frames=int(stage_warp.shape[2]),
                    anchor_enabled=bool(anchor_enabled),
                    camera_enabled=bool(camera_enabled),
                    spatial_warp_enabled=bool(spatial_warp_enabled),
                    warp_confidence_tokens=warp_confidence,
                    memory_tokens=memory_tokens,
                    memory_visibility_tokens=memory_visibility,
                    memory_confidence_tokens=memory_confidence,
                    spatial_attention_enabled=bool(raw.get("spatial_attention_enabled", memory_tokens is not None)),
                )
            self._contexts = contexts
            self._context = next(iter(contexts.values())) if len(contexts) == 1 else None
            self.last_metrics = {}

        def metrics_snapshot(self):
            """Materialize detached GPU scalars only at an explicit log boundary."""
            return {
                key: float(value.detach().cpu()) if hasattr(value, "detach") else float(value)
                for key, value in self.last_metrics.items()
            }

        def clear_context(self):
            self._context = None
            self._contexts = {}

        @staticmethod
        def _apply_rotary(hidden, frequencies):
            first, second = hidden.unflatten(-1, (-1, 2)).unbind(-1)
            cosine, sine = frequencies.unsqueeze(-2).chunk(2, dim=-1)
            output = torch.empty_like(hidden)
            output[..., 0::2] = first * cosine[..., 0::2] - second * sine[..., 1::2]
            output[..., 1::2] = first * sine[..., 1::2] + second * cosine[..., 0::2]
            return output

        def spatial_attention(self, block, hidden, rotary_emb, original_context_length):
            """Second attention pass over fixed [current-W | retrieved-R] temporary K/V."""
            block_index = int(block._spatial_block_index)
            count = int(original_context_length)
            context = self._contexts.get(count) if self._context is None else self._context
            if (
                context is None or not context.spatial_attention_enabled
                or context.memory_tokens is None
            ):
                return hidden
            if count != 540 or hidden.shape[1] < count:
                raise RuntimeError(f"spatial target must be the final 540 tokens, got {count}")
            target = hidden[:, -count:]
            warp = context.warp_tokens + self.spatial_warp_role.to(
                device=target.device, dtype=target.dtype,
            ).view(1, 1, -1)
            memory = context.memory_tokens + self.spatial_memory_role.to(
                device=target.device, dtype=target.dtype,
            ).view(1, 1, -1)
            spatial = torch.cat((warp, memory), dim=1).to(dtype=target.dtype)
            if spatial.shape[1] != 1080:
                raise RuntimeError(f"fixed spatial K/V must contain 1080 tokens, got {spatial.shape[1]}")
            attn = block.attn1
            query = attn.norm_q(attn.to_q(target))
            key = attn.norm_k(
                attn.to_k(spatial) + self.spatial_k_lora[block_index](spatial).to(spatial.dtype)
            )
            value = attn.to_v(spatial) + self.spatial_v_lora[block_index](spatial).to(spatial.dtype)
            query = query.unflatten(2, (attn.heads, -1))
            key = key.unflatten(2, (attn.heads, -1))
            value = value.unflatten(2, (attn.heads, -1))
            if rotary_emb is not None:
                target_rotary = rotary_emb[:, -count:]
                query = self._apply_rotary(query, target_rotary)
                key = self._apply_rotary(key, torch.cat((target_rotary, target_rotary), dim=1))
            query = query.transpose(1, 2)
            key = key.transpose(1, 2)
            value = value.transpose(1, 2)
            visibility = torch.cat(
                (context.visibility_tokens, context.memory_visibility_tokens), dim=1,
            ).to(device=target.device, dtype=torch.float32)
            confidence = torch.cat(
                (context.warp_confidence_tokens, context.memory_confidence_tokens), dim=1,
            ).to(device=target.device, dtype=torch.float32).clamp(0, 1)
            valid = visibility.squeeze(-1) > 0
            key_bias = torch.log(confidence.squeeze(-1).clamp_min(1e-6))[:, None, None, :]
            key_bias = key_bias.masked_fill(~valid[:, None, None, :], -1.0e4)
            has_support = valid.any(dim=-1).view(-1, 1, 1, 1)
            delta = torch.nn.functional.scaled_dot_product_attention(
                query, key, value, attn_mask=key_bias, dropout_p=0.0, is_causal=False,
            ) * has_support
            delta = delta.transpose(1, 2).flatten(2)
            delta = attn.to_out[1](attn.to_out[0](delta))
            gate = self.spatial_gate[block_index].float()
            updated_target = target + (gate * delta.float()).to(target.dtype)
            self.last_metrics[f"spatial_gate_block{block_index}"] = gate.detach()
            self.last_metrics[f"spatial_delta_ratio_block{block_index}"] = (
                delta.float().norm(dim=-1).mean()
                / target.float().norm(dim=-1).mean().clamp_min(1e-8)
            ).detach()
            if self.collect_spatial_metrics:
                with torch.no_grad():
                    scores = torch.matmul(query.float(), key.float().transpose(-1, -2)) * (
                        query.shape[-1] ** -0.5
                    ) + key_bias
                    weights = torch.softmax(scores, dim=-1) * valid[:, None, None, :]
                    weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
                    self.last_metrics[f"warp_attention_mass_block{block_index}"] = (
                        weights[..., :540].sum(dim=-1).mean().detach()
                    )
                    self.last_metrics[f"memory_attention_mass_block{block_index}"] = (
                        weights[..., 540:].sum(dim=-1).mean().detach()
                    )
            self.last_metrics["warp_valid_token_count"] = valid[:, :540].sum().detach()
            self.last_metrics["memory_valid_token_count"] = valid[:, 540:].sum().detach()
            return torch.cat((hidden[:, :-count], updated_target), dim=1)

        def _context_for_hidden(self, hidden, args, kwargs):
            if self._context is not None:
                return self._context
            if not self._contexts:
                return None
            target_count = kwargs.get("original_context_length")
            if target_count is None:
                lengths = kwargs.get("original_context_length_list")
                if lengths is not None:
                    target_count = sum(int(value) for value in lengths)
            if target_count is None:
                # The train_exact transformer places original_context_length at
                # positional index 6, while the official inference transformer
                # has a shorter block signature and places it at index 4.
                # Select only a scalar integer that names one of our stage token
                # counts so attention masks cannot be mistaken for the length.
                for value in args[4:]:
                    if isinstance(value, int) and int(value) in self._contexts:
                        target_count = int(value)
                        break
            if target_count is None:
                raise RuntimeError(
                    "multiple spatial pyramid contexts require the transformer's target token count"
                )
            context = self._contexts.get(int(target_count))
            if context is None:
                raise RuntimeError(
                    f"no spatial pyramid context for target token count {target_count}; "
                    f"available={sorted(self._contexts)} hidden={tuple(hidden.shape)}"
                )
            return context

        def _patch_short_hook(self, _module, _args, output):
            contexts = list(self._contexts.values())
            if not contexts or not any(context.spatial_warp_enabled for context in contexts):
                return output
            warp_latent_frames = {context.warp_latent_frames for context in contexts}
            if len(warp_latent_frames) != 1:
                raise RuntimeError("all spatial pyramid contexts must use the same latent T")
            frame_count = next(iter(warp_latent_frames))
            if output.ndim != 5 or output.shape[1] != self.hidden_size:
                raise RuntimeError(f"unexpected patch_short output {tuple(output.shape)}")
            if output.shape[2] < frame_count:
                # Official pyramid inference injects the current warp only in
                # stage 0. Later stages call patch_short for the shorter normal
                # history, so there is no SPATIAL_WARP suffix to tag there.
                return output
            result = output.clone()
            role = self.spatial_warp_role.to(device=result.device, dtype=result.dtype).view(1, -1, 1, 1, 1)
            result[:, :, -frame_count:] = result[:, :, -frame_count:] + role
            return result

        def _block_hook(self, gate_index, block_index):
            def hook(_module, args, kwargs):
                hidden = args[0]
                context = self._context_for_hidden(hidden, args, kwargs)
                if context is None:
                    return args, kwargs
                count = context.target_token_count
                if hidden.shape[1] < count:
                    raise RuntimeError("transformer hidden state is shorter than target token count")
                target = hidden[:, -count:]
                base_norm = target.float().norm(dim=-1).mean().clamp_min(1e-8)
                if block_index == 0 and context.camera_enabled:
                    camera_delta = self.camera_adapter(context.plucker_tokens)
                    target = target + (self.camera_gate.float() * camera_delta).to(target.dtype)
                    self.last_metrics["camera_ratio"] = (
                        (self.camera_gate.float() * camera_delta).norm(dim=-1).mean() / base_norm
                    ).detach()
                if context.anchor_enabled:
                    anchor_delta = self.anchor_adapter(target, context.warp_tokens)
                    contribution = self.anchor_gates[gate_index].float() * context.visibility_tokens * anchor_delta
                    target = target + contribution.to(target.dtype)
                    self.last_metrics[f"anchor_ratio_block{block_index}"] = (
                        contribution.norm(dim=-1).mean() / base_norm
                    ).detach()
                updated = torch.cat((hidden[:, :-count], target), dim=1)
                return (updated, *args[1:]), kwargs
            return hook

    return Controller()


def install_spatial_reanchor(
    transformer, *, rank=64, refresh_blocks=(0, 10, 20, 30), gate_init=0.05,
    spatial_rank=8,
):
    hidden_size = int(transformer.patch_embedding.out_channels)
    for parameter in transformer.patch_embedding.parameters():
        parameter.requires_grad_(False)
    controller = build_spatial_reanchor_controller(
        hidden_size, rank=rank, refresh_blocks=refresh_blocks, gate_init=gate_init,
        spatial_rank=spatial_rank, block_count=len(transformer.blocks),
    )
    transformer.add_module("spatial_reanchor", controller)
    controller.install(transformer)
    return controller
