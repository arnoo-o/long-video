import numpy as np
import pytest


torch = pytest.importorskip("torch")

from long_video.wah.spatial_reanchor import (
    build_spatial_reanchor_controller,
    install_spatial_reanchor,
    plucker_camera_rays,
    visibility_to_target_tokens,
)


def _cameras():
    poses = np.repeat(np.eye(4, dtype=np.float32)[None], 33, axis=0)
    poses[:, 0, 3] = np.linspace(0, 1, 33)
    intrinsics = np.repeat(
        np.array([[[320, 0, 319.5], [0, 320, 191.5], [0, 0, 1]]], np.float32), 33, axis=0
    )
    return poses, intrinsics


def test_plucker_contract_and_canonical_origin():
    poses, intrinsics = _cameras()
    rays = plucker_camera_rays(
        poses, intrinsics, image_height=384, image_width=640,
        token_height=24, token_width=40, latent_frames=9,
        temporal_scale=4, scene_scale=1.0,
    )
    assert rays.shape == (1, 9, 24, 40, 6)
    torch.testing.assert_close(rays[0, 0, ..., 3:], torch.zeros_like(rays[0, 0, ..., 3:]))
    torch.testing.assert_close(rays[..., :3].norm(dim=-1), torch.ones_like(rays[..., 0]))


def test_plucker_rejects_noncanonical_first_pose():
    poses, intrinsics = _cameras()
    poses[0, 2, 3] = 1
    with pytest.raises(ValueError, match="identity"):
        plucker_camera_rays(
            poses, intrinsics, image_height=384, image_width=640,
            token_height=24, token_width=40, latent_frames=9,
            temporal_scale=4, scene_scale=1.0,
        )


def test_visibility_mapping_uses_exact_vae_temporal_groups():
    visibility = np.ones((33, 384, 640), np.float32)
    visibility[1:5] = 0
    visibility[5:9] = 0.5
    tokens = visibility_to_target_tokens(
        visibility, latent_frames=9, latent_height=48, latent_width=80,
        patch_height=2, patch_width=2, temporal_scale=4,
    ).reshape(1, 9, 24, 40, 1)
    assert tokens.shape == (1, 9, 24, 40, 1)
    assert torch.equal(tokens[:, 1], torch.zeros_like(tokens[:, 1]))
    torch.testing.assert_close(tokens[:, 2], torch.full_like(tokens[:, 2], 0.5))


def test_plucker_temporal_group_is_not_endpoint_sampling():
    poses, intrinsics = _cameras()
    poses[1:5, 0, 3] = np.array([1.0, 2.0, 3.0, 4.0])
    rays = plucker_camera_rays(
        poses, intrinsics, image_height=384, image_width=640,
        token_height=1, token_width=1, latent_frames=9,
        temporal_scale=4, scene_scale=1.0,
    )
    direction = rays[0, 1, 0, 0, :3]
    expected_origin = torch.tensor([2.5, 0.0, 0.0])
    expected_moment = torch.cross(expected_origin, direction, dim=-1)
    torch.testing.assert_close(rays[0, 1, 0, 0, 3:], expected_moment)


def test_adapter_invisible_contribution_is_exactly_zero():
    controller = build_spatial_reanchor_controller(32, rank=8, refresh_blocks=(0,), gate_init=0.05)
    target = torch.randn(1, 12, 32)
    warp = torch.randn(1, 12, 32)
    visibility = torch.zeros(1, 12, 1)
    delta = controller.anchor_adapter(target, warp)
    contribution = controller.anchor_gates[0] * visibility * delta
    assert torch.equal(contribution, torch.zeros_like(contribution))
    contribution.sum().backward()
    assert controller.anchor_gates.grad is not None


class _Block(torch.nn.Module):
    def forward(
        self, hidden_states, encoder_hidden_states, temb, rotary_emb,
        navit_hidden_attention_mask=None, navit_encoder_attention_mask=None,
        original_context_length=None, original_context_length_list=None,
        is_first_denoising_step=False, attention_kwargs=None,
    ):
        return hidden_states


class _Transformer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_embedding = torch.nn.Conv3d(16, 32, (1, 2, 2), (1, 2, 2))
        self.patch_short = torch.nn.Conv3d(16, 32, (1, 2, 2), (1, 2, 2))
        self.blocks = torch.nn.ModuleList([_Block() for _ in range(4)])


def test_hooks_use_frozen_target_patch_and_spatial_role_only_on_warp():
    transformer = _Transformer()
    controller = install_spatial_reanchor(
        transformer, rank=8, refresh_blocks=(0, 1, 2, 3), gate_init=0.05,
    )
    controller.spatial_warp_role.data.fill_(0.25)
    warp = torch.randn(1, 16, 9, 48, 80)
    visibility = torch.ones(1, 8640, 1)
    rays = torch.randn(1, 9, 24, 40, 6)
    controller.prepare_context(warp, visibility, rays)
    short = torch.randn(1, 16, 10, 48, 80)
    raw = transformer.patch_short._conv_forward(short, transformer.patch_short.weight, transformer.patch_short.bias)
    patched = transformer.patch_short(short)
    torch.testing.assert_close(patched[:, :, :1], raw[:, :, :1])
    torch.testing.assert_close(patched[:, :, 1:], raw[:, :, 1:] + 0.25)
    hidden = torch.randn(1, 8700, 32)
    for block in transformer.blocks:
        hidden = block(hidden, None, None, None, None, None, 8640, [8640], False, None)
    (hidden.sum() + patched.sum()).backward()
    assert all(parameter.grad is None for parameter in transformer.patch_embedding.parameters())
    assert controller.camera_gate.grad is not None
    assert controller.anchor_gates.grad is not None
    assert controller.spatial_warp_role.grad is not None


def test_canonical_plucker_is_independent_of_memory_node_switch():
    poses, intrinsics = _cameras()
    first = plucker_camera_rays(
        poses, intrinsics, image_height=384, image_width=640,
        token_height=24, token_width=40, latent_frames=9,
        temporal_scale=4, scene_scale=1.0,
    )
    for _node_id in ("M0", "M1", "M2"):
        current = plucker_camera_rays(
            poses, intrinsics, image_height=384, image_width=640,
            token_height=24, token_width=40, latent_frames=9,
            temporal_scale=4, scene_scale=1.0,
        )
        torch.testing.assert_close(current, first)
