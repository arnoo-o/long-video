import numpy as np
import pytest


torch = pytest.importorskip("torch")

from long_video.wah.spatial_reanchor import (
    build_spatial_reanchor_controller,
    install_spatial_reanchor,
    plucker_camera_rays,
    resize_latents_spatial,
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
    orthogonality = (rays[..., :3] * rays[..., 3:]).sum(dim=-1)
    torch.testing.assert_close(orthogonality, torch.zeros_like(orthogonality), atol=1e-6, rtol=0)


def test_plucker_rejects_noncanonical_first_pose():
    poses, intrinsics = _cameras()
    poses[0, 2, 3] = 1
    with pytest.raises(ValueError, match="identity"):
        plucker_camera_rays(
            poses, intrinsics, image_height=384, image_width=640,
            token_height=24, token_width=40, latent_frames=9,
            temporal_scale=4, scene_scale=1.0,
        )


def test_later_chunk_keeps_sequence_canonical_frame_without_identity_requirement():
    poses, intrinsics = _cameras()
    chunk = poses.copy()
    chunk[:, 0, 3] += 3.0
    rays = plucker_camera_rays(
        chunk, intrinsics, image_height=384, image_width=640,
        token_height=24, token_width=40, latent_frames=9,
        temporal_scale=4, scene_scale=1.0, sequence_frame_start=32,
    )
    assert rays.shape == (1, 9, 24, 40, 6)
    assert not torch.equal(rays[0, 0, ..., 3:], torch.zeros_like(rays[0, 0, ..., 3:]))
    for _node_id in ("M0", "M1", "M2"):
        current = plucker_camera_rays(
            chunk, intrinsics, image_height=384, image_width=640,
            token_height=24, token_width=40, latent_frames=9,
            temporal_scale=4, scene_scale=1.0, sequence_frame_start=32,
        )
        torch.testing.assert_close(current, rays)


def test_local_window_at_zero_does_not_claim_full_sequence_source_origin():
    poses, intrinsics = _cameras()
    poses[:, 0, 3] += 3.0
    rays = plucker_camera_rays(
        poses, intrinsics, image_height=384, image_width=640,
        token_height=24, token_width=40, latent_frames=9,
        temporal_scale=4, scene_scale=1.0, sequence_frame_start=0,
        validate_sequence_source_origin=False,
    )
    assert rays.shape == (1, 9, 24, 40, 6)
    assert not torch.equal(rays[0, 0, ..., 3:], torch.zeros_like(rays[0, 0, ..., 3:]))


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


def test_plucker_temporal_group_uses_pose_slerp_and_mean_translation():
    poses, intrinsics = _cameras()
    poses[1:5, 0, 3] = np.array([1.0, 2.0, 3.0, 4.0])
    angles = np.deg2rad([0.0, 0.0, 0.0, 90.0])
    for frame, angle in zip(range(1, 5), angles):
        poses[frame, :3, :3] = np.array([
            [np.cos(angle), 0.0, np.sin(angle)],
            [0.0, 1.0, 0.0],
            [-np.sin(angle), 0.0, np.cos(angle)],
        ], dtype=np.float32)
    rays = plucker_camera_rays(
        poses, intrinsics, image_height=384, image_width=640,
        token_height=1, token_width=1, latent_frames=9,
        temporal_scale=4, scene_scale=1.0,
    )
    direction = rays[0, 1, 0, 0, :3]
    pixel = torch.tensor([320.0, 192.0, 1.0])
    intrinsic = torch.as_tensor(intrinsics[0])
    camera_direction = torch.nn.functional.normalize(torch.linalg.solve(intrinsic, pixel), dim=0)
    angle = np.deg2rad(45.0)
    expected_rotation = torch.tensor([
        [np.cos(angle), 0.0, np.sin(angle)],
        [0.0, 1.0, 0.0],
        [-np.sin(angle), 0.0, np.cos(angle)],
    ], dtype=torch.float32)
    expected_direction = torch.nn.functional.normalize(expected_rotation @ camera_direction, dim=0)
    torch.testing.assert_close(direction, expected_direction, atol=1e-5, rtol=0)
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
    def __init__(self):
        super().__init__()
        self.norm1 = torch.nn.RMSNorm(32)
        self.attn1 = _Attention()

    def forward(
        self, hidden_states, encoder_hidden_states, temb, rotary_emb,
        navit_hidden_attention_mask=None, navit_encoder_attention_mask=None,
        original_context_length=None, original_context_length_list=None,
        is_first_denoising_step=False, attention_kwargs=None,
    ):
        return hidden_states


class _Attention(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.heads = 4
        self.to_q = torch.nn.Linear(32, 32)
        self.to_k = torch.nn.Linear(32, 32)
        self.to_v = torch.nn.Linear(32, 32)
        self.norm_q = torch.nn.RMSNorm(32)
        self.norm_k = torch.nn.RMSNorm(32)
        self.to_out = torch.nn.ModuleList([torch.nn.Linear(32, 32), torch.nn.Dropout(0.0)])


class _Transformer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_embedding = torch.nn.Conv3d(16, 32, (1, 2, 2), (1, 2, 2))
        self.patch_short = torch.nn.Conv3d(16, 32, (1, 2, 2), (1, 2, 2))
        self.blocks = torch.nn.ModuleList([_Block() for _ in range(4)])


@pytest.mark.parametrize(
    ("stage_id", "latent_height", "latent_width", "token_count"),
    [(0, 12, 20, 540), (1, 24, 40, 2160), (2, 48, 80, 8640)],
)
def test_all_train_exact_pyramid_stage_shapes_align(
    stage_id, latent_height, latent_width, token_count,
):
    transformer = _Transformer()
    controller = install_spatial_reanchor(
        transformer, rank=8, refresh_blocks=(0, 1, 2, 3), gate_init=0.05,
    )
    poses, intrinsics = _cameras()
    full_warp = torch.randn(1, 16, 9, 48, 80)
    stage_warp = resize_latents_spatial(
        full_warp, height=latent_height, width=latent_width,
    )
    visibility = visibility_to_target_tokens(
        np.ones((33, 384, 640), np.float32), latent_frames=9,
        latent_height=latent_height, latent_width=latent_width,
        patch_height=2, patch_width=2, temporal_scale=4,
    )
    rays = plucker_camera_rays(
        poses, intrinsics, image_height=384, image_width=640,
        token_height=latent_height // 2, token_width=latent_width // 2,
        latent_frames=9, temporal_scale=4, scene_scale=1.0,
    )
    controller.prepare_context(stage_warp, visibility, rays)
    expected_tokens = transformer.patch_embedding(stage_warp).flatten(2).transpose(1, 2).detach()
    assert tuple(stage_warp.shape) == (1, 16, 9, latent_height, latent_width)
    assert tuple(visibility.shape) == (1, token_count, 1)
    assert tuple(rays.shape) == (1, 9, latent_height // 2, latent_width // 2, 6)
    assert controller._context.target_token_count == token_count
    torch.testing.assert_close(controller._context.warp_tokens, expected_tokens)
    hidden = torch.randn(1, token_count + 20, 32)
    hidden = transformer.blocks[0](
        hidden, None, None, None,
        original_context_length=token_count,
        original_context_length_list=[token_count],
    )
    assert tuple(hidden.shape) == (1, token_count + 20, 32)


def test_pyramid_context_selects_exact_target_token_count():
    transformer = _Transformer()
    controller = install_spatial_reanchor(
        transformer, rank=8, refresh_blocks=(0, 1, 2, 3), gate_init=0.05,
    )
    contexts = []
    for latent_height, latent_width in ((12, 20), (24, 40), (48, 80)):
        token_height, token_width = latent_height // 2, latent_width // 2
        contexts.append({
            "warp_latents": torch.randn(1, 16, 9, latent_height, latent_width),
            "visibility_tokens": torch.ones(1, 9 * token_height * token_width, 1),
            "plucker_tokens": torch.randn(1, 9, token_height, token_width, 6),
        })
    controller.prepare_context(stage_contexts=contexts)
    assert sorted(controller._contexts) == [540, 2160, 8640]
    for token_count in (540, 2160, 8640):
        hidden = torch.randn(1, token_count + 20, 32)
        output = transformer.blocks[0](
            hidden, None, None, None,
            original_context_length=token_count,
            original_context_length_list=[token_count],
        )
        assert tuple(output.shape) == tuple(hidden.shape)
    with pytest.raises(RuntimeError, match="no spatial pyramid context"):
        transformer.blocks[0](
            torch.randn(1, 110, 32), None, None, None,
            original_context_length=90,
            original_context_length_list=[90],
        )


def test_pyramid_context_accepts_inference_positional_context_length():
    transformer = _Transformer()
    controller = install_spatial_reanchor(
        transformer, rank=8, refresh_blocks=(0, 1, 2, 3), gate_init=0.05,
    )
    contexts = []
    for latent_height, latent_width in ((12, 20), (24, 40), (48, 80)):
        token_height, token_width = latent_height // 2, latent_width // 2
        contexts.append({
            "warp_latents": torch.randn(1, 16, 9, latent_height, latent_width),
            "visibility_tokens": torch.ones(1, 9 * token_height * token_width, 1),
            "plucker_tokens": torch.randn(1, 9, token_height, token_width, 6),
        })
    controller.prepare_context(stage_contexts=contexts)
    for token_count in (540, 2160, 8640):
        hidden = torch.randn(1, token_count + 20, 32)
        context = controller._context_for_hidden(
            hidden, (hidden, None, None, None, token_count), {},
        )
        assert context.target_token_count == token_count


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
    later_stage_short = torch.randn(1, 16, 2, 48, 80)
    later_stage_raw = transformer.patch_short._conv_forward(
        later_stage_short, transformer.patch_short.weight, transformer.patch_short.bias,
    )
    torch.testing.assert_close(transformer.patch_short(later_stage_short), later_stage_raw)
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


def test_second_spatial_attention_has_fixed_slots_hard_memory_mask_and_zero_gate_identity():
    transformer = _Transformer()
    controller = install_spatial_reanchor(
        transformer, rank=8, refresh_blocks=(0, 1, 2, 3), gate_init=0.05,
        spatial_rank=8,
    )
    controller.collect_spatial_metrics = True
    warp = torch.randn(1, 16, 9, 12, 20)
    memory = torch.randn_like(warp)
    visibility = torch.ones(1, 540, 1)
    memory_visibility = torch.zeros(1, 540, 1)
    rays = torch.randn(1, 9, 6, 10, 6)
    controller.prepare_context(
        warp, visibility, rays,
        stage_contexts=[{
            "warp_latents": warp, "visibility_tokens": visibility,
            "warp_confidence_tokens": visibility, "plucker_tokens": rays,
            "memory_warp_latents": memory,
            "memory_visibility_tokens": memory_visibility,
            "memory_confidence_tokens": memory_visibility,
            "spatial_attention_enabled": True,
        }],
    )
    hidden = torch.randn(1, 560, 32)
    rotary = torch.randn(1, 560, 16)
    # Formal train_exact keeps short history at full latent resolution even
    # while the supervised target is stage0; the hook must still capture a
    # complete 540-token stage0 CURRENT_WARP before visible-token dropping.
    transformer.patch_short(torch.randn(1, 16, 9, 48, 80))
    scale_msa = torch.zeros_like(hidden)
    shift_msa = torch.zeros_like(hidden)
    output = controller.spatial_attention(
        transformer.blocks[0], hidden, rotary, 540, scale_msa, shift_msa,
    )
    assert torch.equal(output, hidden)
    assert tuple(controller._context.spatial_warp_tokens.shape) == (1, 540, 32)
    assert tuple(controller._context.memory_tokens.shape) == (1, 540, 32)
    assert int(controller.last_metrics["warp_valid_token_count"]) == 540
    assert int(controller.last_metrics["memory_valid_token_count"]) == 0
    torch.testing.assert_close(
        controller.last_metrics["memory_attention_mass_block0"], torch.tensor(0.0),
    )
    controller.spatial_gate.data[0] = 0.1
    changed = controller.spatial_attention(
        transformer.blocks[0], hidden, rotary, 540, scale_msa, shift_msa,
    )
    assert tuple(changed.shape) == tuple(hidden.shape)
    torch.testing.assert_close(changed[:, :-540], hidden[:, :-540])
    assert not torch.equal(changed[:, -540:], hidden[:, -540:])
    changed[:, -540:].sum().backward()
    assert controller.spatial_k_lora[0].up.weight.grad is not None
    assert controller.spatial_v_lora[0].up.weight.grad is not None


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
