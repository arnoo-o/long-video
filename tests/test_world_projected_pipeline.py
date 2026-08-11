import torch
import pytest
from types import SimpleNamespace

from long_video.wah.world_projected_pipeline import (
    DEFAULT_TEMPORAL_WARMUP,
    WORLD_OWNERSHIP_COVERAGE_THRESHOLD,
    DelayedNodeActivationQueue,
    WorldProjectedWarpAsHistoryPipeline,
    WorldProjectionConfig,
    apply_world_and_boundary_projection,
    apply_world_projection,
    apply_residual_boundary_bridge,
    apply_boundary_then_world_clamp,
    apply_previous_world_boundary,
    build_canonical_world_support,
    build_canonical_world_pyramid,
    build_world_projection_context,
    build_boundary_state_at_sigma,
    build_world_state_at_sigma,
    canonical_support_to_tokens,
    compose_canonical_residual,
    fill_invalid_warp_for_vae,
    encode_canonical_video_latents,
    posterior_mode_or_mean,
    sparse_pixel_constraint_enabled,
    scheduler_align_clean_prediction,
    scheduler_clean_prediction,
    sparse_pixel_constraint,
    temporarily_offload_frozen_transformer_blocks,
    mask_canonical_latent,
    smooth_latent_visibility,
    world_projection_weight,
)


def test_sparse_pixel_constraint_is_stage2_only():
    assert not sparse_pixel_constraint_enabled(0, 3)
    assert not sparse_pixel_constraint_enabled(1, 3)
    assert sparse_pixel_constraint_enabled(2, 3)


def test_clean_prediction_and_native_scheduler_alignment_are_used():
    class FakeScheduler:
        def __init__(self):
            self.convert_calls = 0
            self.noise_calls = 0

        def convert_flow_pred_to_x0(self, *, flow_pred, xt, timestep, sigmas, timesteps):
            self.convert_calls += 1
            return xt - flow_pred

        def add_noise(self, clean, noise, timestep, *, sigmas, timesteps):
            self.noise_calls += 1
            return clean + 0.25 * noise

    scheduler = FakeScheduler()
    sample = torch.full((1, 2, 1, 1, 1), 4.0)
    flow = torch.ones_like(sample)
    sigmas = torch.tensor([1.0, 0.5, 0.0])
    timesteps = torch.tensor([999, 500, 0])
    clean = scheduler_clean_prediction(
        scheduler, flow, torch.tensor(999), sample,
        dmd_sigmas=sigmas, dmd_timesteps=timesteps,
    )
    torch.testing.assert_close(clean, torch.full_like(clean, 3.0))
    aligned = scheduler_align_clean_prediction(
        scheduler, clean, step_index=0, dmd_noisy_tensor=torch.full_like(clean, 8.0),
        dmd_sigmas=sigmas, dmd_timesteps=timesteps, all_timesteps=timesteps[:2],
    )
    torch.testing.assert_close(aligned, torch.full_like(clean, 5.0))
    assert scheduler.convert_calls == 1
    assert scheduler.noise_calls == 1


def test_sparse_pixel_constraint_is_joint_and_only_updates_latent():
    base = torch.zeros(1, 1, 2, 1, 2)
    warp = torch.tensor([[[[[1.0, 5.0]], [[3.0, 7.0]]]]]).expand(1, 3, 2, 1, 2)
    visibility = torch.tensor([[[[[1.0, 0.0]], [[1.0, 0.0]]]]])
    decode_calls = []

    def decode(value):
        decode_calls.append(tuple(value.shape))
        return value.expand(1, 3, 2, 1, 2)

    optimized, metrics = sparse_pixel_constraint(
        base, decode_fn=decode, warp_rgb=warp, visibility=visibility,
        steps=1, lr=0.005, lambda_z=1.0, max_grad_norm=1.0,
    )
    assert decode_calls == [tuple(base.shape)]
    assert optimized.grad_fn is None
    assert not optimized.requires_grad
    assert float(optimized[..., 0].abs().sum()) > 0.0
    assert float(optimized[..., 1].abs().sum()) == 0.0
    assert float(metrics["sparse_clipped_grad_norm"]) <= 1.0 + 1e-6


def test_sparse_pixel_constraint_rejects_soft_visibility():
    base = torch.zeros(1, 1, 1, 1, 1)
    with pytest.raises(ValueError):
        sparse_pixel_constraint(
            base,
            decode_fn=lambda value: value.expand(1, 3, 1, 1, 1),
            warp_rgb=torch.zeros(1, 3, 1, 1, 1),
            visibility=torch.full((1, 1, 1, 1, 1), 0.5),
            steps=1, lr=0.005, lambda_z=1.0, max_grad_norm=1.0,
        )


def test_sparse_constraint_block_offload_requires_frozen_and_restores():
    transformer = torch.nn.Module()
    transformer.blocks = torch.nn.ModuleList([torch.nn.Linear(2, 2) for _ in range(3)])
    for parameter in transformer.parameters():
        parameter.requires_grad_(False)
    with temporarily_offload_frozen_transformer_blocks(
        transformer, restore_device=torch.device("cpu"), block_count=2,
    ) as count:
        assert count == 2
        assert all(parameter.device.type == "cpu" for parameter in transformer.parameters())
    assert all(parameter.device.type == "cpu" for parameter in transformer.parameters())

    next(transformer.blocks[-1].parameters()).requires_grad_(True)
    with pytest.raises(RuntimeError):
        with temporarily_offload_frozen_transformer_blocks(
            transformer, restore_device=torch.device("cpu"), block_count=1,
        ):
            pass


def test_canonical_residual_composition_gives_world_exclusive_known_pixels():
    endpoint = torch.full((1, 2, 2, 1, 3), 4.0)
    residual = torch.full_like(endpoint, 10.0)
    support = torch.tensor([[[[[1.0, 0.0, 0.25]], [[1.0, 0.0, 0.25]]]]])
    composed, base = compose_canonical_residual(endpoint, support, residual)
    torch.testing.assert_close(base[..., 0], torch.full_like(base[..., 0], 4.0))
    torch.testing.assert_close(composed[..., 0], torch.full_like(composed[..., 0], 4.0))
    torch.testing.assert_close(composed[..., 1], torch.full_like(composed[..., 1], 10.0))
    torch.testing.assert_close(composed[..., 2], torch.full_like(composed[..., 2], 8.5))


def test_residual_boundary_only_changes_unknown_slot0():
    raw = torch.zeros(1, 2, 9, 1, 2)
    boundary = torch.ones(1, 2, 1, 1, 2)
    support = torch.zeros(1, 1, 9, 1, 2)
    support[:, :, 0, :, 0] = 1.0
    result, metrics = apply_residual_boundary_bridge(
        raw, boundary, support, sigma=0.0, boundary_beta_max=0.2,
    )
    assert float(result[:, :, 0, :, 0].abs().max()) == 0.0
    torch.testing.assert_close(result[:, :, 0, :, 1], torch.full_like(result[:, :, 0, :, 1], 0.2))
    assert float(result[:, :, 1:].abs().max()) == 0.0
    assert float(metrics["projection_delta_ratio"]) == 0.0


def test_boundary_then_world_clamp_has_fixed_order_and_exact_regions():
    raw = torch.zeros(1, 1, 2, 1, 3)
    world = torch.full_like(raw, 8.0)
    support = torch.tensor([[[[[1.0, 0.0, 1.0]], [[1.0, 0.0, 1.0]]]]])
    boundary = torch.full((1, 1, 1, 1, 3), 4.0)
    result, metrics = apply_boundary_then_world_clamp(
        raw, world, support, boundary, sigma=0.0, boundary_beta_max=0.5,
    )
    # Known is exactly world; unknown slot0 receives Boundary before clamp;
    # Ownership is strictly binary.
    torch.testing.assert_close(result[:, :, 0, :, 0], torch.full((1, 1, 1), 8.0))
    torch.testing.assert_close(result[:, :, 0, :, 1], torch.full((1, 1, 1), 2.0))
    torch.testing.assert_close(result[:, :, 0, :, 2], torch.full((1, 1, 1), 8.0))
    assert float(result[:, :, 1, :, 1].abs().max()) == 0.0
    assert float(metrics["unknown_projection_delta_max"]) == 0.0
    assert float(metrics["world_clamp_formula_max_error"]) == 0.0
    assert float(metrics["per_step_world_clamp"]) == 1.0
    assert set(torch.unique(support).tolist()) == {0.0, 1.0}


def test_world_projection_is_exact_identity_in_unknown_region_and_bounded():
    raw = torch.randn(1, 4, 2, 3, 5)
    world = torch.randn_like(raw)
    visibility = torch.zeros(1, 1, 2, 3, 5)
    visibility[..., :2] = 1.0
    confidence = torch.ones_like(visibility)
    projected, metrics = apply_world_projection(
        raw, world, visibility, confidence,
        sigma=0.25, lambda_max=0.5, gamma=1.0,
        confidence_ramp_min=0.2, confidence_ramp_max=0.5,
    )
    unknown = (visibility == 0).expand_as(raw)
    assert torch.equal(projected[unknown], raw[unknown])
    expected_strength = 0.5 * (1.0 - 0.25)
    torch.testing.assert_close(
        projected[~unknown],
        (raw + expected_strength * (world - raw))[~unknown],
    )
    assert float(metrics["unknown_projection_delta_max"]) == 0.0
    assert 0.0 <= float(metrics["projection_mask_ratio"]) <= 1.0


def test_soft_confidence_ramp_and_sigma_schedule_disable_projection():
    raw = torch.randn(1, 2, 1, 2, 2)
    world = torch.randn_like(raw)
    visible = torch.ones(1, 1, 1, 2, 2)
    minimum_confidence = torch.full_like(visible, 0.2)
    ramped_out, _ = apply_world_projection(
        raw, world, visible, minimum_confidence, sigma=0.0,
        confidence_ramp_min=0.2, confidence_ramp_max=0.5,
    )
    early, _ = apply_world_projection(
        raw, world, visible, torch.ones_like(visible), sigma=1.0,
    )
    assert torch.equal(ramped_out, raw)
    assert torch.equal(early, raw)


def test_soft_confidence_ramp_is_linear_and_stage_lambda_can_be_zero():
    visible = torch.ones(1, 1, 1, 1, 3)
    confidence = torch.tensor([[[[[0.2, 0.35, 0.5]]]]])
    weight, schedule = world_projection_weight(
        visible, confidence, sigma=0.0, lambda_max=0.3, gamma=1.0,
        confidence_ramp_min=0.2, confidence_ramp_max=0.5,
    )
    torch.testing.assert_close(weight, torch.tensor([[[[[0.0, 0.15, 0.3]]]]]))
    torch.testing.assert_close(schedule, torch.tensor(0.3))
    disabled, _ = world_projection_weight(
        visible, confidence, sigma=0.0, lambda_max=0.0, gamma=1.0,
        confidence_ramp_min=0.2, confidence_ramp_max=0.5,
    )
    assert torch.count_nonzero(disabled) == 0


def test_world_state_uses_real_stage_start_not_generic_noise():
    start = torch.full((1, 2, 1, 2, 2), 4.0)
    endpoint = torch.full_like(start, 10.0)
    state = build_world_state_at_sigma(
        stage_id=2, current_sigma=0.8, next_sigma=0.25,
        canonical_endpoint=endpoint, stage_start_state=start,
    )
    torch.testing.assert_close(state, torch.full_like(state, 8.5))


def test_three_stage_world_pyramid_and_support_shapes_match():
    clean = torch.randn(1, 16, 9, 48, 80)
    pyramid = build_canonical_world_pyramid(clean, 3)
    assert [tuple(item.shape) for item in pyramid] == [
        (1, 16, 9, 12, 20),
        (1, 16, 9, 24, 40),
        (1, 16, 9, 48, 80),
    ]
    visibility = torch.ones(33, 384, 640)
    confidence = torch.full_like(visibility, 0.75)
    context = build_world_projection_context(
        clean, visibility, confidence,
        config=WorldProjectionConfig(lambda_max_by_stage=(0.0, 0.15, 0.30)),
        previous_clean_boundary_latent=torch.randn(1, 16, 1, 48, 80),
    )
    for latent, visible, conf in zip(
        context.canonical_latents, context.visibility, context.confidence,
    ):
        assert tuple(visible.shape) == (1, 1, 9, latent.shape[-2], latent.shape[-1])
        assert tuple(conf.shape) == tuple(visible.shape)
        assert torch.isfinite(latent).all()
        assert torch.isfinite(visible).all()
        assert torch.isfinite(conf).all()
    assert [tuple(item.shape) for item in context.previous_boundary_latents] == [
        (1, 16, 1, 12, 20),
        (1, 16, 1, 24, 40),
        (1, 16, 1, 48, 80),
    ]


def test_visibility_smoothing_erodes_edges_and_never_expands_unknown_region():
    value = torch.zeros(1, 1, 1, 7, 7)
    value[..., 1:6, 1:6] = 1.0
    softened = smooth_latent_visibility(value)
    assert softened.min() >= 0 and softened.max() <= 1
    assert torch.equal(softened[value == 0], torch.zeros_like(softened[value == 0]))
    assert float(softened[..., 3, 3]) > float(softened[..., 1, 1])


def test_invalid_warp_fill_uses_nearest_visible_without_enabling_support():
    import numpy as np

    rgb = np.zeros((1, 3, 3, 3), np.float32)
    visibility = np.zeros((1, 3, 3), bool)
    visibility[0, 1, 1] = True
    rgb[0, 1, 1] = (0.2, 0.4, 0.6)
    filled = fill_invalid_warp_for_vae(rgb, visibility)
    np.testing.assert_allclose(filled, np.broadcast_to((0.2, 0.4, 0.6), filled.shape))
    assert not visibility[0, 0, 0]


def test_boundary_and_world_projection_share_raw_origin_with_slot0_priority():
    raw = torch.zeros(1, 2, 5, 1, 1)
    world = torch.full_like(raw, 2.0)
    boundary = torch.full((1, 2, 1, 1, 1), 4.0)
    visible = torch.ones(1, 1, 5, 1, 1)
    confidence = torch.ones_like(visible)
    projected, metrics = apply_world_and_boundary_projection(
        raw, world, visible, confidence,
        boundary_state=boundary,
        boundary_beta_max=0.2,
        sigma=0.0, lambda_max=0.25, gamma=1.0,
        confidence_ramp_min=0.2, confidence_ramp_max=0.5,
    )
    # Boundary owns slot0; later slots receive WPF from the same raw origin.
    expected = torch.tensor([0.8, 0.5, 0.5, 0.5, 0.5]).view(1, 1, 5, 1, 1)
    torch.testing.assert_close(projected, expected.expand_as(projected))
    assert float(metrics["boundary_active"]) == 1.0
    assert float(metrics["boundary_non_slot0_delta_max"]) == 0.0
    assert float(metrics["unknown_projection_delta_max"]) == 0.0
    assert float(metrics["wpf_slot0_strength_max"]) == 0.0


def test_boundary_only_changes_temporal_slot0_and_stage0_is_disabled():
    raw = torch.randn(1, 2, 5, 2, 2)
    boundary = torch.randn(1, 2, 1, 2, 2)
    projected, metrics = apply_world_and_boundary_projection(
        raw, torch.randn_like(raw),
        torch.zeros(1, 1, 5, 2, 2),
        torch.ones(1, 1, 5, 2, 2),
        boundary_state=boundary,
        boundary_beta_max=0.2, sigma=0.0,
    )
    torch.testing.assert_close(projected[:, :, 1:], raw[:, :, 1:])
    assert float(metrics["boundary_non_slot0_delta_max"]) == 0.0
    stage0, stage0_metrics = apply_world_and_boundary_projection(
        raw, torch.randn_like(raw),
        torch.zeros(1, 1, 5, 2, 2), torch.ones(1, 1, 5, 2, 2),
        boundary_state=boundary, boundary_beta_max=0.0, sigma=0.0,
    )
    torch.testing.assert_close(stage0, raw)
    assert float(stage0_metrics["boundary_delta_ratio"]) == 0.0


def test_boundary_strength_increases_as_next_sigma_approaches_zero():
    raw = torch.zeros(1, 1, 3, 1, 1)
    boundary = torch.ones(1, 1, 1, 1, 1)
    support = torch.zeros(1, 1, 3, 1, 1)
    _, early = apply_world_and_boundary_projection(
        raw, raw, support, support,
        boundary_state=boundary, boundary_beta_max=0.2, sigma=0.8,
    )
    _, late = apply_world_and_boundary_projection(
        raw, raw, support, support,
        boundary_state=boundary, boundary_beta_max=0.2, sigma=0.0,
    )
    assert float(early["boundary_strength"]) < float(late["boundary_strength"])


def test_clean_boundary_uses_same_scheduler_sigma_coordinate():
    start = torch.full((1, 2, 9, 2, 2), 4.0)
    endpoint = torch.full((1, 2, 1, 2, 2), 10.0)
    state = build_boundary_state_at_sigma(
        next_sigma=0.25,
        clean_boundary_endpoint=endpoint,
        stage_start_state=start,
    )
    torch.testing.assert_close(state, torch.full_like(state, 8.5))


def test_delayed_node_activation_waits_full_chunk_and_schedule_cannot_be_replaced():
    node0 = SimpleNamespace(node_id="node_000")
    node1 = SimpleNamespace(node_id="node_001")
    node2 = SimpleNamespace(node_id="node_002")
    queue = DelayedNodeActivationQueue()
    entry = queue.schedule(node1, created_after_chunk=5)
    assert entry.activate_at_chunk == 7
    assert queue.activate_due(5) is None
    assert queue.activate_due(6) is None
    # chunk6 must still render node0 while node1 remains scheduled.
    render_node = node0
    assert render_node.node_id == "node_000"
    try:
        queue.schedule(node2, created_after_chunk=6)
    except RuntimeError as error:
        assert "cannot replace pending activation node_001" in str(error)
    else:
        raise AssertionError("a later candidate replaced the pending activation")
    activated = queue.activate_due(7)
    render_node = activated.node
    assert render_node.node_id == "node_001"


def test_one_chunk_node_activation_delay_is_supported_for_inference_ablation():
    node = SimpleNamespace(node_id="node_001")
    queue = DelayedNodeActivationQueue(delay_chunks=1)
    entry = queue.schedule(node, created_after_chunk=5)
    assert entry.activate_at_chunk == 6
    assert queue.activate_due(5) is None
    assert queue.activate_due(6).node is node


def test_temporal_warmup_only_scales_final_wpf_strength():
    raw = torch.zeros(1, 2, 9, 1, 1)
    world = torch.ones_like(raw)
    visibility = torch.ones(1, 1, 9, 1, 1)
    confidence = torch.ones_like(visibility)
    projected, metrics = apply_world_projection(
        raw, world, visibility, confidence,
        sigma=0.0, lambda_max=0.5,
        temporal_warmup=DEFAULT_TEMPORAL_WARMUP,
    )
    expected = torch.tensor(DEFAULT_TEMPORAL_WARMUP).view(1, 1, 9, 1, 1) * 0.5
    torch.testing.assert_close(projected[:, :1], expected)
    assert float(metrics["wpf_slot0_strength_max"]) == 0.0
    assert torch.equal(visibility, torch.ones_like(visibility))
    assert torch.equal(confidence, torch.ones_like(confidence))


def test_boundary_active_wpf_slot0_is_strictly_zero_even_with_custom_warmup():
    raw = torch.zeros(1, 1, 9, 1, 1)
    world = torch.ones_like(raw)
    boundary = torch.ones(1, 1, 1, 1, 1)
    support = torch.ones(1, 1, 9, 1, 1)
    projected, metrics = apply_world_and_boundary_projection(
        raw, world, support, support,
        boundary_state=boundary, boundary_beta_max=0.2, sigma=0.0,
        lambda_max=0.5, temporal_warmup=(1.0,) * 9,
    )
    torch.testing.assert_close(projected[:, :, 0], torch.full((1, 1, 1, 1), 0.2))
    assert float(metrics["wpf_slot0_strength_max"]) == 0.0


def test_deterministic_canonical_encode_uses_mode_and_does_not_consume_generator():
    class Posterior:
        sample_calls = 0

        def __init__(self, value):
            self.mean = value
            self.mode_calls = 0

        def mode(self):
            self.mode_calls += 1
            return self.mean + 1.0

        def sample(self, *args, **kwargs):  # pragma: no cover - must never run
            type(self).sample_calls += 1
            raise AssertionError("canonical VAE encode must not sample")

    class VAE:
        dtype = torch.float32

        def __init__(self):
            self.calls = []

        def encode(self, video):
            self.calls.append(video.detach().clone())
            value = video.mean(dim=1, keepdim=True)
            return type("Encoded", (), {"latent_dist": Posterior(value)})()

    class Pipe:
        vae_scale_factor_temporal = 4

        def __init__(self):
            self.vae = VAE()

    pipe = Pipe()
    video = torch.linspace(-1.0, 1.0, 33 * 2 * 2 * 3).reshape(1, 3, 33, 2, 2)
    generator = torch.Generator().manual_seed(123)
    before = generator.get_state()
    first_a, clean_a = encode_canonical_video_latents(
        pipe, video, latents_mean=torch.zeros(1, 1, 1, 1, 1),
        latents_std=torch.ones(1, 1, 1, 1, 1), num_latent_frames_per_chunk=9,
        dtype=torch.float32, device="cpu",
    )
    first_b, clean_b = encode_canonical_video_latents(
        pipe, video, latents_mean=torch.zeros(1, 1, 1, 1, 1),
        latents_std=torch.ones(1, 1, 1, 1, 1), num_latent_frames_per_chunk=9,
        dtype=torch.float32, device="cpu",
    )
    assert torch.equal(first_a, first_b)
    assert torch.equal(clean_a, clean_b)
    assert torch.equal(before, generator.get_state())
    assert len(pipe.vae.calls) == 4
    assert Posterior.sample_calls == 0


def test_posterior_mode_falls_back_to_mean_without_sampling():
    mean = torch.randn(1, 2, 1, 1, 1)
    torch.testing.assert_close(
        posterior_mode_or_mean(type("Encoded", (), {"latent_dist": type("Posterior", (), {"mean": mean})()})()),
        mean,
    )


def test_cached_wah_conditioning_reuses_exact_first_frame_and_clean_latents():
    pipe = WorldProjectedWarpAsHistoryPipeline.__new__(WorldProjectedWarpAsHistoryPipeline)
    pipe._wah_execution_device = lambda: torch.device("cpu")
    pipe._coerce_warp_video_tensor = lambda value, height, width, device: value.detach().clone()
    rgb = torch.zeros(1, 3, 9, 2, 2)
    first = torch.full((1, 2, 1, 1, 1), 7.0)
    clean = torch.arange(18, dtype=torch.float32).reshape(1, 2, 9, 1, 1)
    pipe.set_canonical_warp_conditioning(
        rgb, clean, first_frame_latent=first, height=2, width=2,
    )
    cached_first, cached_clean = pipe.prepare_video_latents(
        rgb, dtype=torch.float32, device=torch.device("cpu"), generator=object(),
    )
    assert torch.equal(cached_first, first)
    assert torch.equal(cached_clean, clean)


def test_canonical_support_uses_exact_33_to_9_groups_once():
    visibility = torch.zeros(33, 16, 16)
    confidence = torch.zeros_like(visibility)
    groups = [(0, 1)] + [(1 + 4 * index, 5 + 4 * index) for index in range(8)]
    for index, (start, stop) in enumerate(groups):
        visibility[start:stop, 2:14, 2:14] = (index + 1) / 9.0
        confidence[start:stop, 2:14, 2:14] = 0.2 + index * 0.1
    support = build_canonical_world_support(
        visibility, confidence, latent_frames=9, latent_height=8, latent_width=8,
        temporal_scale=4,
    )
    assert tuple(support.visibility.shape) == (1, 1, 9, 8, 8)
    assert tuple(support.confidence.shape) == (1, 1, 9, 8, 8)
    assert tuple(support.safe_support.shape) == (1, 1, 9, 8, 8)
    center_visibility = support.visibility[0, 0, :, 4, 4]
    torch.testing.assert_close(center_visibility, torch.arange(1, 10) / 9.0)
    center_ownership = support.world_ownership_mask[0, 0, :, 4, 4]
    assert center_ownership.tolist() == [0.0] * 8 + [1.0]
    assert set(torch.unique(support.world_ownership_mask).tolist()) <= {0.0, 1.0}
    center_confidence = support.confidence[0, 0, :, 4, 4]
    assert center_confidence[:-1].eq(0).all()
    torch.testing.assert_close(center_confidence[-1], torch.tensor(1.0))


def test_safe_support_masks_fill_only_latent_and_tokens_share_grouped_support():
    visibility = torch.zeros(33, 16, 16)
    visibility[:, 3:13, 3:13] = 1.0
    confidence = torch.ones_like(visibility)
    support = build_canonical_world_support(
        visibility, confidence, latent_frames=9, latent_height=8, latent_width=8,
    )
    latent = torch.ones(1, 4, 9, 8, 8)
    safe = mask_canonical_latent(latent, support.safe_support)
    invalid = (support.safe_support == 0).expand_as(safe)
    assert torch.equal(safe[invalid], torch.zeros_like(safe[invalid]))
    tokens = canonical_support_to_tokens(
        support.safe_support,
        latent_height=4, latent_width=4, patch_height=2, patch_width=2,
    )
    assert tuple(tokens.shape) == (1, 9 * 2 * 2, 1)


def test_previous_world_boundary_replaces_slot0_latent_visibility_and_confidence_exactly():
    latent = torch.randn(1, 4, 9, 8, 8)
    first = torch.randn(1, 4, 1, 8, 8)
    support = build_canonical_world_support(
        torch.ones(33, 16, 16), torch.full((33, 16, 16), 0.7),
        latent_frames=9, latent_height=8, latent_width=8,
    )
    previous_latent = torch.randn(1, 4, 1, 8, 8)
    previous_visibility = torch.ones((1, 1, 1, 8, 8))
    previous_confidence = torch.full((1, 1, 1, 8, 8), 0.6)
    clean, cached_first, replaced, applied = apply_previous_world_boundary(
        latent, first, support,
        previous_latent=previous_latent,
        previous_visibility=previous_visibility,
        previous_confidence=previous_confidence,
    )
    assert applied
    assert torch.equal(clean[:, :, 0:1], previous_latent)
    assert torch.equal(cached_first, previous_latent)
    assert torch.equal(replaced.safe_support[:, :, 0:1], previous_visibility)
    assert torch.equal(replaced.visibility[:, :, 0:1], previous_visibility)
    assert torch.equal(replaced.confidence[:, :, 0:1], previous_confidence)


def test_fill_only_and_subthreshold_coverage_never_gain_world_ownership():
    visibility = torch.zeros(33, 10, 10)
    visibility[:, :9] = 1.0
    confidence = torch.ones_like(visibility)
    support = build_canonical_world_support(
        visibility, confidence, latent_frames=9, latent_height=1, latent_width=1,
    )
    # Every temporal group has exactly 90% real renderer coverage and is owned.
    assert torch.equal(support.world_ownership_mask, torch.ones_like(support.world_ownership_mask))
    visibility[:, 0, 0] = 0.0
    support = build_canonical_world_support(
        visibility, confidence, latent_frames=9, latent_height=1, latent_width=1,
    )
    assert torch.equal(support.world_ownership_mask, torch.zeros_like(support.world_ownership_mask))
    assert WORLD_OWNERSHIP_COVERAGE_THRESHOLD == 0.9


def test_wah_history_mapping_reuses_cached_canonical_support_without_regrouping():
    pipe = WorldProjectedWarpAsHistoryPipeline.__new__(WorldProjectedWarpAsHistoryPipeline)
    pipe._wah_execution_device = lambda: torch.device("cpu")
    pipe._coerce_warp_video_tensor = lambda value, height, width, device: value.clone()
    pipe._coerce_visibility_mask = lambda value: torch.as_tensor(value, dtype=torch.float32)
    pixel_visibility = torch.ones(1, 1, 33, 16, 16)
    pixel_visibility[:, :, :, :2] = 0
    pixel_confidence = torch.full_like(pixel_visibility, 0.6) * pixel_visibility
    support = build_canonical_world_support(
        pixel_visibility, pixel_confidence,
        latent_frames=9, latent_height=8, latent_width=8,
    )
    pipe.set_canonical_warp_conditioning(
        torch.zeros(1, 3, 33, 16, 16),
        torch.zeros(1, 4, 9, 8, 8),
        canonical_support=support,
        pixel_visibility=pixel_visibility,
        pixel_confidence=pixel_confidence,
        height=16,
        width=16,
    )
    mapped_visibility = pipe._visibility_mask_to_history_latents(
        pixel_visibility,
        latent_frames=9, latent_height=8, latent_width=8, temporal_scale=4,
    )
    mapped_weighted_confidence = pipe._visibility_mask_to_history_latents(
        pixel_confidence,
        latent_frames=9, latent_height=8, latent_width=8, temporal_scale=4,
    )
    assert torch.equal(mapped_visibility, support.safe_support)
    assert torch.equal(
        mapped_weighted_confidence, support.safe_support * support.confidence,
    )
