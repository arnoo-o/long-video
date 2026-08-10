import torch

from long_video.wah.world_projected_pipeline import (
    WorldProjectionConfig,
    apply_world_and_boundary_projection,
    apply_world_projection,
    build_canonical_world_pyramid,
    build_world_projection_context,
    build_world_state_at_sigma,
    fill_invalid_warp_for_vae,
    smooth_latent_visibility,
    world_projection_weight,
)


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
    )
    for latent, visible, conf in zip(
        context.canonical_latents, context.visibility, context.confidence,
    ):
        assert tuple(visible.shape) == (1, 1, 9, latent.shape[-2], latent.shape[-1])
        assert tuple(conf.shape) == tuple(visible.shape)
        assert torch.isfinite(latent).all()
        assert torch.isfinite(visible).all()
        assert torch.isfinite(conf).all()


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


def test_boundary_and_world_projection_share_raw_origin_and_decay_over_three_latents():
    raw = torch.zeros(1, 2, 5, 1, 1)
    world = torch.full_like(raw, 2.0)
    previous = torch.full((1, 2, 3, 1, 1), 4.0)
    visible = torch.ones(1, 1, 5, 1, 1)
    confidence = torch.ones_like(visible)
    projected, metrics = apply_world_and_boundary_projection(
        raw, world, visible, confidence,
        previous_boundary=previous,
        boundary_beta=(0.6, 0.3, 0.1),
        sigma=0.0, lambda_max=0.25, gamma=1.0,
        confidence_ramp_min=0.2, confidence_ramp_max=0.5,
    )
    # z = raw + beta*(prev-raw) + lambda*(world-raw)
    expected = torch.tensor([2.9, 1.7, 0.9, 0.5, 0.5]).view(1, 1, 5, 1, 1)
    torch.testing.assert_close(projected, expected.expand_as(projected))
    assert float(metrics["boundary_active"]) == 1.0
    assert float(metrics["unknown_projection_delta_max"]) == 0.0


def test_boundary_projection_does_not_change_later_temporal_latents_without_world_support():
    raw = torch.randn(1, 2, 5, 2, 2)
    previous = torch.randn(1, 2, 3, 2, 2)
    projected, _ = apply_world_and_boundary_projection(
        raw, torch.randn_like(raw),
        torch.zeros(1, 1, 5, 2, 2),
        torch.ones(1, 1, 5, 2, 2),
        previous_boundary=previous,
        boundary_beta=(0.6, 0.3, 0.1), sigma=0.0,
    )
    torch.testing.assert_close(projected[:, :, 3:], raw[:, :, 3:])
