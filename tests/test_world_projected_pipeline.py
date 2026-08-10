import torch

from long_video.wah.world_projected_pipeline import (
    WorldProjectionConfig,
    apply_world_projection,
    build_canonical_world_pyramid,
    build_world_projection_context,
    build_world_state_at_sigma,
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
        confidence_power=1.0, confidence_threshold=0.3,
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


def test_confidence_threshold_and_sigma_schedule_disable_projection():
    raw = torch.randn(1, 2, 1, 2, 2)
    world = torch.randn_like(raw)
    visible = torch.ones(1, 1, 1, 2, 2)
    low_confidence = torch.full_like(visible, 0.29)
    thresholded, _ = apply_world_projection(
        raw, world, visible, low_confidence, sigma=0.0, confidence_threshold=0.3,
    )
    early, _ = apply_world_projection(
        raw, world, visible, torch.ones_like(visible), sigma=1.0,
    )
    assert torch.equal(thresholded, raw)
    assert torch.equal(early, raw)


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
        config=WorldProjectionConfig(lambda_max=0.5),
    )
    for latent, visible, conf in zip(
        context.canonical_latents, context.visibility, context.confidence,
    ):
        assert tuple(visible.shape) == (1, 1, 9, latent.shape[-2], latent.shape[-1])
        assert tuple(conf.shape) == tuple(visible.shape)
        assert torch.isfinite(latent).all()
        assert torch.isfinite(visible).all()
        assert torch.isfinite(conf).all()
