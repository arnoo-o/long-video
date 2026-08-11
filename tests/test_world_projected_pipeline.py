import torch
from types import SimpleNamespace

from long_video.wah.world_projected_pipeline import (
    DelayedNodeActivationQueue,
    WorldProjectionConfig,
    apply_world_and_boundary_projection,
    apply_world_projection,
    build_canonical_world_pyramid,
    build_world_projection_context,
    build_boundary_state_at_sigma,
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
    # slot0: beta*boundary + (1-beta)*lambda*world; later slots: lambda*world.
    expected = torch.tensor([1.2, 0.5, 0.5, 0.5, 0.5]).view(1, 1, 5, 1, 1)
    torch.testing.assert_close(projected, expected.expand_as(projected))
    assert float(metrics["boundary_active"]) == 1.0
    assert float(metrics["boundary_non_slot0_delta_max"]) == 0.0
    assert float(metrics["unknown_projection_delta_max"]) == 0.0


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
