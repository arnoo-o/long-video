from pathlib import Path
import pytest

import torch
import torch.nn.functional as F


def test_hard_negatives_are_paired_at_each_positive_key_time():
    from scripts.train_sightline_rgbd import _hard_negative_indices

    # Query global time 10 has two positives at different key times.  The
    # camera baseline makes the epipolar line horizontal, so same-line tokens
    # are ignored and only off-epipolar candidates can be selected.
    identities = (
        ('current', (10,), 4, 4, 'current'),
        ('current', (9,), 4, 4, 'current'),
        ('current', (8,), 3, 3, 'current'),
        ('current', (9,), 4, 5, 'current'),  # d_epi=0: ignore
        ('current', (9,), 2, 5, 'current'),  # d_epi=2: mid band
        ('current', (9,), 0, 4, 'current'),  # d_epi=4: far band
        ('current', (9,), 1, 1, 'current'),  # d_epi=3: mid band
        ('current', (9,), 0, 6, 'current'),  # d_epi=4: far band
        ('current', (8,), 3, 4, 'current'),  # d_epi=0: ignore
        ('current', (8,), 1, 4, 'current'),  # d_epi=2: mid band
        ('current', (8,), 0, 3, 'current'),  # d_epi=3: mid band
        ('current', (8,), 7, 3, 'current'),  # d_epi=4: far band
    )
    c2w = torch.eye(4).repeat(33, 1, 1)
    c2w[8, 0, 3] = 0.2
    intrinsics = torch.eye(3).repeat(33, 1, 1)
    intrinsics[:, 0, 0] = 100.0
    intrinsics[:, 1, 1] = 100.0
    intrinsics[:, 0, 2] = 4.0
    intrinsics[:, 1, 2] = 4.0
    negatives, masks, matched, pair_count = _hard_negative_indices(
        [0], [[1, 2]], identities, (3, 8, 8), 33,
        c2w=c2w, intrinsics=intrinsics, max_negatives=4
    )

    assert matched is True
    assert pair_count == 2
    assert all(all(mask) for mask in masks[0])
    assert 3 not in negatives[0][0] and 8 not in negatives[0][1]
    assert all(identities[index][1][0] == positive_time for pair, positive_time in zip(negatives[0], (9, 8)) for index in pair)


def test_rgbd_ranking_loss_keeps_negative_axis_paired_and_trains_augmented_qk():
    from long_video.training.sightline import CorrespondencePlan, SightlineTrainable

    trainable = SightlineTrainable(4, layers=(0,), heads=1)
    augmented_q = torch.randn(1, 1, 1, 4, requires_grad=True)
    augmented_k = torch.randn(1, 7, 1, 4, requires_grad=True)
    native_q = torch.randn(1, 1, 1, 4, requires_grad=True)
    native_k = torch.randn(1, 7, 1, 4, requires_grad=True)
    plan = CorrespondencePlan(
        query_indices=torch.tensor([0]),
        positive_indices=torch.tensor([[1, 2]]),
        positive_mask=torch.tensor([[True, True]]),
        weights=torch.tensor([0.7]),
        identities=(),
        flags=(),
        negative_indices=torch.tensor([[[3, 4], [5, 6]]]),
        negative_mask=torch.ones(1, 2, 2, dtype=torch.bool),
        negative_key_t_match=True,
        negative_pair_count=2,
    )

    loss = trainable.rgbd_ranking_loss(
        augmented_q, augmented_k, native_q, native_k, plan, margin=0.1, temperature=0.1
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert augmented_q.grad is not None and augmented_q.grad.abs().sum() > 0
    assert augmented_k.grad is not None and augmented_k.grad.abs().sum() > 0
    assert native_q.grad is None and native_k.grad is None


def test_disabled_scale_augmentation_does_not_consume_rng():
    from long_video.sightline.conditioning import SightlineConditioner

    conditioner = SightlineConditioner(4, scale_aug_prob=0.0)
    rays = torch.randn(2, 3, 7)
    state = torch.random.get_rng_state()
    assert conditioner.sample_scale_delta(rays, training=True) is None
    assert torch.equal(state, torch.random.get_rng_state())


def test_streaming_rgbd_ranking_matches_dense_reference_with_confidence_and_upstream_scale():
    from long_video.training.sightline import CorrespondencePlan, SightlineTrainable

    torch.manual_seed(404)
    batch, rows, heads, dim, key_count = 3, 4, 3, 5, 19
    positive = torch.tensor([[1, 2, 3], [4, 5, -1], [6, 6, 7], [8, 9, 10]])
    positive_mask = torch.tensor([[True, True, True], [True, True, False], [True, True, True], [True, True, True]])
    positive_weights = torch.tensor([[0.2, 1.7, 3.1], [2.2, 0.4, 0.0], [0.6, 4.0, 0.9], [1.3, 0.1, 2.5]])
    negative = torch.tensor([
        [[11, 12, 12, -1], [13, -1, -1, -1], [14, 15, 16, 17]],
        [[11, 12, -1, -1], [13, 13, 14, -1], [-1, -1, -1, -1]],
        [[11, -1, -1, -1], [12, 13, 13, 14], [15, 16, -1, -1]],
        [[11, 12, 13, -1], [14, -1, -1, -1], [15, 15, 16, -1]],
    ])
    negative_mask = negative.ge(0)
    row_weights = torch.tensor([0.7, 1.6, 0.3, 2.4])
    plan = CorrespondencePlan(
        query_indices=torch.arange(rows), positive_indices=positive, positive_mask=positive_mask,
        weights=row_weights, identities=(), flags=(), negative_indices=negative,
        negative_mask=negative_mask, positive_weights=positive_weights,
        negative_key_t_match=True, negative_pair_count=int(positive_mask.sum()),
    )
    native_q = torch.randn(batch, rows, heads, dim)
    native_k = torch.randn(batch, key_count, heads, dim)
    q = torch.randn(batch, rows, heads, dim, requires_grad=True)
    k = torch.randn(batch, key_count, heads, dim, requires_grad=True)

    def dense_reference(aq, ak):
        scale = dim ** -0.5
        safe_positive = positive.clamp_min(0)
        safe_negative = negative.clamp_min(0)
        positive_aug = torch.gather(
            ak[:, None, None].expand(batch, rows, positive.shape[1], key_count, heads, dim),
            3, safe_positive[None, :, :, None, None, None].expand(batch, rows, positive.shape[1], 1, heads, dim),
        ).squeeze(3)
        positive_native = torch.gather(
            native_k[:, None, None].expand(batch, rows, positive.shape[1], key_count, heads, dim),
            3, safe_positive[None, :, :, None, None, None].expand(batch, rows, positive.shape[1], 1, heads, dim),
        ).squeeze(3)
        negative_aug = torch.gather(
            ak[:, None, None, None].expand(batch, rows, positive.shape[1], negative.shape[2], key_count, heads, dim),
            4, safe_negative[None, :, :, :, None, None, None].expand(batch, rows, positive.shape[1], negative.shape[2], 1, heads, dim),
        ).squeeze(4)
        negative_native = torch.gather(
            native_k[:, None, None, None].expand(batch, rows, positive.shape[1], negative.shape[2], key_count, heads, dim),
            4, safe_negative[None, :, :, :, None, None, None].expand(batch, rows, positive.shape[1], negative.shape[2], 1, heads, dim),
        ).squeeze(4)
        query = aq.unsqueeze(2)
        native_query = native_q.unsqueeze(2)
        positive_delta = (query * positive_aug).sum(-1) * scale - (native_query * positive_native).sum(-1) * scale
        negative_delta = ((query.unsqueeze(3) * negative_aug).sum(-1) * scale - (native_query.unsqueeze(3) * negative_native).sum(-1) * scale).permute(0, 1, 2, 4, 3)
        count = negative_mask.sum(-1).clamp_min(1).to(negative_delta.dtype)
        negative_mean = negative_delta.masked_fill(~negative_mask[None, :, :, None, :], 0.0).sum(-1) / count[None, :, :, None]
        gap = positive_delta - negative_mean
        pair_loss = F.softplus((0.1 - gap) / 0.1)
        pair_mask = positive_mask & negative_mask.any(-1)
        confidence = positive_weights * pair_mask
        per_row = (pair_loss * confidence[None, :, :, None]).sum(2) / confidence.sum(-1).clamp_min(1e-8)[None, :, None]
        valid_rows = pair_mask.any(-1)
        return (per_row.mean((0, 2)) * row_weights).sum() / row_weights[valid_rows].sum().clamp_min(1e-8)

    dense_q = q.detach().clone().requires_grad_(True)
    dense_k = k.detach().clone().requires_grad_(True)
    dense = dense_reference(dense_q, dense_k)
    streamed = SightlineTrainable(4, layers=(0,), heads=heads).rgbd_ranking_loss(
        q, k, native_q, native_k, plan, margin=0.1, temperature=0.1
    )
    upstream = 0.37
    dense_grad = torch.autograd.grad(upstream * dense, (dense_q, dense_k), retain_graph=True)
    (upstream * streamed).backward()
    assert torch.allclose(streamed, dense, atol=2e-6, rtol=2e-6)
    assert torch.allclose(q.grad, dense_grad[0], atol=3e-6, rtol=3e-6)
    assert torch.allclose(k.grad, dense_grad[1], atol=3e-6, rtol=3e-6)


def test_training_source_uses_only_rgbd_stage1_and_stage2():
    source = (Path(__file__).parents[1] / 'scripts' / 'train_sightline_rgbd.py').read_text()
    config = (Path(__file__).parents[1] / 'configs' / 'sightline.yaml').read_text()
    assert "for rgbd_stage_index in (1,2)" in source
    assert "stage_rgbd_plan=rgbd_plans.get(stage_index)" in source
    assert "rgbd_stage_index=max(" not in source
    assert "lambda_rgbd: 0.01" in config
    assert "scale_augmentation_probability: 0.0" in config
    assert "rgbd_stage1_negative_key_t_match" in source
    assert "rgbd_stage2_negative_key_t_match" in source
    assert "rgbd_scale_sum" not in source
    assert "1.0/len(valid_rgbd_stages)" in source


def test_total_metric_contains_all_fm_stages_without_a_second_backward_path():
    from scripts.train_sightline_rgbd import _total_metric

    fm = torch.tensor(0.7)
    rgbd_term = torch.tensor(0.02, requires_grad=True)
    cross_term = torch.tensor(0.03, requires_grad=True)
    total = _total_metric(fm, rgbd_term, cross_term)
    assert total.item() == pytest.approx(0.75)
    assert not total.requires_grad
