from pathlib import Path

import torch


def test_hard_negatives_are_paired_at_each_positive_key_time():
    from scripts.train_sightline_rgbd import _hard_negative_indices

    # Query global time 10 has two positives at different key times.  Each
    # time has spatially nearby non-positive current tokens available.
    identities = (
        ('current', (10,), 4, 4, 'current'),
        ('current', (8,), 5, 5, 'current'),
        ('current', (9,), 2, 2, 'current'),
        ('current', (8,), 5, 6, 'current'),
        ('current', (8,), 7, 7, 'current'),
        ('current', (9,), 2, 3, 'current'),
        ('current', (9,), 4, 4, 'current'),
    )
    negatives, masks, matched, pair_count = _hard_negative_indices(
        [0], [[1, 2]], identities, (8, 8, 8), 8, max_negatives=2
    )

    assert matched is True
    assert pair_count == 2
    assert masks == [[[True, True], [True, True]]]
    assert [[identities[index][1][0] for index in pair] for pair in negatives[0]] == [[8, 8], [9, 9]]


def test_rgbd_ranking_loss_keeps_negative_axis_paired_and_trains_augmented_qk():
    from long_video.training.sightline import CorrespondencePlan, SightlineTrainable

    trainable = SightlineTrainable(4, layers=(0,), heads=1)
    augmented_q = torch.randn(1, 1, 1, 4, requires_grad=True)
    augmented_k = torch.randn(1, 7, 1, 4, requires_grad=True)
    native_q = torch.randn(1, 1, 1, 4)
    native_k = torch.randn(1, 7, 1, 4)
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


def test_training_source_uses_only_rgbd_stage1_and_stage2():
    source = (Path(__file__).parents[1] / 'scripts' / 'train_sightline_rgbd.py').read_text()
    config = (Path(__file__).parents[1] / 'configs' / 'sightline.yaml').read_text()
    assert "for rgbd_stage_index in (1,2)" in source
    assert "stage_rgbd_plan=rgbd_plans.get(stage_index)" in source
    assert "rgbd_stage_index=max(" not in source
    assert "lambda_rgbd: 0.006" in config
    assert "rgbd_stage1_negative_key_t_match" in source
    assert "rgbd_stage2_negative_key_t_match" in source
