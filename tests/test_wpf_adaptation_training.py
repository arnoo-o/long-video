import torch

from long_video.training.wpf_adaptation import (
    WPFAdaptationObserver,
    WPFTrainingSampler,
    curriculum_max_chunks,
    renderer_visibility_to_latent_mask,
    weighted_mse,
)


def test_sampler_is_uniform_random_and_resumable():
    assert [curriculum_max_chunks(step) for step in (1, 400, 401, 800, 801, 1001, 1201)] == [1, 1, 2, 2, 3, 4, 6]
    sampler = WPFTrainingSampler(seed=9)
    choices = [sampler.choose_position() for _ in range(4000)]
    assert set(choices) == {(1, 0), (1, 1), (2, 0), (2, 1)}
    assert max(choices.count(value) for value in set(choices)) - min(
        choices.count(value) for value in set(choices)
    ) < 160
    state = sampler.state_dict()
    expected = [sampler.choose_position() for _ in range(30)]
    resumed = WPFTrainingSampler(seed=999)
    resumed.load_state_dict(state)
    assert [resumed.choose_position() for _ in range(30)] == expected


def test_visibility_mapping_is_soft_exact_grouping_without_morphology():
    visibility = torch.zeros(1, 1, 33, 8, 8)
    visibility[:, :, 1, 0, 0] = 1
    mask = renderer_visibility_to_latent_mask(visibility, (9, 2, 2))
    assert tuple(mask.shape) == (1, 1, 9, 2, 2)
    assert mask[0, 0, 0].sum() == 0
    assert torch.isclose(mask[0, 0, 1, 0, 0], torch.tensor(1 / 64))
    assert torch.count_nonzero(mask) == 1


def test_clean_space_observer_backprops_once_and_detaches_scheduler_state():
    shapes = [(1, 2, 9, 1, 1)] * 3
    gt = [torch.zeros(shape) for shape in shapes]
    observer = WPFAdaptationObserver(gt, selected_stage=1, selected_step=1)
    visibility = torch.zeros(1, 1, 33, 2, 2)
    visibility[:, :, :, :, 0] = 1
    for stage in range(3):
        for step in range(2):
            prediction = torch.ones(shapes[stage], requires_grad=True)
            sample = torch.full(shapes[stage], 2.0)
            result = observer({
                "stage_id": stage, "step_id": step,
                "model_output": prediction, "base_model_output": torch.zeros_like(prediction),
                "sample": sample, "dmd_sigmas": torch.tensor([1.0, 0.5, 0.0]),
                "point_visibility": visibility,
            })
            if (stage, step) == (1, 1):
                assert prediction.grad is not None and torch.isfinite(prediction.grad).all()
            else:
                assert prediction.grad is None
            assert not result[0].requires_grad and not result[1].requires_grad
    observer.assert_complete()
    assert observer.losses[(1, 1)]["fill"] > 0
    assert observer.losses[(1, 1)]["keep"] > 0


def test_weighted_mse_normalizes_by_weight_and_channels():
    prediction = torch.tensor([[[[[2.0, 9.0]]]]])
    target = torch.zeros_like(prediction)
    weight = torch.tensor([[[[[1.0, 0.0]]]]])
    assert weighted_mse(prediction, target, weight) == 4
