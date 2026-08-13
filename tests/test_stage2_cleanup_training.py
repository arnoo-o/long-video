import json

import torch

from long_video.training.stage2_cleanup import (
    BalancedChunkSampler, Stage2CompletionObserver, curriculum_max_chunks,
    renderer_visibility_to_latent_mask, select_balanced_training_records,
    weighted_mse,
)
from scripts.query_stage2_training_progress import progress_summary


def test_curriculum_and_balanced_supervision():
    assert [curriculum_max_chunks(x) for x in (1, 400, 401, 800, 801, 1001, 1201, 1400)] == [1, 1, 2, 2, 3, 4, 6, 6]
    sampler = BalancedChunkSampler(seed=7)
    selected = [sampler.choose(i, forced_max_chunks=2)[1] for i in range(1, 11)]
    assert abs(selected.count(0) - selected.count(1)) <= 1
    choices = [sampler.choose_stage2_step() for _ in range(1000)]
    assert set(choices) == {2, 3}
    assert abs(choices.count(2) - choices.count(3)) < 100
    state = sampler.state_dict()
    expected = [sampler.choose_stage2_step() for _ in range(20)]
    resumed = BalancedChunkSampler(seed=999)
    resumed.load_state_dict(state)
    assert [resumed.choose_stage2_step() for _ in range(20)] == expected


def test_balanced_training_record_selection():
    records = [
        {"trajectory_id": f"{environment}_{index:03d}", "split": "train",
         "environment": environment}
        for environment in ("indoor", "outdoor") for index in range(75)
    ]
    selected = select_balanced_training_records(records, 100)
    assert len(selected) == 100
    assert sum(item["environment"] == "indoor" for item in selected) == 50


def test_visibility_mapping_uses_exact_vae_groups_without_dilation():
    visibility = torch.zeros(1, 1, 33, 8, 8)
    visibility[:, :, 1, 0, 0] = 1
    mask = renderer_visibility_to_latent_mask(visibility, (9, 2, 2))
    assert tuple(mask.shape) == (1, 1, 9, 2, 2)
    assert mask[0, 0, 0].sum() == 0
    assert torch.isclose(mask[0, 0, 1, 0, 0], torch.tensor(1 / 64))
    assert torch.count_nonzero(mask) == 1


def test_weighted_mse_only_uses_selected_region():
    prediction = torch.tensor([[[[[2.0, 9.0]]]]])
    target = torch.zeros_like(prediction)
    weight = torch.tensor([[[[[1.0, 0.0]]]]])
    assert weighted_mse(prediction, target, weight) == 4


def test_completion_observer_backprops_only_selected_step_and_detaches():
    z_gt = torch.zeros(1, 1, 9, 1, 1)
    visibility = torch.zeros(1, 1, 33, 8, 8)
    visibility[:, :, :, :4] = 1
    observer = Stage2CompletionObserver(z_gt, scheduler=None, selected_step=2)
    for step in range(4):
        prediction = torch.ones_like(z_gt, requires_grad=True)
        sample = torch.full_like(z_gt, 2.0)
        base = torch.zeros_like(z_gt)
        result = observer({"stage_id": 2, "step_id": step, "model_output": prediction,
                           "sample": sample, "timestep": torch.tensor(10),
                           "dmd_timesteps": torch.tensor([10, 0]),
                           "dmd_sigmas": torch.tensor([0.5, 0.0]),
                           "completion_adapter_active": step >= 2,
                           "base_model_output": base if step == 2 else None,
                           "point_visibility": visibility})
        if step == 2:
            assert prediction.grad is not None and torch.isfinite(prediction.grad).all()
        else:
            assert prediction.grad is None
        assert not result[0].requires_grad and not result[1].requires_grad
    observer.assert_complete()
    assert set(observer.losses) == {2}
    assert observer.losses[2]["gen"] > 0
    assert observer.losses[2]["keep"] > 0


def test_progress_query_is_read_only(tmp_path):
    metrics = tmp_path / "metrics"; checkpoints = tmp_path / "checkpoints"
    metrics.mkdir(); checkpoints.mkdir()
    payload = {"global_step": 20, "total_loss": 1.5,
               **{f"stage2_step{i}_loss": i + 0.5 for i in range(4)},
               "lr": 5e-5, "grad_norm": 0.8, "trajectory_length": 2,
               "world_point_count": 123, "elapsed_training_time_sec": 40,
               "optimizer_step_time_sec": 2.0}
    (metrics / "step_0020.json").write_text(json.dumps(payload))
    (checkpoints / "checkpoint_step_0100.pt").touch()
    result = progress_summary(tmp_path)
    assert result["global_step"] == 20
    assert result["stage2_losses"] == [2.5, 3.5]
    assert result["recent_20_average_step_sec"] == 2.0
