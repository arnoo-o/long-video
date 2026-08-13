import json

import torch

from long_video.training.stage2_cleanup import (
    BalancedChunkSampler, Stage2FlowObserver, curriculum_max_chunks,
    select_balanced_training_records,
)
from scripts.query_stage2_training_progress import progress_summary


def test_curriculum_and_balanced_supervision():
    assert [curriculum_max_chunks(x) for x in (1, 400, 401, 800, 801, 1001, 1201, 1401)] == [1, 1, 2, 2, 3, 4, 5, 6]
    sampler = BalancedChunkSampler(seed=7)
    selected = [sampler.choose(i, forced_max_chunks=2)[1] for i in range(1, 11)]
    assert abs(selected.count(0) - selected.count(1)) <= 1


def test_balanced_training_record_selection():
    records = [
        {"trajectory_id": f"{environment}_{index:03d}", "split": "train",
         "environment": environment}
        for environment in ("indoor", "outdoor") for index in range(75)
    ]
    selected = select_balanced_training_records(records, 100)
    assert len(selected) == 100
    assert sum(item["environment"] == "indoor" for item in selected) == 50


def test_stage2_observer_backprops_only_selected_step_and_detaches():
    z_gt = torch.zeros(1, 1, 1, 1, 1)
    observer = Stage2FlowObserver(z_gt, scheduler=None, selected_step=2)
    for step in range(4):
        prediction = torch.ones_like(z_gt, requires_grad=True)
        sample = torch.full_like(z_gt, 2.0)
        result = observer({"stage_id": 2, "step_id": step, "model_output": prediction,
                           "sample": sample, "timestep": torch.tensor(10),
                           "dmd_timesteps": torch.tensor([10, 0]),
                           "dmd_sigmas": torch.tensor([0.5, 0.0])})
        if step == 2:
            assert prediction.grad is not None and torch.isfinite(prediction.grad).all()
        else:
            assert prediction.grad is None
        assert not result[0].requires_grad and not result[1].requires_grad
    observer.assert_complete()
    assert set(observer.losses) == {2}


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
    assert result["stage2_losses"] == [0.5, 1.5, 2.5, 3.5]
    assert result["recent_20_average_step_sec"] == 2.0
