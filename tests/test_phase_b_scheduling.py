import numpy as np
import pytest

from long_video.oracle_training.history_bank import HistoryBankKey, history_bank_cache_key
from long_video.oracle_training.round_robin import (
    RoundRobinChunkScheduler,
    eligible_current_chunks,
)
from long_video.oracle_training.spatial_memory_prefix import SpatialMemoryPrefixBank, choose_prefix
from long_video.oracle_training.supervision import validate_current_chunk_supervision


def _pose(x=0.0, yaw=0.0):
    c, s = np.cos(yaw), np.sin(yaw)
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = ((c, -s, 0.0), (s, c, 0.0), (0.0, 0.0, 1.0))
    pose[0, 3] = x
    return pose


def test_round_robin_is_deterministic_and_uniform():
    assert eligible_current_chunks(5) == (1, 2, 3, 4)
    scheduler = RoundRobinChunkScheduler()
    observed = [scheduler.next_chunk("traj", 5) for _ in range(10)]
    assert observed == [1, 2, 3, 4, 1, 2, 3, 4, 1, 2]
    assert scheduler.counts_for("traj", 5) == {"1": 3, "2": 3, "3": 2, "4": 2}
    restored = RoundRobinChunkScheduler()
    restored.restore(scheduler.snapshot())
    assert restored.next_chunk("traj", 5) == 3
    with pytest.raises(ValueError):
        eligible_current_chunks(1)


def test_history_cache_key_carries_trajectory_and_chunk():
    key_a = history_bank_cache_key("trajectory-a", 1, "scene", "revisit")
    key_b = history_bank_cache_key("trajectory-a", 2, "scene", "revisit")
    assert key_a[:2] == ("trajectory-a", 1)
    assert key_a != key_b
    digest_a = HistoryBankKey(
        checkpoint_sha="x", global_step=1, scene_id="s", source_id="src",
        trajectory_id="trajectory-a", history_chunk_index=1,
        generation_config=(), prompt="p", seed=1,
    ).digest()
    digest_b = HistoryBankKey(
        checkpoint_sha="x", global_step=1, scene_id="s", source_id="src",
        trajectory_id="trajectory-a", history_chunk_index=2,
        generation_config=(), prompt="p", seed=1,
    ).digest()
    assert digest_a != digest_b


def test_shared_boundary_is_excluded_but_non_boundary_target_is_effective():
    weights = np.ones(9, dtype=np.float32)
    weights[0] = 0.0
    assert validate_current_chunk_supervision(weights, 9) == list(range(1, 9))
    with pytest.raises(ValueError):
        bad = np.zeros(9, dtype=np.float32)
        validate_current_chunk_supervision(bad, 9)
    with pytest.raises(ValueError):
        validate_current_chunk_supervision(np.ones(8, dtype=np.float32), 9)


def test_spatial_memory_thresholds_and_m0_priority():
    bank = SpatialMemoryPrefixBank(translation_threshold=3.0, rotation_threshold=30.0)
    support = np.ones((1, 1), np.float32)
    bank.add_if_novel(
        pose=_pose(), intrinsics=np.eye(3), latent=np.array([[[[1.0]]]]),
        visibility=support, confidence=support, frame_id=0, chunk_id=0, source_type="M0",
    )
    # A generated entry at the same pose cannot displace M0.
    result = bank.add_if_novel(
        pose=_pose(), intrinsics=np.eye(3), latent=np.array([[[[2.0]]]]),
        visibility=support, confidence=support, frame_id=1, chunk_id=1, source_type="generated",
    )
    assert result["created"] is False
    selected, _ = bank.query(_pose(x=2.9))
    assert selected is not None and selected[3].source_type == "M0"
    selected, _ = bank.query(_pose(x=3.1))
    assert selected is None
    latent, visibility, confidence, report = choose_prefix(
        bank, pose=_pose(x=4.0), m0_latent=np.array([[[[3.0]]]]),
        m0_visibility=np.ones((1, 1), np.float32),
    )
    assert report["prefix_source"] == "current_M0_boundary"
    assert float(latent.ravel()[0]) == 3.0
    _, _, _, invalid = choose_prefix(
        bank, pose=_pose(x=4.0), m0_latent=np.array([[[[3.0]]]]),
        m0_visibility=np.zeros((1, 1), np.float32),
    )
    assert invalid["prefix_source"] == "masked_invalid"
