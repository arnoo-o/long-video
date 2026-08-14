import numpy as np
import pytest

torch = pytest.importorskip("torch")

from long_video.geometry.geotoken import (
    GEOTOKEN_BLOCKS,
    GeometryTokenizer,
    GeoTokenConditioner,
    TEMPORAL_GROUPS,
    geometry_channels_from_render,
)
from long_video.training.geotoken import (
    BalancedRolloutSampler,
    augment_partial_voxels,
    stable_voxel_hash,
    voxel_uniform_stream,
    checkpoint_names,
    max_chunks_for_step,
    phase_for_step,
)
from long_video.geometry.geotoken_runtime import PointWorldGeoTokenProvider
from long_video.initialization.recal3r_geometry_backend import ReCal3RGeometryBackend


def test_temporal_layout_and_invalid_geometry_are_exact_zero():
    assert TEMPORAL_GROUPS == ((0,), (1, 2, 3, 4), (5, 6, 7, 8),
                               (9, 10, 11, 12), (13, 14, 15, 16),
                               (17, 18, 19, 20), (21, 22, 23, 24),
                               (25, 26, 27, 28), (29, 30, 31, 32))
    tokenizer = GeometryTokenizer(24)
    value = torch.randn(2, 12, 33, 3, 5)
    value[:, 10:] = 0
    encoded, support = tokenizer(value)
    assert encoded.shape == (2, 24, 9, 3, 5)
    assert support.shape == (2, 1, 9, 3, 5)
    assert torch.count_nonzero(encoded) == 0
    assert torch.count_nonzero(support) == 0


def test_zero_gates_are_strict_noop_and_blocks_are_fixed():
    module = GeoTokenConditioner(16)
    assert module.block_indices == GEOTOKEN_BLOCKS
    hidden = torch.randn(1, 7, 16)
    module.set_active(torch.randn_like(hidden), torch.ones(1, 7, 1))
    args, _ = module._make_hook(8)(None, (hidden, None, None, None, 7), {})
    assert torch.equal(args[0], hidden)


def test_render_channels_use_fixed_source_frame_and_mask_invalid():
    xyz = np.ones((1, 2, 2, 3), np.float32)
    depth = np.ones((1, 2, 2), np.float32) * 2
    visibility = np.array([[[1, 0], [1, 1]]], bool)
    confidence = np.ones((1, 2, 2), np.float32)
    pose = np.eye(4, dtype=np.float32)[None]
    value = geometry_channels_from_render(
        xyz, depth, visibility, confidence, pose, np.zeros(3), 2.0,
    )
    assert value.shape == (1, 2, 2, 12)
    assert np.count_nonzero(value[0, 0, 1]) == 0
    assert value[0, 0, 0, 10] == 1
    assert value[0, 0, 0, 11] == 1


def test_curriculum_balances_lengths_and_checkpoint_boundaries():
    expected = {1: ("A", 1), 161: ("A", 2), 321: ("A", 3),
                501: ("B", 1), 901: ("B", 3), 1101: ("C", 1),
                1701: ("C", 4), 1861: ("C", 6)}
    for step, (phase, maximum) in expected.items():
        assert phase_for_step(step) == phase
        assert max_chunks_for_step(step) == maximum
    sampler = BalancedRolloutSampler(7)
    values = [sampler.choose_length(1701 + index) for index in range(40)]
    counts = [values.count(index) for index in range(1, 5)]
    assert max(counts) - min(counts) <= 1
    assert checkpoint_names(80) == ("checkpoint_step_0080.pt",)
    assert checkpoint_names(500) == ("phase_a_final_step_0500.pt",)
    assert set(checkpoint_names(2000)) == {
        "checkpoint_step_2000.pt", "phase_c_final_step_2000.pt",
    }


def test_runtime_reads_actual_patch_grid_without_hardcoding():
    provider = object.__new__(PointWorldGeoTokenProvider)
    provider.current_c2w = np.eye(4, dtype=np.float32)[None].repeat(33, 0)
    provider.current_k = np.eye(3, dtype=np.float32)[None].repeat(33, 0)
    provider._encode_chunk = lambda _c2w, _k, h, w: (
        torch.zeros(1, 16, 9, h, w), torch.zeros(1, 1, 9, h, w),
    )
    provider._history_part = lambda *_args: None
    current, history, support = provider._build({
        "hidden_states": torch.zeros(1, 16, 9, 24, 40),
        "_geotoken_patch_size": (1, 2, 2),
    })
    assert current.tokens.shape == (1, 9 * 12 * 20, 16)
    assert history is None and support is None


def test_phase_b_voxel_corruption_is_rollout_consistent():
    class Args:
        point_dropout = 0.2
        confidence_dropout = 0.1
        depth_noise = 0.01
        xyz_jitter = 0.005
    points = np.array([[1, 0, 2], [2, 0, 3], [3, 0, 4]], np.float32)
    confidence = np.ones(3, np.float32)
    keys = np.array([[1, 0, 2], [2, 0, 3], [3, 0, 4]], np.int64)
    first = augment_partial_voxels(points, confidence, keys, source_center=np.zeros(3), scene_scale=2, args=Args(), seed=9)
    second = augment_partial_voxels(points, confidence, keys, source_center=np.zeros(3), scene_scale=2, args=Args(), seed=9)
    assert all(np.array_equal(a, b) for a, b in zip(first, second))


def test_phase_b_random_streams_are_independent():
    keys = np.arange(24, dtype=np.int64).reshape(8, 3)
    base = stable_voxel_hash(17, keys)
    point_dropout = voxel_uniform_stream(base, 1)
    confidence_dropout = voxel_uniform_stream(base, 2)
    jitter_x = voxel_uniform_stream(base, 3)
    jitter_y = voxel_uniform_stream(base, 5)
    jitter_z = voxel_uniform_stream(base, 7)
    assert not np.array_equal(point_dropout, confidence_dropout)
    assert not np.array_equal(jitter_x, jitter_y)
    assert not np.array_equal(jitter_y, jitter_z)


def test_recal3r_backend_reset_clears_trajectory_state_without_loading_model():
    backend = ReCal3RGeometryBackend("unused", "unused", "cpu")
    backend._frames = [object()]
    backend._state_args = object()
    backend._last_predictions = [object()]
    backend._sequence_version = 3
    backend.reset()
    assert backend.get_state()["frame_count"] == 0
    assert backend.get_state()["has_recurrent_state"] is False
    assert backend.get_state()["sequence_version"] == 0
