import numpy as np

from long_video.oracle_training.spatial_memory_warp import SpatialMemoryWarpBank


def _camera(tx=0.0):
    pose = np.eye(4, dtype=np.float32)
    pose[0, 3] = tx
    intrinsics = np.array([[20.0, 0.0, 7.5], [0.0, 20.0, 5.5], [0.0, 0.0, 1.0]], np.float32)
    return pose, intrinsics


def test_generated_boundary_uses_causal_depth_and_renders_fixed_query():
    height, width = 12, 16
    pose, intrinsics = _camera()
    rgb = np.full((height, width, 3), 127, np.uint8)
    depth = np.full((height, width), 2.0, np.float32)
    visibility = np.ones((height, width), bool)
    confidence = np.full((height, width), 0.75, np.float32)
    bank = SpatialMemoryWarpBank(translation_threshold=3.0, rotation_threshold_degrees=30.0)
    entry = bank.add_generated_boundary(
        rgb=rgb, depth=depth, visibility=visibility, confidence=confidence,
        pose=pose, intrinsics=intrinsics, frame_id=32, chunk_id=0,
        provenance={"uses_future_gt": False},
    )
    assert len(entry.points_xyz) == height * width
    result = bank.render_query(
        poses=np.stack([pose] * 33), intrinsics=np.stack([intrinsics] * 33),
        height=height, width=width, device="cpu", point_radius=0,
    )
    assert result["report"]["memory_hit"]
    assert result["report"]["uses_future_gt"] is False
    assert result["rgb"].shape == (33, height, width, 3)
    assert result["visibility"].shape == (33, height, width)
    assert np.isfinite(result["confidence"]).all()


def test_memory_miss_keeps_full_invalid_slots_and_does_not_mutate_query_inputs():
    pose, intrinsics = _camera()
    query_pose, _ = _camera(4.0)
    bank = SpatialMemoryWarpBank(translation_threshold=3.0, rotation_threshold_degrees=30.0)
    bank.add_generated_boundary(
        rgb=np.zeros((12, 16, 3), np.uint8), depth=np.ones((12, 16), np.float32),
        visibility=np.ones((12, 16), bool), confidence=np.ones((12, 16), np.float32),
        pose=pose, intrinsics=intrinsics, frame_id=32, chunk_id=0,
        provenance={"uses_future_gt": False},
    )
    poses = np.stack([query_pose] * 33)
    result = bank.render_query(
        poses=poses, intrinsics=np.stack([intrinsics] * 33),
        height=12, width=16, device="cpu",
    )
    assert result["report"]["memory_hit"] is False
    assert result["rgb"].shape == (33, 12, 16, 3)
    assert not result["visibility"].any()
    assert not result["confidence"].any()
    np.testing.assert_array_equal(poses, np.stack([query_pose] * 33))
