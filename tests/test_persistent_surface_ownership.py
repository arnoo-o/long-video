from types import SimpleNamespace

import numpy as np

from long_video.initialization.geometry_backend import GeometryPrediction
from long_video.initialization.recal3r_world_accumulator import (
    MATCH_BASE_TOLERANCE, ONLINE_FUSION_VOXEL_SIZE,
    SOURCE_FREE_SPACE_BASE_TOLERANCE, ReCal3RWorldAccumulator,
)
from long_video.types import CameraBatch, ScaleMetadata, SpatialNode


H = W = 2


def prediction(z, *, x=0.0, y=0.0, confidence=1.0, pixel=(0, 0)):
    xyz = np.full((H, W, 3), np.nan, np.float32)
    conf = np.zeros((H, W), np.float32)
    xyz[pixel] = (x, y, z); conf[pixel] = confidence
    return GeometryPrediction(np.array([conf]), np.array([conf]),
        point_maps=xyz[None], geometry_confidence=conf[None])


class Backend:
    def __init__(self, predictions): self.predictions = [prediction(1.0)] + list(predictions)
    def reset(self): pass
    def replay_prefix(self, rgb, c2w, intrinsics, **kwargs): return self.predictions[:len(rgb)]
    def get_state(self): return {"alignment": {"status": "locked", "scale": 1.0}}


def node():
    xyz = np.array([[0, 0, 1]], np.float32); rgb = np.array([[10, 20, 30]], np.uint8)
    pose = np.eye(4, dtype=np.float32); k = np.eye(3, dtype=np.float32)
    result = SpatialNode("node_000", "active", None, pose, 0, 0, xyz[0], xyz[0],
        np.zeros((1, H, W, 3), np.uint8), np.ones((1, H, W), np.float32), pose[None], k[None],
        xyz, rgb, np.ones(1, np.float32), np.ones(1, np.int8), np.ones(1, np.uint16), scale=ScaleMetadata())
    result.appearance_anchors = {"anchor_rgb": rgb.copy(), "anchor_confidence": np.ones(1, np.float32),
        "anchor_frame": np.zeros(1, np.int32), "source_locked": np.ones(1, bool)}
    return result


def association(count, *, visible=True, depth=1.0, point_index=0):
    dep = np.full((count, H, W), np.nan, np.float32); vis = np.zeros((count, H, W), bool)
    index = np.full((count, H, W), -1, np.int64); winning = np.zeros((count, H, W, 3), np.float32)
    if visible:
        dep[:, 0, 0] = depth; vis[:, 0, 0] = True; index[:, 0, 0] = point_index; winning[:, 0, 0] = (0, 0, depth)
    cameras = CameraBatch(np.repeat(np.eye(4, dtype=np.float32)[None], count, 0),
        np.repeat(np.eye(3, dtype=np.float32)[None], count, 0), H, W)
    warp = SimpleNamespace(depth=dep, visibility=vis, point_index=index, winning_xyz_world=winning)
    return warp, cameras


def run(predictions, warp, cameras):
    accumulator = ReCal3RWorldAccumulator(Backend(predictions), node(), trajectory_id="test")
    accumulator.prepare_chunk_association(warp, cameras)
    count = len(predictions)
    accumulator.update_chunk(np.zeros((count, H, W, 3), np.uint8), cameras.c2w, cameras.intrinsics, range(1, count + 1))
    return accumulator


def test_plane_depth_jitter_plateaus_and_never_moves_owned_xyz():
    predictions = [prediction(1.0 + ((index % 3) - 1) * .005) for index in range(32)]
    warp, cameras = association(32)
    accumulator = run(predictions, warp, cameras)
    assert len(accumulator.get_point_world().points_xyz) == 1
    np.testing.assert_array_equal(accumulator.get_point_world().points_xyz, np.array([[0, 0, 1]], np.float32))
    assert int(accumulator.get_point_world().observation_count[0]) == 33
    assert accumulator.last_update_metrics["existing_xyz_moved_count"] == 0
    assert accumulator.last_update_metrics["association_match_pixels"] == 32


def test_initial_recal_chunk_does_not_conflict_with_pi3x_w0():
    predictions = [prediction(.5), prediction(1.5)]
    warp, cameras = association(2)
    accumulator = run(predictions, warp, cameras)
    assert len(accumulator.get_point_world().points_xyz) == 2
    assert int(accumulator.get_point_world().observation_count[0]) == 1
    assert accumulator.last_update_metrics["association_conflict_pixels"] == 0
    assert accumulator.last_update_metrics["association_novel_pixels"] == 2
    assert accumulator.last_update_metrics["source_free_space_rejected"] == 1
    assert accumulator.last_update_metrics["novel_committed_points"] == 1


def test_front_and_back_current_view_conflicts_are_rejected_after_chunk0():
    backend = Backend([prediction(1.0)])
    accumulator = ReCal3RWorldAccumulator(backend, node(), trajectory_id="test")
    warp, cameras = association(1)
    accumulator.prepare_chunk_association(warp, cameras)
    accumulator.update_chunk(np.zeros((1, H, W, 3), np.uint8), cameras.c2w, cameras.intrinsics, [1])
    backend.predictions.extend([prediction(.5), prediction(1.5)])
    warp, cameras = association(2)
    accumulator.prepare_chunk_association(warp, cameras)
    accumulator.update_chunk(np.zeros((2, H, W, 3), np.uint8), cameras.c2w, cameras.intrinsics, [2, 3])
    assert len(accumulator.get_point_world().points_xyz) == 1
    assert accumulator.last_update_metrics["association_conflict_pixels"] == 2


def test_source_free_space_violation_is_hard_rejected():
    warp, cameras = association(1, visible=False)
    accumulator = run([prediction(.5)], warp, cameras)
    assert len(accumulator.get_point_world().points_xyz) == 1
    assert accumulator.last_update_metrics["source_free_space_rejected"] == 1
    assert accumulator.last_update_metrics["novel_committed_points"] == 0


def test_single_frame_novel_surface_commits_immediately():
    warp, cameras = association(1, visible=False)
    accumulator = run([prediction(2.0)], warp, cameras)
    assert len(accumulator.get_point_world().points_xyz) == 2
    assert accumulator.last_update_metrics["novel_committed_points"] == 1
    assert int(accumulator.get_point_world().observation_count[1]) == 1


def test_surface_behind_source_or_outside_source_fov_commits():
    warp, cameras = association(2, visible=False)
    accumulator = run([prediction(2.0), prediction(2.0, x=10.0)], warp, cameras)
    assert accumulator.last_update_metrics["source_free_space_rejected"] == 0
    assert accumulator.last_update_metrics["novel_committed_points"] == 2


def test_confirmed_surface_reobserves_as_match_and_source_rgb_stays_locked():
    predictions = [prediction(2.0), prediction(2.0), prediction(2.0)]
    warp, cameras = association(3, visible=False)
    accumulator = run(predictions, warp, cameras)
    assert len(accumulator.get_point_world().points_xyz) == 2
    committed = accumulator.get_point_world().points_xyz[1].copy()
    # Next frame sees the newly owned point and the source point with arbitrarily
    # high confidence/color; neither XYZ nor source appearance can change.
    accumulator.backend.predictions.append(prediction(2.001, confidence=100.0))
    warp, cameras = association(1, visible=True, depth=2.0, point_index=1)
    accumulator.prepare_chunk_association(warp, cameras)
    accumulator.update_chunk(np.full((1, H, W, 3), 255, np.uint8), cameras.c2w, cameras.intrinsics, [4])
    np.testing.assert_array_equal(accumulator.get_point_world().points_xyz[1], committed)
    np.testing.assert_array_equal(accumulator.get_point_world().points_rgb[0], [10, 20, 30])
    assert accumulator.last_update_metrics["association_match_pixels"] == 1
    assert accumulator.last_update_metrics["novel_committed_points"] == 0


def test_source_locked_rgb_rejects_higher_confidence_generated_color():
    backend = Backend([prediction(1.0, confidence=100.0)])
    accumulator = ReCal3RWorldAccumulator(backend, node(), trajectory_id="test")
    warp, cameras = association(1, visible=True, depth=1.0, point_index=0)
    accumulator.prepare_chunk_association(warp, cameras)
    accumulator.update_chunk(np.full((1, H, W, 3), 255, np.uint8), cameras.c2w, cameras.intrinsics, [1])
    np.testing.assert_array_equal(accumulator.get_point_world().points_rgb[0], [10, 20, 30])
    assert bool(accumulator.get_point_world().appearance_anchors["source_locked"][0])


def test_replay_is_exactly_once():
    accumulator = ReCal3RWorldAccumulator(Backend([prediction(2.0)]), node(), trajectory_id="test")
    warp, cameras = association(1, visible=False); accumulator.prepare_chunk_association(warp, cameras)
    accumulator.update_chunk(np.zeros((1, H, W, 3), np.uint8), cameras.c2w, cameras.intrinsics, [1])
    accumulator.backend.predictions.append(prediction(2.0)); accumulator.prepare_chunk_association(warp, cameras)
    try:
        accumulator.update_chunk(np.zeros((1, H, W, 3), np.uint8), cameras.c2w, cameras.intrinsics, [1])
    except RuntimeError as error:
        assert "processed twice" in str(error)
    else:
        raise AssertionError("one global frame was submitted twice")


def test_three_by_three_search_recovers_one_pixel_reprojection_error():
    pred = prediction(1.0, pixel=(0, 1))
    warp, cameras = association(1, visible=True, depth=1.0, point_index=0)
    accumulator = run([pred], warp, cameras)
    assert accumulator.last_update_metrics["association_match_pixels"] == 1
    assert accumulator.last_update_metrics["association_conflict_pixels"] == 0
    assert accumulator.last_update_metrics["association_novel_pixels"] == 0


def test_chunk_local_voxel_median_best_rgb_and_distinct_frame_count():
    predictions = [
        prediction(2.001, x=.001, confidence=.2),
        prediction(2.004, x=.004, confidence=.9),
        prediction(2.007, x=.007, confidence=.4),
    ]
    warp, cameras = association(3, visible=False)
    accumulator = ReCal3RWorldAccumulator(Backend(predictions), node(), trajectory_id="test")
    accumulator.prepare_chunk_association(warp, cameras)
    images = np.zeros((3, H, W, 3), np.uint8)
    images[0, 0, 0] = (20, 0, 0); images[1, 0, 0] = (90, 0, 0); images[2, 0, 0] = (40, 0, 0)
    accumulator.update_chunk(images, cameras.c2w, cameras.intrinsics, [1, 2, 3])
    world = accumulator.get_point_world()
    np.testing.assert_allclose(world.points_xyz[1], [.004, 0, 2.004], atol=1e-6)
    np.testing.assert_array_equal(world.points_rgb[1], [90, 0, 0])
    assert int(world.observation_count[1]) == 3
    assert accumulator.last_update_metrics["novel_committed_points"] == 1
    assert "novel_fusion_seconds" in accumulator.last_update_metrics


def test_online_voxel005_does_not_expand_association_tolerances():
    assert ONLINE_FUSION_VOXEL_SIZE == .05
    assert MATCH_BASE_TOLERANCE == .04
    assert SOURCE_FREE_SPACE_BASE_TOLERANCE == .06
    backend = Backend([prediction(1.0)])
    accumulator = ReCal3RWorldAccumulator(backend, node(), trajectory_id="test")
    warp, cameras = association(1)
    accumulator.prepare_chunk_association(warp, cameras)
    accumulator.update_chunk(np.zeros((1, H, W, 3), np.uint8), cameras.c2w, cameras.intrinsics, [1])
    backend.predictions.append(prediction(1.07))
    accumulator.prepare_chunk_association(warp, cameras)
    accumulator.update_chunk(np.zeros((1, H, W, 3), np.uint8), cameras.c2w, cameras.intrinsics, [2])
    assert accumulator.last_update_metrics["association_conflict_pixels"] == 1
    assert accumulator.last_update_metrics["association_match_pixels"] == 0


def test_source_free_space_keeps_original_point06_tolerance_with_voxel005():
    warp, cameras = association(1, visible=False)
    accumulator = run([prediction(.92)], warp, cameras)
    assert accumulator.last_update_metrics["source_free_space_rejected"] == 1
    assert accumulator.last_update_metrics["novel_committed_points"] == 0
