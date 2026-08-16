import numpy as np
import torch

from long_video.initialization.recal3r_geometry_backend import ReCal3RGeometryBackend, _Frame


def _backend(scale=2.0):
    backend = ReCal3RGeometryBackend.__new__(ReCal3RGeometryBackend)
    backend.confidence_threshold = 0.85
    backend.confidence_temperature = 0.35
    backend._geometry_validation = {}
    backend._alignment_scale = float(scale)
    backend._alignment_metadata = {
        "status": "locked",
        "anchor": "test_source_geometry",
        "scale": float(scale),
    }
    return backend


def _prediction(height=384, width=512, depth=1.0):
    points = np.zeros((height, width, 3), np.float32)
    points[..., 2] = float(depth)
    confidence = np.full((height, width), 1.0, np.float32)
    return {
        "pts3d_in_self_view": torch.from_numpy(points)[None],
        "conf_self": torch.from_numpy(confidence)[None],
    }


def test_recal_self_view_points_use_commanded_camera_pose():
    height, width = 384, 512
    commanded = np.eye(4, dtype=np.float32)
    commanded[:3, 3] = np.array([10.0, -3.0, 5.0], np.float32)
    frame = _Frame(
        rgb=np.zeros((height, width, 3), np.uint8),
        c2w=commanded,
        intrinsics=np.eye(3, dtype=np.float32),
        identity="traj:1",
    )

    (depth, calibrated, world), raw_depth = _backend(scale=2.0)._geometry_for(
        frame, _prediction(height, width)
    )

    assert np.allclose(raw_depth, 1.0)
    assert np.allclose(depth, 2.0)
    assert np.all(np.isfinite(world))
    assert np.allclose(world[..., 0], 10.0)
    assert np.allclose(world[..., 1], -3.0)
    assert np.allclose(world[..., 2], 7.0)
    assert np.all(calibrated > 0)


def test_source_geometry_alignment_sets_scale_without_camera_baseline():
    height, width = 384, 512
    backend = ReCal3RGeometryBackend.__new__(ReCal3RGeometryBackend)
    backend._alignment_scale = None
    backend._alignment_metadata = {"status": "pending"}
    backend._frames = [
        _Frame(
            rgb=np.zeros((height, width, 3), np.uint8),
            c2w=np.eye(4, dtype=np.float32),
            intrinsics=np.eye(3, dtype=np.float32),
            identity="traj:0",
        )
    ]
    backend._last_predictions = [object()]
    backend.raw_recal_depth = lambda trajectory_id, index: np.ones(
        (height, width), np.float32
    )
    backend._cache_all_results = lambda: None
    backend.replay_predictions = lambda: []

    source_world_depth = np.full((height, width), 2.0, np.float32)
    backend.lock_source_geometry_alignment(source_world_depth)

    assert np.isclose(backend._alignment_scale, 2.0)
    assert backend._alignment_metadata["status"] == "locked"
    assert backend._alignment_metadata["anchor"] == "pi3x_w0_source_geometry_alignment_v2"
    assert backend._alignment_metadata["placement"] == "commanded_c2w"


def test_commanded_pose_remains_authoritative_after_source_scale_lock():
    height, width = 384, 512
    backend = _backend(scale=1.5)
    commanded = np.eye(4, dtype=np.float32)
    commanded[:3, 3] = np.array([-4.0, 2.0, 8.0], np.float32)
    frame = _Frame(
        rgb=np.zeros((height, width, 3), np.uint8),
        c2w=commanded,
        intrinsics=np.eye(3, dtype=np.float32),
        identity="traj:17",
    )

    depth, _, world = backend._geometry_for(frame, _prediction(height, width))[0]
    assert np.allclose(depth, 1.5)
    assert np.allclose(world[..., 0], -4.0)
    assert np.allclose(world[..., 1], 2.0)
    assert np.allclose(world[..., 2], 9.5)
