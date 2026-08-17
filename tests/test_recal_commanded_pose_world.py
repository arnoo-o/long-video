import numpy as np
import torch

from long_video.geometry.backprojection import backproject_z_depth
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


def _intrinsics(height, width):
    return np.array([[420.0, 0.0, width / 2.0], [0.0, 395.0, height / 2.0], [0.0, 0.0, 1.0]], np.float32)


def test_recal_self_view_points_use_commanded_camera_pose():
    height, width = 384, 512
    commanded = np.eye(4, dtype=np.float32)
    commanded[:3, 3] = np.array([10.0, -3.0, 5.0], np.float32)
    frame = _Frame(
        rgb=np.zeros((height, width, 3), np.uint8),
        c2w=commanded,
        intrinsics=_intrinsics(height, width),
        identity="traj:1",
    )

    (depth, calibrated, world), raw_depth = _backend(scale=2.0)._geometry_for(
        frame, _prediction(height, width)
    )

    assert np.allclose(raw_depth, 1.0)
    assert np.allclose(depth, 2.0)
    assert np.all(np.isfinite(world))
    expected = backproject_z_depth(np.full((height, width), 2.0, np.float32), frame.intrinsics)
    expected = expected @ commanded[:3, :3].T + commanded[:3, 3]
    assert np.allclose(world, expected)
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
    assert backend._alignment_metadata["placement"] == "commanded_intrinsics_and_c2w"


def test_commanded_pose_remains_authoritative_after_source_scale_lock():
    height, width = 384, 512
    backend = _backend(scale=1.5)
    commanded = np.eye(4, dtype=np.float32)
    commanded[:3, 3] = np.array([-4.0, 2.0, 8.0], np.float32)
    frame = _Frame(
        rgb=np.zeros((height, width, 3), np.uint8),
        c2w=commanded,
        intrinsics=_intrinsics(height, width),
        identity="traj:17",
    )

    depth, _, world = backend._geometry_for(frame, _prediction(height, width))[0]
    assert np.allclose(depth, 1.5)
    expected = backproject_z_depth(np.full((height, width), 1.5, np.float32), frame.intrinsics)
    expected = expected @ commanded[:3, :3].T + commanded[:3, 3]
    assert np.allclose(world, expected)


def test_depth_backprojection_round_trips_through_same_commanded_camera():
    height, width = 97, 151
    depth = np.linspace(0.4, 9.0, height * width, dtype=np.float32).reshape(height, width)
    intrinsics = _intrinsics(height, width)
    angle = 0.37
    c2w = np.array([
        [np.cos(angle), 0.0, np.sin(angle), 1.2],
        [0.0, 1.0, 0.0, -0.7],
        [-np.sin(angle), 0.0, np.cos(angle), 3.1],
        [0.0, 0.0, 0.0, 1.0],
    ], np.float32)
    local = backproject_z_depth(depth, intrinsics)
    world = local @ c2w[:3, :3].T + c2w[:3, 3]
    recovered = (world - c2w[:3, 3]) @ c2w[:3, :3]
    projected = recovered @ intrinsics.T
    uv = projected[..., :2] / projected[..., 2:3]
    y, x = np.indices((height, width), dtype=np.float32)
    error = np.linalg.norm(uv - np.stack([x, y], -1), axis=-1)
    assert float(np.median(error)) < 0.25


def test_geometry_for_ignores_recal_self_view_xy():
    height, width = 384, 512
    frame = _Frame(
        rgb=np.zeros((height, width, 3), np.uint8),
        c2w=np.eye(4, dtype=np.float32),
        intrinsics=_intrinsics(height, width),
        identity="traj:xy",
    )
    backend = _backend(scale=1.7)
    reference = _prediction(height, width, depth=2.0)
    perturbed = _prediction(height, width, depth=2.0)
    noisy = perturbed["pts3d_in_self_view"].numpy()
    noisy[..., 0] = 5000.0
    noisy[..., 1] = -7000.0
    _, _, world_reference = backend._geometry_for(frame, reference)[0]
    _, _, world_perturbed = backend._geometry_for(frame, perturbed)[0]
    assert np.allclose(world_reference, world_perturbed, equal_nan=True)


def test_geometry_for_uses_p40_raw_confidence_over_valid_grid_as_threshold():
    height, width = 384, 512
    frame = _Frame(
        rgb=np.zeros((height, width, 3), np.uint8),
        c2w=np.eye(4, dtype=np.float32),
        intrinsics=_intrinsics(height, width),
        identity="traj:19",
    )
    prediction = _prediction(height, width, depth=2.0)
    confidence = prediction["conf_self"].numpy()
    confidence[..., : width // 4] = 0.1
    confidence[..., width // 4 :] = 1.5
    backend = _backend(scale=1.0)
    _, calibrated, world = backend._geometry_for(frame, prediction)[0]
    validation = backend.geometry_validation("traj", 19)
    assert np.isclose(validation["effective_confidence_threshold"], 1.5)
    assert validation["confidence_threshold_mode"] == "p40_valid_grid_raw_confidence"
    assert not np.isfinite(world[:, : width // 4]).any()
    assert np.isfinite(world[:, width // 4 :]).all()
    assert np.all(calibrated[:, : width // 4] == 0)


def test_geometry_for_supports_configurable_raw_confidence_quantile():
    height, width = 384, 512
    frame = _Frame(np.zeros((height, width, 3), np.uint8), np.eye(4, dtype=np.float32),
                   _intrinsics(height, width), "traj:23")
    prediction = _prediction(height, width, depth=2.0)
    confidence = prediction["conf_self"].numpy()
    confidence[..., : width // 2] = 0.2
    confidence[..., width // 2:] = 1.8
    backend = _backend(scale=1.0)
    backend.confidence_quantile = 0.75
    backend._geometry_for(frame, prediction)
    validation = backend.geometry_validation("traj", 23)
    assert np.isclose(validation["effective_confidence_threshold"], 1.8)
    assert validation["confidence_threshold_mode"] == "p75_valid_grid_raw_confidence"
    assert validation["confidence_quantile"] == .75
