import numpy as np
import torch

from long_video.data.recal3r_full_scene import Sim3Alignment
from long_video.initialization.recal3r_geometry_backend import ReCal3RGeometryBackend, _Frame


def _backend(scale=2.0):
    backend = ReCal3RGeometryBackend.__new__(ReCal3RGeometryBackend)
    backend.confidence_threshold = 0.85
    backend.confidence_temperature = 0.35
    backend._geometry_validation = {}
    backend._alignment = Sim3Alignment(
        scale=float(scale),
        rotation=np.eye(3, dtype=np.float32),
        translation=np.zeros(3, dtype=np.float32),
        camera_alignment_error=0.0,
        camera_alignment_error_ratio=0.0,
        median_rotation_error_degrees=0.0,
        max_rotation_error_degrees=0.0,
    )
    return backend


def test_recal_self_view_points_use_commanded_camera_pose():
    # 384x512 maps are already on ReCal's resize/crop grid.
    height, width = 384, 512
    points = np.zeros((height, width, 3), np.float32)
    points[..., 2] = 1.0
    confidence = np.full((height, width), 1.0, np.float32)
    prediction = {
        "pts3d_in_self_view": torch.from_numpy(points)[None],
        "conf_self": torch.from_numpy(confidence)[None],
    }

    commanded = np.eye(4, dtype=np.float32)
    commanded[:3, 3] = np.array([10.0, -3.0, 5.0], np.float32)
    frame = _Frame(
        rgb=np.zeros((height, width, 3), np.uint8),
        c2w=commanded,
        intrinsics=np.eye(3, dtype=np.float32),
        identity="traj:1",
    )

    (depth, calibrated, world), raw_depth = _backend(scale=2.0)._geometry_for(frame, prediction)

    # Local z=1 at scale 2 must stay two units in front of the commanded
    # camera, independent of whatever camera pose ReCal itself predicted.
    assert np.allclose(raw_depth, 1.0)
    assert np.allclose(depth, 2.0)
    assert np.all(np.isfinite(world))
    assert np.allclose(world[..., 0], 10.0)
    assert np.allclose(world[..., 1], -3.0)
    assert np.allclose(world[..., 2], 7.0)
    assert np.all(calibrated > 0)


def test_causal_scale_lock_does_not_control_world_pose():
    target = np.repeat(np.eye(4, dtype=np.float32)[None], 3, axis=0)
    target[1, 0, 3] = 1.0
    target[2, 0, 3] = 2.0

    # ReCal sees the same motion in arbitrary units and with an unrelated
    # source translation. The scale lock should recover only the factor 2.
    recal = np.repeat(np.eye(4, dtype=np.float32)[None], 3, axis=0)
    recal[:, 1, 3] = 100.0
    recal[1, 0, 3] = 0.5
    recal[2, 0, 3] = 1.0

    alignment = ReCal3RGeometryBackend._lock_causal_recal_to_world(recal, target)
    assert np.isclose(alignment.scale, 2.0)
