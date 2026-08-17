import json
import math
import subprocess
import sys

import numpy as np

from long_video.data.controls import integrate_controls
from long_video.inference.trajectory import TrajectorySegment, build_controls, trajectory_document


def _distance(controls, positive, negative):
    return sum((item[positive] - item[negative]) * item["delta_time"]
               for chunk in controls for item in chunk)


def test_segment_controls_have_exact_chunks_and_totals():
    segments = [
        TrajectorySegment("right", 90, "backward", 20, 2),
        TrajectorySegment("left", 180, "front_left", 40, 3),
    ]
    controls = build_controls(segments)
    assert len(controls) == 5
    assert all(len(chunk) == 32 for chunk in controls)
    first = controls[:2]
    assert np.isclose(sum(item["yaw_delta"] for chunk in first for item in chunk), math.pi / 2)
    assert np.isclose(_distance(first, "backward", "forward"), 20)
    second = controls[2:]
    forward = _distance(second, "forward", "backward")
    left = _distance(second, "strafe_left", "strafe_right")
    assert np.isclose(math.hypot(forward, left), 40)
    assert np.isclose(forward, left)


def test_each_segment_eases_to_small_boundary_velocity():
    controls = build_controls([
        TrajectorySegment("none", 0, "forward", 8, 2),
        TrajectorySegment("right", 90, "right", 12, 2),
    ])
    flat = [item for chunk in controls for item in chunk]
    boundary = 64
    assert flat[boundary - 1]["forward"] < flat[boundary // 2]["forward"]
    assert flat[boundary]["strafe_right"] < flat[boundary + 32]["strafe_right"]


def test_controls_integrate_in_current_camera_basis():
    controls = build_controls([
        TrajectorySegment("right", 90, "none", 0, 1),
        TrajectorySegment("none", 0, "forward", 10, 1),
    ])
    poses = integrate_controls(np.eye(4, dtype=np.float32), [item for chunk in controls for item in chunk])
    displacement = poses[-1, :3, 3] - poses[31, :3, 3]
    assert np.isclose(np.linalg.norm(displacement), 10, atol=1e-4)
    assert abs(displacement[0]) > 9.9
    assert abs(displacement[2]) < 1e-3


def test_trajectory_document_is_json_serializable():
    document = trajectory_document([TrajectorySegment("left", 30, "back_right", 2, 1)])
    assert json.loads(json.dumps(document))["segments"][0]["movement"] == "back_right"


def test_invalid_segment_is_rejected():
    with np.testing.assert_raises(ValueError):
        build_controls([TrajectorySegment("none", 10, "none", 0, 1)])
