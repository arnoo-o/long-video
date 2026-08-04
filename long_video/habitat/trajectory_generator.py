"""Deterministic camera trajectories for Habitat validation."""
from __future__ import annotations

import numpy as np

from ..data.controls import integrate_controls


def standard_validation_controls(delta_time=0.1):
    dt = float(delta_time)
    controls = []
    controls += [{"yaw_delta": np.deg2rad(10), "delta_time": dt} for _ in range(9)]
    controls += [{"forward": 1, "delta_time": dt} for _ in range(12)]
    controls += [{"backward": 1, "delta_time": dt} for _ in range(5)]
    controls += [{"strafe_left": 1, "delta_time": dt} for _ in range(5)]
    controls += [{"strafe_right": 1, "delta_time": dt} for _ in range(10)]
    controls += [{"forward": 1, "yaw_delta": np.deg2rad(-3), "delta_time": dt} for _ in range(12)]
    controls += [{"forward": 1, "delta_time": dt} for _ in range(10)]
    controls += [{"backward": 1, "delta_time": dt} for _ in range(22)]
    return controls


def generate_validation_trajectory(initial_c2w=None, move_speed=1.0, delta_time=0.1):
    initial = np.eye(4, dtype=np.float32) if initial_c2w is None else np.asarray(initial_c2w, np.float32)
    controls = standard_validation_controls(delta_time)
    return integrate_controls(initial, controls, move_speed=move_speed), controls
