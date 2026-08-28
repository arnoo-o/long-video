import importlib.util
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_rgbd_memory_dataset.py"
SPEC = importlib.util.spec_from_file_location("build_rgbd_memory_dataset", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_bonn_marker_rotation_is_rigid_so3():
    rotation = MODULE.BONN_T_MARKER[:3, :3]
    assert np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-8)
    assert np.isclose(np.linalg.det(rotation), 1.0, atol=1e-8)
