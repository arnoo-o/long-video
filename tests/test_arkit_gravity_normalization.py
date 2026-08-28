import importlib.util
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "normalize_arkit_gravity.py"
SPEC = importlib.util.spec_from_file_location("normalize_arkit_gravity", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_camera_rotation_is_proper_for_each_supported_image_rotation():
    for clockwise in (0, 90, 180, 270):
        rotation = MODULE._camera_rotation(clockwise)
        assert np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-8)
        assert np.isclose(np.linalg.det(rotation), 1.0, atol=1e-8)


def test_180_rotation_updates_intrinsics_and_pixels_together():
    rgb = np.zeros((480, 832, 3), dtype=np.uint8)
    rgb[0, 0] = (1, 2, 3)
    depth = np.zeros((480, 832), dtype=np.uint16)
    depth[0, 0] = 1234
    K = np.asarray(((500.0, 0, 123.0), (0, 501.0, 234.0), (0, 0, 1)))
    out_rgb, out_depth, out_K = MODULE._rotate_pair(rgb, depth, K, 180)
    assert out_rgb.shape == rgb.shape and out_depth.shape == depth.shape
    assert out_K[0, 0] > 0 and out_K[1, 1] > 0
