import numpy as np

from long_video.data.rgbd_memory import (
    associate_timestamp_streams, center_crop_resize_geometry, localize_c2w,
)


def test_crop_resize_updates_intrinsics_exactly():
    K = np.array([[525.0, 0, 319.5], [0, 525.0, 239.5], [0, 0, 1]])
    crop, transformed = center_crop_resize_geometry(480, 640, K)
    left, top, right, bottom = crop
    assert (right - left, bottom - top) == (640, 369)
    assert transformed[0, 0] == 525.0 * 832 / 640
    assert transformed[1, 1] == 525.0 * 480 / 369
    assert transformed[1, 2] == (239.5 - top) * 480 / 369


def test_three_way_timestamp_association_is_unique_and_bounded():
    rgb = [(0.0, "r0"), (0.033, "r1"), (0.066, "r2")]
    depth = [(0.002, "d0"), (0.035, "d1"), (0.2, "bad")]
    pose = [(0.001, 0, 0, 0, 0, 0, 0, 1), (0.034, 1, 0, 0, 0, 0, 0, 1)]
    rows, stats = associate_timestamp_streams(rgb, depth, pose, max_difference=.02)
    assert len(rows) == 2 and stats["dropped_rgb"] == 1
    assert rows[0]["rgb"] == "r0" and rows[1]["depth"] == "d1"


def test_local_pose_starts_at_identity():
    poses = np.repeat(np.eye(4)[None], 3, axis=0)
    poses[:, 0, 3] = (4, 5, 7)
    local = localize_c2w(poses)
    assert np.allclose(local[0], np.eye(4))
    assert np.allclose(local[:, 0, 3], (0, 1, 3))
