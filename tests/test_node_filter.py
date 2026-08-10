import numpy as np

from long_video.memory.node_filter import filter_node_to_observed_erp
from long_video.types import SpatialNode


def test_observed_erp_filter_is_non_destructive_and_keeps_only_masked_rays():
    points = np.array([[0, 0, 1], [1, 0, 0], [0, 0, -1]], np.float32)
    node = SpatialNode(
        "node_000", "active", None, np.eye(4, dtype=np.float32), 0, 1.0,
        points.min(0), points.max(0),
        np.zeros((1, 2, 2, 3), np.uint8), np.ones((1, 2, 2), np.float32),
        np.eye(4, dtype=np.float32)[None], np.eye(3, dtype=np.float32)[None],
        points, np.arange(9, dtype=np.uint8).reshape(3, 3), np.ones(3, np.float32),
        np.array([0, 1, 1], np.int8), np.ones(3, np.int16),
        point_view_mask=np.arange(3, dtype=np.uint64),
    )
    mask = np.zeros((8, 16), bool)
    mask[4, 8] = True  # longitude=0, latitude=0: the +z source direction.

    filtered, report = filter_node_to_observed_erp(node, mask)

    np.testing.assert_array_equal(filtered.points_xyz, points[:1])
    np.testing.assert_array_equal(filtered.points_rgb, node.points_rgb[:1])
    np.testing.assert_array_equal(filtered.point_view_mask, node.point_view_mask[:1])
    assert report["kept_point_count"] == 1
    assert report["removed_completion_point_count"] == 2
    assert len(node.points_xyz) == 3
    assert node.quality_metrics == {}
