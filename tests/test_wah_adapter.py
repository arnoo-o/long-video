from types import SimpleNamespace

import numpy as np

from long_video.wah.adapter import WAHAdapter


def test_wah_warp_rgb_fills_only_invisible_pixels_with_boundary_mean():
    rgb = np.asarray([
        [[[0.1, 0.2, 0.3], [0.9, 0.8, 0.7]],
         [[0.4, 0.5, 0.6], [0.3, 0.2, 0.1]]],
    ], dtype=np.float32)
    visibility = np.asarray([[[1, 0], [0, 1]]], dtype=bool)
    confidence = np.full((1, 2, 2), 0.75, dtype=np.float32)
    boundary = np.asarray([
        [[0, 30, 60], [30, 60, 90]],
        [[60, 90, 120], [90, 120, 150]],
    ], dtype=np.uint8)

    inputs = WAHAdapter.warp_inputs(
        SimpleNamespace(rgb=rgb, visibility=visibility, confidence=confidence),
        fill_frame=boundary,
    )

    expected_mean = boundary.astype(np.float32).mean(axis=(0, 1)) / 255.0
    np.testing.assert_array_equal(inputs["warp_video"][visibility], rgb[visibility])
    np.testing.assert_allclose(
        inputs["warp_video"][~visibility],
        np.broadcast_to(expected_mean, (2, 3)),
        rtol=0,
        atol=1e-7,
    )
    np.testing.assert_array_equal(inputs["warp_visibility_mask"], visibility[None, None])
