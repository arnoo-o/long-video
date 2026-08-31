from __future__ import annotations

import numpy as np

from scripts.resample_bonn_tum_24fps import nearest_real_indices


def test_30fps_source_selects_unique_real_frames_for_24fps_target():
    source = np.arange(301, dtype=np.float64) / 30.0
    target = np.arange(241, dtype=np.float64) / 24.0
    selected = nearest_real_indices(source, target)
    assert np.all(np.diff(selected) > 0)
    assert len(np.unique(selected)) == len(target)
    assert np.max(np.abs(source[selected] - target)) <= 1.0 / 60.0 + 1e-12


def test_nearest_real_indices_never_fabricates_fractional_identity():
    source = np.asarray([10.0, 10.033, 10.067, 10.100], np.float64)
    target = np.asarray([10.0, 10.0416666667, 10.0833333333], np.float64)
    selected = nearest_real_indices(source, target)
    assert selected.dtype == np.int64
    assert selected.tolist() == [0, 1, 2]
