from types import SimpleNamespace

import numpy as np

from long_video.memory.memory_manager import MemoryManager
from long_video.types import Z_DEPTH


def _frame(index):
    return SimpleNamespace(
        global_frame_index=index,
        camera_c2w=np.eye(4, dtype=np.float32),
        intrinsics=np.asarray([[1, 0, 1], [0, 1, 1], [0, 0, 1]], np.float32),
    )


def test_depth_consistency_requires_three_observations_five_frames_apart():
    manager = MemoryManager(max_overlap_depth_error=0.5)
    frames = [_frame(index) for index in (0, 4, 5, 10)]
    depth = np.full((4, 3, 3), np.nan, np.float32)
    depth[:, 1, 1] = 2.0
    candidate = SimpleNamespace(
        points_xyz=np.asarray([[0, 0, 2], [0, 0, 3]], np.float32),
        view_depth=depth,
        depth_convention=Z_DEPTH,
    )

    valid, counts = manager._depth_consistent_observation_mask(candidate, frames)

    assert counts.tolist() == [3, 0]
    assert valid.tolist() == [True, False]
