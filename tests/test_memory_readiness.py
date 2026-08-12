from types import SimpleNamespace

import numpy as np
import pytest

from long_video.memory.memory_manager import MemoryManager


class _ReadinessBuffer:
    def __init__(self, frames, translation, view_change, new_area, coverage=0.01):
        self.frames = [SimpleNamespace(coverage=coverage) for _ in range(frames)]
        self.translation_baseline = translation
        self.view_diversity = view_change
        self.mean_new_area_ratio = new_area

    def __len__(self):
        return len(self.frames)


def test_any_readiness_requires_twelve_frames_then_accepts_one_condition():
    manager = MemoryManager(
        transition_readiness_mode="any",
        min_transition_frames=12,
        min_translation_baseline=2.5,
        min_view_diversity=0.4363323129985824,
        min_new_area_ratio=0.05,
        max_world_overlap=0.50,
    )
    manager.buffer = _ReadinessBuffer(11, translation=3.0, view_change=0.0, new_area=0.0)
    assert not manager._ready(0.01)
    manager.buffer = _ReadinessBuffer(12, translation=0.0, view_change=0.5, new_area=0.0)
    report = manager.readiness_report(0.49)
    assert report["ready"]
    assert report["conditions"] == {
        "translation": False,
        "view_change": True,
        "new_area": False,
    }
    assert report["world_overlap_below_max"] is True


def test_world_overlap_must_be_strictly_below_fifty_percent():
    manager = MemoryManager()
    manager.buffer = _ReadinessBuffer(
        12, translation=3.0, view_change=1.0, new_area=0.5, coverage=0.20,
    )
    assert not manager._ready(0.501)
    manager.buffer = _ReadinessBuffer(
        12, translation=3.0, view_change=0.0, new_area=0.0, coverage=0.50,
    )
    assert not manager._ready(0.50)
    assert manager._ready(0.499)


def test_readiness_uses_current_chunk_overlap_not_history_mean():
    manager = MemoryManager()
    manager.buffer = _ReadinessBuffer(
        12, translation=0.0, view_change=0.5, new_area=0.5, coverage=0.95,
    )
    report = manager.readiness_report(0.25)
    assert report["ready"]
    assert report["values"]["current_chunk_world_overlap"] == 0.25


def test_readiness_policy_rejects_conflicting_mode_or_thresholds():
    with pytest.raises(ValueError, match="permanently 'any'"):
        MemoryManager(transition_readiness_mode="all")
    with pytest.raises(ValueError, match="thresholds are permanent"):
        MemoryManager(min_translation_baseline=1.0)


def test_candidate_created_frame_is_last_inclusive_generated_frame():
    manager = MemoryManager(low_coverage_chunks=0)
    captured = {}
    manager.buffer = SimpleNamespace(can_attempt=lambda frame: True)
    manager._append_chunk = lambda *args, **kwargs: None
    manager.readiness_report = lambda _overlap=None: {"ready": True}

    def build_candidate(_active, created_frame):
        captured["created_frame"] = created_frame
        return SimpleNamespace(node_id="node_001"), [], []

    manager.build_candidate = build_candidate
    manager.validate_candidate = lambda *_args: (True, {})
    manager.promote = lambda _active, candidate: candidate
    active = SimpleNamespace(node_id="node_000")
    generated = np.zeros((32, 1, 1, 3), np.uint8)
    warp = SimpleNamespace(coverage_per_frame=np.ones(32, np.float32), visibility=np.zeros((32, 1, 1), bool))

    manager.process_chunk(active, generated, None, warp, frame_start=33)

    assert captured["created_frame"] == 64


def test_pending_activation_blocks_candidate_without_replacing_history_path():
    manager = MemoryManager(low_coverage_chunks=0)
    manager.buffer = SimpleNamespace(can_attempt=lambda frame: True)
    manager._append_chunk = lambda *args, **kwargs: None
    manager.readiness_report = lambda _overlap=None: {"ready": True}
    manager.build_candidate = lambda *_args: (_ for _ in ()).throw(
        AssertionError("pending activation must prevent candidate construction")
    )
    active = SimpleNamespace(node_id="node_000")
    generated = np.zeros((32, 1, 1, 3), np.uint8)
    warp = SimpleNamespace(coverage_per_frame=np.ones(32, np.float32), visibility=np.zeros((32, 1, 1), bool))
    returned, event = manager.process_chunk(
        active, generated, None, warp, frame_start=65,
        allow_candidate_promotion=False,
    )
    assert returned is active
    assert event["parent_pixel_visibility_ratio"] == 0.0
    assert event["candidate_promotion_blocked"] is True
    assert "candidate_id" not in event
