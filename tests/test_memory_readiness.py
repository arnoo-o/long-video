from types import SimpleNamespace

import numpy as np

from long_video.memory.memory_manager import MemoryManager


class _ReadinessBuffer:
    def __init__(self, frames, translation, view_change, new_area):
        self.frames = [SimpleNamespace(coverage=0.01) for _ in range(frames)]
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
        min_overlap_coverage=0.03,
    )
    manager.buffer = _ReadinessBuffer(11, translation=3.0, view_change=0.0, new_area=0.0)
    assert not manager._ready()
    manager.buffer = _ReadinessBuffer(12, translation=0.0, view_change=0.5, new_area=0.0)
    report = manager.readiness_report()
    assert report["ready"]
    assert report["conditions"] == {
        "translation": False,
        "view_change": True,
        "new_area": False,
        "world_overlap": False,
    }


def test_all_readiness_keeps_original_and_semantics():
    manager = MemoryManager(transition_readiness_mode="all")
    manager.buffer = _ReadinessBuffer(12, translation=1.0, view_change=1.0, new_area=0.5)
    assert not manager._ready()  # World overlap is only 1%, below 3%.


def test_candidate_created_frame_is_last_inclusive_generated_frame():
    manager = MemoryManager(low_coverage_chunks=0)
    captured = {}
    manager.buffer = SimpleNamespace(can_attempt=lambda frame: True)
    manager._append_chunk = lambda *args, **kwargs: None
    manager._ready = lambda: True

    def build_candidate(_active, created_frame):
        captured["created_frame"] = created_frame
        return SimpleNamespace(node_id="node_001"), [], []

    manager.build_candidate = build_candidate
    manager.validate_candidate = lambda *_args: (True, {})
    manager.promote = lambda _active, candidate: candidate
    active = SimpleNamespace(node_id="node_000")
    generated = np.zeros((32, 1, 1, 3), np.uint8)
    warp = SimpleNamespace(coverage_per_frame=np.zeros(32, np.float32))

    manager.process_chunk(active, generated, None, warp, frame_start=33)

    assert captured["created_frame"] == 64
