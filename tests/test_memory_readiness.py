from types import SimpleNamespace

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
