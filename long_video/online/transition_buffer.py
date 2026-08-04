"""Frames retained while the active node is losing coverage."""
from dataclasses import dataclass

import numpy as np


@dataclass
class TransitionFrame:
    generated_rgb: np.ndarray
    camera_c2w: np.ndarray
    intrinsics: np.ndarray
    old_node_warp: np.ndarray
    warp_visibility: np.ndarray
    warp_confidence: np.ndarray
    coverage: float
    global_frame_index: int

    @property
    def high_conf_coverage(self):
        mask = np.asarray(self.warp_visibility, bool)
        confidence = np.asarray(self.warp_confidence, np.float32)
        return float((mask & (confidence >= 0.5)).mean())


class TransitionBuffer:
    def __init__(self):
        self.frames = []

    def append(self, **kwargs):
        frame = TransitionFrame(**kwargs)
        self.frames.append(frame)
        return frame

    def clear(self):
        self.frames.clear()

    def __len__(self):
        return len(self.frames)

    @property
    def translation_baseline(self):
        if len(self.frames) < 2:
            return 0.0
        positions = np.stack([frame.camera_c2w[:3, 3] for frame in self.frames])
        return float(np.linalg.norm(positions[:, None] - positions[None, :], axis=-1).max())

    @property
    def view_diversity(self):
        if len(self.frames) < 2:
            return 0.0
        forward = np.stack([frame.camera_c2w[:3, 2] for frame in self.frames])
        forward /= np.linalg.norm(forward, axis=1, keepdims=True).clip(1e-8)
        cosines = np.clip(forward @ forward.T, -1.0, 1.0)
        return float(np.arccos(cosines).max())

    @property
    def mean_new_area_ratio(self):
        if not self.frames:
            return 0.0
        return float(np.mean([1.0 - frame.coverage for frame in self.frames]))

    def select_keyframes(self, count=8):
        if not self.frames:
            return []
        count = min(int(count), len(self.frames))
        selected = [0]
        positions = np.stack([frame.camera_c2w[:3, 3] for frame in self.frames])
        directions = np.stack([frame.camera_c2w[:3, 2] for frame in self.frames])
        directions /= np.linalg.norm(directions, axis=1, keepdims=True).clip(1e-8)
        while len(selected) < count:
            scores = np.full(len(self.frames), -np.inf, np.float32)
            for index in range(len(self.frames)):
                if index in selected:
                    continue
                translation = min(np.linalg.norm(positions[index] - positions[j]) for j in selected)
                angle = min(
                    np.arccos(np.clip(directions[index] @ directions[j], -1.0, 1.0))
                    for j in selected
                )
                scores[index] = translation + 0.25 * angle
            selected.append(int(np.argmax(scores)))
        return [self.frames[index] for index in sorted(selected)]
