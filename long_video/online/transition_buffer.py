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
    old_node_warp_depth: np.ndarray
    old_node_warp_source: np.ndarray
    old_node_warp_rgb_content_origin: np.ndarray
    old_node_warp_depth_content_origin: np.ndarray
    old_node_warp_evidence_role: np.ndarray
    old_node_warp_rgb_evidence_role: np.ndarray
    old_node_warp_depth_evidence_role: np.ndarray
    old_node_depth_convention: str
    warp_confidence: np.ndarray
    coverage: float
    global_frame_index: int

    @property
    def high_conf_coverage(self):
        mask = np.asarray(self.warp_visibility, bool)
        confidence = np.asarray(self.warp_confidence, np.float32)
        return float((mask & (confidence >= 0.5)).mean())


class TransitionBuffer:
    def __init__(self, max_length=96, max_age_frames=240, cooldown_frames=32):
        self.frames = []
        self.max_length = int(max_length)
        self.max_age_frames = int(max_age_frames)
        self.cooldown_frames = int(cooldown_frames)
        self.cooldown_until = -1
        self.rejection_reasons = []
        self._indices = set()

    def append(self, **kwargs):
        index = int(kwargs["global_frame_index"])
        if index in self._indices:
            return None
        if kwargs.get("old_node_depth_convention") != "Z_DEPTH":
            raise ValueError("TransitionFrame old-node depth must be Z_DEPTH")
        frame = TransitionFrame(**kwargs)
        self.frames.append(frame)
        self._indices.add(index)
        newest = frame.global_frame_index
        self.frames = [item for item in self.frames
                       if newest-item.global_frame_index <= self.max_age_frames]
        if len(self.frames) > self.max_length:
            self.frames = self.frames[-self.max_length:]
        self._indices = {item.global_frame_index for item in self.frames}
        return frame

    def clear(self):
        self.frames.clear()
        self._indices.clear()

    def reject(self, frame_index, reason):
        self.rejection_reasons.append({"frame_index": int(frame_index), "reason": str(reason)})
        self.cooldown_until = int(frame_index) + self.cooldown_frames

    def can_attempt(self, frame_index):
        return int(frame_index) >= self.cooldown_until

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

    def select_keyframes(self, count=8, heldout_count=4):
        if not self.frames:
            return [], []
        if len(self.frames) < count + heldout_count:
            return [], []
        mapping_candidates = self.frames[:-heldout_count]
        count = min(int(count), len(mapping_candidates))
        selected = [0]
        positions = np.stack([frame.camera_c2w[:3, 3] for frame in mapping_candidates])
        directions = np.stack([frame.camera_c2w[:3, 2] for frame in mapping_candidates])
        directions /= np.linalg.norm(directions, axis=1, keepdims=True).clip(1e-8)
        while len(selected) < count:
            scores = np.full(len(mapping_candidates), -np.inf, np.float32)
            for index in range(len(mapping_candidates)):
                if index in selected:
                    continue
                translation = min(np.linalg.norm(positions[index] - positions[j]) for j in selected)
                angle = min(
                    np.arccos(np.clip(directions[index] @ directions[j], -1.0, 1.0))
                    for j in selected
                )
                scores[index] = translation + 0.25 * angle
            selected.append(int(np.argmax(scores)))
        mapping = [mapping_candidates[index] for index in sorted(selected)]
        heldout = self.frames[-heldout_count:]
        return mapping, heldout
