"""24 FPS anchor interpolation, trajectory interpolation, and supervision weights."""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class DenseTiming:
    data_fps_nominal: int = 3
    model_fps: int = 24
    anchor_stride: int = 8
    chunk_frames: int = 33
    chunk_stride: int = 32
    vae_temporal_scale: int = 4

    def __post_init__(self):
        if self.model_fps != self.data_fps_nominal * self.anchor_stride:
            raise ValueError("model_fps must equal data_fps_nominal * anchor_stride")
        if self.chunk_frames != self.chunk_stride + 1:
            raise ValueError("chunk_frames must equal chunk_stride + 1")
        if self.chunk_stride % self.anchor_stride:
            raise ValueError("chunk boundary must land on a real anchor")

    def anchor_count(self, num_chunks: int) -> int:
        return 1 + int(num_chunks) * self.chunk_stride // self.anchor_stride

    def dense_count(self, anchor_count: int) -> int:
        return 1 + (int(anchor_count) - 1) * self.anchor_stride

    def anchor_indices(self, anchor_count: int) -> np.ndarray:
        return np.arange(int(anchor_count), dtype=np.int64) * self.anchor_stride

    def alphas(self) -> np.ndarray:
        return np.arange(1, self.anchor_stride, dtype=np.float32) / self.anchor_stride


def continuous_runs(timestamps, gap_factor: float = 2.5):
    values = np.asarray(timestamps, np.float64)
    if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("timestamps must be a finite 1D array with at least two entries")
    delta = np.diff(values)
    if np.any(delta <= 0):
        raise ValueError("timestamps must increase strictly")
    median = float(np.median(delta))
    breaks = np.flatnonzero(delta > float(gap_factor) * median) + 1
    bounds = np.r_[0, breaks, len(values)]
    return [(int(a), int(b)) for a, b in zip(bounds[:-1], bounds[1:])], median


def validate_window(start: int, count: int, runs):
    end = int(start) + int(count)
    if not any(int(a) <= int(start) and end <= int(b) for a, b in runs):
        raise ValueError(f"anchor window [{start}, {end}) crosses an acquisition gap")


def allocate_disjoint_windows(runs, *, train_count=8, diagnostic_count=2, rollout_anchors=17):
    """Allocate one rollout and non-overlapping five-anchor windows deterministically."""
    available = [[a, b] for a, b in sorted(runs, key=lambda pair: pair[1] - pair[0], reverse=True)]
    rollout = None
    for item in available:
        if item[1] - item[0] >= rollout_anchors:
            rollout = item[0]
            item[0] += rollout_anchors
            break
    if rollout is None:
        raise ValueError("no gap-free run contains 17 rollout anchors")
    singles = []
    for item in available:
        while item[1] - item[0] >= 5 and len(singles) < train_count + diagnostic_count:
            singles.append(item[0]); item[0] += 5
    if len(singles) < train_count + diagnostic_count:
        raise ValueError("insufficient disjoint five-anchor windows")
    return {"train": singles[:train_count], "diagnostic": singles[train_count:], "rollout": [rollout]}


def _rotation_to_quaternion(rotation):
    from scipy.spatial.transform import Rotation
    return Rotation.from_matrix(np.asarray(rotation, np.float64)).as_quat()


def interpolate_c2w(anchor_c2w, anchor_stride: int = 8):
    from scipy.spatial.transform import Rotation, Slerp
    anchors = np.asarray(anchor_c2w, np.float64)
    if anchors.ndim != 3 or anchors.shape[1:] != (4, 4):
        raise ValueError("anchor_c2w must have shape [A,4,4]")
    dense = []
    for index in range(len(anchors) - 1):
        pair = anchors[index:index + 2]
        rotations = Rotation.from_matrix(pair[:, :3, :3])
        slerp = Slerp([0.0, 1.0], rotations)
        for step in range(anchor_stride):
            alpha = step / anchor_stride
            pose = np.eye(4, dtype=np.float64)
            pose[:3, :3] = slerp([alpha]).as_matrix()[0]
            pose[:3, 3] = (1.0 - alpha) * pair[0, :3, 3] + alpha * pair[1, :3, 3]
            dense.append(pose)
    dense.append(anchors[-1])
    result = np.stack(dense).astype(np.float32)
    np.testing.assert_allclose(result[::anchor_stride], anchors, atol=2e-5)
    error = np.max(np.abs(result[:, :3, :3] @ np.swapaxes(result[:, :3, :3], 1, 2) - np.eye(3)))
    if error > 2e-5 or np.max(np.abs(np.linalg.det(result[:, :3, :3]) - 1)) > 2e-5:
        raise ValueError("SLERP produced an invalid rotation")
    return result


def dense_rgb_weights(anchor_count: int, timing: DenseTiming = DenseTiming()):
    weights = np.full(timing.dense_count(anchor_count), 0.25, np.float32)
    weights[timing.anchor_indices(anchor_count)] = 1.0
    weights[0] = 0.0
    return weights


def temporal_weights_to_latent(rgb_weights, vae_temporal_scale: int = 4):
    values = np.asarray(rgb_weights, np.float32)
    if values.ndim != 1 or not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("RGB temporal weights must be finite, non-negative, and one-dimensional")
    groups = [values[:1]]
    groups.extend(values[start:start + vae_temporal_scale] for start in range(1, len(values), vae_temporal_scale))
    result = np.asarray([group.mean() for group in groups], np.float32)
    if not np.any(result > 0):
        raise ValueError("temporal supervision weights are all zero")
    return result


class PracticalRIFE425:
    """File/subprocess adapter for the official Practical-RIFE 4.25 full model."""

    def __init__(self, repo, checkpoint_dir, python_executable, device="cuda:0"):
        self.repo = Path(repo); self.checkpoint_dir = Path(checkpoint_dir)
        self.python_executable = str(python_executable); self.device = str(device)
        required = [self.repo / "model" / "warplayer.py", self.checkpoint_dir / "RIFE_HDv3.py", self.checkpoint_dir / "IFNet_HDv3.py", self.checkpoint_dir / "flownet.pkl"]
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise FileNotFoundError(f"Practical-RIFE 4.25 full files are missing: {missing}")

    def interpolate(self, anchors: np.ndarray, work_dir, *, multiplier: int = 8) -> np.ndarray:
        from PIL import Image
        frames = np.asarray(anchors)
        multiplier = int(multiplier)
        if multiplier < 2:
            raise ValueError("RIFE multiplier must be at least 2")
        work = Path(work_dir); inputs = work / "anchors"; output = work / "dense.npy"
        inputs.mkdir(parents=True, exist_ok=True)
        for index, frame in enumerate(frames):
            Image.fromarray(frame).save(inputs / f"{index:06d}.png")
        runner = Path(__file__).resolve().parents[2] / "scripts" / "run_practical_rife.py"
        command = [self.python_executable, str(runner), "--rife-root", str(self.repo),
                   "--checkpoint", str(self.checkpoint_dir), "--input", str(inputs),
                   "--output", str(output), "--multiplier", str(multiplier), "--device", self.device]
        completed = subprocess.run(command, text=True, capture_output=True, check=False)
        if completed.returncode:
            raise RuntimeError(json.dumps({"command": command, "stdout": completed.stdout,
                                           "stderr": completed.stderr}, indent=2))
        dense = np.load(output)
        expected_count = 1 + (len(frames) - 1) * multiplier
        if dense.shape != (expected_count, *frames.shape[1:]):
            raise ValueError(f"RIFE returned unexpected shape {dense.shape}")
        if not np.array_equal(dense[::multiplier], frames):
            raise ValueError("RIFE adapter did not preserve anchor frames byte-for-byte")
        return dense
