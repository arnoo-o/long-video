"""Canonical on-disk WAH sample reader with legacy confidence fallback."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


REQUIRED_FILES = (
    "first_frame.png",
    "target_video.mp4",
    "camera_poses.npy",
    "warp_video.mp4",
    "warp_visibility_mask.npy",
    "prompt.txt",
)


def load_wah_conditioning(sample_dir):
    root = Path(sample_dir)
    missing = [name for name in REQUIRED_FILES if not (root / name).exists()]
    if missing:
        raise FileNotFoundError(f"Incomplete WAH sample {root}: missing {missing}")
    visibility = np.load(root / "warp_visibility_mask.npy").astype(np.float32)
    confidence_path = root / "warp_confidence.npy"
    confidence = (
        np.load(confidence_path).astype(np.float32)
        if confidence_path.exists()
        else visibility.copy()
    )
    source_path = root / "warp_source.npy"
    source = (
        np.load(source_path).astype(np.int8)
        if source_path.exists()
        else np.where(visibility > 0, 0, 4).astype(np.int8)
    )
    if visibility.shape != confidence.shape or visibility.shape != source.shape:
        raise ValueError("visibility, confidence, and source arrays must share [T,H,W]")
    return {
        "camera_poses": np.load(root / "camera_poses.npy").astype(np.float32),
        "warp_visibility_mask": visibility,
        "warp_confidence_mask": np.clip(confidence, 0, 1) * (visibility > 0),
        "warp_source": source,
        "prompt": (root / "prompt.txt").read_text(encoding="utf-8").strip(),
        "metadata": json.loads((root / "metadata.json").read_text(encoding="utf-8"))
        if (root / "metadata.json").exists() else {},
    }
