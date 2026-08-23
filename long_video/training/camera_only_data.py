"""Strict loader for the independent 65-frame, two-chunk camera-only corpus."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import numpy as np


@dataclass(frozen=True)
class CameraOnlyRecord:
    raw: dict
    root: Path

    @property
    def trajectory_id(self) -> str:
        return str(self.raw["trajectory_id"])

    def path(self, key: str) -> Path:
        return self.root / self.raw[key]

    def rgb_paths(self) -> list[Path]:
        paths = sorted(self.path("rgb_dir").glob("*.jpg"))
        if len(paths) != 65:
            raise ValueError(f"{self.trajectory_id}: expected exactly 65 RGB frames, found {len(paths)}")
        return paths

    def load_cameras(self):
        c2w = np.load(self.path("target_c2w_local"), mmap_mode="r")
        raw = np.load(self.path("target_c2w_local_raw"), mmap_mode="r")
        k = np.load(self.path("intrinsics"), mmap_mode="r")
        if c2w.shape != (65, 4, 4) or raw.shape != (65, 4, 4):
            raise ValueError(f"{self.trajectory_id}: camera poses must be [65,4,4]")
        if k.shape == (3, 3):
            k = np.repeat(k[None], 65, axis=0)
        if k.shape != (65, 3, 3):
            raise ValueError(f"{self.trajectory_id}: intrinsics must be [65,3,3]")
        if not (np.isfinite(c2w).all() and np.isfinite(raw).all() and np.isfinite(k).all()):
            raise ValueError(f"{self.trajectory_id}: non-finite camera cache")
        if not np.allclose(c2w[0], np.eye(4), atol=1e-5):
            raise ValueError(f"{self.trajectory_id}: local pose does not start at identity")
        det = np.linalg.det(c2w[:, :3, :3])
        if not np.allclose(det, 1.0, atol=1e-3):
            raise ValueError(f"{self.trajectory_id}: invalid rotation determinant")
        return c2w, raw, k


def load_camera_only_manifest(path: str | Path, *, expected_count: int | None = None) -> list[CameraOnlyRecord]:
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload.get("records")
    if not isinstance(records, list) or (expected_count is not None and len(records) != expected_count):
        raise ValueError("invalid camera-only manifest")
    result = []
    for row in records:
        required = ("trajectory_id", "scene_hash", "rgb_dir", "target_c2w_local",
                    "target_c2w_local_raw", "intrinsics")
        missing = [key for key in required if key not in row]
        if missing:
            raise ValueError(f"camera-only record missing keys: {missing}")
        record = CameraOnlyRecord(row, path.parent)
        record.rgb_paths()
        record.load_cameras()
        result.append(record)
    return result
