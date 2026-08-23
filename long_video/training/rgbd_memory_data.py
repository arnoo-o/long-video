"""Strict loader for the canonical 97-frame RGB-D memory dataset."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterator

import numpy as np


FRAME_COUNT = 97
CHUNK_COUNT = 3
HEIGHT = 480
WIDTH = 832
REQUIRED_KEYS = (
    "dataset", "scene_id", "sequence_id", "rgb_dir", "depth_dir",
    "c2w_abs", "c2w_local", "intrinsics", "timestamps",
    "frame_count", "chunk_count", "height", "width",
)


@dataclass(frozen=True)
class RGBDMemoryRecord:
    raw: dict
    root: Path

    @property
    def record_id(self) -> str:
        return str(self.raw.get("record_id") or f"{self.raw['dataset']}:{self.raw['sequence_id']}")

    @property
    def trajectory_id(self) -> str:
        """Compatibility name used by the Sightline training loop."""
        return self.record_id

    @property
    def training_scope(self) -> str:
        return str(self.raw.get("training_scope", "rgbd_memory"))

    @property
    def memory_eligible(self) -> bool:
        return bool(self.raw.get("memory_eligible", self.training_scope == "rgbd_memory"))

    def path(self, key: str) -> Path:
        value = Path(self.raw[key])
        return value if value.is_absolute() else self.root / value

    @staticmethod
    def _frames(directory: Path) -> tuple[Path, ...]:
        return tuple(sorted(p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg"}))

    def rgb_paths(self) -> tuple[Path, ...]:
        paths = self._frames(self.path("rgb_dir"))
        if len(paths) != FRAME_COUNT:
            raise ValueError(f"{self.record_id}: expected {FRAME_COUNT} RGB frames, found {len(paths)}")
        return paths

    def depth_paths(self) -> tuple[Path, ...]:
        paths = self._frames(self.path("depth_dir"))
        if len(paths) != FRAME_COUNT:
            raise ValueError(f"{self.record_id}: expected {FRAME_COUNT} depth frames, found {len(paths)}")
        return paths

    def load_cameras(self, *, local: bool = True) -> tuple[np.ndarray, np.ndarray]:
        key = "c2w_local" if local else "c2w_abs"
        c2w = np.load(self.path(key), mmap_mode="r")
        intrinsics = np.load(self.path("intrinsics"), mmap_mode="r")
        if c2w.shape != (FRAME_COUNT, 4, 4):
            raise ValueError(f"{self.record_id}: {key} must be [{FRAME_COUNT},4,4]")
        if intrinsics.shape != (FRAME_COUNT, 3, 3):
            raise ValueError(f"{self.record_id}: intrinsics must be [{FRAME_COUNT},3,3]")
        return c2w, intrinsics

    def load_timestamps(self) -> np.ndarray:
        timestamps = np.load(self.path("timestamps"), mmap_mode="r")
        if timestamps.shape != (FRAME_COUNT,) or not np.isfinite(timestamps).all():
            raise ValueError(f"{self.record_id}: timestamps must be finite [{FRAME_COUNT}]")
        if np.any(np.diff(timestamps) <= 0):
            raise ValueError(f"{self.record_id}: timestamps must be strictly increasing")
        return timestamps

    def load_correspondences(self) -> dict[str, np.ndarray]:
        if not self.memory_eligible:
            return {}
        if "correspondence_cache" not in self.raw:
            raise ValueError(f"{self.record_id}: memory-eligible record has no correspondence cache")
        with np.load(self.path("correspondence_cache"), allow_pickle=False) as cache:
            arrays = {key: np.asarray(cache[key]) for key in cache.files}
        required = {"query_frame", "key_frame", "query_chunk", "key_chunk", "query_t", "key_t", "query_y", "query_x", "key_y", "key_x", "weight"}
        missing = required.difference(arrays)
        if missing:
            raise ValueError(f"{self.record_id}: correspondence cache missing {sorted(missing)}")
        count = len(arrays["query_frame"])
        if any(len(value) != count for value in arrays.values() if value.ndim == 1):
            raise ValueError(f"{self.record_id}: correspondence columns have different lengths")
        if count and (np.any(arrays["key_frame"] >= arrays["query_frame"]) or np.any(arrays["key_chunk"] >= arrays["query_chunk"])):
            raise ValueError(f"{self.record_id}: correspondence cache is not strictly causal")
        return arrays

    def correspondence_rows(self) -> Iterator[dict]:
        cache = self.load_correspondences()
        if not cache:
            return
        count = len(cache["query_frame"])
        for index in range(count):
            yield {
                "query_frame": int(cache["query_frame"][index]),
                "key_frame": int(cache["key_frame"][index]),
                "query_chunk": int(cache["query_chunk"][index]),
                "key_chunk": int(cache["key_chunk"][index]),
                "query_latent_temporal": int(cache["query_t"][index]),
                "key_latent_temporal": int(cache["key_t"][index]),
                "query_y": int(cache["query_y"][index]),
                "query_x": int(cache["query_x"][index]),
                "key_y": int(cache["key_y"][index]),
                "key_x": int(cache["key_x"][index]),
                "weight": float(cache["weight"][index]),
                "matched_count": int(cache.get("matched_count", np.ones(count, np.int32))[index]),
                "valid_count": int(cache.get("valid_count", np.ones(count, np.int32))[index]),
                "coverage": float(cache.get("coverage", np.ones(count, np.float32))[index]),
                "vote": float(cache.get("vote", cache["weight"])[index]),
            }

    def validate(self) -> None:
        missing = [key for key in REQUIRED_KEYS if key not in self.raw]
        if missing:
            raise ValueError(f"record missing keys: {missing}")
        expected = {"frame_count": FRAME_COUNT, "chunk_count": CHUNK_COUNT, "height": HEIGHT, "width": WIDTH}
        wrong = {key: (self.raw.get(key), value) for key, value in expected.items() if int(self.raw.get(key, -1)) != value}
        if wrong:
            raise ValueError(f"{self.record_id}: fixed geometry mismatch {wrong}")
        rgb, depth = self.rgb_paths(), self.depth_paths()
        if [p.stem for p in rgb] != [p.stem for p in depth]:
            raise ValueError(f"{self.record_id}: RGB/depth frame names are not aligned")
        c2w_abs, intrinsics = self.load_cameras(local=False)
        c2w_local, _ = self.load_cameras(local=True)
        if not np.isfinite(c2w_abs).all() or not np.isfinite(c2w_local).all() or not np.isfinite(intrinsics).all():
            raise ValueError(f"{self.record_id}: non-finite camera geometry")
        if not np.allclose(c2w_local[0], np.eye(4), atol=1e-5):
            raise ValueError(f"{self.record_id}: first local pose is not identity")
        self.load_timestamps()
        if self.memory_eligible:
            self.load_correspondences()


def load_rgbd_memory_manifest(path: str | Path, *, expected_count: int | None = None) -> list[RGBDMemoryRecord]:
    manifest = Path(path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    records = payload.get("records")
    if not isinstance(records, list) or (expected_count is not None and len(records) != expected_count):
        raise ValueError("invalid RGB-D memory manifest record count")
    result = [RGBDMemoryRecord(row, manifest.parent) for row in records]
    for record in result:
        record.validate()
    return result
