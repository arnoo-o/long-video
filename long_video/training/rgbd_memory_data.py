"""Strict loader for variable-length canonical RGB-D memory records.

Records own their 3- or 6-chunk geometry.  This deliberately permits mixed
97/193-frame manifests without padding or reordering observations.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Iterator

import numpy as np


CHUNK_STRIDE = 32
HEIGHT = 480
WIDTH = 832
REQUIRED_KEYS = (
    "dataset", "scene_id", "sequence_id", "rgb_dir", "depth_dir",
    "c2w_abs", "c2w_local", "intrinsics", "timestamps",
    "frame_count", "chunk_count", "height", "width",
)

@dataclass(frozen=True)
class CorrespondenceSlice:
    arrays:dict[str,np.ndarray]
    indices:np.ndarray

    def __len__(self): return int(self.indices.size)
    def column(self,name): return self.arrays[name][self.indices]


def expected_frame_count(chunk_count: int) -> int:
    if not 1 <= int(chunk_count) <= 6:
        raise ValueError("chunk_count must be in 1..6")
    return 1 + CHUNK_STRIDE * int(chunk_count)


@dataclass(frozen=True)
class RGBDMemoryRecord:
    raw: dict
    root: Path
    _correspondence_arrays:dict[str,np.ndarray]|None=field(default=None,init=False,repr=False,compare=False)
    _correspondence_by_query:tuple[np.ndarray,...]|None=field(default=None,init=False,repr=False,compare=False)

    @property
    def record_id(self) -> str:
        return str(self.raw.get("record_id") or f"{self.raw['dataset']}:{self.raw['sequence_id']}")

    @property
    def trajectory_id(self) -> str:
        """Compatibility name used by the Sightline training loop."""
        return self.record_id

    @property
    def frame_count(self) -> int:
        return int(self.raw["frame_count"])

    @property
    def chunk_count(self) -> int:
        return int(self.raw["chunk_count"])

    @property
    def source_frame_start(self) -> int:
        """Offset for a legal 3-chunk view of a 6-chunk parent record."""
        return int(self.raw.get("source_frame_start", 0))

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
        return tuple(sorted((p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg"}), key=lambda p: p.name))

    def _slice(self, values, name: str):
        start, stop = self.source_frame_start, self.source_frame_start + self.frame_count
        if len(values) < stop:
            raise ValueError(f"{self.record_id}: {name} has {len(values)} frames, requires [{start}:{stop})")
        return values[start:stop]

    def rgb_paths(self) -> tuple[Path, ...]:
        paths = self._slice(self._frames(self.path("rgb_dir")), "RGB")
        if len(paths) != self.frame_count:
            raise ValueError(f"{self.record_id}: RGB count mismatch")
        return paths

    def depth_paths(self) -> tuple[Path, ...]:
        paths = self._slice(self._frames(self.path("depth_dir")), "depth")
        if len(paths) != self.frame_count:
            raise ValueError(f"{self.record_id}: depth count mismatch")
        return paths

    def load_cameras(self, *, local: bool = True) -> tuple[np.ndarray, np.ndarray]:
        absolute = self._slice(np.load(self.path("c2w_abs"), mmap_mode="r"), "c2w_abs")
        intrinsics = self._slice(np.load(self.path("intrinsics"), mmap_mode="r"), "intrinsics")
        if absolute.shape != (self.frame_count, 4, 4) or intrinsics.shape != (self.frame_count, 3, 3):
            raise ValueError(f"{self.record_id}: camera shape does not match frame_count")
        c2w = np.linalg.inv(np.asarray(absolute[0])) @ np.asarray(absolute) if local else absolute
        return c2w, intrinsics

    def load_timestamps(self) -> np.ndarray:
        timestamps = self._slice(np.load(self.path("timestamps"), mmap_mode="r"), "timestamps")
        if timestamps.shape != (self.frame_count,) or not np.isfinite(timestamps).all():
            raise ValueError(f"{self.record_id}: timestamps must match frame_count")
        if np.any(np.diff(timestamps) <= 0):
            raise ValueError(f"{self.record_id}: timestamps must be strictly increasing")
        return timestamps

    def load_correspondences(self) -> dict[str, np.ndarray]:
        if not self.memory_eligible:
            return {}
        if "correspondence_cache" not in self.raw:
            raise ValueError(f"{self.record_id}: memory-eligible record has no correspondence cache")
        if self._correspondence_arrays is not None: return self._correspondence_arrays
        with np.load(self.path("correspondence_cache"), allow_pickle=False) as cache:
            arrays = {key: np.ascontiguousarray(cache[key]) for key in cache.files}
        required = {"query_frame", "key_frame", "query_chunk", "key_chunk", "query_t", "key_t", "query_y", "query_x", "key_y", "key_x", "weight"}
        missing = required.difference(arrays)
        if missing:
            raise ValueError(f"{self.record_id}: correspondence cache missing {sorted(missing)}")
        count = len(arrays["query_frame"])
        if any(len(value) != count for value in arrays.values() if value.ndim == 1):
            raise ValueError(f"{self.record_id}: correspondence columns have different lengths")
        if count and (np.any(arrays["key_frame"] >= arrays["query_frame"]) or np.any(arrays["key_chunk"] >= arrays["query_chunk"]) or np.any(arrays["query_frame"] >= self.frame_count) or np.any(arrays["key_frame"] < 0) or np.any(arrays["query_chunk"] >= self.chunk_count) or np.any(arrays["key_chunk"] < 0)):
            raise ValueError(f"{self.record_id}: correspondence cache is not strictly causal or out of bounds")
        object.__setattr__(self,"_correspondence_arrays",arrays)
        object.__setattr__(self,"_correspondence_by_query",tuple(np.flatnonzero(arrays["query_chunk"]==chunk).astype(np.int64,copy=False) for chunk in range(self.chunk_count)))
        return arrays

    def correspondences_for_chunk(self,query_chunk:int) -> CorrespondenceSlice:
        if not 0<=int(query_chunk)<self.chunk_count: raise ValueError('query_chunk outside record')
        arrays=self.load_correspondences()
        return CorrespondenceSlice(arrays,self._correspondence_by_query[int(query_chunk)])

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
        if self.frame_count != expected_frame_count(self.chunk_count):
            raise ValueError(f"{self.record_id}: frame_count must equal 1 + 32 * chunk_count")
        if (int(self.raw.get("height", -1)), int(self.raw.get("width", -1))) != (HEIGHT, WIDTH):
            raise ValueError(f"{self.record_id}: unsupported image geometry")
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
            if "correspondence_cache" not in self.raw or not self.path("correspondence_cache").is_file():
                raise ValueError(f"{self.record_id}: memory-eligible record has no correspondence cache")


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
