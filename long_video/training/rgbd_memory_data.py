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
    "frame_count", "chunk_count", "height", "width", "near_depth",
)
SCANNET_REQUIRED_KEYS = (
    "source_timestamps", "source_frame_indices", "pointcloud", "metadata", "fps",
)
ARKIT_RRD_REQUIRED_KEYS = SCANNET_REQUIRED_KEYS
RESAMPLED_RGBD_REQUIRED_KEYS = SCANNET_REQUIRED_KEYS

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

    @property
    def near_depth(self) -> float:
        value=float(self.raw["near_depth"])
        if not np.isfinite(value) or value<=0: raise ValueError(f"{self.record_id}: near_depth must be finite and positive")
        return value

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

    def load_source_identity(self) -> tuple[np.ndarray, np.ndarray]:
        if "source_timestamps" not in self.raw or "source_frame_indices" not in self.raw:
            raise ValueError(f"{self.record_id}: source timestamp/frame identity is missing")
        timestamps = np.asarray(np.load(self.path("source_timestamps"), mmap_mode="r"))
        indices = np.asarray(np.load(self.path("source_frame_indices"), mmap_mode="r"))
        start, stop = self.source_frame_start, self.source_frame_start + self.frame_count
        timestamps, indices = timestamps[start:stop], indices[start:stop]
        if timestamps.shape != (self.frame_count,) or indices.shape != (self.frame_count,):
            raise ValueError(f"{self.record_id}: source identity must match frame_count")
        if timestamps.dtype.kind != "f" or indices.dtype.kind not in "iu":
            raise ValueError(f"{self.record_id}: invalid source identity dtypes")
        if not np.isfinite(timestamps).all() or np.any(np.diff(timestamps) <= 0):
            raise ValueError(f"{self.record_id}: source timestamps are not strictly increasing")
        if np.any(np.diff(indices) <= 0) or len(np.unique(indices)) != self.frame_count:
            raise ValueError(f"{self.record_id}: source frame indices must be unique and increasing")
        return timestamps.astype(np.float64, copy=False), indices.astype(np.int64, copy=False)

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
        _=self.near_depth
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
        if self.raw.get("dataset") == "scannet":
            self._validate_scannet()
        if self.raw.get("dataset") == "arkitscenes":
            self._validate_arkitscenes_rrd()
        if self.raw.get("dataset") in {"bonn", "tum"} and "fps" in self.raw:
            self._validate_resampled_bonn_tum()

    def _validate_resampled_bonn_tum(self) -> None:
        missing = [key for key in RESAMPLED_RGBD_REQUIRED_KEYS if key not in self.raw]
        if missing:
            raise ValueError(f"{self.record_id}: 24 FPS Bonn/TUM record missing {missing}")
        if (self.frame_count, self.chunk_count, self.source_frame_start) != (97, 3, 0):
            raise ValueError(f"{self.record_id}: Bonn/TUM requires a full 97-frame/3-chunk record")
        if float(self.raw["fps"]) != 24.0:
            raise ValueError(f"{self.record_id}: Bonn/TUM target FPS must be 24")
        target = self.load_timestamps()
        expected = np.arange(self.frame_count, dtype=np.float64) / 24.0
        if not np.allclose(target, expected, rtol=0.0, atol=1e-8):
            raise ValueError(f"{self.record_id}: Bonn/TUM target timestamps are not an exact 24 FPS axis")
        source_timestamps, source_indices = self.load_source_identity()
        if len(np.unique(source_indices)) != self.frame_count or np.any(np.diff(source_indices) <= 0):
            raise ValueError(f"{self.record_id}: Bonn/TUM source frames are repeated or reordered")
        metadata = json.loads(self.path("metadata").read_text(encoding="utf-8"))
        origin = float(metadata.get("target_time_origin_seconds", float("nan")))
        if not np.isfinite(origin) or np.max(np.abs(source_timestamps - (origin + target))) > .0200005:
            raise ValueError(f"{self.record_id}: Bonn/TUM source observation exceeds 20 ms target tolerance")
        if metadata.get("interpolation") != "none" or metadata.get("source_selection") != "nearest real frame; strictly increasing and unique":
            raise ValueError(f"{self.record_id}: Bonn/TUM real-frame provenance is invalid")
        c2w, intrinsics = self.load_cameras(local=False); rotations = c2w[:, :3, :3]
        if (not np.allclose(c2w[:, 3], np.asarray((0, 0, 0, 1)), atol=1e-5)
                or not np.allclose(rotations.transpose(0, 2, 1) @ rotations, np.eye(3), atol=2e-4)
                or not np.allclose(np.linalg.det(rotations), 1.0, atol=2e-4)):
            raise ValueError(f"{self.record_id}: Bonn/TUM c2w is not rigid")
        if (np.any(intrinsics[:, 0, 0] <= 0) or np.any(intrinsics[:, 1, 1] <= 0)
                or np.any(intrinsics[:, 0, 2] < 0) or np.any(intrinsics[:, 0, 2] >= WIDTH)
                or np.any(intrinsics[:, 1, 2] < 0) or np.any(intrinsics[:, 1, 2] >= HEIGHT)):
            raise ValueError(f"{self.record_id}: invalid Bonn/TUM intrinsics")
        with np.load(self.path("pointcloud"), allow_pickle=False) as cloud:
            required = {"xyz_world", "offsets", "source_frame_indices", "timestamps"}
            if not required.issubset(cloud.files):
                raise ValueError(f"{self.record_id}: Bonn/TUM pointcloud metadata is incomplete")
            xyz, offsets = cloud["xyz_world"], cloud["offsets"]
            if (xyz.dtype != np.float32 or xyz.ndim != 2 or xyz.shape[1] != 3 or not np.isfinite(xyz).all()
                    or offsets.dtype != np.int64 or offsets.shape != (98,) or offsets[0] != 0
                    or offsets[-1] != len(xyz) or np.any(np.diff(offsets) <= 0)
                    or not np.array_equal(cloud["source_frame_indices"].astype(np.int64), source_indices)
                    or not np.array_equal(cloud["timestamps"].astype(np.float64), target)):
                raise ValueError(f"{self.record_id}: invalid Bonn/TUM pointcloud frame identity")
        cache = self.load_correspondences()
        if not len(cache.get("query_frame", ())):
            raise ValueError(f"{self.record_id}: Bonn/TUM correspondence cache is empty")

    def _validate_arkitscenes_rrd(self) -> None:
        missing = [key for key in ARKIT_RRD_REQUIRED_KEYS if key not in self.raw]
        if missing:
            raise ValueError(f"{self.record_id}: ARKitScenes RRD record missing {missing}")
        derived = bool(self.raw.get("parent_record_id"))
        legal_geometry = ((self.frame_count, self.chunk_count, self.source_frame_start) == (193, 6, 0) if not derived
                          else (self.frame_count, self.chunk_count, self.source_frame_start) in ((97, 3, 0), (97, 3, 96)))
        if not legal_geometry or float(self.raw["fps"]) != 24.0:
            raise ValueError(f"{self.record_id}: ARKitScenes RRD requires a full 193/6 record or a legal parent-derived 97/3 unit")
        target = self.load_timestamps()
        expected = target[0] + np.arange(self.frame_count, dtype=np.float64) / 24.0
        if not np.allclose(target, expected, rtol=0.0, atol=1e-8):
            raise ValueError(f"{self.record_id}: ARKitScenes target timestamps are not an exact 24 FPS axis")
        source_timestamps, source_indices = self.load_source_identity()
        all_source=np.load(self.path("source_timestamps"),mmap_mode="r")
        all_target=np.load(self.path("timestamps"),mmap_mode="r")
        parent_clock_offset=float(all_source[0]-all_target[0])
        if np.max(np.abs((source_timestamps-target)-parent_clock_offset)) > .0100001:
            raise ValueError(f"{self.record_id}: ARKitScenes RGB observations exceed 10 ms target tolerance")
        if np.max(np.diff(source_timestamps)) > .060:
            raise ValueError(f"{self.record_id}: ARKitScenes source time discontinuity")
        c2w, intrinsics = self.load_cameras(local=False); rotations = c2w[:, :3, :3]
        if (not np.allclose(c2w[:, 3], np.asarray((0, 0, 0, 1)), atol=1e-5)
                or not np.allclose(rotations.transpose(0, 2, 1) @ rotations, np.eye(3), atol=1e-4)
                or not np.allclose(np.linalg.det(rotations), 1.0, atol=1e-4)):
            raise ValueError(f"{self.record_id}: ARKitScenes c2w is not rigid")
        if (np.any(intrinsics[:, 0, 0] <= 0) or np.any(intrinsics[:, 1, 1] <= 0)
                or np.any(intrinsics[:, 0, 2] < 0) or np.any(intrinsics[:, 0, 2] >= WIDTH)
                or np.any(intrinsics[:, 1, 2] < 0) or np.any(intrinsics[:, 1, 2] >= HEIGHT)
                or not np.allclose(intrinsics[:, 2], np.asarray((0, 0, 1)), atol=1e-6)):
            raise ValueError(f"{self.record_id}: invalid ARKitScenes intrinsics")
        metadata = json.loads(self.path("metadata").read_text(encoding="utf-8"))
        if metadata.get("pose_source") != "mebx_stream_4_vision_transform" or metadata.get("orientation_source") != "measured_gravity":
            raise ValueError(f"{self.record_id}: invalid ARKitScenes RRD provenance")
        timing = metadata.get("timing_validation", {})
        timing_keys=("rgb_target_max_error_seconds", "depth_rgb_max_error_seconds", "pose_rgb_max_error_seconds", "intrinsics_rgb_max_error_seconds", "depth_intrinsics_rgb_max_error_seconds", "confidence_rgb_max_error_seconds")
        if any(float(timing.get(key, 1.0)) > .0100001 for key in timing_keys):
            raise ValueError(f"{self.record_id}: ARKitScenes cross-stream timing validation failed")
        with np.load(self.path("pointcloud"), allow_pickle=False) as cloud:
            required = {"xyz_world", "offsets", "source_frame_indices", "timestamps"}
            if not required.issubset(cloud.files):
                raise ValueError(f"{self.record_id}: ARKitScenes pointcloud metadata is incomplete")
            xyz, offsets = cloud["xyz_world"], cloud["offsets"]
            if (xyz.dtype != np.float32 or xyz.ndim != 2 or xyz.shape[1] != 3 or not np.isfinite(xyz).all()
                    or offsets.dtype != np.int64 or offsets.shape != (194,) or offsets[0] != 0
                    or offsets[-1] != len(xyz) or np.any(np.diff(offsets) <= 0)):
                raise ValueError(f"{self.record_id}: invalid ARKitScenes pointcloud frame layout")
            start,stop=self.source_frame_start,self.source_frame_start+self.frame_count
            if not np.array_equal(cloud["source_frame_indices"][start:stop].astype(np.int64), source_indices):
                raise ValueError(f"{self.record_id}: ARKitScenes pointcloud/source identity mismatch")
            if not np.array_equal(cloud["timestamps"][start:stop].astype(np.float64), target):
                raise ValueError(f"{self.record_id}: ARKitScenes pointcloud timestamp mismatch")
        if not len(self.load_correspondences().get("query_frame", ())):
            raise ValueError(f"{self.record_id}: ARKitScenes correspondence cache is empty")

    def _validate_scannet(self) -> None:
        missing = [key for key in SCANNET_REQUIRED_KEYS if key not in self.raw]
        if missing:
            raise ValueError(f"{self.record_id}: ScanNet record missing {missing}")
        derived = bool(self.raw.get("parent_record_id"))
        legal_geometry = ((self.frame_count, self.chunk_count, self.source_frame_start) == (193, 6, 0) if not derived
                          else (self.frame_count, self.chunk_count, self.source_frame_start) in ((97, 3, 0), (97, 3, 96)))
        if not legal_geometry or float(self.raw["fps"]) != 24.0:
            raise ValueError(f"{self.record_id}: ScanNet requires a full 193/6 record or a legal parent-derived 97/3 unit")
        target = self.load_timestamps()
        expected = target[0] + np.arange(self.frame_count, dtype=np.float64) / 24.0
        if not np.allclose(target, expected, rtol=0.0, atol=1e-8):
            raise ValueError(f"{self.record_id}: ScanNet target timestamps are not an exact 24 FPS axis")
        source_timestamps, source_indices = self.load_source_identity()
        if np.max(np.abs(source_timestamps - target)) > (1.0 / 60.0 + 1e-6):
            raise ValueError(f"{self.record_id}: source frames are not nearest 30 FPS observations")
        if np.max(np.diff(source_timestamps)) > (2.0 / 30.0 + 1e-6):
            raise ValueError(f"{self.record_id}: ScanNet source time discontinuity")
        c2w, intrinsics = self.load_cameras(local=False)
        rotations = c2w[:, :3, :3]
        should_be_identity = rotations.transpose(0, 2, 1) @ rotations
        if (not np.allclose(c2w[:, 3], np.asarray((0, 0, 0, 1)), atol=1e-5)
                or not np.allclose(should_be_identity, np.eye(3), atol=1e-4)
                or not np.allclose(np.linalg.det(rotations), 1.0, atol=1e-4)):
            raise ValueError(f"{self.record_id}: ScanNet c2w is not a rigid OpenCV transform")
        if (np.any(intrinsics[:, 0, 0] <= 0) or np.any(intrinsics[:, 1, 1] <= 0)
                or np.any(intrinsics[:, 0, 2] < 0) or np.any(intrinsics[:, 0, 2] >= WIDTH)
                or np.any(intrinsics[:, 1, 2] < 0) or np.any(intrinsics[:, 1, 2] >= HEIGHT)
                or not np.allclose(intrinsics[:, 2], np.asarray((0, 0, 1)), atol=1e-6)):
            raise ValueError(f"{self.record_id}: invalid ScanNet intrinsics")
        metadata = json.loads(self.path("metadata").read_text(encoding="utf-8"))
        continuity = metadata.get("continuity_validation", {})
        orientation = metadata.get("orientation_validation", {})
        if continuity.get("passed") is not True or orientation.get("passed") is not True:
            raise ValueError(f"{self.record_id}: ScanNet continuity/orientation validation did not pass")
        if int(orientation.get("rotation_cw_degrees", -1)) not in (0, 90, 180, 270):
            raise ValueError(f"{self.record_id}: invalid ScanNet record orientation")
        with np.load(self.path("pointcloud"), allow_pickle=False) as cloud:
            required = {"xyz_world", "offsets", "source_frame_indices", "timestamps"}
            if not required.issubset(cloud.files):
                raise ValueError(f"{self.record_id}: ScanNet pointcloud metadata is incomplete")
            xyz, offsets = cloud["xyz_world"], cloud["offsets"]
            if (xyz.dtype != np.float32 or xyz.ndim != 2 or xyz.shape[1] != 3 or not np.isfinite(xyz).all()
                    or offsets.dtype != np.int64 or offsets.shape != (194,) or offsets[0] != 0
                    or offsets[-1] != len(xyz) or np.any(np.diff(offsets) <= 0)):
                raise ValueError(f"{self.record_id}: invalid ScanNet pointcloud frame layout")
            start,stop=self.source_frame_start,self.source_frame_start+self.frame_count
            if not np.array_equal(cloud["source_frame_indices"][start:stop].astype(np.int64), source_indices):
                raise ValueError(f"{self.record_id}: pointcloud/source frame identity mismatch")
            if not np.array_equal(cloud["timestamps"][start:stop].astype(np.float64), target):
                raise ValueError(f"{self.record_id}: pointcloud timestamp identity mismatch")
        cache = self.load_correspondences()
        if not len(cache.get("query_frame", ())):
            raise ValueError(f"{self.record_id}: ScanNet correspondence cache is empty")


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
