import json
from pathlib import Path

import numpy as np
import pytest

from long_video.training.camera_only_data import load_camera_only_manifest


def _record(root: Path, count=65):
    sample = root / "sample"
    rgb = sample / "rgb"
    rgb.mkdir(parents=True)
    for index in range(count):
        (rgb / f"{index:06d}.jpg").write_bytes(b"jpeg")
    c2w = np.repeat(np.eye(4, dtype=np.float32)[None], count, axis=0)
    raw = c2w.copy()
    k = np.repeat(np.eye(3, dtype=np.float32)[None], count, axis=0)
    np.save(sample / "target_c2w_local.npy", c2w)
    np.save(sample / "target_c2w_local_raw.npy", raw)
    np.save(sample / "intrinsics.npy", k)
    return {
        "trajectory_id": "scene_camera2_000000", "scene_hash": "a" * 64,
        "rgb_dir": "sample/rgb", "target_c2w_local": "sample/target_c2w_local.npy",
        "target_c2w_local_raw": "sample/target_c2w_local_raw.npy", "intrinsics": "sample/intrinsics.npy",
    }


def test_camera_only_loader_requires_exact_65(tmp_path):
    row = _record(tmp_path)
    (tmp_path / "manifest.json").write_text(json.dumps({"records": [row]}))
    records = load_camera_only_manifest(tmp_path / "manifest.json")
    assert len(records) == 1
    assert records[0].load_cameras()[0].shape == (65, 4, 4)


def test_camera_only_loader_rejects_193(tmp_path):
    row = _record(tmp_path, count=193)
    (tmp_path / "manifest.json").write_text(json.dumps({"records": [row]}))
    with pytest.raises(ValueError, match="65"):
        load_camera_only_manifest(tmp_path / "manifest.json")
