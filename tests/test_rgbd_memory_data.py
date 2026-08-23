import json
import importlib.util
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from long_video.training.rgbd_memory_data import load_rgbd_memory_manifest


def _record(tmp_path: Path):
    root = tmp_path / "record"
    (root / "rgb").mkdir(parents=True)
    (root / "depth").mkdir()
    for index in range(97):
        Image.new("RGB", (832, 480)).save(root / "rgb" / f"{index:06d}.png")
        Image.fromarray(np.ones((480, 832), np.uint16)).save(root / "depth" / f"{index:06d}.png")
    poses = np.repeat(np.eye(4)[None], 97, axis=0)
    intrinsics = np.repeat(np.eye(3)[None], 97, axis=0)
    np.save(root / "c2w_abs.npy", poses)
    np.save(root / "c2w_local.npy", poses)
    np.save(root / "intrinsics.npy", intrinsics)
    np.save(root / "timestamps.npy", np.arange(97, dtype=np.float64))
    np.savez_compressed(root / "correspondence_cache.npz", **{
        "query_frame": np.array([32]), "key_frame": np.array([0]),
        "query_chunk": np.array([1]), "key_chunk": np.array([0]),
        "query_t": np.array([0]), "key_t": np.array([0]),
        "query_y": np.array([1]), "query_x": np.array([2]),
        "key_y": np.array([1]), "key_x": np.array([2]), "weight": np.array([1.0]),
    })
    return {
        "record_id": "test:scene:000000", "dataset": "test", "scene_id": "scene", "sequence_id": "seq",
        "rgb_dir": "record/rgb", "depth_dir": "record/depth", "c2w_abs": "record/c2w_abs.npy",
        "c2w_local": "record/c2w_local.npy", "intrinsics": "record/intrinsics.npy",
        "timestamps": "record/timestamps.npy", "correspondence_cache": "record/correspondence_cache.npz",
        "frame_count": 97, "chunk_count": 3, "height": 480, "width": 832,
    }


def test_strict_rgbd_manifest_loads_97_frame_record(tmp_path):
    row = _record(tmp_path)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"records": [row]}))
    record = load_rgbd_memory_manifest(manifest)[0]
    assert record.load_cameras()[0].shape == (97, 4, 4)
    assert len(list(record.correspondence_rows())) == 1


def test_rgbd_manifest_rejects_noncausal_cache(tmp_path):
    row = _record(tmp_path)
    cache = tmp_path / "record" / "correspondence_cache.npz"
    with np.load(cache) as old:
        data = {key: old[key] for key in old.files}
    data["key_frame"] = np.array([32])
    np.savez_compressed(cache, **data)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"records": [row]}))
    with pytest.raises(ValueError, match="not strictly causal"):
        load_rgbd_memory_manifest(manifest)


def test_camera_only_record_does_not_require_correspondence(tmp_path):
    row = _record(tmp_path)
    row.pop("correspondence_cache")
    row.update(training_scope="camera_only", memory_eligible=False)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"records": [row]}))
    record = load_rgbd_memory_manifest(manifest)[0]
    assert not record.memory_eligible
    assert record.load_correspondences() == {}
    assert list(record.correspondence_rows()) == []


def test_sequence_split_hits_exact_train_count_without_leakage():
    script = Path(__file__).parents[1] / "scripts" / "split_rgbd_train_val.py"
    spec = importlib.util.spec_from_file_location("rgbd_split", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    records = []
    for dataset, sizes in {"tum": (3, 7), "bonn": (3, 7), "nrgbd": (3, 7)}.items():
        for sequence_index, size in enumerate(sizes):
            records.extend({"dataset": dataset, "sequence_id": f"seq-{sequence_index}", "record_id": f"{dataset}-{sequence_index}-{clip}"} for clip in range(size))
    all_rows, train, val = module.split_records(records, 21)
    assert len(all_rows) == 30 and len(train) == 21 and len(val) == 9
    assert {(row["dataset"], row["sequence_id"]) for row in train}.isdisjoint({(row["dataset"], row["sequence_id"]) for row in val})
