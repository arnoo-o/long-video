import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from long_video.training.rgbd_memory_data import load_rgbd_memory_manifest
from scripts.build_scannet_rgbd_records import (
    FRAME_COUNT,
    nearest_source_indices,
    rotate_image_geometry,
    select_split,
)


def _scannet_record(tmp_path: Path) -> tuple[dict, Path]:
    root = tmp_path / "record"
    (root / "rgb").mkdir(parents=True); (root / "depth").mkdir()
    for index in range(FRAME_COUNT):
        Image.new("RGB", (832, 480), color=(20, 30, 40)).save(root / "rgb" / f"{index:06d}.png")
        Image.fromarray(np.full((480, 832), 1000, np.uint16)).save(root / "depth" / f"{index:06d}.png")
    c2w = np.repeat(np.eye(4, dtype=np.float64)[None], FRAME_COUNT, axis=0)
    K = np.repeat(np.asarray(((500, 0, 416), (0, 500, 240), (0, 0, 1)), np.float64)[None], FRAME_COUNT, axis=0)
    target = np.arange(FRAME_COUNT, dtype=np.float64) / 24.0
    indices = np.rint(np.arange(FRAME_COUNT) * 30.0 / 24.0).astype(np.int32)
    source = indices.astype(np.float64) / 30.0
    np.save(root / "c2w_abs.npy", c2w); np.save(root / "c2w_local.npy", c2w)
    np.save(root / "intrinsics.npy", K); np.save(root / "timestamps.npy", target)
    np.save(root / "source_timestamps.npy", source); np.save(root / "source_frame_indices.npy", indices)
    np.savez(root / "pointcloud.npz", xyz_world=np.zeros((FRAME_COUNT, 3), np.float32), offsets=np.arange(FRAME_COUNT + 1, dtype=np.int64), source_frame_indices=indices, timestamps=target)
    np.savez_compressed(root / "correspondence_cache.npz", query_frame=np.asarray([32]), key_frame=np.asarray([0]), query_chunk=np.asarray([1]), key_chunk=np.asarray([0]), query_t=np.asarray([0]), key_t=np.asarray([0]), query_y=np.asarray([0]), query_x=np.asarray([0]), key_y=np.asarray([0]), key_x=np.asarray([0]), weight=np.asarray([1], np.float32))
    (root / "metadata.json").write_text(json.dumps({"continuity_validation": {"passed": True}, "orientation_validation": {"passed": True, "rotation_cw_degrees": 0}}))
    row = {
        "record_id": "scannet__scene0000_00__000000", "dataset": "scannet", "scene_id": "scene0000", "sequence_id": "scene0000_00",
        "rgb_dir": "record/rgb", "depth_dir": "record/depth", "c2w_abs": "record/c2w_abs.npy", "c2w_local": "record/c2w_local.npy",
        "intrinsics": "record/intrinsics.npy", "timestamps": "record/timestamps.npy", "source_timestamps": "record/source_timestamps.npy",
        "source_frame_indices": "record/source_frame_indices.npy", "pointcloud": "record/pointcloud.npz", "metadata": "record/metadata.json",
        "correspondence_cache": "record/correspondence_cache.npz", "frame_count": 193, "chunk_count": 6, "fps": 24,
        "height": 480, "width": 832, "near_depth": 1.0, "memory_eligible": True,
    }
    return row, root


def test_nearest_24fps_mapping_is_real_unique_and_eight_seconds():
    source = np.arange(400, dtype=np.float64) / 30.0
    indices, target, selected = nearest_source_indices(source, 15)
    assert len(indices) == 193 and np.all(np.diff(indices) > 0)
    assert len(np.unique(indices)) == 193 and target[-1] - target[0] == 8.0
    assert np.max(np.abs(target - selected)) <= 1 / 60 + 1e-8


def test_rotation_updates_intrinsics_before_resize():
    image = np.zeros((480, 640, 3), np.uint8)
    K = np.asarray(((500, 0, 310), (0, 510, 230), (0, 0, 1)), np.float64)
    rotated, transformed = rotate_image_geometry(image, K, 90)
    assert rotated.shape[:2] == (640, 480)
    assert np.allclose(transformed, ((510, 0, 249), (0, 500, 310), (0, 0, 1)))


def test_strict_scannet_record_validator(tmp_path):
    row, _ = _scannet_record(tmp_path)
    manifest = tmp_path / "manifest.json"; manifest.write_text(json.dumps({"records": [row]}))
    record = load_rgbd_memory_manifest(manifest, expected_count=1)[0]
    assert record.frame_count == 193 and record.chunk_count == 6
    source, indices = record.load_source_identity()
    assert source.shape == indices.shape == (193,)


def test_scannet_validator_accepts_parent_derived_second_unit(tmp_path):
    row, _ = _scannet_record(tmp_path)
    row.update(record_id="scannet__scene0000_00__000000__frames_096_192",
               parent_record_id="scannet__scene0000_00__000000",
               source_frame_start=96, frame_count=97, chunk_count=3)
    manifest = tmp_path / "unit_manifest.json"; manifest.write_text(json.dumps({"records": [row]}))
    record = load_rgbd_memory_manifest(manifest, expected_count=1)[0]
    assert record.source_frame_start == 96 and record.load_timestamps()[0] == 4.0


def test_scannet_validator_rejects_nonunique_source_frames(tmp_path):
    row, root = _scannet_record(tmp_path)
    indices = np.load(root / "source_frame_indices.npy"); indices[10] = indices[9]
    np.save(root / "source_frame_indices.npy", indices)
    manifest = tmp_path / "manifest.json"; manifest.write_text(json.dumps({"records": [row]}))
    with pytest.raises(ValueError, match="unique and increasing"):
        load_rgbd_memory_manifest(manifest)


def test_scannet_split_is_exact_and_scene_isolated():
    records = []
    for scene in range(8):
        for index in range(100):
            records.append({"record_id": f"r{scene}-{index}", "scene_id": f"scene{scene:04d}", "sequence_id": f"scene{scene:04d}_00"})
    train, val = select_split(records, 540, 60)
    assert len(train) == 540 and len(val) == 60
    assert {row["scene_id"] for row in train}.isdisjoint({row["scene_id"] for row in val})
