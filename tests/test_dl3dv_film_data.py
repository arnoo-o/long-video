import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import torch

from long_video.data.dl3dv import (
    CHUNK_STRIDE, center_crop_resize_geometry, chunk_real_indices,
    load_dl3dv_scene, ranked_candidates, read_official_metadata,
    resample_real_frame_indices, source_relative_opencv_c2w,
)
from long_video.training.causal_rollout import (
    AllChunkRoundRobin, build_boundary_states_once, current_chunk_loss_weights,
    validate_boundary_cache,
)


def test_official_metadata_is_joined_and_balanced(tmp_path):
    scene_hashes = [f"{i:064x}" for i in range(4)]
    csv_path = tmp_path / "valid.csv"
    csv_path.write_text("hash,sensibility label,batch,duration\n" + "\n".join(
        f"{h},Without any sensitivity information,1K,{'' if h == scene_hashes[0] else '60'}"
        for h in scene_hashes))
    rows = []
    for index, scene_hash in enumerate(scene_hashes):
        poi, category = (("Living-Room", "Indoor") if index % 2 == 0 else
                         ("Campus", "Nature & Outdoors"))
        rows.append(f"<tr><td><figcaption>{scene_hash}</figcaption><td>1K</td><td>bd</td>"
                    f"<td>abs nonreflection</td><td>abs nontransparent</td><td>nlight</td>"
                    f"<td>{poi}</td><td>{category}</td><td>phone</td></td></tr>")
    html = tmp_path / "index.html"; html.write_text("<table>" + "".join(rows) + "</table>")
    records = read_official_metadata(csv_path, html)
    ranked = ranked_candidates(records, seed=1)
    assert len(ranked) == 4
    assert any(x["duration"] == 0 for x in ranked)
    assert [x["environment"] for x in ranked] == ["indoor", "outdoor", "indoor", "outdoor"]


def test_real_frame_resampling_never_synthesizes_indices():
    times = np.arange(20, dtype=np.float64) / 10
    indices = resample_real_frame_indices(times, target_fps=24)
    assert indices.dtype == np.int64
    assert np.all(np.diff(indices) >= 0)
    assert set(indices).issubset(set(range(20)))
    assert len(np.unique(indices)) < len(indices)  # duplicates are real frames, never RIFE frames


def test_nerfstudio_pose_becomes_source_relative_opencv(tmp_path):
    frames = []
    for index in range(3):
        Image.new("RGB", (800, 480), (index * 20, 0, 0)).save(tmp_path / f"{index}.png")
        pose = np.eye(4); pose[0, 3] = index
        frames.append({"file_path": f"{index}.png", "transform_matrix": pose.tolist(), "time": index / 30})
    (tmp_path / "transforms.json").write_text(json.dumps({
        "fl_x": 500, "fl_y": 510, "cx": 399.5, "cy": 239.5, "frames": frames,
    }))
    scene = load_dl3dv_scene(tmp_path)
    local = source_relative_opencv_c2w(scene.c2w_opencv, 1)
    np.testing.assert_allclose(local[1], np.eye(4), atol=1e-6)
    np.testing.assert_allclose(local[:, 0, 3], [-1, 0, 1], atol=1e-6)
    assert np.allclose(np.linalg.det(local[:, :3, :3]), 1)


def test_official_stale_images_path_uses_480p_images_directory(tmp_path):
    image_root = tmp_path / "images_8"; image_root.mkdir()
    frames = []
    for index in range(2):
        Image.new("RGB", (800, 480)).save(image_root / f"frame_{index:05d}.png")
        frames.append({"file_path": f"images/frame_{index:05d}.png",
                       "transform_matrix": np.eye(4).tolist()})
    (tmp_path / "transforms.json").write_text(json.dumps({
        "fl_x": 500, "fl_y": 500, "frames": frames,
    }))
    scene = load_dl3dv_scene(tmp_path, duration=2.0)
    assert all(path.parent.name == "images_8" for path in scene.image_paths)
    np.testing.assert_allclose(scene.frame_times, [0, 2])


def test_crop_resize_intrinsics_tracks_pixel_centers():
    k = np.array([[500, 0, 399.5], [0, 500, 239.5], [0, 0, 1]], np.float32)
    crop, out = center_crop_resize_geometry((480, 854), k, (384, 640))
    assert crop == (27, 0, 827, 480)
    np.testing.assert_allclose(out[:2, 2], [297.9, 191.5], atol=1e-4)


@pytest.mark.parametrize("count", [8, 12])
def test_chunk_layout_has_one_shared_boundary(count):
    indices = list(range(count * CHUNK_STRIDE + 1))
    chunks = chunk_real_indices(indices, count)
    assert all(len(x) == 33 for x in chunks)
    for left, right in zip(chunks, chunks[1:]):
        assert left[-1] == right[0]
        assert set(left).intersection(right) == {left[-1]}


@pytest.mark.parametrize("count", [8, 12])
def test_round_robin_covers_chunk_zero_and_is_balanced(count):
    scheduler = AllChunkRoundRobin()
    values = [scheduler.next("trajectory", count) for _ in range(count * 3 + 2)]
    assert set(values) == set(range(count))
    counts = scheduler.coverage_report([{"trajectory_id": "trajectory", "chunk_count": count}])["trajectory"]
    assert max(counts) - min(counts) <= 1
    state = scheduler.state_dict(); restored = AllChunkRoundRobin(); restored.load_state_dict(state)
    assert restored.state_dict() == state


def test_one_rollout_caches_every_boundary_without_future_gt():
    record = {"trajectory_id": "t", "chunk_count": 8, "uses_future_gt": False}
    calls = []
    def rollout(index, state):
        calls.append(index)
        return {"generated_chunks": state["generated_chunks"] + [index]}
    cache = build_boundary_states_once(record, {"generated_chunks": []}, rollout,
                                       model_fingerprint="sha")
    assert calls == list(range(7))
    assert validate_boundary_cache(cache, record)
    histories = sorted(len(x["history_chunks"]) for x in cache.values())
    assert histories == list(range(8))
    weights = current_chunk_loss_weights(8, 0)
    assert weights[0] == 0 and torch.all(weights[1:] == 1)


def test_future_gt_is_rejected_by_boundary_builder():
    with pytest.raises(ValueError, match="future GT"):
        build_boundary_states_once({"trajectory_id": "x", "chunk_count": 8,
                                    "uses_future_gt": True}, {}, lambda *_: {},
                                   model_fingerprint="sha")
