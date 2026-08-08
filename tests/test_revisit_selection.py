import numpy as np
import pytest
from zipfile import ZipFile

from long_video.oracle_training.revisit import (
    MultiChunkContract,
    add_renderer_overlap,
    choose_independent_final_candidates,
    bidirectional_depth_reprojection_overlap,
    scan_holo360d_zip,
    score_large_motion_window,
    score_revisit_window,
)
from long_video.data.camera import resize_intrinsics


def _overlap_cameras(height=4, width=6):
    c2w = np.repeat(np.eye(4, dtype=np.float32)[None], 2, axis=0)
    intrinsics = np.repeat(np.array([[4.0, 0.0, 2.5], [0.0, 4.0, 1.5], [0.0, 0.0, 1.0]], np.float32)[None], 2, axis=0)
    depth = np.full((2, height, width), 2.0, np.float32)
    visibility = np.ones((2, height, width), bool)
    return depth, visibility, c2w, intrinsics


def test_bidirectional_depth_reprojection_same_pose_is_near_one():
    depth, visibility, c2w, intrinsics = _overlap_cameras()
    overlap = bidirectional_depth_reprojection_overlap(depth, visibility, c2w, intrinsics)
    assert overlap == pytest.approx(1.0)


def test_bidirectional_depth_reprojection_rejects_depth_inconsistency():
    depth, visibility, c2w, intrinsics = _overlap_cameras()
    c2w[1, 0, 3] = 0.8
    depth[1].fill(9.0)
    overlap = bidirectional_depth_reprojection_overlap(depth, visibility, c2w, intrinsics)
    assert np.isfinite(overlap) and 0.0 <= overlap < 0.2


def test_low_resolution_intrinsics_use_pixel_center_scaling():
    _depth, _visibility, _c2w, intrinsics = _overlap_cameras(384, 640)
    scaled = resize_intrinsics(intrinsics, (384, 640), (192, 320))
    np.testing.assert_allclose(scaled[:, 0, 0], intrinsics[:, 0, 0] * 0.5)
    np.testing.assert_allclose(scaled[:, 1, 1], intrinsics[:, 1, 1] * 0.5)
    np.testing.assert_allclose(scaled[:, 0, 2], (intrinsics[:, 0, 2] + 0.5) * 0.5 - 0.5)
    np.testing.assert_allclose(scaled[:, 1, 2], (intrinsics[:, 1, 2] + 0.5) * 0.5 - 0.5)


def test_multi_chunk_contract():
    expected = {8: (257, 33), 12: (385, 49), 16: (513, 65)}
    for chunks, values in expected.items():
        contract = MultiChunkContract(chunks).validate()
        assert (contract.dense_frames, contract.anchors) == values


def test_revisit_prefers_matching_orientation():
    poses = np.repeat(np.eye(4, dtype=np.float64)[None], 33, axis=0)
    poses[:, 0, 3] = np.sin(np.linspace(0, 2 * np.pi, len(poses)))
    matching = score_revisit_window(poses, 0, len(poses))
    poses[-1, :3, :3] = np.diag([-1, 1, -1])
    opposite = score_revisit_window(poses, 0, len(poses))
    assert matching["pose_prefilter_score"] > opposite["pose_prefilter_score"]


def test_revisit_later_anchor_lands_in_training_chunk():
    poses = np.repeat(np.eye(4, dtype=np.float64)[None], 33, axis=0)
    poses[:, 0, 3] = np.sin(np.linspace(0, 2 * np.pi, len(poses)))
    candidate = score_revisit_window(poses, 0, 33)
    chunk = candidate["training_chunk_index"]
    assert 4 * chunk <= candidate["later_anchor_offset"] <= 4 * chunk + 4


def test_large_motion_selects_chunk_where_motion_occurs():
    poses = np.repeat(np.eye(4, dtype=np.float64)[None], 33, axis=0)
    poses[13:17, 0, 3] = np.linspace(0, 5, 4)
    poses[17:, 0, 3] = 5
    candidate = score_large_motion_window(poses, 0, 33)
    assert candidate["training_chunk_index"] == 3
    assert candidate["selection_translation"] == pytest.approx(5.0)


def test_renderer_overlap_controls_final_score_and_independent_choice():
    base = {
        "start": 0, "anchor_count": 33, "chunks": 8,
        "max_translation": 2.0, "max_rotation_degrees": 20.0,
        "selection_translation": 1.0, "selection_rotation_degrees": 10.0,
        "selection_temporal_gap_anchors": 12, "pose_overlap_proxy": 0.8,
        "training_chunk_index": 3,
    }
    revisit = add_renderer_overlap(base, 0.7, sample_type="revisit")
    overlapping_motion = add_renderer_overlap(base, 0.1, sample_type="large_motion")
    independent_motion = add_renderer_overlap(
        {**base, "start": 100}, 0.2, sample_type="large_motion",
    )
    selected_revisit, selected_motion = choose_independent_final_candidates(
        [revisit], [overlapping_motion, independent_motion],
    )
    assert selected_revisit[0]["renderer_overlap"] == 0.7
    assert selected_motion[0]["start"] == 100
    assert selected_motion[0]["independent_from_revisit"] is True


def test_final_candidates_are_selected_per_scene_and_chunk_length():
    base = {
        "start": 0, "anchor_count": 33, "chunks": 8,
        "max_translation": 2.0, "max_rotation_degrees": 20.0,
        "selection_translation": 1.0, "selection_rotation_degrees": 10.0,
        "selection_temporal_gap_anchors": 12, "pose_overlap_proxy": 0.8,
        "training_chunk_index": 3,
    }
    revisits, motions = [], []
    for scene_id in ("Indoor_013", "Outdoor_008"):
        candidate = {**base, "scene_id": scene_id}
        revisits.append(add_renderer_overlap(candidate, 0.7, sample_type="revisit"))
        motions.append(add_renderer_overlap(candidate, 0.1, sample_type="large_motion"))
    selected_revisit, selected_motion = choose_independent_final_candidates(revisits, motions)
    assert {item["scene_id"] for item in selected_revisit} == {"Indoor_013", "Outdoor_008"}
    assert {item["scene_id"] for item in selected_motion} == {"Indoor_013", "Outdoor_008"}


def test_scan_accepts_official_consolidated_pose_file(tmp_path):
    archive = tmp_path / "Outdoor_008.zip"
    timestamps = [f"{index + 1:.6f}" for index in range(4)]
    pose_rows = ["image x y z r0 r1 r2 r3 r4 r5 r6 r7 r8"]
    with ZipFile(archive, "w") as handle:
        for index, stem in enumerate(timestamps):
            for relative in (
                f"rgb/{stem}.jpg", f"depth/mesh_depth/{stem}.exr", f"mask/{stem}.jpg",
            ):
                handle.writestr(f"Outdoor_008/{relative}", b"x")
            pose_rows.append(f"{stem}.jpg {index} 0 0 1 0 0 0 1 0 0 0 1")
        handle.writestr("Outdoor_008/poses/pose.txt", "\n".join(pose_rows))
    report, frame_ids, poses, runs = scan_holo360d_zip(archive)
    assert report["pose_format"] == "consolidated"
    assert report["matched_counts"] == {"rgb": 4, "depth": 4, "mask": 4, "pose": 4}
    assert frame_ids == timestamps
    assert poses.shape == (4, 4, 4)
    assert runs == [(0, 4)]
