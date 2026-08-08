import numpy as np
from zipfile import ZipFile

from long_video.oracle_training.revisit import MultiChunkContract, scan_holo360d_zip, score_revisit_window


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
    assert matching["revisit_score"] > opposite["revisit_score"]


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
