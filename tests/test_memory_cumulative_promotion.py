from types import SimpleNamespace

import numpy as np

from long_video.memory.memory_manager import MemoryManager


def _node(node_id, xyz, source, confidence=None):
    xyz = np.asarray(xyz, np.float32)
    count = len(xyz)
    return SimpleNamespace(
        node_id=node_id,
        status="active" if node_id == "node_000" else "candidate",
        parent_id=None,
        center_c2w=np.eye(4, dtype=np.float32),
        created_frame=0,
        coverage_radius=1.0,
        bbox_min=xyz.min(axis=0),
        bbox_max=xyz.max(axis=0),
        points_xyz=xyz,
        points_rgb=np.arange(count * 3, dtype=np.uint8).reshape(count, 3),
        points_confidence=(np.ones(count, np.float32) if confidence is None
                           else np.asarray(confidence, np.float32)),
        points_source=np.asarray(source, np.int8),
        observation_count=np.ones(count, np.int16),
        points_normal=None,
        point_view_mask=None,
        points_rgb_content_origin=None,
        points_depth_content_origin=None,
        points_evidence_role=None,
        points_rgb_evidence_role=None,
        points_depth_evidence_role=None,
        quality_metrics={},
    )


def test_promotion_preserves_parent_and_appends_only_verified_novel_voxels():
    manager = MemoryManager(voxel_size=0.1)
    parent = _node("node_000", [[0, 0, 1], [1, 0, 1]], [0, 0])
    parent_xyz = parent.points_xyz.copy()
    candidate = _node(
        "node_001",
        [
            [0.01, 0, 1],  # verified, but duplicates a committed parent voxel
            [2.00, 0, 1],  # unverified and must never enter the committed world
            [3.00, 0, 1],  # verified and novel
            [3.01, 0, 1],  # same novel voxel, lower confidence
        ],
        [3, 2, 3, 3],
        [0.9, 1.0, 0.8, 0.5],
    )
    candidate._verified_new_point_mask = np.asarray([True, False, True, True])

    promoted = manager.promote(parent, candidate)

    assert np.array_equal(promoted.points_xyz[: len(parent_xyz)], parent_xyz)
    assert len(promoted.points_xyz) == len(parent_xyz) + 1
    assert np.allclose(promoted.points_xyz[-1], [3.0, 0.0, 1.0])
    assert promoted.quality_metrics["parent_points_preserved"] is True
    assert promoted.quality_metrics["eligible_candidate_point_count"] == 3
    assert promoted.quality_metrics["appended_eligible_point_count"] == 1
    assert promoted.quality_metrics["discarded_ineligible_candidate_point_count"] == 1
    assert parent.status == "archived"
    assert promoted.status == "active"
    assert promoted.parent_id == parent.node_id


def test_depth_anchor_failure_retries_unanchored_only_when_depth_rejection_is_ignored():
    class Backend:
        def __init__(self):
            self.calls = []

        def predict(self, *_args, **kwargs):
            self.calls.append(kwargs)
            if kwargs["known_depth"] is not None:
                raise ValueError("Scale anchor rejected: 0 valid pixels < 32")
            return SimpleNamespace(
                diagnostics={}, scale_info={"anchor_source": "relative"},
            )

    backend = Backend()
    manager = MemoryManager(
        geometry_backend=backend,
    )
    prediction = manager._predict_geometry(
        np.zeros((8, 2, 2, 3), np.float32),
        np.repeat(np.eye(4, dtype=np.float32)[None], 8, axis=0),
        np.repeat(np.eye(3, dtype=np.float32)[None], 8, axis=0),
        known_depth=np.ones((8, 2, 2), np.float32),
        known_mask=np.ones((8, 2, 2), bool),
        known_scale=None,
    )

    assert len(backend.calls) == 2
    assert backend.calls[1]["known_depth"] is None
    assert prediction.diagnostics["depth_anchor_fallback"] is True
    assert prediction.scale_info["anchor_source"] == "unanchored_pi3_mandatory_promotion"


def test_generated_points_keep_nine_pixels_from_parent_interior_except_boundary():
    candidate = SimpleNamespace(
        points_xyz=np.asarray([
            [15, 15, 1],  # parent interior pixel
            [22, 15, 1],  # within nine pixels of parent interior
            [29, 15, 1],  # ten pixels from the nearest parent interior pixel
            [10, 15, 1],  # parent boundary pixel is explicitly exempt
            [21, 15, 1],  # generated points do not exclude each other
        ], np.float32),
        points_source=np.asarray([2, 2, 2, 2, 2], np.int8),
    )
    parent_visible = np.zeros((32, 32), bool)
    parent_visible[10:21, 10:21] = True
    frame = SimpleNamespace(
        camera_c2w=np.eye(4, dtype=np.float32),
        intrinsics=np.eye(3, dtype=np.float32),
        warp_visibility=parent_visible,
    )

    keep = MemoryManager._outside_parent_projection_mask(candidate, [frame])

    assert keep.tolist() == [False, False, True, True, False]


def test_source_and_new_world_voxel_default_is_point_zero_two():
    assert MemoryManager().voxel_size == 0.008
