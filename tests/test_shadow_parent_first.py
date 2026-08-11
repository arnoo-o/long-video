from types import SimpleNamespace

import numpy as np
import pytest

from long_video.geometry.point_renderer import render
from long_video.memory.memory_manager import MemoryManager
from long_video.online.transition_buffer import TransitionBuffer
from long_video.online.causal_renderer import CausalActiveNodeRenderer
from long_video.types import CameraBatch, Z_DEPTH
from long_video.online.delayed_activation import DelayedNodeActivationQueue


def _transition_frame(index):
    pose = np.eye(4, dtype=np.float32)
    pose[0, 3] = float(index)
    shape = (2, 2)
    return dict(
        generated_rgb=np.zeros((*shape, 3), np.uint8),
        camera_c2w=pose,
        intrinsics=np.eye(3, dtype=np.float32),
        old_node_warp=np.zeros((*shape, 3), np.float32),
        warp_visibility=np.ones(shape, bool),
        old_node_warp_depth=np.ones(shape, np.float32),
        old_node_warp_source=np.zeros(shape, np.int8),
        old_node_warp_rgb_content_origin=np.full(shape, "oracle_source", dtype="U24"),
        old_node_warp_depth_content_origin=np.full(shape, "oracle_source", dtype="U24"),
        old_node_warp_evidence_role=np.full(shape, "parent_warp", dtype="U24"),
        old_node_warp_rgb_evidence_role=np.full(shape, "parent_warp", dtype="U24"),
        old_node_warp_depth_evidence_role=np.full(shape, "parent_warp", dtype="U24"),
        old_node_depth_convention=Z_DEPTH,
        warp_confidence=np.ones(shape, np.float32),
        coverage=0.5,
        global_frame_index=index,
    )


def _node(node_id, xyz, source, status="candidate"):
    xyz = np.asarray(xyz, np.float32)
    count = len(xyz)
    return SimpleNamespace(
        node_id=node_id,
        status=status,
        parent_id=None,
        center_c2w=np.eye(4, dtype=np.float32),
        created_frame=0,
        coverage_radius=1.0,
        bbox_min=xyz.min(axis=0),
        bbox_max=xyz.max(axis=0),
        points_xyz=xyz,
        points_rgb=np.arange(count * 3, dtype=np.uint8).reshape(count, 3),
        points_confidence=np.ones(count, np.float32),
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


def test_boundary_is_mapping_and_previous_four_are_held_out():
    buffer = TransitionBuffer(max_length=256)
    for index in range(193):
        buffer.append(**_transition_frame(index))
    mapping, heldout = buffer.select_keyframes(8, 4)
    mapping_indices = [frame.global_frame_index for frame in mapping]
    heldout_indices = [frame.global_frame_index for frame in heldout]
    assert max(mapping_indices) == 192
    assert 192 in mapping_indices
    assert all(index <= 192 for index in mapping_indices)
    assert heldout_indices == [188, 189, 190, 191]
    assert 192 not in heldout_indices


def test_shadow_freezes_hash_and_commits_only_when_due():
    manager = MemoryManager(voxel_size=0.1)
    parent = _node("node_000", [[0, 0, 1]], [0], status="active")
    candidate = _node("node_001", [[2, 0, 1]], [3])
    candidate._verified_new_point_mask = np.asarray([True])
    shadow = manager.prepare_shadow(parent, candidate)
    assert parent.status == "active"
    assert shadow.status == "shadow"
    assert shadow.quality_metrics["parent_point_count"] == 1
    creation_hash = shadow.quality_metrics["shadow_hash_at_creation"]
    assert creation_hash == manager.shadow_points_sha256(shadow)
    assert shadow.points_xyz.flags.writeable is False

    queue = DelayedNodeActivationQueue(delay_chunks=2, max_pending=1)
    queue.schedule(shadow, created_after_chunk=5)
    assert queue.activate_due(6) is None
    assert parent.status == "active"
    due = queue.activate_due(7)
    assert due.node is shadow
    activation_hash = manager.verify_shadow(shadow)
    manager.commit_shadow(parent, shadow, verified_hash=activation_hash)
    assert parent.status == "archived"
    assert shadow.status == "active"
    assert shadow.quality_metrics["shadow_hash_at_activation"] == creation_hash
    assert shadow.quality_metrics["shadow_hash_equal"] is True


def test_deferred_process_keeps_parent_through_pending_chunk():
    manager = MemoryManager(voxel_size=0.1)
    parent = _node("node_000", [[0, 0, 1]], [0], status="active")
    candidate = _node("node_001", [[2, 0, 1]], [3])
    candidate._verified_new_point_mask = np.asarray([True])
    manager.buffer = SimpleNamespace(can_attempt=lambda _frame: True)
    manager._append_chunk = lambda *_args, **_kwargs: None
    manager.readiness_report = lambda _overlap=None: {"ready": True}
    manager.build_candidate = lambda _active, _created: (candidate, [], [])
    manager.validate_candidate = lambda _candidate, _frames, _heldout: (True, {})
    warp = SimpleNamespace(coverage_per_frame=np.zeros(1, np.float32))
    generated = np.zeros((1, 1, 1, 3), np.uint8)

    returned, event = manager.process_chunk(
        parent, generated, None, warp, frame_start=0,
        allow_candidate_promotion=True, defer_candidate_promotion=True,
    )
    shadow = event["shadow_node"]
    assert returned is parent
    assert parent.status == "active"
    assert shadow.status == "shadow"
    frozen_hash = shadow.quality_metrics["shadow_hash_at_creation"]

    # A pending activation may continue collecting history but cannot rebuild or
    # mutate the frozen shadow.
    returned, pending_event = manager.process_chunk(
        parent, generated, None, warp, frame_start=1,
        allow_candidate_promotion=False, defer_candidate_promotion=True,
    )
    assert returned is parent
    assert "shadow_node" not in pending_event
    assert shadow.quality_metrics["shadow_hash_at_creation"] == frozen_hash
    assert manager.shadow_points_sha256(shadow) == frozen_hash


def test_shadow_tamper_is_detected_before_commit():
    manager = MemoryManager(voxel_size=0.1)
    parent = _node("node_000", [[0, 0, 1]], [0], status="active")
    candidate = _node("node_001", [[2, 0, 1]], [3])
    candidate._verified_new_point_mask = np.asarray([True])
    shadow = manager.prepare_shadow(parent, candidate)
    shadow.points_xyz.setflags(write=True)
    shadow.points_xyz[1, 0] += 0.25
    with pytest.raises(RuntimeError, match="shadow hash mismatch"):
        manager.verify_shadow(shadow)
    assert parent.status == "active"


def test_parent_first_renderer_keeps_parent_and_allows_distant_delta_hole():
    pose = np.eye(4, dtype=np.float32)
    intrinsics = np.eye(3, dtype=np.float32)
    intrinsics[0, 2] = intrinsics[1, 2] = 4.0
    cameras = CameraBatch(pose[None], intrinsics[None], 8, 8)
    node = _node(
        "node_001",
        [[0, 0, 2], [0, 0, 1], [3, 0, 1]],
        [0, 3, 3],
        status="active",
    )
    node.points_rgb[:] = np.asarray([[20, 0, 0], [220, 0, 0], [0, 220, 0]], np.uint8)
    node.points_confidence[:] = [0.9, 0.1, 0.8]
    node.quality_metrics["parent_point_count"] = 1
    warp = render(node, cameras, point_radius=0, device="cpu")
    # The near generated point projects exactly onto the parent pixel and is
    # excluded by parent-first compositing/dilation.
    assert warp.source[0, 4, 4] == 0
    assert warp.point_index[0, 4, 4] == 0
    np.testing.assert_allclose(warp.winning_xyz_world[0, 4, 4], node.points_xyz[0])
    np.testing.assert_allclose(warp.rgb[0, 4, 4], [20 / 255.0, 0.0, 0.0])
    assert not warp.delta_allowed_visibility[0, 4, 4]
    assert warp.delta_output_on_parent_visible == 0
    assert warp.delta_output_on_parent_protection_mask == 0
    # The distant generated point is three pixels away and fills a true hole.
    assert warp.source[0, 4, 7] == 3
    assert warp.point_index[0, 4, 7] == 2
    np.testing.assert_allclose(warp.winning_xyz_world[0, 4, 7], node.points_xyz[2])
    assert warp.delta_allowed_visibility[0, 4, 7]


def test_causal_renderer_exposes_the_composited_warp_to_conditioning():
    pose = np.eye(4, dtype=np.float32)
    intrinsics = np.eye(3, dtype=np.float32)
    intrinsics[0, 2] = intrinsics[1, 2] = 6.0
    cameras = CameraBatch(pose[None], intrinsics[None], 8, 12)
    node = _node("node_001", [[0, 0, 2], [4, 0, 1]], [0, 3], status="active")
    node.parent_id = "node_000"
    node.quality_metrics["parent_point_count"] = 1

    class Store:
        def load(self, _node_id):
            return node

    renderer = CausalActiveNodeRenderer(Store(), node_id="node_001")
    result = renderer.render(cameras, frame_start=0, allow_reactivation=False)
    assert result.warp.parent_first is True
    assert result.warp.rgb_content_origin[0, 6, 6] == "oracle_source"
    assert result.warp.rgb_content_origin[0, 6, 10] == "model_generated"
    assert result.provenance["selection"] == "explicit_scheduled_render_node"


def test_second_promotion_uses_all_previous_points_as_parent():
    manager = MemoryManager(voxel_size=0.1)
    node0 = _node("node_000", [[0, 0, 1]], [0], status="active")
    node1_candidate = _node("node_001", [[2, 0, 1]], [3])
    node1_candidate._verified_new_point_mask = np.asarray([True])
    node1 = manager.promote(node0, node1_candidate)
    assert node1.quality_metrics["parent_point_count"] == 1
    node2_candidate = _node("node_002", [[4, 0, 1]], [3])
    node2_candidate._verified_new_point_mask = np.asarray([True])
    node2 = manager.promote(node1, node2_candidate)
    assert node2.quality_metrics["parent_point_count"] == len(node1.points_xyz)
    assert node2.parent_point_count == len(node1.points_xyz)
