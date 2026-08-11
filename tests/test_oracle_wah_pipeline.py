import numpy as np
import pytest

from long_video.data.camera import rgb_to_uint8
from long_video.data.erp_geometry import (
    backproject_erp_ray_distance, erp_unit_rays, perspective_unit_rays,
    ray_distance_to_z_depth, source_relative_c2w, z_depth_to_ray_distance,
)
from long_video.data.panorama_projection import intrinsics_from_fov
from long_video.geometry.point_renderer import render_numpy_reference
from long_video.initialization.geometry_backend import GeometryPrediction, MultiViewGeometryBackend
from long_video.memory.memory_manager import MemoryManager
from long_video.memory.node_store import NodeStore
from long_video.oracle_training.contracts import (
    GeneratedMemoryBatch, SupervisionBatch, assert_history_frames_are_generated,
    assert_no_supervision_content, validate_content_labels,
)
from long_video.oracle_training.dataset import attach_warp_provenance
from long_video.oracle_training.oracle_node import build_oracle_erp_node
from long_video.oracle_training.temporal import ChunkContract, build_primary_loss_masks
from long_video.types import CameraBatch, ScaleMetadata, Z_DEPTH


def test_erp_opencv_directions_and_periodic_boundary():
    rays = erp_unit_rays(4, 8, pixel_center=0.0)
    np.testing.assert_allclose(rays[2, 4], [0, 0, 1], atol=1e-6)
    np.testing.assert_allclose(rays[2, 6], [1, 0, 0], atol=1e-6)
    np.testing.assert_allclose(rays[2, 2], [-1, 0, 0], atol=1e-6)
    assert rays[1, 4, 1] < 0
    assert rays[3, 4, 1] > 0
    dense = erp_unit_rays(256, 512, pixel_center=0.5)
    assert np.max(np.linalg.norm(dense, axis=-1) - 1) < 1e-6
    assert np.linalg.norm(dense[:, 0] - dense[:, -1], axis=-1).max() < 0.013


def test_full_erp_rgbd_backprojection_and_c2w():
    rgb = np.zeros((4, 8, 3), np.uint8); rgb[..., 0] = 127
    depth = np.full((4, 8), 2.0, np.float32); mask = np.ones((4, 8), bool)
    pose = np.eye(4, dtype=np.float32); pose[:3, 3] = [1, 2, 3]
    xyz, colors = backproject_erp_ray_distance(rgb, depth, mask, pose, pixel_center=0.0)
    assert xyz.shape == (32, 3) and colors.shape == (32, 3)
    np.testing.assert_allclose(xyz[2 * 8 + 4], [1, 2, 5], atol=1e-6)


def test_ray_distance_z_depth_roundtrip():
    k = intrinsics_from_fov(90, 8, 6)
    rays = perspective_unit_rays(k, 6, 8)
    ray = np.full((6, 8), 3.0, np.float32)
    z = ray_distance_to_z_depth(ray, rays)
    np.testing.assert_allclose(z_depth_to_ray_distance(z, rays), ray, atol=1e-5)


def test_source_relative_world_frame():
    source = np.eye(4, dtype=np.float32); source[:3, 3] = [2, 0, 0]
    target = source.copy(); target[:3, 3] += [0, 0, 3]
    local = source_relative_c2w(source, target)
    np.testing.assert_allclose(local[:3, 3], [0, 0, 3], atol=1e-6)


def test_oracle_m0_metric_and_cross_frame_render():
    height, width = 32, 64
    rgb = np.zeros((height, width, 3), np.uint8)
    rgb[..., 0] = np.arange(width, dtype=np.uint8)[None]
    depth = np.full((height, width), 2.0, np.float32)
    node = build_oracle_erp_node(rgb, depth, np.ones((height, width), bool), voxel_size=0)
    assert node.scale.mode == "dataset_calibrated"
    assert node.scale.meters_per_world_unit == 1.0
    assert np.all(node.points_confidence == 1)
    k = intrinsics_from_fov(90, 32, 24)
    poses = np.repeat(np.eye(4, dtype=np.float32)[None], 2, axis=0)
    poses[1, 2, 3] = 0.1
    warp = attach_warp_provenance(render_numpy_reference(
        node, CameraBatch(poses, np.repeat(k[None], 2, 0), 24, 32), point_radius=1
    ), node)
    assert warp.visibility[0].any() and warp.visibility[1].any()
    assert np.nanmedian(warp.depth[1]) < np.nanmedian(warp.depth[0])
    assert np.all(warp.rgb_content_origin[warp.visibility] == "oracle_source")


def test_float_rgb_conversion_is_unambiguous():
    value = rgb_to_uint8(np.array([0.0, 0.5, 1.0], np.float32))
    np.testing.assert_array_equal(value, [0, 128, 255])
    with pytest.raises(ValueError):
        rgb_to_uint8(np.array([np.nan], np.float32))
    with pytest.raises(ValueError):
        rgb_to_uint8(np.array([2.0], np.float32))


def test_supervision_cannot_enter_history_memory_or_candidate():
    target = np.zeros((1, 2, 2, 3), np.uint8)
    supervision = SupervisionBatch(target, np.ones((1, 2, 2), np.float32), np.ones((1, 2, 2), bool))
    with pytest.raises(RuntimeError): supervision.as_memory_content()
    with pytest.raises(ValueError): assert_no_supervision_content({"target_rgb_for_loss": target}, "MemoryManager")
    with pytest.raises(ValueError): assert_no_supervision_content({"target_z_depth_for_eval": target[..., 0]}, "candidate builder")
    with pytest.raises(ValueError): assert_history_frames_are_generated(target, target)
    GeneratedMemoryBatch(target.copy())


def test_content_origin_and_evidence_role_contract():
    validate_content_labels(
        np.array(["oracle_source", "model_generated"]),
        np.array(["oracle_source", "pi3_prediction"]),
        np.array(["parent_warp", "current_generation"]),
        np.array(["parent_warp", "geometry_prediction"]),
    )
    with pytest.raises(ValueError):
        validate_content_labels(
            np.array(["ground_truth_supervision_only"]),
            np.array(["oracle_source"]), np.array(["direct_source"]),
        )


def test_chunk_zero_boundary_global_indices_and_primary_masks():
    contract = ChunkContract(33, 4)
    assert contract.latent_frames == 9
    assert contract.shared_boundary_rule == "reuse_previous_boundary_as_next_chunk_frame_zero"
    global_indices = [np.arange(chunk * 32, chunk * 32 + 33) for chunk in range(4)]
    assert global_indices[0][0] == 0
    for chunk in range(1, 4):
        assert global_indices[chunk][0] == global_indices[chunk - 1][-1]
    assert global_indices[-1][-1] == 128
    rgb, latent = build_primary_loss_masks(contract)
    assert not rgb[0] and rgb[1:].all()
    assert not latent[0] and latent[1:].all()
    assert latent.sum() == 8
    padding = np.zeros(33, bool); padding[-1] = True
    _, masked = build_primary_loss_masks(contract, padding_frames=padding)
    assert not masked[-1]
    with pytest.raises(ValueError): build_primary_loss_masks(contract, valid_target_frames=np.ones(32, bool))


class _Geometry(MultiViewGeometryBackend):
    def predict(self, view_rgb, view_c2w, intrinsics, known_depth=None, known_mask=None,
                known_depth_convention=None, known_scale=None):
        assert known_depth_convention == Z_DEPTH
        depth = np.where(np.asarray(known_mask), known_depth, 2.0).astype(np.float32)
        confidence = np.where(np.asarray(known_mask), 1.0, 0.6).astype(np.float32)
        return GeometryPrediction(
            depth=depth, depth_confidence=confidence,
            predicted_c2w=np.asarray(view_c2w),
            scale_info={"mode":"dataset_calibrated","meters_per_world_unit":1.0,
                        "uncertainty":0.01,"anchor_source":"known_metric_depth_overlap"},
            diagnostics={"pose_error":0.0,"confidence_source":"local_depth_continuity",
                         "confidence_type":"heuristic"}, depth_convention=Z_DEPTH,
        )


def test_m1_parent_rgb_and_generated_new_region_provenance(tmp_path):
    manager = MemoryManager(geometry_backend=_Geometry(), keyframe_count=8, heldout_count=4)
    h, w = 8, 8
    for index in range(12):
        pose = np.eye(4, dtype=np.float32); pose[0, 3] = index * 0.05
        visible = np.zeros((h, w), bool); visible[:, :4] = True
        manager.buffer.append(
            generated_rgb=np.full((h, w, 3), 240 if index == 11 else 200, np.uint8), camera_c2w=pose,
            intrinsics=intrinsics_from_fov(90, w, h),
            old_node_warp=np.full((h, w, 3), 0.2, np.float32),
            warp_visibility=visible, old_node_warp_depth=np.full((h, w), 1.5, np.float32),
            old_node_warp_source=np.zeros((h, w), np.int8),
            old_node_warp_rgb_content_origin=np.full((h, w), "oracle_source", dtype="U24"),
            old_node_warp_depth_content_origin=np.full((h, w), "oracle_source", dtype="U24"),
            old_node_warp_evidence_role=np.full((h, w), "direct_source", dtype="U24"),
            old_node_warp_rgb_evidence_role=np.full((h, w), "direct_source", dtype="U24"),
            old_node_warp_depth_evidence_role=np.full((h, w), "direct_source", dtype="U24"),
            old_node_depth_convention=Z_DEPTH, warp_confidence=np.ones((h, w), np.float32),
            coverage=0.5, global_frame_index=index,
        )
    parent = build_oracle_erp_node(
        np.zeros((4, 8, 3), np.uint8), np.ones((4, 8), np.float32),
        np.ones((4, 8), bool), voxel_size=0,
    )
    manager.register(parent)
    # The latest buffered frame is the causal boundary and is included in the
    # mapping set, so candidate creation is stamped with global frame 11.
    candidate, mapping, heldout = manager.build_candidate(parent, 11)
    assert len(mapping) == 8 and len(heldout) == 4
    assert candidate.quality_metrics["shadow_boundary_frame"] == 11
    assert max(candidate.quality_metrics["shadow_mapping_frame_indices"]) == 11
    assert all(index <= 11 for index in candidate.quality_metrics["shadow_mapping_frame_indices"])
    assert 11 not in candidate.quality_metrics["shadow_heldout_frame_indices"]
    assert np.all(candidate.view_rgb[:, :, :4] == 51)
    boundary_position = candidate.quality_metrics["persistent_surface_mapping_position"]
    assert np.all(candidate.view_rgb[boundary_position, :, 4:] == 240)
    assert np.all(np.delete(candidate.view_rgb[:, :, 4:], boundary_position, axis=0) == 200)
    assert np.all(candidate.view_rgb_content_origin[:, :, :4] == "oracle_source")
    assert np.all(candidate.view_rgb_content_origin[:, :, 4:] == "model_generated")
    assert np.all(candidate.view_depth_content_origin[:, :, 4:] == "pi3_prediction")
    assert np.all(candidate.view_rgb_evidence_role[:, :, 4:] == "current_generation")
    assert np.all(candidate.view_depth_evidence_role[:, :, 4:] == "geometry_prediction")
    assert np.all(candidate.view_rgb_evidence_role[:, :, :4] == "parent_warp")
    assert np.all(candidate.view_depth_evidence_role[:, :, :4] == "parent_warp")
    assert np.ptp(candidate.view_image_confidence[:, :, 4:]) > 0
    assert candidate.quality_metrics["geometry_diagnostics"]["confidence_type"] == "heuristic"
    assert candidate.quality_metrics["canonical_surface_commit"] is True
    assert candidate.quality_metrics["geometry_input_view_count"] == 8
    assert candidate.quality_metrics["persistent_surface_view_count"] == 1
    assert candidate.quality_metrics["persistent_surface_global_frame"] == 11
    generated_points = candidate.points_source == 2
    assert generated_points.any()
    assert np.all(candidate.points_rgb[generated_points] == 240)
    assert np.all(candidate.observation_count == 1)
    assert np.all(candidate.point_view_mask == 1)
    accepted, metrics = manager.validate_candidate(candidate, mapping, heldout)
    assert accepted
    assert metrics["mandatory_acceptance_by_readiness"] is True
    assert metrics["legacy_rgb_pose_depth_confidence_gates_used"] is False
    assert metrics["legacy_point_verification_gates_used"] is False
    assert metrics["eligible_nonoverlap_point_count"] >= 0
    store = NodeStore(tmp_path / "session")
    store.save(candidate)
    restored = store.load(candidate.node_id)
    assert restored.schema_version == 4
    np.testing.assert_array_equal(
        restored.view_rgb_evidence_role, candidate.view_rgb_evidence_role)
    np.testing.assert_array_equal(
        restored.view_depth_evidence_role, candidate.view_depth_evidence_role)
