"""Online spatial-memory state machine and candidate-node validation."""
from __future__ import annotations

import hashlib

import numpy as np

from ..geometry.point_renderer import render
from ..data.camera import rgb_to_float01, rgb_to_uint8
from ..types import ScaleMetadata, ViewSet, Z_DEPTH
from .node_builder import build_from_views
from ..online.transition_buffer import TransitionBuffer


class MemoryManager:
    REQUIRED_HISTORY_FRAMES = 12
    REQUIRED_TRANSLATION = 2.5
    REQUIRED_VIEW_CHANGE_RADIANS = float(np.deg2rad(25.0))
    REQUIRED_NEW_AREA_RATIO = 0.15
    REQUIRED_MAX_WORLD_OVERLAP = 0.10
    PARENT_PROJECTION_CLEARANCE_PIXELS = 9
    ACTIVE = "ACTIVE"
    TRANSITION = "TRANSITION"
    CANDIDATE = "CANDIDATE"
    VALIDATING = "VALIDATING"
    ACTIVE_NEW_NODE = "ACTIVE_NEW_NODE"

    def __init__(
        self,
        geometry_backend=None,
        node_store=None,
        coverage_threshold=0.35,
        high_confidence_threshold=0.5,
        low_coverage_chunks=2,
        min_transition_frames=12,
        min_translation_baseline=2.5,
        min_view_diversity=float(np.deg2rad(25.0)),
        min_new_area_ratio=0.15,
        max_world_overlap=0.10,
        max_overlap_rgb_error=0.25,
        max_overlap_depth_error=0.5,
        generated_confidence=0.25,
        keyframe_count=8,
        voxel_size=0.02,
        heldout_count=4,
        max_buffer_length=96,
        max_age_frames=240,
        failure_cooldown_frames=32,
        reactivation_hysteresis=0.1,
        transition_readiness_mode="any",
    ):
        self.geometry_backend = geometry_backend
        self.node_store = node_store
        self.coverage_threshold = float(coverage_threshold)
        self.high_confidence_threshold = float(high_confidence_threshold)
        self.low_coverage_chunks = int(low_coverage_chunks)
        required = {
            "min_transition_frames": (int(min_transition_frames), self.REQUIRED_HISTORY_FRAMES),
            "min_translation_baseline": (float(min_translation_baseline), self.REQUIRED_TRANSLATION),
            "min_view_diversity": (float(min_view_diversity), self.REQUIRED_VIEW_CHANGE_RADIANS),
            "min_new_area_ratio": (float(min_new_area_ratio), self.REQUIRED_NEW_AREA_RATIO),
            "max_world_overlap": (float(max_world_overlap), self.REQUIRED_MAX_WORLD_OVERLAP),
        }
        conflicting = {
            name: {"requested": requested, "required": mandated}
            for name, (requested, mandated) in required.items()
            if not np.isclose(requested, mandated, rtol=0.0, atol=1e-9)
        }
        if conflicting:
            raise ValueError(
                "Validated Causal World readiness thresholds are permanent: "
                f"{conflicting}"
            )
        if str(transition_readiness_mode) != "any":
            raise ValueError("Validated Causal World readiness mode is permanently 'any'")
        self.min_transition_frames = self.REQUIRED_HISTORY_FRAMES
        self.min_translation_baseline = self.REQUIRED_TRANSLATION
        self.min_view_diversity = self.REQUIRED_VIEW_CHANGE_RADIANS
        self.min_new_area_ratio = self.REQUIRED_NEW_AREA_RATIO
        self.max_world_overlap = self.REQUIRED_MAX_WORLD_OVERLAP
        self.max_overlap_rgb_error=float(max_overlap_rgb_error)
        self.max_overlap_depth_error = float(max_overlap_depth_error)
        self.generated_confidence = float(generated_confidence)
        self.keyframe_count = int(keyframe_count)
        self.voxel_size = float(voxel_size)
        self.heldout_count = int(heldout_count)
        self.low_count = 0
        self.state = self.ACTIVE
        self.buffer=TransitionBuffer(
            max_length=max_buffer_length,max_age_frames=max_age_frames,
            cooldown_frames=failure_cooldown_frames)
        self.reactivation_hysteresis=float(reactivation_hysteresis)
        self.transition_readiness_mode = "any"
        self.current_world_overlap = 1.0
        self.nodes = {}
        # Accepted candidates may remain in this shadow registry until the
        # inference scheduler reaches their activation chunk.  A shadow is
        # immutable while pending and is committed exactly once at activation.
        self.shadow_candidates = {}
        self.events = []

    def register(self, node):
        self.nodes[node.node_id] = node

    def observe(self, coverage):
        self.low_count = self.low_count + 1 if coverage < self.coverage_threshold else 0
        return self.low_count

    @staticmethod
    def parent_pixel_visibility_per_frame(warp):
        """Measure parent-world screen coverage only on renderer RGB pixels."""
        visibility = np.asarray(warp.visibility)
        if visibility.ndim != 3:
            raise ValueError(
                f"renderer visibility must be [T,H,W], got {visibility.shape}"
            )
        if visibility.dtype != np.bool_:
            visibility = visibility > 0
        return visibility.reshape(len(visibility), -1).mean(axis=1, dtype=np.float64)

    def _ready(self, current_world_overlap=None):
        return bool(self.readiness_report(current_world_overlap)["ready"])

    def readiness_report(self, current_world_overlap=None):
        overlap = float(
            self.current_world_overlap
            if current_world_overlap is None else current_world_overlap
        )
        if not np.isfinite(overlap) or not 0.0 <= overlap <= 1.0:
            raise ValueError(f"current world overlap must be finite in [0,1], got {overlap}")
        conditions = {
            "translation": self.buffer.translation_baseline >= self.min_translation_baseline,
            "view_change": self.buffer.view_diversity >= self.min_view_diversity,
            "new_area": self.buffer.mean_new_area_ratio >= self.min_new_area_ratio,
        }
        enough_frames = len(self.buffer) >= self.min_transition_frames
        world_overlap_below_max = overlap < self.max_world_overlap
        condition_ready = any(conditions.values())
        return {
            "ready": bool(enough_frames and world_overlap_below_max and condition_ready),
            "enough_frames": bool(enough_frames),
            "world_overlap_below_max": bool(world_overlap_below_max),
            "frame_count": int(len(self.buffer)),
            "mode": self.transition_readiness_mode,
            "conditions": conditions,
            "values": {
                "translation": float(self.buffer.translation_baseline),
                "view_change_radians": float(self.buffer.view_diversity),
                "view_change_degrees": float(np.rad2deg(self.buffer.view_diversity)),
                "new_area": float(self.buffer.mean_new_area_ratio),
                "current_chunk_world_overlap": overlap,
            },
            "thresholds": {
                "translation": float(self.min_translation_baseline),
                "view_change_radians": float(self.min_view_diversity),
                "view_change_degrees": float(np.rad2deg(self.min_view_diversity)),
                "new_area": float(self.min_new_area_ratio),
                "maximum_world_overlap_exclusive": float(self.max_world_overlap),
                "minimum_frames": int(self.min_transition_frames),
            },
        }

    def _append_chunk(self, generated_rgb_for_memory, cameras, warp, frame_start):
        from ..online.causal_contracts import assert_no_supervision_content
        assert_no_supervision_content({"generated_rgb_for_memory": generated_rgb_for_memory}, "MemoryManager")
        parent_pixel_visibility = self.parent_pixel_visibility_per_frame(warp)
        for index in range(len(generated_rgb_for_memory)):
            self.buffer.append(
                generated_rgb=np.asarray(generated_rgb_for_memory[index]),
                camera_c2w=np.asarray(cameras.c2w[index], np.float32),
                intrinsics=np.asarray(cameras.intrinsics[index], np.float32),
                old_node_warp=np.asarray(warp.rgb[index]),
                warp_visibility=np.asarray(warp.visibility[index]),
                old_node_warp_depth=np.asarray(warp.depth[index],np.float32),
                old_node_warp_source=np.asarray(warp.source[index],np.int8),
                old_node_warp_rgb_content_origin=np.asarray(
                    warp.rgb_content_origin[index] if warp.rgb_content_origin is not None
                    else np.where(warp.source[index] == 0, "oracle_source", "model_generated")
                ),
                old_node_warp_depth_content_origin=np.asarray(
                    warp.depth_content_origin[index] if warp.depth_content_origin is not None
                    else np.where(warp.source[index] == 0, "oracle_source", "pi3_prediction")
                ),
                old_node_warp_evidence_role=np.asarray(
                    warp.evidence_role[index] if warp.evidence_role is not None
                    else np.full(warp.source[index].shape, "parent_warp", dtype="U24")
                ),
                old_node_warp_rgb_evidence_role=np.asarray(
                    warp.rgb_evidence_role[index] if warp.rgb_evidence_role is not None
                    else (warp.evidence_role[index] if warp.evidence_role is not None
                          else np.full(warp.source[index].shape,"parent_warp",dtype="U24"))
                ),
                old_node_warp_depth_evidence_role=np.asarray(
                    warp.depth_evidence_role[index] if warp.depth_evidence_role is not None
                    else (warp.evidence_role[index] if warp.evidence_role is not None
                          else np.full(warp.source[index].shape,"parent_warp",dtype="U24"))
                ),
                old_node_depth_convention=Z_DEPTH,
                warp_confidence=np.asarray(warp.confidence[index]),
                coverage=float(parent_pixel_visibility[index]),
                global_frame_index=int(frame_start + index),
            )

    @staticmethod
    def _generated_image_confidence(rgb, known_mask, base_confidence):
        """Per-pixel generated-RGB confidence from texture and parent proximity."""
        from scipy.ndimage import distance_transform_edt
        value = rgb_to_float01(rgb)
        gray = value.mean(axis=-1)
        grad_y, grad_x = np.gradient(gray)
        gradient = np.sqrt(grad_x * grad_x + grad_y * grad_y)
        scale = float(np.percentile(gradient, 95))
        texture = np.clip(gradient / max(scale, 1e-6), 0.0, 1.0)
        distance = distance_transform_edt(~np.asarray(known_mask, bool))
        height, width = known_mask.shape
        proximity = np.exp(-distance / max(0.25 * np.hypot(height, width), 1.0))
        confidence = float(base_confidence) * (0.4 + 0.3 * texture + 0.3 * proximity)
        return np.clip(confidence, 0.0, 1.0).astype(np.float32)

    def _predict_geometry(self, rgb, c2w, intrinsics, *, known_depth, known_mask,
                          known_scale):
        try:
            return self.geometry_backend.predict(
                rgb, c2w, intrinsics,
                known_depth=known_depth,
                known_mask=known_mask,
                known_depth_convention=Z_DEPTH,
                known_scale=known_scale,
            )
        except ValueError as error:
            if not str(error).startswith("Scale anchor rejected:"):
                raise
            prediction = self.geometry_backend.predict(
                rgb, c2w, intrinsics,
                known_depth=None,
                known_mask=None,
                known_depth_convention=None,
                known_scale=known_scale,
            )
            prediction.diagnostics["depth_anchor_fallback"] = True
            prediction.diagnostics["depth_anchor_failure"] = str(error)
            prediction.scale_info["anchor_source"] = "unanchored_pi3_mandatory_promotion"
            return prediction

    def build_candidate(self, active_node, created_frame):
        if self.geometry_backend is None:
            raise RuntimeError("MemoryManager requires a geometry backend to construct M1")
        frames, heldout = self.buffer.select_keyframes(self.keyframe_count,self.heldout_count)
        if len(frames) != self.keyframe_count or len(heldout) != self.heldout_count:
            raise RuntimeError("Need 8 mapping and at least 4 held-out transition frames")
        created_frame = int(created_frame)
        if not self.buffer.frames:
            raise RuntimeError("candidate construction requires a non-empty transition buffer")
        boundary_frame = self.buffer.frames[-1]
        if int(boundary_frame.global_frame_index) != created_frame:
            raise RuntimeError(
                "transition boundary does not match candidate creation frame: "
                f"{boundary_frame.global_frame_index} != {created_frame}"
            )
        mapping_indices = [int(frame.global_frame_index) for frame in frames]
        heldout_indices = [int(frame.global_frame_index) for frame in heldout]
        if not mapping_indices or max(mapping_indices) != created_frame:
            raise RuntimeError("mapping keyframes must include the causal boundary frame")
        if any(index > created_frame for index in mapping_indices + heldout_indices):
            raise RuntimeError("candidate keyframes cannot use frames after the causal boundary")
        if created_frame in heldout_indices:
            raise RuntimeError("causal boundary frame cannot be held out")
        generated_rgb = rgb_to_float01(np.stack([frame.generated_rgb for frame in frames]))
        parent_rgb = rgb_to_float01(np.stack([frame.old_node_warp for frame in frames]))
        known_mask = np.stack([frame.warp_visibility for frame in frames])
        rgb = np.where(known_mask[..., None], parent_rgb, generated_rgb)
        c2w = np.stack([frame.camera_c2w for frame in frames]).astype(np.float32)
        intrinsics = np.stack([frame.intrinsics for frame in frames]).astype(np.float32)
        known_depth=np.stack([frame.old_node_warp_depth for frame in frames])
        prediction = self._predict_geometry(
            rgb, c2w, intrinsics,
            known_depth=known_depth, known_mask=known_mask,
            known_scale=active_node.scale,
        )
        source=np.stack([
            np.where(frame.warp_visibility,frame.old_node_warp_source,2)
            for frame in frames]).astype(np.int8)
        # Each factor is applied exactly once by node_builder:
        # source_prior x image_confidence x depth_confidence.
        generated_image_confidence=np.stack([
            self._generated_image_confidence(generated_rgb[index],frame.warp_visibility,
                                             self.generated_confidence)
            for index,frame in enumerate(frames)]).astype(np.float32)
        image_confidence=np.stack([
            np.where(frame.warp_visibility,1.0,generated_image_confidence[index])
            for index,frame in enumerate(frames)]).astype(np.float32)
        depth=np.asarray(prediction.depth,np.float32).copy()
        depth_confidence=np.asarray(prediction.depth_confidence,np.float32).copy()
        from ..geometry.confidence import DEFAULT_SOURCE_PRIOR
        for index,frame in enumerate(frames):
            valid=frame.warp_visibility & np.isfinite(frame.old_node_warp_depth)
            depth[index][valid]=frame.old_node_warp_depth[valid]
            prior=np.asarray([
                DEFAULT_SOURCE_PRIOR.get(int(item),0.0)
                for item in frame.old_node_warp_source.ravel()
            ],np.float32).reshape(frame.old_node_warp_source.shape)
            inherited=np.divide(
                frame.warp_confidence,prior,
                out=np.zeros_like(frame.warp_confidence,dtype=np.float32),
                where=prior>0,
            )
            # Multiplication by source_prior in node_builder reconstructs the
            # inherited parent confidence instead of applying that prior twice.
            depth_confidence[index][valid]=np.clip(inherited[valid],0.0,1.0)
        views = ViewSet(
            rgb=rgb,
            depth=depth,
            depth_confidence=depth_confidence,
            c2w=c2w,
            intrinsics=intrinsics,
            source=source,
            image_confidence=image_confidence,
            depth_convention=prediction.depth_convention or Z_DEPTH,
        )
        boundary_mapping_position = mapping_indices.index(created_frame)
        node_index = max([int(key.split("_")[-1]) for key in self.nodes] + [0]) + 1
        candidate = build_from_views(
            views,
            node_id=f"node_{node_index:03d}",
            center_c2w=c2w[boundary_mapping_position],
            created_frame=created_frame,
            voxel_size=self.voxel_size,
            status="candidate",
            parent_id=active_node.node_id,
        )
        candidate.view_rgb = rgb_to_uint8(views.rgb)
        candidate.view_depth = np.asarray(views.depth)
        candidate.view_c2w = np.asarray(views.c2w)
        candidate.intrinsics = np.asarray(views.intrinsics)
        candidate.view_source = np.asarray(views.source)
        candidate.view_image_confidence = np.asarray(views.image_confidence)
        candidate.view_depth_confidence = np.asarray(views.depth_confidence)
        candidate.quality_metrics.update({
            "shadow_boundary_frame": int(created_frame),
            "shadow_mapping_frame_indices": mapping_indices,
            "shadow_heldout_frame_indices": heldout_indices,
            "canonical_surface_commit": False,
            "multi_view_surface_commit": True,
            "geometry_input_view_count": int(len(frames)),
            "persistent_surface_view_count": int(len(frames)),
            "persistent_surface_mapping_positions": list(range(len(frames))),
            "persistent_surface_global_frames": mapping_indices,
        })
        view_rgb_origin=np.stack([
            np.where(frame.warp_visibility,frame.old_node_warp_rgb_content_origin,
                     "model_generated") for frame in frames])
        view_depth_origin=np.stack([
            np.where(frame.warp_visibility,frame.old_node_warp_depth_content_origin,
                     "pi3_prediction") for frame in frames])
        view_rgb_evidence=np.stack([
            np.where(frame.warp_visibility,"parent_warp","current_generation")
            for frame in frames])
        view_depth_evidence=np.stack([
            np.where(frame.warp_visibility,"parent_warp","geometry_prediction")
            for frame in frames])
        candidate.view_rgb_content_origin=view_rgb_origin
        candidate.view_depth_content_origin=view_depth_origin
        candidate.view_evidence_role=view_rgb_evidence
        candidate.view_rgb_evidence_role=view_rgb_evidence
        candidate.view_depth_evidence_role=view_depth_evidence
        generated_points=np.isin(candidate.points_source,(2,3))
        candidate.points_rgb_content_origin=np.where(
            generated_points,"model_generated","oracle_source").astype("U24")
        candidate.points_depth_content_origin=np.where(
            generated_points,"pi3_prediction","oracle_source").astype("U24")
        candidate.points_rgb_evidence_role=np.where(
            generated_points,"current_generation","parent_warp").astype("U24")
        candidate.points_depth_evidence_role=np.where(
            generated_points,"geometry_prediction","parent_warp").astype("U24")
        candidate.points_evidence_role=candidate.points_rgb_evidence_role.copy()
        from ..online.causal_contracts import validate_content_labels
        validate_content_labels(
            candidate.view_rgb_content_origin,candidate.view_depth_content_origin,
            candidate.view_rgb_evidence_role,candidate.view_depth_evidence_role)
        generated_region=candidate.view_rgb_evidence_role=="current_generation"
        generated_confidence_values=candidate.view_image_confidence[generated_region]
        candidate.quality_metrics["generated_image_confidence_min"]=float(
            generated_confidence_values.min() if generated_confidence_values.size else 0.0)
        candidate.quality_metrics["generated_image_confidence_max"]=float(
            generated_confidence_values.max() if generated_confidence_values.size else 0.0)
        candidate.quality_metrics["generated_image_confidence_std"]=float(
            generated_confidence_values.std() if generated_confidence_values.size else 0.0)
        candidate.quality_metrics["distinct_view_ratio"] = float((candidate.observation_count>=2).mean())
        candidate.quality_metrics["relative_pose"] = (
            np.linalg.inv(active_node.center_c2w) @ candidate.center_c2w
        ).tolist()
        candidate.quality_metrics["geometry_diagnostics"] = prediction.diagnostics
        candidate.quality_metrics["new_point_ratio"] = float((candidate.points_source==2).mean())
        info=prediction.scale_info
        candidate.scale=ScaleMetadata(
            mode=info.get("mode","relative"),
            meters_per_world_unit=info.get("meters_per_world_unit"),
            uncertainty=float(info.get("uncertainty",1.0)),
            anchor_source=info.get("anchor_source","parent_overlap"),
            diagnostics={k:v for k,v in info.items()
                         if k not in {"mode","meters_per_world_unit","uncertainty","anchor_source"}},
        )
        self.state = self.CANDIDATE
        return candidate, frames, heldout

    @classmethod
    def _parent_projection_exclusion_mask(cls, parent_visible):
        """Protect a 9-pixel neighborhood around non-boundary parent pixels."""
        parent = np.asarray(parent_visible, bool)
        if parent.ndim != 2:
            raise ValueError(f"parent visibility must be [H,W], got {parent.shape}")
        boundary = np.zeros_like(parent)
        for row in range(parent.shape[0]):
            columns = np.flatnonzero(parent[row])
            if columns.size:
                boundary[row, columns[0]] = True
                boundary[row, columns[-1]] = True
        for column in range(parent.shape[1]):
            rows = np.flatnonzero(parent[:, column])
            if rows.size:
                boundary[rows[0], column] = True
                boundary[rows[-1], column] = True
        interior = parent & ~boundary
        radius = cls.PARENT_PROJECTION_CLEARANCE_PIXELS
        protected = np.zeros_like(parent)
        padded_interior = np.pad(interior, radius, mode="constant")
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if dx * dx + dy * dy > radius * radius:
                    continue
                protected |= padded_interior[
                    radius + dy:radius + dy + parent.shape[0],
                    radius + dx:radius + dx + parent.shape[1],
                ]
        return protected & ~boundary

    @classmethod
    def _outside_parent_projection_mask(cls, candidate, frames):
        """Keep generated points at least 9 pixels from non-boundary parent pixels."""
        points = np.asarray(candidate.points_xyz, np.float32)
        overlaps_parent = np.zeros(len(points), bool)
        for frame in frames:
            c2w = np.asarray(frame.camera_c2w, np.float32)
            camera = (points - c2w[:3, 3]) @ c2w[:3, :3]
            z = camera[:, 2]
            projected = camera @ np.asarray(frame.intrinsics, np.float32).T
            u = np.rint(projected[:, 0] / np.maximum(z, 1e-8)).astype(np.int64)
            v = np.rint(projected[:, 1] / np.maximum(z, 1e-8)).astype(np.int64)
            parent_visible = np.asarray(frame.warp_visibility, bool)
            parent_exclusion = cls._parent_projection_exclusion_mask(parent_visible)
            height, width = parent_visible.shape
            inside = (z > 0) & (u >= 0) & (u < width) & (v >= 0) & (v < height)
            if inside.any():
                indices = np.flatnonzero(inside)
                overlaps_parent[indices] |= parent_exclusion[v[indices], u[indices]]
        return (np.asarray(candidate.points_source) == 2) & ~overlaps_parent

    def validate_candidate(self, candidate, frames, heldout):
        """Commit every finite, parent-nonoverlapping generated point.

        Candidate admission already happened in ``readiness_report``.  No RGB,
        depth, pose, confidence, support-view, occlusion, or camera-baseline
        metric is allowed to veto either the node or its otherwise valid novel
        points.
        """
        self.state = self.VALIDATING
        del heldout
        generated = np.asarray(candidate.points_source) == 2
        finite = np.isfinite(np.asarray(candidate.points_xyz)).all(axis=1)
        nonoverlap = self._outside_parent_projection_mask(candidate, frames)
        eligible = generated & finite & nonoverlap
        candidate.points_source[eligible] = 3
        candidate.points_confidence[eligible] = np.maximum(
            candidate.points_confidence[eligible], 0.9
        )
        candidate._verified_new_point_mask = eligible.copy()
        metrics = {
            "scale_dispersion": float(candidate.scale.uncertainty),
            "pose_error": float(candidate.quality_metrics["geometry_diagnostics"].get("pose_error",0.0)),
            "valid_depth_ratio": float(np.isfinite(candidate.view_depth).mean()),
            "new_point_ratio": float(candidate.quality_metrics.get("new_point_ratio",0.0)),
            "distinct_view_count": int(candidate.observation_count.max(initial=0)),
            "mandatory_acceptance_by_readiness": True,
            "legacy_rgb_pose_depth_confidence_gates_used": False,
            "legacy_point_verification_gates_used": False,
            "eligible_nonoverlap_point_count": int(eligible.sum()),
            "parent_projection_overlap_rejected_point_count": int(
                (generated & ~nonoverlap).sum()
            ),
            "nonfinite_generated_point_count": int((generated & ~finite).sum()),
        }
        candidate.quality_metrics.update({
            "verified_point_ratio": float(eligible.mean()),
            "generated_point_count_before_parent_overlap_filter": int(generated.sum()),
            "generated_point_count_outside_parent_projection": int(nonoverlap.sum()),
            "generated_point_count_rejected_parent_projection_overlap": int(
                (generated & ~nonoverlap).sum()
            ),
            "eligible_nonoverlap_point_count": int(eligible.sum()),
        })
        candidate.quality_metrics.update(metrics)
        return True, metrics

    @staticmethod
    def _point_field(node, name, count, *, dtype=None, fill=0):
        value = getattr(node, name, None)
        if value is not None:
            return np.asarray(value)
        if dtype is None:
            raise ValueError(f"missing required point field {name}")
        return np.full((count,), fill, dtype=dtype)

    def _merge_verified_points(self, active, candidate):
        """Preserve every committed parent point and append verified novel voxels only."""
        if getattr(candidate, "quality_metrics", None) is None:
            candidate.quality_metrics = {}
        parent_count = len(active.points_xyz)
        verified = getattr(candidate, "_verified_new_point_mask", None)
        if verified is None:
            verified = np.asarray(candidate.points_source) == 3
        else:
            verified = np.asarray(verified, bool).copy()
            if verified.shape != (len(candidate.points_xyz),):
                raise ValueError("verified new-point mask shape mismatch")
        verified &= np.isfinite(np.asarray(candidate.points_xyz)).all(axis=1)
        verified_indices = np.flatnonzero(verified)

        if len(verified_indices):
            candidate_xyz = np.asarray(candidate.points_xyz)[verified_indices]
            candidate_keys = np.floor(candidate_xyz / self.voxel_size).astype(np.int64)
            unique_keys, inverse = np.unique(candidate_keys, axis=0, return_inverse=True)
            confidence = np.asarray(candidate.points_confidence)[verified_indices]
            best = np.full(len(unique_keys), -1, np.int64)
            for local_index, group in enumerate(inverse):
                previous = best[group]
                if previous < 0 or confidence[local_index] > confidence[previous]:
                    best[group] = local_index

            parent_keys = np.floor(
                np.asarray(active.points_xyz) / self.voxel_size
            ).astype(np.int64)
            key_dtype = np.dtype((np.void, unique_keys.dtype.itemsize * unique_keys.shape[1]))
            unique_key_view = np.ascontiguousarray(unique_keys).view(key_dtype).ravel()
            parent_key_view = np.ascontiguousarray(parent_keys).view(key_dtype).ravel()
            novel_groups = ~np.isin(unique_key_view, parent_key_view)
            append_indices = verified_indices[best[novel_groups]]
        else:
            append_indices = np.empty((0,), np.int64)

        point_fields = (
            "points_xyz", "points_rgb", "points_confidence", "points_source",
            "observation_count",
        )
        for name in point_fields:
            parent = np.asarray(getattr(active, name))
            child = np.asarray(getattr(candidate, name))[append_indices]
            setattr(candidate, name, np.concatenate([parent, child], axis=0))
            if not np.array_equal(np.asarray(getattr(candidate, name))[:parent_count], parent):
                raise RuntimeError(f"cumulative promotion modified parent field {name}")

        optional_fields = {
            "points_normal": (np.float32, 0.0, (3,)),
            "point_view_mask": (np.uint64, 0, ()),
            "points_rgb_content_origin": ("U24", "unknown", ()),
            "points_depth_content_origin": ("U24", "unknown", ()),
            "points_evidence_role": ("U24", "unknown", ()),
            "points_rgb_evidence_role": ("U24", "unknown", ()),
            "points_depth_evidence_role": ("U24", "unknown", ()),
        }
        for name, (dtype, fill, tail_shape) in optional_fields.items():
            parent_value = getattr(active, name, None)
            child_value = getattr(candidate, name, None)
            if parent_value is None and child_value is None:
                continue
            parent = (np.asarray(parent_value) if parent_value is not None else
                      np.full((parent_count, *tail_shape), fill, dtype=dtype))
            child = (np.asarray(child_value)[append_indices] if child_value is not None else
                     np.full((len(append_indices), *tail_shape), fill, dtype=dtype))
            setattr(candidate, name, np.concatenate([parent, child], axis=0))

        points = np.asarray(candidate.points_xyz)
        candidate.bbox_min = points.min(axis=0).astype(np.float32)
        candidate.bbox_max = points.max(axis=0).astype(np.float32)
        candidate.coverage_radius = float(
            np.linalg.norm(candidate.bbox_max - candidate.bbox_min) * 0.5
        )
        candidate.quality_metrics.update({
            "parent_points_preserved": True,
            "parent_point_count": int(parent_count),
            "eligible_candidate_point_count": int(len(verified_indices)),
            "appended_eligible_point_count": int(len(append_indices)),
            "discarded_ineligible_candidate_point_count": int(len(verified) - len(verified_indices)),
            "discarded_duplicate_eligible_point_count": int(
                len(verified_indices) - len(append_indices)
            ),
            "cumulative_point_count": int(len(candidate.points_xyz)),
        })
        # Keep the split boundary directly on the node as well as in the
        # quality report.  The direct attribute is intentionally dynamic for
        # compatibility with schema-v3/v4 SpatialNode instances; renderers
        # fall back to quality_metrics when loading an older node.
        candidate.parent_point_count = int(parent_count)
        if hasattr(candidate, "_verified_new_point_mask"):
            del candidate._verified_new_point_mask
        return candidate

    @staticmethod
    def shadow_points_sha256(node):
        """Return a deterministic digest of the committed point payload.

        A shadow is immutable between candidate creation and scheduled
        activation.  Hashing the four persisted point arrays catches both
        accidental writes and hostile/tampered replacements while avoiding
        non-deterministic object metadata.
        """
        digest = hashlib.sha256()
        for name in (
            "points_xyz", "points_rgb", "points_confidence", "points_source",
        ):
            value = np.ascontiguousarray(np.asarray(getattr(node, name)))
            digest.update(name.encode("utf-8"))
            digest.update(value.dtype.str.encode("ascii"))
            digest.update(repr(tuple(value.shape)).encode("ascii"))
            digest.update(value.tobytes(order="C"))
        return digest.hexdigest()

    @classmethod
    def _freeze_shadow_points(cls, shadow):
        for name in (
            "points_xyz", "points_rgb", "points_confidence", "points_source",
        ):
            value = np.asarray(getattr(shadow, name))
            # ``setflags`` is best effort for exotic array wrappers; all normal
            # NumPy arrays used by SpatialNode are made read-only here.
            try:
                value.setflags(write=False)
            except ValueError as error:
                raise RuntimeError(f"cannot freeze shadow field {name}") from error
            setattr(shadow, name, value)
        return shadow

    def prepare_shadow(self, active, candidate):
        """Merge and freeze an accepted candidate without changing the parent.

        This is the deferred counterpart to :meth:`promote`.  It performs the
        cumulative parent+delta merge exactly once, stores the immutable shadow
        and leaves ``active`` ACTIVE until :meth:`commit_shadow` runs.
        """
        if getattr(active, "status", None) != "active":
            raise RuntimeError("shadow preparation requires an ACTIVE parent")
        if getattr(candidate, "status", None) not in {"candidate", "shadow"}:
            raise RuntimeError(
                f"cannot prepare candidate {getattr(candidate, 'node_id', None)} "
                f"from status {getattr(candidate, 'status', None)!r}"
            )
        if getattr(candidate, "status", None) == "shadow":
            # Never rebuild or mutate an already pending shadow.
            return candidate
        shadow = self._merge_verified_points(active, candidate)
        shadow.status = "shadow"
        shadow.parent_id = active.node_id
        if getattr(shadow, "quality_metrics", None) is None:
            shadow.quality_metrics = {}
        shadow_hash = self.shadow_points_sha256(shadow)
        shadow.quality_metrics.update({
            "shadow_status": "frozen",
            "shadow_hash_at_creation": shadow_hash,
        })
        shadow.shadow_hash_at_creation = shadow_hash
        shadow.shadow_hash_at_activation = None
        self._freeze_shadow_points(shadow)
        self.shadow_candidates[shadow.node_id] = shadow
        self.register(shadow)
        self.state = self.CANDIDATE
        if self.node_store is not None:
            self.node_store.save(shadow)
        return shadow

    def verify_shadow(self, shadow):
        """Verify the frozen point payload immediately before activation."""
        expected = getattr(shadow, "shadow_hash_at_creation", None)
        if expected is None:
            expected = shadow.quality_metrics.get("shadow_hash_at_creation")
        if not expected:
            raise RuntimeError(f"shadow {shadow.node_id} has no creation SHA256")
        actual = self.shadow_points_sha256(shadow)
        if actual != expected:
            raise RuntimeError(
                f"shadow hash mismatch for {shadow.node_id}: expected {expected}, got {actual}"
            )
        shadow.shadow_hash_at_activation = actual
        shadow.shadow_hash_equal = True
        shadow.quality_metrics["shadow_hash_at_activation"] = actual
        shadow.quality_metrics["shadow_hash_equal"] = True
        return actual

    def commit_shadow(self, active, shadow, *, verified_hash=None):
        """Verify and activate a scheduled shadow at its exact due chunk."""
        if getattr(shadow, "status", None) != "shadow":
            raise RuntimeError(
                f"shadow {getattr(shadow, 'node_id', None)} is not pending activation"
            )
        if getattr(shadow, "parent_id", None) != getattr(active, "node_id", None):
            raise RuntimeError(
                f"shadow parent mismatch: {getattr(shadow, 'parent_id', None)} != "
                f"{getattr(active, 'node_id', None)}"
            )
        if verified_hash is None:
            verified_hash = self.verify_shadow(shadow)
        elif verified_hash != getattr(shadow, "shadow_hash_at_activation", None):
            raise RuntimeError("commit_shadow received an unverified shadow hash")
        active.status = "archived"
        shadow.status = "active"
        self.register(active)
        self.register(shadow)
        if self.node_store is not None:
            self.node_store.save(active)
            self.node_store.save(shadow)
        self.shadow_candidates.pop(shadow.node_id, None)
        self.buffer.clear()
        self.low_count = 0
        self.state = self.ACTIVE_NEW_NODE
        return shadow

    def promote(self, active, candidate, offline_gt_metrics=None):
        if offline_gt_metrics is not None:
            raise ValueError("promotion cannot consume offline ground-truth metrics")
        if getattr(candidate, "status", None) == "shadow":
            return self.commit_shadow(active, candidate)
        candidate = self._merge_verified_points(active, candidate)
        active.status = "archived"
        candidate.status = "active"
        candidate.parent_id = active.node_id
        self.register(active)
        self.register(candidate)
        if self.node_store is not None:
            self.node_store.save(active)
            self.node_store.save(candidate)
        self.buffer.clear()
        self.low_count = 0
        self.state = self.ACTIVE_NEW_NODE
        return candidate

    def process_chunk(self, active_node, generated_rgb_for_memory, cameras, warp, frame_start,
                      *, allow_candidate_promotion=True,
                      defer_candidate_promotion=False, **forbidden):
        from ..online.causal_contracts import assert_no_supervision_content
        assert_no_supervision_content(forbidden, "MemoryManager")
        generated_frame_count = len(generated_rgb_for_memory)
        if generated_frame_count < 1:
            raise ValueError("MemoryManager cannot process an empty generated chunk")
        # ``frame_start`` is inclusive.  A candidate built after this chunk is
        # causally available at its final generated frame, not at the following
        # frame (the right-open range endpoint).
        candidate_created_frame = int(frame_start) + generated_frame_count - 1
        self.register(active_node)
        parent_pixel_visibility = self.parent_pixel_visibility_per_frame(warp)
        current_pixel_coverage = float(parent_pixel_visibility[-1])
        self.current_world_overlap = current_pixel_coverage
        self.observe(current_pixel_coverage)
        event = {"state": self.state, "coverage": current_pixel_coverage,
                 "parent_pixel_visibility_ratio": current_pixel_coverage,
                 "parent_pixel_visibility_ratio_per_frame":
                     parent_pixel_visibility.tolist()}
        # Permanent policy: every causal generated frame contributes to readiness.
        # Low coverage is diagnostic only and never gates candidate construction.
        self.state = self.TRANSITION
        self._append_chunk(generated_rgb_for_memory, cameras, warp, frame_start)
        readiness = self.readiness_report(current_pixel_coverage)
        event["readiness"] = readiness
        event["candidate_promotion_blocked"] = not bool(allow_candidate_promotion)
        event["candidate_promotion_deferred"] = bool(defer_candidate_promotion)
        if (allow_candidate_promotion and self.state==self.TRANSITION and readiness["ready"] and
                self.buffer.can_attempt(candidate_created_frame)):
            candidate, frames, heldout = self.build_candidate(active_node, candidate_created_frame)
            accepted, metrics = self.validate_candidate(candidate, frames, heldout)
            if not accepted:
                raise RuntimeError("readiness-qualified candidate must be accepted")
            event.update(candidate_id=candidate.node_id, accepted=accepted, metrics=metrics)
            if defer_candidate_promotion:
                shadow = self.prepare_shadow(active_node, candidate)
                event.update(
                    shadow_node=shadow,
                    shadow_frozen=True,
                    shadow_hash_at_creation=shadow.quality_metrics.get(
                        "shadow_hash_at_creation"
                    ),
                    shadow_parent_point_count=shadow.quality_metrics.get(
                        "parent_point_count"
                    ),
                )
            else:
                active_node = self.promote(active_node, candidate)
        event["state"] = self.state
        self.events.append(event)
        return active_node, event

    def maybe_reactivate(self,active_node,cameras,improvement=None):
        if active_node.quality_metrics.get("parent_points_preserved", False):
            return active_node, None
        hysteresis=(self.reactivation_hysteresis if improvement is None else float(improvement))
        active_warp=render(active_node,cameras,point_radius=0,device="cpu")
        active_score=float((active_warp.visibility*active_warp.confidence).mean())
        camera_center=np.asarray(cameras.c2w[0,:3,3])
        best=active_node; best_score=active_score; best_metrics=None
        for node in self.nodes.values():
            if node.status!="archived": continue
            adjacent=(node.parent_id==active_node.node_id or
                      active_node.parent_id==node.node_id)
            distance=float(np.linalg.norm(camera_center-node.center_c2w[:3,3]))
            radius=max(float(node.coverage_radius),float(active_node.coverage_radius),1e-6)
            if not adjacent and distance>2*radius: continue
            candidate=render(node,cameras,point_radius=0,device="cpu")
            score=float((candidate.visibility*candidate.confidence).mean())
            overlap=active_warp.visibility&candidate.visibility
            if overlap.any():
                rgb_error=float(np.abs(active_warp.rgb[overlap]-candidate.rgb[overlap]).mean())
                depth_mask=overlap&np.isfinite(active_warp.depth)&np.isfinite(candidate.depth)
                depth_error=(float(np.median(np.abs(active_warp.depth[depth_mask]-
                                                    candidate.depth[depth_mask])))
                             if depth_mask.any() else float("inf"))
            else:
                rgb_error=depth_error=float("inf")
            consistent=(rgb_error<=self.max_overlap_rgb_error and
                        depth_error<=self.max_overlap_depth_error)
            if consistent and score>best_score:
                best=node; best_score=score
                best_metrics={"graph_adjacent":adjacent,"spatial_distance":distance,
                              "rgb_error":rgb_error,"depth_error":depth_error}
        if best is active_node or best_score<active_score+hysteresis:
            return active_node,None
        active_node.status="archived"; best.status="active"
        if self.node_store is not None:
            self.node_store.save(active_node); self.node_store.save(best)
        event={"type":"reactivate","from":active_node.node_id,"to":best.node_id,
               "old_confidence_coverage":active_score,
               "new_confidence_coverage":best_score,**best_metrics}
        self.events.append(event)
        return best,event

    @classmethod
    def from_config(cls,config,**dependencies):
        return cls(**dependencies,**dict(config))
