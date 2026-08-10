"""Online spatial-memory state machine and candidate-node validation."""
from __future__ import annotations

import numpy as np

from ..geometry.point_renderer import render
from ..data.camera import rgb_to_float01
from ..types import CameraBatch, ScaleMetadata, ViewSet, Z_DEPTH
from .node_builder import build_from_views
from ..online.transition_buffer import TransitionBuffer


class MemoryManager:
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
        min_translation_baseline=0.15,
        min_view_diversity=0.15,
        min_new_area_ratio=0.2,
        min_overlap_coverage=0.03,
        min_confidence_weighted_coverage=0.5,
        max_overlap_rgb_error=0.25,
        max_overlap_depth_error=0.5,
        max_heldout_rgb_error=0.25,
        max_heldout_depth_error=0.5,
        min_new_point_ratio=0.1,
        generated_confidence=0.25,
        keyframe_count=8,
        voxel_size=0.02,
        heldout_count=4,
        max_scale_dispersion=0.25,
        max_pose_error=0.25,
        max_buffer_length=96,
        max_age_frames=240,
        failure_cooldown_frames=32,
        reactivation_hysteresis=0.1,
        min_verified_views=2,
        min_verified_baseline=0.03,
        max_verified_rgb_error=0.15,
        max_verified_depth_error=0.1,
        transition_readiness_mode="all",
    ):
        self.geometry_backend = geometry_backend
        self.node_store = node_store
        self.coverage_threshold = float(coverage_threshold)
        self.high_confidence_threshold = float(high_confidence_threshold)
        self.low_coverage_chunks = int(low_coverage_chunks)
        self.min_transition_frames = int(min_transition_frames)
        self.min_translation_baseline = float(min_translation_baseline)
        self.min_view_diversity = float(min_view_diversity)
        self.min_new_area_ratio = float(min_new_area_ratio)
        self.min_overlap_coverage = float(min_overlap_coverage)
        self.min_confidence_weighted_coverage=float(min_confidence_weighted_coverage)
        self.max_overlap_rgb_error=float(max_overlap_rgb_error)
        self.max_overlap_depth_error = float(max_overlap_depth_error)
        self.max_heldout_rgb_error=float(max_heldout_rgb_error)
        self.max_heldout_depth_error=float(max_heldout_depth_error)
        self.min_new_point_ratio = float(min_new_point_ratio)
        self.generated_confidence = float(generated_confidence)
        self.keyframe_count = int(keyframe_count)
        self.voxel_size = float(voxel_size)
        self.heldout_count = int(heldout_count)
        self.low_count = 0
        self.max_scale_dispersion=float(max_scale_dispersion)
        self.max_pose_error=float(max_pose_error)
        self.state = self.ACTIVE
        self.buffer=TransitionBuffer(
            max_length=max_buffer_length,max_age_frames=max_age_frames,
            cooldown_frames=failure_cooldown_frames)
        self.reactivation_hysteresis=float(reactivation_hysteresis)
        self.min_verified_views=int(min_verified_views)
        self.min_verified_baseline=float(min_verified_baseline)
        self.max_verified_rgb_error=float(max_verified_rgb_error)
        self.max_verified_depth_error=float(max_verified_depth_error)
        self.transition_readiness_mode = str(transition_readiness_mode)
        if self.transition_readiness_mode not in {"all", "any"}:
            raise ValueError("transition_readiness_mode must be 'all' or 'any'")
        self.nodes = {}
        self.events = []

    def register(self, node):
        self.nodes[node.node_id] = node

    def observe(self, coverage):
        self.low_count = self.low_count + 1 if coverage < self.coverage_threshold else 0
        return self.low_count

    def _ready(self):
        return bool(self.readiness_report()["ready"])

    def readiness_report(self):
        overlap = (
            float(np.mean([frame.coverage for frame in self.buffer.frames]))
            if self.buffer.frames else 0.0
        )
        conditions = {
            "translation": self.buffer.translation_baseline >= self.min_translation_baseline,
            "view_change": self.buffer.view_diversity >= self.min_view_diversity,
            "new_area": self.buffer.mean_new_area_ratio >= self.min_new_area_ratio,
            "world_overlap": overlap >= self.min_overlap_coverage,
        }
        enough_frames = len(self.buffer) >= self.min_transition_frames
        condition_ready = (
            any(conditions.values())
            if self.transition_readiness_mode == "any"
            else all(conditions.values())
        )
        return {
            "ready": bool(enough_frames and condition_ready),
            "enough_frames": bool(enough_frames),
            "frame_count": int(len(self.buffer)),
            "mode": self.transition_readiness_mode,
            "conditions": conditions,
            "values": {
                "translation": float(self.buffer.translation_baseline),
                "view_change_radians": float(self.buffer.view_diversity),
                "view_change_degrees": float(np.rad2deg(self.buffer.view_diversity)),
                "new_area": float(self.buffer.mean_new_area_ratio),
                "world_overlap": overlap,
            },
            "thresholds": {
                "translation": float(self.min_translation_baseline),
                "view_change_radians": float(self.min_view_diversity),
                "view_change_degrees": float(np.rad2deg(self.min_view_diversity)),
                "new_area": float(self.min_new_area_ratio),
                "world_overlap": float(self.min_overlap_coverage),
                "minimum_frames": int(self.min_transition_frames),
            },
        }

    def _append_chunk(self, generated_rgb_for_memory, cameras, warp, frame_start):
        from ..oracle_training.contracts import assert_no_supervision_content
        assert_no_supervision_content({"generated_rgb_for_memory": generated_rgb_for_memory}, "MemoryManager")
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
                coverage=float(warp.coverage_per_frame[index]),
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
    def build_candidate(self, active_node, created_frame):
        if self.geometry_backend is None:
            raise RuntimeError("MemoryManager requires a geometry backend to construct M1")
        frames, heldout = self.buffer.select_keyframes(self.keyframe_count,self.heldout_count)
        if len(frames) != self.keyframe_count or len(heldout) != self.heldout_count:
            raise RuntimeError("Need 8 mapping and at least 4 held-out transition frames")
        generated_rgb = rgb_to_float01(np.stack([frame.generated_rgb for frame in frames]))
        parent_rgb = rgb_to_float01(np.stack([frame.old_node_warp for frame in frames]))
        known_mask = np.stack([frame.warp_visibility for frame in frames])
        rgb = np.where(known_mask[..., None], parent_rgb, generated_rgb)
        c2w = np.stack([frame.camera_c2w for frame in frames]).astype(np.float32)
        intrinsics = np.stack([frame.intrinsics for frame in frames]).astype(np.float32)
        known_depth=np.stack([frame.old_node_warp_depth for frame in frames])
        prediction = self.geometry_backend.predict(
            rgb,c2w,intrinsics,known_depth=known_depth,known_mask=known_mask,
            known_depth_convention=Z_DEPTH,known_scale=active_node.scale,
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
        node_index = max([int(key.split("_")[-1]) for key in self.nodes] + [0]) + 1
        candidate = build_from_views(
            views,
            node_id=f"node_{node_index:03d}",
            center_c2w=c2w[len(c2w) // 2],
            created_frame=created_frame,
            voxel_size=self.voxel_size,
            status="candidate",
            parent_id=active_node.node_id,
        )
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
        from ..oracle_training.contracts import validate_content_labels
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

    def _verified_point_mask(self, candidate):
        """Require distinct translated views plus RGB/depth/occlusion agreement per point."""
        points=np.asarray(candidate.points_xyz,np.float32)
        colors=np.asarray(candidate.points_rgb,np.float32)/255.0
        supports=[]; camera_centers=[]
        for view_index in range(len(candidate.view_rgb)):
            c2w=np.asarray(candidate.view_c2w[view_index],np.float32)
            camera=(points-c2w[:3,3])@c2w[:3,:3]
            z=camera[:,2]
            projected=camera@np.asarray(candidate.view_intrinsics[view_index],np.float32).T
            u=np.rint(projected[:,0]/np.maximum(z,1e-8)).astype(np.int64)
            v=np.rint(projected[:,1]/np.maximum(z,1e-8)).astype(np.int64)
            height,width=candidate.view_depth[view_index].shape
            inside=(z>0)&(u>=0)&(u<width)&(v>=0)&(v<height)
            sampled_depth=np.full(len(points),np.nan,np.float32)
            sampled_rgb=np.zeros((len(points),3),np.float32)
            if inside.any():
                sampled_depth[inside]=candidate.view_depth[view_index,v[inside],u[inside]]
                value=np.asarray(candidate.view_rgb[view_index],np.float32)
                if value.size and value.max()>1:
                    value=value/255.0
                sampled_rgb[inside]=value[v[inside],u[inside]]
            tolerance=np.maximum(self.max_verified_depth_error,0.05*np.nan_to_num(sampled_depth,nan=0.0))
            depth_error=np.abs(z-sampled_depth)
            rgb_error=np.abs(colors-sampled_rgb).mean(axis=1)
            depth_consistent=np.isfinite(sampled_depth)&(depth_error<=tolerance)
            occlusion_consistent=np.isfinite(sampled_depth)&(z<=sampled_depth+tolerance)
            supports.append(inside&depth_consistent&occlusion_consistent&(rgb_error<=self.max_verified_rgb_error))
            camera_centers.append(c2w[:3,3])
        support=np.stack(supports); distinct=support.sum(axis=0)
        baseline=np.zeros(len(points),np.float32); centers=np.asarray(camera_centers,np.float32)
        for left in range(len(centers)):
            for right in range(left+1,len(centers)):
                pair=support[left]&support[right]
                baseline[pair]=np.maximum(baseline[pair],float(np.linalg.norm(centers[left]-centers[right])))
        candidate.quality_metrics["verified_support_mean"]=float(distinct.mean())
        candidate.quality_metrics["verified_baseline_mean"]=float(
            baseline[distinct>=self.min_verified_views].mean()
            if np.any(distinct>=self.min_verified_views) else 0.0)
        return ((candidate.points_source==2)&(distinct>=self.min_verified_views)&
                (baseline>=self.min_verified_baseline))
    def validate_candidate(self, candidate, frames, heldout):
        self.state = self.VALIDATING
        all_frames=list(frames)+list(heldout)
        height, width = all_frames[0].generated_rgb.shape[:2]
        cameras = CameraBatch(
            np.stack([frame.camera_c2w for frame in all_frames]),
            np.stack([frame.intrinsics for frame in all_frames]),
            height,
            width,
        )
        validation_frames=list(heldout[:4])+list(frames[:4])
        if len(validation_frames)!=8:
            raise RuntimeError("held-out Pi3 validation requires four held-out and four mapping references")
        validation_generated=rgb_to_float01(np.stack([
            frame.generated_rgb for frame in validation_frames]))
        validation_parent=rgb_to_float01(np.stack([
            frame.old_node_warp for frame in validation_frames]))
        validation_known_mask=np.stack([
            frame.warp_visibility for frame in validation_frames])
        validation_rgb=np.where(
            validation_known_mask[...,None],validation_parent,validation_generated)
        validation_prediction=self.geometry_backend.predict(
            validation_rgb,
            np.stack([frame.camera_c2w for frame in validation_frames]).astype(np.float32),
            np.stack([frame.intrinsics for frame in validation_frames]).astype(np.float32),
            known_depth=np.stack([
                frame.old_node_warp_depth for frame in validation_frames]).astype(np.float32),
            known_mask=validation_known_mask,
            known_depth_convention=Z_DEPTH,
            known_scale=candidate.scale,
        )
        rendered = render(candidate, cameras, point_radius=0, device="cpu")
        def rgb01(value):
            value=np.asarray(value,np.float32)
            return value/255.0 if value.size and value.max()>1 else value
        overlap_rgb=[]; overlap_depth=[]; held_rgb=[]; held_depth=[]; held_depth_pixels=[]
        for index,frame in enumerate(all_frames):
            overlap=(rendered.visibility[index] & frame.warp_visibility &
                     np.isfinite(rendered.depth[index]) &
                     np.isfinite(frame.old_node_warp_depth))
            if overlap.any():
                overlap_rgb.append(float(np.abs(rendered.rgb[index][overlap]-
                                                  rgb01(frame.old_node_warp)[overlap]).mean()))
                overlap_depth.append(float(np.median(np.abs(rendered.depth[index][overlap]-
                                                              frame.old_node_warp_depth[overlap]))))
            if index>=len(frames):
                valid=rendered.visibility[index]
                if valid.any():
                    held_rgb.append(float(np.abs(rendered.rgb[index][valid]-
                                               rgb01(frame.generated_rgb)[valid]).mean()))
                heldout_index=index-len(frames)
                predicted_depth=np.asarray(
                    validation_prediction.depth[heldout_index],np.float32)
                depth_valid=(rendered.visibility[index] & ~frame.warp_visibility &
                             np.isfinite(rendered.depth[index]) &
                             np.isfinite(predicted_depth) & (predicted_depth>0))
                held_depth_pixels.append(int(depth_valid.sum()))
                if depth_valid.any():
                    held_depth.append(float(np.median(np.abs(
                        rendered.depth[index][depth_valid]-predicted_depth[depth_valid]))))
        metrics = {
            "overlap_rgb_error": float(np.mean(overlap_rgb)) if overlap_rgb else float("inf"),
            "overlap_depth_error": float(np.mean(overlap_depth)) if overlap_depth else float("inf"),
            "heldout_rgb_error": float(np.mean(held_rgb)) if held_rgb else float("inf"),
            "heldout_depth_error": float(np.mean(held_depth)) if held_depth else float("inf"),
            "heldout_depth_valid_pixels": int(sum(held_depth_pixels)),
            "scale_dispersion": float(candidate.scale.uncertainty),
            "pose_error": float(candidate.quality_metrics["geometry_diagnostics"].get("pose_error",0.0)),
            "valid_depth_ratio": float(np.isfinite(candidate.view_depth).mean()),
            "new_point_ratio": float(candidate.quality_metrics.get("new_point_ratio",0.0)),
            "pixel_coverage": float(rendered.visibility.mean()),
            "confidence_weighted_coverage": float((rendered.visibility*rendered.confidence).mean()),
            "distinct_view_count": int(candidate.observation_count.max(initial=0)),
            "confidence_source": candidate.quality_metrics["geometry_diagnostics"].get(
                "confidence_source","unknown"),
            "confidence_type": candidate.quality_metrics["geometry_diagnostics"].get(
                "confidence_type","unknown"),
            "heldout_confidence_source": validation_prediction.diagnostics.get(
                "confidence_source","unknown"),
            "heldout_confidence_type": validation_prediction.diagnostics.get(
                "confidence_type","unknown"),
            "known_rgb_content_origins": sorted(np.unique(
                candidate.view_rgb_content_origin[
                    candidate.view_rgb_evidence_role=="parent_warp"]).tolist()),
            "known_depth_content_origins": sorted(np.unique(
                candidate.view_depth_content_origin[
                    candidate.view_depth_evidence_role=="parent_warp"]).tolist()),
            "known_evidence_role": "parent_warp",
            "new_rgb_content_origin": "model_generated",
            "new_rgb_evidence_role": "current_generation",
            "new_depth_content_origin": "pi3_prediction",
            "new_depth_evidence_role": "geometry_prediction",
            "generated_image_confidence_min": candidate.quality_metrics[
                "generated_image_confidence_min"],
            "generated_image_confidence_max": candidate.quality_metrics[
                "generated_image_confidence_max"],
            "generated_image_confidence_std": candidate.quality_metrics[
                "generated_image_confidence_std"],
        }
        candidate.quality_metrics.update(metrics)
        accepted = (
            metrics["confidence_weighted_coverage"]>=self.min_confidence_weighted_coverage
            and metrics["heldout_rgb_error"]<=self.max_heldout_rgb_error
            and metrics["overlap_rgb_error"]<=self.max_overlap_rgb_error
            and metrics["overlap_depth_error"] <= self.max_overlap_depth_error
            and metrics["heldout_depth_error"]<=self.max_heldout_depth_error
            and metrics["new_point_ratio"] >= self.min_new_point_ratio
            and metrics["scale_dispersion"]<=self.max_scale_dispersion
            and metrics["pose_error"]<=self.max_pose_error
        )
        if accepted:
            verified=self._verified_point_mask(candidate)
            candidate.points_source[verified]=3
            candidate.points_confidence[verified]=np.maximum(candidate.points_confidence[verified],0.9)
            candidate.quality_metrics["verified_point_ratio"]=float(verified.mean())
        return accepted, metrics

    def promote(self, active, candidate, offline_gt_metrics=None):
        if offline_gt_metrics is not None:
            raise ValueError("promotion cannot consume offline ground-truth metrics")
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

    def process_chunk(self, active_node, generated_rgb_for_memory, cameras, warp, frame_start, **forbidden):
        from ..oracle_training.contracts import assert_no_supervision_content
        assert_no_supervision_content(forbidden, "MemoryManager")
        generated_frame_count = len(generated_rgb_for_memory)
        if generated_frame_count < 1:
            raise ValueError("MemoryManager cannot process an empty generated chunk")
        # ``frame_start`` is inclusive.  A candidate built after this chunk is
        # causally available at its final generated frame, not at the following
        # frame (the right-open range endpoint).
        candidate_created_frame = int(frame_start) + generated_frame_count - 1
        self.register(active_node)
        mean_coverage = float(np.mean(warp.coverage_per_frame))
        self.observe(mean_coverage)
        event = {"state": self.state, "coverage": mean_coverage}
        if self.low_count >= self.low_coverage_chunks:
            self.state = self.TRANSITION
            self._append_chunk(generated_rgb_for_memory, cameras, warp, frame_start)
        if (self.state==self.TRANSITION and self._ready() and
                self.buffer.can_attempt(candidate_created_frame)):
            candidate, frames, heldout = self.build_candidate(active_node, candidate_created_frame)
            accepted, metrics = self.validate_candidate(candidate, frames, heldout)
            event.update(candidate_id=candidate.node_id, accepted=accepted, metrics=metrics)
            if accepted:
                active_node = self.promote(active_node, candidate)
            else:
                rejection_reasons=[
                    key for key, failed in {
                        "confidence_weighted_coverage":metrics["confidence_weighted_coverage"]<self.min_confidence_weighted_coverage,
                        "heldout_rgb_error":metrics["heldout_rgb_error"]>self.max_heldout_rgb_error,
                        "overlap_rgb_error":metrics["overlap_rgb_error"]>self.max_overlap_rgb_error,
                        "overlap_depth_error":metrics["overlap_depth_error"]>self.max_overlap_depth_error,
                        "heldout_depth_error":metrics["heldout_depth_error"]>self.max_heldout_depth_error,
                        "new_point_ratio":metrics["new_point_ratio"]<self.min_new_point_ratio,
                        "scale_dispersion":metrics["scale_dispersion"]>self.max_scale_dispersion,
                        "pose_error":metrics["pose_error"]>self.max_pose_error,
                    }.items() if failed]
                self.buffer.reject(candidate_created_frame, rejection_reasons)
                event["rejection_reason"] = rejection_reasons
                self.state = self.TRANSITION
        event["state"] = self.state
        self.events.append(event)
        return active_node, event

    def maybe_reactivate(self,active_node,cameras,improvement=None):
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
