"""Online spatial-memory state machine and candidate-node validation."""
from __future__ import annotations

import numpy as np

from ..geometry.point_renderer import render
from ..types import CameraBatch, ViewSet, Z_DEPTH
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
        min_transition_frames=8,
        min_translation_baseline=0.15,
        min_view_diversity=0.15,
        min_new_area_ratio=0.2,
        min_overlap_coverage=0.03,
        candidate_coverage_threshold=0.5,
        max_reprojection_error=0.25,
        max_overlap_depth_error=0.5,
        min_new_point_ratio=0.1,
        generated_confidence=0.25,
        inherited_confidence=0.6,
        keyframe_count=8,
        voxel_size=0.02,
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
        self.candidate_coverage_threshold = float(candidate_coverage_threshold)
        self.max_reprojection_error = float(max_reprojection_error)
        self.max_overlap_depth_error = float(max_overlap_depth_error)
        self.min_new_point_ratio = float(min_new_point_ratio)
        self.generated_confidence = float(generated_confidence)
        self.inherited_confidence = float(inherited_confidence)
        self.keyframe_count = int(keyframe_count)
        self.voxel_size = float(voxel_size)
        self.low_count = 0
        self.state = self.ACTIVE
        self.buffer = TransitionBuffer()
        self.nodes = {}
        self.events = []

    def register(self, node):
        self.nodes[node.node_id] = node

    def observe(self, coverage):
        self.low_count = self.low_count + 1 if coverage < self.coverage_threshold else 0
        return self.low_count

    def _ready(self):
        overlap = float(np.mean([frame.coverage for frame in self.buffer.frames]))
        return (
            len(self.buffer) >= self.min_transition_frames
            and self.buffer.translation_baseline >= self.min_translation_baseline
            and self.buffer.view_diversity >= self.min_view_diversity
            and self.buffer.mean_new_area_ratio >= self.min_new_area_ratio
            and overlap >= self.min_overlap_coverage
        )

    def _append_chunk(self, generated, cameras, warp, frame_start):
        for index in range(len(generated)):
            self.buffer.append(
                generated_rgb=np.asarray(generated[index]),
                camera_c2w=np.asarray(cameras.c2w[index], np.float32),
                intrinsics=np.asarray(cameras.intrinsics[index], np.float32),
                old_node_warp=np.asarray(warp.rgb[index]),
                warp_visibility=np.asarray(warp.visibility[index]),
                warp_confidence=np.asarray(warp.confidence[index]),
                coverage=float(warp.coverage_per_frame[index]),
                global_frame_index=int(frame_start + index),
            )

    def _inherit_parent_points(self, parent, candidate):
        keep = np.asarray(parent.points_confidence) >= self.inherited_confidence
        inherited = int(keep.sum())
        new_count = len(candidate.points_xyz)
        if inherited:
            for name in ("points_xyz", "points_rgb", "points_confidence", "points_source", "observation_count"):
                setattr(candidate, name, np.concatenate([getattr(candidate, name), getattr(parent, name)[keep]]))
            candidate.bbox_min = candidate.points_xyz.min(0).astype(np.float32)
            candidate.bbox_max = candidate.points_xyz.max(0).astype(np.float32)
        candidate.quality_metrics["new_point_ratio"] = float(new_count / max(1, new_count + inherited))

    def build_candidate(self, active_node, created_frame):
        if self.geometry_backend is None:
            raise RuntimeError("MemoryManager requires a geometry backend to construct M1")
        frames = self.buffer.select_keyframes(self.keyframe_count)
        if len(frames) != self.keyframe_count:
            raise RuntimeError(f"Need {self.keyframe_count} transition keyframes, got {len(frames)}")
        rgb = np.stack([frame.generated_rgb for frame in frames])
        c2w = np.stack([frame.camera_c2w for frame in frames]).astype(np.float32)
        intrinsics = np.stack([frame.intrinsics for frame in frames]).astype(np.float32)
        prediction = self.geometry_backend.predict(rgb, c2w, intrinsics)
        source = np.full(rgb.shape[:3], 2, np.int8)
        image_confidence = np.full(rgb.shape[:3], self.generated_confidence, np.float32)
        views = ViewSet(
            rgb=rgb,
            depth=prediction.depth,
            depth_confidence=prediction.depth_confidence,
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
        verified = candidate.observation_count >= 2
        candidate.points_source[verified] = 3
        candidate.points_confidence[verified] = np.maximum(
            candidate.points_confidence[verified], 0.9
        )
        candidate.quality_metrics["verified_point_ratio"] = float(verified.mean())
        candidate.quality_metrics["relative_pose"] = (
            np.linalg.inv(active_node.center_c2w) @ candidate.center_c2w
        ).tolist()
        candidate.quality_metrics["geometry_diagnostics"] = prediction.diagnostics
        self._inherit_parent_points(active_node, candidate)
        self.state = self.CANDIDATE
        return candidate, frames

    def validate_candidate(self, candidate, frames):
        self.state = self.VALIDATING
        height, width = frames[0].generated_rgb.shape[:2]
        cameras = CameraBatch(
            np.stack([frame.camera_c2w for frame in frames]),
            np.stack([frame.intrinsics for frame in frames]),
            height,
            width,
        )
        rendered = render(candidate, cameras, point_radius=1)
        visible = rendered.visibility
        targets = np.stack([frame.generated_rgb for frame in frames]).astype(np.float32)
        if targets.max() > 1.0:
            targets /= 255.0
        reprojection = float(np.abs(rendered.rgb[visible] - targets[visible]).mean()) if visible.any() else float("inf")
        old_depth = np.stack([
            np.where(frame.warp_visibility, np.nan, np.nan) for frame in frames
        ])
        overlap_errors = []
        for index, frame in enumerate(frames):
            overlap = rendered.visibility[index] & frame.warp_visibility & np.isfinite(rendered.depth[index])
            if overlap.any() and hasattr(frame, "old_node_warp"):
                rgb_old = np.asarray(frame.old_node_warp, np.float32)
                overlap_errors.append(float(np.abs(rendered.rgb[index][overlap] - rgb_old[overlap]).mean()))
        overlap_error = float(np.mean(overlap_errors)) if overlap_errors else 0.0
        metrics = {
            "candidate_coverage": float(rendered.coverage_per_frame.mean()),
            "multi_view_reprojection_error": reprojection,
            "overlap_error": overlap_error,
            "new_point_ratio": float(candidate.quality_metrics.get("new_point_ratio", 1.0)),
        }
        candidate.quality_metrics.update(metrics)
        accepted = (
            metrics["candidate_coverage"] >= self.candidate_coverage_threshold
            and metrics["multi_view_reprojection_error"] <= self.max_reprojection_error
            and metrics["overlap_error"] <= self.max_overlap_depth_error
            and metrics["new_point_ratio"] >= self.min_new_point_ratio
        )
        return accepted, metrics

    def promote(self, active, candidate):
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

    def process_chunk(self, active_node, generated, cameras, warp, frame_start):
        self.register(active_node)
        mean_coverage = float(np.mean(warp.coverage_per_frame))
        self.observe(mean_coverage)
        event = {"state": self.state, "coverage": mean_coverage}
        if self.low_count >= self.low_coverage_chunks:
            self.state = self.TRANSITION
            self._append_chunk(generated, cameras, warp, frame_start)
        if self.state == self.TRANSITION and self._ready():
            candidate, frames = self.build_candidate(active_node, frame_start + len(generated))
            accepted, metrics = self.validate_candidate(candidate, frames)
            event.update(candidate_id=candidate.node_id, accepted=accepted, metrics=metrics)
            if accepted:
                active_node = self.promote(active_node, candidate)
            else:
                self.state = self.TRANSITION
        event["state"] = self.state
        self.events.append(event)
        return active_node, event

    def maybe_reactivate(self, active_node, cameras, improvement=0.1):
        candidates = [node for node in self.nodes.values() if node.status == "archived"]
        if not candidates:
            return active_node, None
        active_coverage = float(render(active_node, cameras, point_radius=0).coverage_per_frame.mean())
        best, best_coverage = active_node, active_coverage
        for node in candidates:
            coverage = float(render(node, cameras, point_radius=0).coverage_per_frame.mean())
            if coverage > best_coverage:
                best, best_coverage = node, coverage
        if best is not active_node and best_coverage >= active_coverage + improvement:
            active_node.status = "archived"
            best.status = "active"
            event = {"type": "reactivate", "from": active_node.node_id, "to": best.node_id,
                     "old_coverage": active_coverage, "new_coverage": best_coverage}
            self.events.append(event)
            return best, event
        return active_node, None
