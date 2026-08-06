"""Small deterministic M0 -> candidate M1 -> active M1 state-machine test."""
import numpy as np

from long_video.initialization.geometry_backend import GeometryPrediction
from long_video.memory.memory_manager import MemoryManager
from long_video.memory.node_builder import build_from_views
from long_video.types import CameraBatch, ViewSet, WarpBatch, Z_DEPTH


class PlaneGeometry:
    def predict(self, view_rgb, view_c2w, intrinsics, **_kwargs):
        depth = np.full(view_rgb.shape[:3], 2.0, np.float32)
        confidence = np.ones_like(depth)
        return GeometryPrediction(
            depth=depth,
            depth_confidence=confidence,
            diagnostics={"backend": "deterministic_plane"},
            depth_convention=Z_DEPTH,
            scale_info={"mode":"metric_anchor","meters_per_world_unit":1.0,
                        "uncertainty":0.0,"anchor_source":"parent_overlap"},
        )


def main():
    frames, height, width = 12, 24, 24
    rgb = np.full((frames, height, width, 3), 128, np.uint8)
    intrinsics = np.repeat(
        np.array([[20.0, 0, width / 2], [0, 20.0, height / 2], [0, 0, 1]], np.float32)[None],
        frames,
        axis=0,
    )
    poses = np.repeat(np.eye(4, dtype=np.float32)[None], frames, axis=0)
    for index in range(frames):
        angle = 0.03 * index
        poses[index, 0, 3] = 0.03 * index
        poses[index, :3, :3] = np.array(
            [[np.cos(angle), 0, np.sin(angle)], [0, 1, 0],
             [-np.sin(angle), 0, np.cos(angle)]],
            np.float32,
        )
    depth = np.full((frames, height, width), 2.0, np.float32)
    views = ViewSet(
        rgb, depth, np.ones_like(depth), poses, intrinsics,
        np.zeros_like(depth, np.int8), np.ones_like(depth), Z_DEPTH,
    )
    m0 = build_from_views(views, voxel_size=0.04)
    visibility = np.zeros((frames, height, width), bool)
    visibility[:, :, : width // 4] = True
    warp = WarpBatch(
        rgb.astype(np.float32) / 255,
        np.where(visibility, 2.0, np.nan).astype(np.float32),
        visibility,
        visibility.astype(np.float32),
        np.where(visibility, 0, 4).astype(np.int8),
        visibility.reshape(frames, -1).mean(1).astype(np.float32),
    )
    manager = MemoryManager(
        geometry_backend=PlaneGeometry(),
        coverage_threshold=0.5,
        low_coverage_chunks=1,
        min_transition_frames=12,
        min_translation_baseline=0.1,
        min_view_diversity=0.1,
        min_new_area_ratio=0.5,
        min_overlap_coverage=0.1,
        min_confidence_weighted_coverage=0.01,
        max_overlap_rgb_error=1.0,max_heldout_rgb_error=1.0,
        max_overlap_depth_error=1.0,
        max_heldout_depth_error=1.0,
        min_new_point_ratio=0.01,
        keyframe_count=8,
        heldout_count=4,
    )
    active, event = manager.process_chunk(
        m0, rgb, CameraBatch(poses, intrinsics, height, width), warp, 0
    )
    assert event["accepted"] is True, event
    assert m0.status == "archived"
    assert active.node_id == "node_001" and active.status == "active"
    assert active.parent_id == "node_000"
    assert active.quality_metrics["verified_point_ratio"] > 0
    print("memory transition passed", event["metrics"], {
        "verified_point_ratio": active.quality_metrics["verified_point_ratio"],
        "verified_support_mean": active.quality_metrics["verified_support_mean"],
        "verified_baseline_mean": active.quality_metrics["verified_baseline_mean"],
    })


if __name__ == "__main__":
    main()
