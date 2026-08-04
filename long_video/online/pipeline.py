"""End-to-end online spatial-history generation."""
from __future__ import annotations

import numpy as np

from ..data.controls import integrate_controls
from ..geometry.point_renderer import render
from ..initialization.initial_node_pipeline import initialize_spatial_node
from ..types import CameraBatch
from ..wah.adapter import WAHAdapter


class OnlineSpatialHistoryPipeline:
    def __init__(
        self,
        wah_pipeline=None,
        active_node=None,
        memory_manager=None,
        prompt="",
        renderer_kwargs=None,
        wah_state_kwargs=None,
    ):
        self.wah_pipeline = wah_pipeline
        self.wah_adapter = WAHAdapter(wah_pipeline) if wah_pipeline is not None else None
        self.active_node = active_node
        self.memory_manager = memory_manager
        self.prompt = str(prompt)
        self.renderer_kwargs = dict(renderer_kwargs or {})
        self.wah_state_kwargs = dict(wah_state_kwargs or {})
        self.current_camera_c2w = (
            active_node.center_c2w.copy() if active_node is not None else np.eye(4, dtype=np.float32)
        )
        self.recent_video_history = []
        self.autoregressive_state = None
        self.frame_index = 0

    def initialize(
        self,
        observed_images,
        camera_specs,
        prompt,
        completion_backend,
        geometry_backend,
        config,
        first_image=None,
    ):
        self.prompt = str(prompt)
        self.active_node = initialize_spatial_node(
            observed_images,
            camera_specs,
            prompt,
            completion_backend,
            geometry_backend,
            config,
        )
        self.current_camera_c2w = self.active_node.center_c2w.copy()
        if self.memory_manager is not None:
            self.memory_manager.geometry_backend = geometry_backend
            self.memory_manager.register(self.active_node)
        if self.wah_pipeline is not None:
            if first_image is None:
                first_image = np.asarray(self.active_node.view_rgb[0])
            kwargs = {
                "conditioning_type": "warp",
                "warp_history_downsample_mode": "short",
                "rope_alignment": True,
                **self.wah_state_kwargs,
            }
            self.autoregressive_state = self.wah_pipeline.init_autoregressive_state(
                prompt=self.prompt,
                image=first_image,
                **kwargs,
            )
            self.wah_adapter.configure_state(self.autoregressive_state)
        return self.active_node

    @staticmethod
    def _video_array(video):
        value = np.asarray(video)
        if value.ndim == 5 and value.shape[0] == 1:
            value = value[0]
        if value.ndim != 4:
            raise ValueError(f"WAH generated video must be [T,H,W,C] or [1,T,H,W,C], got {value.shape}")
        if value.shape[-1] != 3 and value.shape[1] == 3:
            value = np.moveaxis(value, 1, -1)
        return value

    def generate_chunk(self, controls, intrinsics, height, width):
        if self.active_node is None:
            raise RuntimeError("initialize() must establish M0 before generation")
        if self.wah_adapter is None or self.autoregressive_state is None:
            raise RuntimeError("A loaded patched WAH pipeline and autoregressive state are required")
        poses = integrate_controls(self.current_camera_c2w, controls)
        if not len(poses):
            raise ValueError("controls must contain at least one output frame")
        intrinsics = np.asarray(intrinsics, np.float32)
        if intrinsics.ndim == 2:
            intrinsics = np.repeat(intrinsics[None], len(poses), axis=0)
        if len(intrinsics) != len(poses):
            raise ValueError("intrinsics count must match generated camera poses")
        cameras = CameraBatch(poses, intrinsics, int(height), int(width))
        warp = render(self.active_node, cameras, **self.renderer_kwargs)
        generated_video, self.autoregressive_state = self.wah_adapter.generate_next_chunk(
            self.autoregressive_state, warp, output_type="np"
        )
        generated = self._video_array(generated_video)
        if len(generated) != len(poses):
            usable = min(len(generated), len(poses))
            generated, poses = generated[:usable], poses[:usable]
            cameras = CameraBatch(poses, intrinsics[:usable], int(height), int(width))
            for name in ("rgb", "depth", "visibility", "confidence", "source"):
                setattr(warp, name, getattr(warp, name)[:usable])
            warp.coverage_per_frame = warp.coverage_per_frame[:usable]
        frame_start = self.frame_index
        self.frame_index += len(poses)
        self.current_camera_c2w = poses[-1].copy()
        self.recent_video_history.append(generated)
        self.recent_video_history = self.recent_video_history[-2:]
        memory_event = None
        if self.memory_manager is not None:
            self.active_node, memory_event = self.memory_manager.process_chunk(
                self.active_node, generated, cameras, warp, frame_start
            )
        high_conf = warp.visibility & (warp.confidence >= (
            self.memory_manager.high_confidence_threshold if self.memory_manager else 0.5
        ))
        statistics = {
            "frame_start": frame_start,
            "frame_end": self.frame_index,
            "coverage": warp.coverage_per_frame.tolist(),
            "mean_coverage": float(warp.coverage_per_frame.mean()),
            "high_conf_coverage": high_conf.reshape(len(poses), -1).mean(1).tolist(),
            "new_area_ratio": (1.0 - warp.coverage_per_frame).tolist(),
            "active_node_id": self.active_node.node_id,
            "memory_event": memory_event,
        }
        return generated, poses, warp, statistics
