"""End-to-end online spatial-history generation."""
from __future__ import annotations

from dataclasses import fields, replace
import numpy as np

from ..data.controls import integrate_controls
from ..geometry.point_renderer import render
from ..initialization.initial_node_pipeline import initialize_spatial_node
from ..types import CameraBatch
from ..wah.adapter import WAHAdapter
from ..wah.stage0_causal_world_film import (
    aggregate_winning_points,
    fixed_source_scale,
    install_stage0_causal_world_film,
)
from .delayed_activation import DelayedNodeActivationQueue


class OnlineSpatialHistoryPipeline:
    def __init__(
        self,
        wah_pipeline=None,
        active_node=None,
        memory_manager=None,
        prompt="",
        renderer_kwargs=None,
        control_kwargs=None,
        wah_state_kwargs=None,
    ):
        self.wah_pipeline = wah_pipeline
        self.wah_adapter = WAHAdapter(wah_pipeline) if wah_pipeline is not None else None
        self.stage0_film = (
            install_stage0_causal_world_film(wah_pipeline.transformer)
            if wah_pipeline is not None else None
        )
        self.active_node = active_node
        self.memory_manager = memory_manager
        if self.memory_manager is not None and active_node is not None:
            self.memory_manager.register(active_node)
        self.prompt = str(prompt)
        self.renderer_kwargs = {"device":"cpu",**dict(renderer_kwargs or {})}
        self.control_kwargs = dict(control_kwargs or {})
        self.wah_state_kwargs = dict(wah_state_kwargs or {})
        self.current_camera_c2w = (
            active_node.center_c2w.copy() if active_node is not None else np.eye(4, dtype=np.float32)
        )
        self.recent_video_history = []
        self.autoregressive_state = None
        self.frame_index = 0
        self.chunk_index = 0
        self.activation_queue = DelayedNodeActivationQueue(delay_chunks=2)
        self.source_c2w_fixed = (
            active_node.center_c2w.copy() if active_node is not None else None
        )
        self.source_depth_scale = (
            fixed_source_scale(np.asarray(active_node.view_depth[0]))
            if active_node is not None else None
        )

    def initialize(
        self,
        views,
        prompt,
        geometry_backend,
        config,
        first_image=None,
    ):
        self.prompt = str(prompt)
        self.active_node = initialize_spatial_node(
            views, geometry_backend, config,
        )
        self.current_camera_c2w = self.active_node.center_c2w.copy()
        self.source_c2w_fixed = self.active_node.center_c2w.copy()
        self.source_depth_scale = fixed_source_scale(np.asarray(self.active_node.view_depth[0]))
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
            self.stage0_film.bind_inference_scheduler(self.wah_pipeline.scheduler)
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

    @staticmethod
    def _slice_warp_frames(warp, start):
        updates = {}
        frame_count = len(warp.rgb)
        for item in fields(warp):
            value = getattr(warp, item.name)
            if isinstance(value, np.ndarray) and value.ndim and len(value) == frame_count:
                updates[item.name] = value[int(start):]
        return replace(warp, **updates)

    def _generate_cameras(self, cameras):
        if self.active_node is None:
            raise RuntimeError("initialize() must establish M0 before generation")
        if self.wah_adapter is None or self.autoregressive_state is None:
            raise RuntimeError("A loaded patched WAH pipeline and autoregressive state are required")
        poses = np.asarray(cameras.c2w, np.float32)
        activation = self.activation_queue.activate_due(self.chunk_index)
        if activation is not None:
            verified_hash = self.memory_manager.verify_shadow(activation.node)
            self.active_node = self.memory_manager.commit_shadow(
                self.active_node, activation.node, verified_hash=verified_hash,
            )
        reactivation_event=None
        if self.memory_manager is not None:
            self.active_node,reactivation_event=self.memory_manager.maybe_reactivate(
                self.active_node,cameras)
        warp = render(self.active_node,cameras,**self.renderer_kwargs)
        import torch
        with torch.no_grad():
            point_feature, visibility0 = aggregate_winning_points(
                warp, self.source_c2w_fixed, self.source_depth_scale,
                self.stage0_film.point_encoder,
                device=self.wah_pipeline._wah_execution_device(),
                dtype=getattr(self.wah_pipeline.transformer, "dtype", None),
            )
        self.stage0_film.set_point_context(point_feature, visibility0)
        self.stage0_film.bind_inference_scheduler(self.wah_pipeline.scheduler)
        generated_video, self.autoregressive_state = self.wah_adapter.generate_next_chunk(
            self.autoregressive_state, warp, output_type="np"
        )
        generated = self._video_array(generated_video)
        frame_offset = len(poses) - len(generated)
        if frame_offset not in (0, 1) or (frame_offset == 1 and self.chunk_index == 0):
            raise ValueError(
                f"WAH generated {len(generated)} frames but trajectory/warp has {len(poses)}; "
                "expected 33 frames for chunk0 or the 32 new frames after a shared boundary"
            )
        memory_cameras = cameras
        memory_warp = warp
        if frame_offset:
            memory_cameras = CameraBatch(
                cameras.c2w[frame_offset:], cameras.intrinsics[frame_offset:],
                cameras.height, cameras.width,
            )
            memory_warp = self._slice_warp_frames(warp, frame_offset)
        frame_start = self.frame_index + frame_offset
        # A 33-frame chunk advances by stride 32; its final frame is the next
        # chunk's shared boundary and must not be counted twice.
        self.frame_index += len(poses) - 1
        self.current_camera_c2w = poses[-1].copy()
        self.recent_video_history.append(generated)
        self.recent_video_history = self.recent_video_history[-2:]
        memory_event = None
        if self.memory_manager is not None:
            self.active_node, memory_event = self.memory_manager.process_chunk(
                self.active_node, generated_rgb_for_memory=generated,
                cameras=memory_cameras, warp=memory_warp,
                frame_start=frame_start, allow_candidate_promotion=len(self.activation_queue) == 0,
                defer_candidate_promotion=True,
            )
            shadow = memory_event.pop("shadow_node", None)
            if shadow is not None:
                scheduled = self.activation_queue.schedule(
                    shadow, created_after_chunk=self.chunk_index,
                )
                memory_event["pending_node_id"] = scheduled.node_id
                memory_event["activate_at_chunk"] = scheduled.activate_at_chunk
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
            "reactivation_event":reactivation_event,
            "active_node_id": self.active_node.node_id,
            "memory_event": memory_event,
            "chunk_index": self.chunk_index,
            "stage0_film_applied": self.stage0_film.applied_calls > 0,
            "uses_future_gt": False,
        }
        self.chunk_index += 1
        return generated, poses, warp, statistics

    def generate_chunk_at_cameras(self, c2w, intrinsics, height, width):
        """Generate one chunk at exact dataset cameras without control integration."""
        poses = np.asarray(c2w, np.float32)
        if poses.ndim != 3 or poses.shape[1:] != (4, 4) or len(poses) < 1:
            raise ValueError("c2w must be [T,4,4] with at least one frame")
        intrinsics = np.asarray(intrinsics, np.float32)
        if intrinsics.ndim == 2:
            intrinsics = np.repeat(intrinsics[None], len(poses), axis=0)
        if intrinsics.shape != (len(poses), 3, 3):
            raise ValueError("intrinsics must be [T,3,3] and match c2w")
        return self._generate_cameras(CameraBatch(poses, intrinsics, int(height), int(width)))

    def prepare_supervised_chunk(self, c2w, intrinsics, height, width):
        """Build the exact causal inference conditioning without generating GT.

        This is the training boundary: it renders only the world committed by
        earlier model generations, installs Point-FiLM context, and asks the
        pinned WAH implementation to build its native TEMP/CURRENT_WARP
        histories.  It neither advances AR state nor writes current GT into the
        world.
        """
        poses = np.asarray(c2w, np.float32)
        intrinsics = np.asarray(intrinsics, np.float32)
        if intrinsics.ndim == 2:
            intrinsics = np.repeat(intrinsics[None], len(poses), axis=0)
        cameras = CameraBatch(poses, intrinsics, int(height), int(width))
        activation = self.activation_queue.activate_due(self.chunk_index)
        if activation is not None:
            verified_hash = self.memory_manager.verify_shadow(activation.node)
            self.active_node = self.memory_manager.commit_shadow(
                self.active_node, activation.node, verified_hash=verified_hash,
            )
        if self.memory_manager is not None:
            self.active_node, _ = self.memory_manager.maybe_reactivate(self.active_node, cameras)
        warp = render(self.active_node, cameras, **self.renderer_kwargs)
        import torch
        point_feature, visibility0 = aggregate_winning_points(
            warp, self.source_c2w_fixed, self.source_depth_scale,
            self.stage0_film.point_encoder,
            device=self.wah_pipeline._wah_execution_device(),
            dtype=getattr(self.wah_pipeline.transformer, "dtype", None),
        )
        self.stage0_film.set_point_context(point_feature, visibility0)
        self.stage0_film.bind_inference_scheduler(self.wah_pipeline.scheduler)
        state = self.autoregressive_state
        inputs = self.wah_adapter.warp_inputs(warp)
        self.wah_adapter.configure_state(state)
        self.wah_pipeline._prepare_autoregressive_warp_chunk(
            state, inputs["warp_video"], inputs["warp_visibility_mask"],
            inputs["warp_confidence_mask"],
        )
        history_sizes = list(state["history_sizes"])
        count = int(state["num_history_latent_frames"])
        long_history, mid_history, short_history = state["history_latents"][
            :, :, -count:
        ].split(history_sizes, dim=2)
        prefix = state.get("image_latents")
        if prefix is None:
            prefix = torch.zeros_like(short_history[:, :, :1])
        base_short = torch.cat([prefix, short_history], dim=2)
        histories = self.wah_pipeline._build_pyramid_base_histories(
            state=state,
            device=self.wah_pipeline._wah_execution_device(),
            history_dtype=base_short.dtype,
            generator=state.get("generator"),
            base_latents_history_short=base_short,
        )
        return warp, point_feature, visibility0, histories

    def generate_chunk(self, controls, intrinsics, height, width):
        poses = integrate_controls(
            self.current_camera_c2w, controls,
            scale=self.active_node.scale, **self.control_kwargs,
        )
        if not len(poses):
            raise ValueError("controls must contain at least one output frame")
        intrinsics = np.asarray(intrinsics, np.float32)
        if intrinsics.ndim == 2:
            intrinsics = np.repeat(intrinsics[None], len(poses), axis=0)
        return self._generate_cameras(CameraBatch(poses, intrinsics, int(height), int(width)))
