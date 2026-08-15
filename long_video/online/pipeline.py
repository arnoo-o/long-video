"""End-to-end online spatial-history generation."""
from __future__ import annotations

from dataclasses import fields, replace
import numpy as np

from ..data.controls import integrate_controls
from ..geometry.point_renderer import render
from ..initialization.initial_node_pipeline import initialize_spatial_node
from ..types import CameraBatch
from ..wah.adapter import WAHAdapter
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
        pre_render_world_hook=None,
        world_accumulator=None,
    ):
        self.wah_pipeline = wah_pipeline
        self.wah_adapter = WAHAdapter(wah_pipeline) if wah_pipeline is not None else None
        self.active_node = active_node
        self.memory_manager = memory_manager
        if self.memory_manager is not None and active_node is not None:
            self.memory_manager.register(active_node)
        self.prompt = str(prompt)
        self.renderer_kwargs = {"device":"cpu",**dict(renderer_kwargs or {})}
        self.control_kwargs = dict(control_kwargs or {})
        self.wah_state_kwargs = dict(wah_state_kwargs or {})
        self.pre_render_world_hook = pre_render_world_hook
        self.world_accumulator = world_accumulator
        self.current_camera_c2w = (
            active_node.center_c2w.copy() if active_node is not None else np.eye(4, dtype=np.float32)
        )
        self.recent_video_history = []
        self.wah_fill_frame = (
            np.asarray(active_node.view_rgb[0]).copy() if active_node is not None else None
        )
        self.autoregressive_state = None
        self.frame_index = 0
        self.chunk_index = 0
        self.activation_queue = DelayedNodeActivationQueue(delay_chunks=1)

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
        if self.memory_manager is not None:
            self.memory_manager.geometry_backend = geometry_backend
            self.memory_manager.register(self.active_node)
        if self.wah_pipeline is not None:
            if first_image is None:
                first_image = np.asarray(self.active_node.view_rgb[0])
            self.wah_fill_frame = np.asarray(first_image).copy()
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

    def _decode_last_latent_chunk(self):
        """Decode the exact nine-latent AR chunk to its canonical 33 frames.

        The public WAH delta is cumulative-output bookkeeping and can contain
        32 or occasionally 36 newly finalized frames.  World construction must
        instead consume the current nine-latent chunk at its fixed 33-frame
        camera alignment.
        """
        state = self.autoregressive_state
        latents = state["last_latents"]
        vae_dtype = self.wah_pipeline.vae.dtype
        mean = state["latents_mean"].to(device=latents.device, dtype=vae_dtype)
        std = state["latents_std"].to(device=latents.device, dtype=vae_dtype)
        vae_latents = latents.to(vae_dtype) / std + mean
        decoded = self.wah_pipeline._decode_autoregressive_latents(
            diffusion_latents=latents, vae_latents=vae_latents,
        )
        video = self.wah_pipeline.video_processor.postprocess_video(decoded, output_type="np")
        result = self._video_array(video)
        if len(result) != 33:
            raise RuntimeError(f"nine WAH latents must decode to 33 frames, got {len(result)}")
        return result

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
        activation = None
        if self.world_accumulator is None:
            activation = self.activation_queue.activate_due(self.chunk_index)
            if activation is not None:
                verified_hash = self.memory_manager.verify_shadow(activation.node)
                self.active_node = self.memory_manager.commit_shadow(
                    self.active_node, activation.node, verified_hash=verified_hash,
                )
        reactivation_event=None
        if self.memory_manager is not None and self.world_accumulator is None:
            self.active_node,reactivation_event=self.memory_manager.maybe_reactivate(
                self.active_node,cameras)
        # Consumers that build conditioning from the point world must observe
        # exactly the snapshot that WAH is about to render, after all delayed
        # activation and reactivation decisions for this chunk.
        pre_render_snapshot = None
        if self.pre_render_world_hook is not None:
            pre_render_snapshot = self.pre_render_world_hook(self.active_node, cameras)
        warp = render(self.active_node,cameras,**self.renderer_kwargs)
        if pre_render_snapshot is not None:
            expected = (
                getattr(self.active_node, "node_id", id(self.active_node)),
                int(getattr(self.active_node, "quality_metrics", {}).get("recal3r_world_version", 0)),
            )
            observed = pre_render_snapshot.get("world_version") if isinstance(pre_render_snapshot, dict) else pre_render_snapshot
            # Legacy non-accumulator hooks returned a node id; ReCal worlds
            # additionally carry the monotonically changing accumulator version.
            if observed != expected and observed != expected[0]:
                raise RuntimeError(
                    f"pre-render conditioning world mismatch: {observed!r} != {expected!r}"
                )
        if hasattr(self.wah_pipeline, "set_world_projection_from_renderer"):
            self.wah_pipeline.set_world_projection_from_renderer(
                warp.rgb, warp.visibility, warp.confidence,
                height=cameras.height, width=cameras.width,
            )
        try:
            _generated_delta, self.autoregressive_state = self.wah_adapter.generate_next_chunk(
                self.autoregressive_state, warp, output_type="np", fill_frame=self.wah_fill_frame,
            )
        finally:
            context = getattr(self.wah_pipeline, "_world_projection_context", None)
            world_projection_diagnostics = list(context.diagnostics) if context is not None else []
            if hasattr(self.wah_pipeline, "clear_world_projection_context"):
                self.wah_pipeline.clear_world_projection_context()
        # Geometry history follows the same autoregressive-state ownership as
        # WAH latents.  The hook captures the pre-render active world and each
        # snapshot is detached by the provider before any ReCal3R/world mutation.
        if isinstance(pre_render_snapshot, dict):
            freeze = pre_render_snapshot.get("freeze_history")
            if freeze is not None:
                history = self.autoregressive_state.setdefault("_geotoken_history_snapshots", [])
                history.append(freeze(chunk_index=self.chunk_index, frame_start=self.frame_index))
                snapshot = history[-1]
                source_slot = self.autoregressive_state.setdefault(
                    "_geotoken_source_geometry", (snapshot, 0),
                )
                del source_slot  # state ownership documents the immutable source slot.
                previous = self.autoregressive_state.setdefault("_geotoken_prev_history_window", [])
                previous.extend((snapshot, slot) for slot in range(9))
                # Match official WAH prev_history_latent_window exactly: 16+2+1.
                self.autoregressive_state["_geotoken_prev_history_window"] = previous[-19:]
        generated_chunk = self._decode_last_latent_chunk()
        self.wah_fill_frame = np.asarray(generated_chunk[-1]).copy()
        if len(poses) != len(generated_chunk):
            raise ValueError("current WAH latent chunk and target cameras must both contain 33 frames")
        # P0 is the shared source/boundary pose.  W0 already owns source
        # geometry, so ReCal fusion always receives only P1..P32.
        frame_offset = 1
        generated = generated_chunk[frame_offset:]
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
        self.recent_video_history.append(generated_chunk)
        self.recent_video_history = self.recent_video_history[-2:]
        memory_event = None
        if self.world_accumulator is not None:
            self.active_node = self.world_accumulator.update_chunk(
                generated, memory_cameras.c2w, memory_cameras.intrinsics,
                range(frame_start, frame_start + len(generated)),
            )
            memory_event = {
                "backend": "recal3r_world_accumulator",
                "updated_frames": int(len(generated)),
                "world_point_count": int(len(self.active_node.points_xyz)),
            }
        elif self.memory_manager is not None:
            backend = self.memory_manager.geometry_backend
            if backend is not None and hasattr(backend, "update"):
                # Advance a recurrent geometry backend on every causal
                # generated chunk before candidate selection/promotion.
                backend.update(generated, memory_cameras.c2w, memory_cameras.intrinsics)
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
            "world_projection_diagnostics": world_projection_diagnostics,
            "uses_future_gt": False,
        }
        self.chunk_index += 1
        return generated_chunk, poses, warp, statistics

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

    def generate_chunk(self, controls, intrinsics, height, width):
        start = self.current_camera_c2w.copy()
        advanced = integrate_controls(
            self.current_camera_c2w, controls,
            # PointWorld is deliberately relative; control speed is never
            # rescaled by ReCal alignment metadata.
            scale=None, **self.control_kwargs,
        )
        if len(advanced) != 32:
            raise ValueError("one 33-frame chunk requires exactly 32 controls")
        poses = np.concatenate([start[None], advanced], axis=0)
        intrinsics = np.asarray(intrinsics, np.float32)
        if intrinsics.ndim == 2:
            intrinsics = np.repeat(intrinsics[None], len(poses), axis=0)
        return self._generate_cameras(CameraBatch(poses, intrinsics, int(height), int(width)))

