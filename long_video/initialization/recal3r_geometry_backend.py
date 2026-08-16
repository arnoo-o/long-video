"""Causal recurrent adapter for the official ReCal3R/CUT3R inference stack."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np

from ..data.recal3r_full_scene import (
    Sim3Alignment, calibrate_recal3r_confidence, official_resize_crop, remap_model_map,
)
from ..types import Z_DEPTH
from .geometry_backend import GeometryPrediction, MultiViewGeometryBackend


@dataclass
class _Frame:
    rgb: np.ndarray
    c2w: np.ndarray
    intrinsics: np.ndarray
    identity: str


class ReCal3RGeometryBackend(MultiViewGeometryBackend):
    """Frozen official ReCal3R geometry with a trajectory-owned causal state.

    ReCal3R predicts local self-view geometry and its own camera trajectory from
    generated RGB. The generated chunk already has an authoritative commanded
    camera trajectory, so ReCal camera poses are used only to recover the one
    relative reconstruction scale. Once that causal alignment locks, every
    self-view point map is placed with the commanded c2w for the same frame.
    """

    def __init__(self, checkpoint, repo_path, device, confidence_threshold=1.5,
                 confidence_temperature=0.35, min_alignment_frames=3):
        self.checkpoint = str(checkpoint)
        self.repo_path = str(repo_path)
        self.device = str(device)
        self.confidence_threshold = float(confidence_threshold)
        self.confidence_temperature = float(confidence_temperature)
        self.min_alignment_frames = int(min_alignment_frames)
        self._model = None
        self._inference = None
        self._pose = None
        self._img_norm = None
        self.reset()

    def _load(self):
        if self._model is not None:
            return
        import torch
        repo = Path(self.repo_path).resolve()
        if not repo.is_dir() or not Path(self.checkpoint).exists():
            raise FileNotFoundError("official ReCal3R repo/checkpoint is required")
        for root in (repo, repo / "src"):
            if str(root) not in sys.path:
                sys.path.insert(0, str(root))
        from src.dust3r.inference import inference_recurrent_lighter
        from src.dust3r.model import ARCroco3DStereo
        from src.dust3r.utils.image import ImgNorm
        model = ARCroco3DStereo.from_pretrained(self.checkpoint).to(self.device).eval()
        model.config.model_update_type = "recal3r"
        model.beta_base = model.config.beta_base = 0.1
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        self._model, self._inference, self._img_norm = model, inference_recurrent_lighter, ImgNorm

    def reset(self, *, preserve_alignment=False):
        self._frames: list[_Frame] = []
        self._state_args = None
        self._last_predictions = []
        self._results = {}
        self._raw_depth = {}
        self._geometry_validation = {}
        self._recal_poses = []
        self._seen = set()
        self._sequence_version = 0
        if not preserve_alignment:
            self._alignment = None
            self._alignment_metadata = {"status": "pending"}

    def initialize(self, rgb, c2w, intrinsics, **_kwargs):
        self.reset()
        return self.update(rgb, c2w, intrinsics)

    def get_state(self):
        return {
            "sequence_version": self._sequence_version,
            "frame_count": len(self._frames),
            "has_recurrent_state": self._state_args is not None,
            "backend": "official_recal3r_recurrent",
            "geometry_placement": "scaled_self_view_at_commanded_c2w",
            "alignment": dict(self._alignment_metadata),
        }

    def _view(self, frame: _Frame, index: int):
        import torch
        from PIL import Image

        image = Image.fromarray(frame.rgb.astype(np.uint8), "RGB")
        width, height = image.size
        transform = official_resize_crop(height, width, 512)
        resized = image.resize((transform["resized_width"], transform["resized_height"]), Image.Resampling.BICUBIC)
        left, top = transform["crop_left"], transform["crop_top"]
        cropped = resized.crop((left, top, left + transform["crop_width"], top + transform["crop_height"]))
        tensor = self._img_norm(cropped)[None]
        return {
            "img": tensor,
            "ray_map": torch.full((1, 6, tensor.shape[-2], tensor.shape[-1]), torch.nan),
            "true_shape": torch.tensor([[tensor.shape[-2], tensor.shape[-1]]], dtype=torch.int32),
            "idx": index, "instance": str(index),
            "camera_pose": torch.eye(4, dtype=torch.float32).unsqueeze(0),
            "img_mask": torch.tensor([True]), "ray_mask": torch.tensor([False]),
            "update": torch.tensor([True]), "reset": torch.tensor([False]),
        }

    def update_frame(self, rgb, c2w, intrinsics, *, trajectory_id, global_frame_index):
        """Process exactly one causal frame, keyed by trajectory/frame identity."""
        identity = f"{trajectory_id}:{int(global_frame_index)}"
        if identity in self._seen:
            raise RuntimeError(f"ReCal3R frame was submitted twice: {identity}")
        return self._update_frames([(np.asarray(rgb, np.uint8), np.asarray(c2w, np.float32),
                                     np.asarray(intrinsics, np.float32), identity)])[0]

    def update_chunk(self, rgb, c2w, intrinsics, *, trajectory_id, global_frame_indices):
        """Submit one Helios chunk once, in causal frame order."""
        rgb, c2w, intrinsics = np.asarray(rgb), np.asarray(c2w), np.asarray(intrinsics)
        ids = [int(value) for value in global_frame_indices]
        if len(rgb) != len(c2w) or len(rgb) != len(intrinsics) or len(rgb) != len(ids):
            raise ValueError("ReCal3R chunk RGB/cameras/indices must align")
        if ids != sorted(ids) or len(set(ids)) != len(ids):
            raise ValueError("ReCal3R chunk indices must be strictly unique and ordered")
        return self._update_frames([(np.asarray(image, np.uint8), np.asarray(pose, np.float32), np.asarray(k, np.float32),
                                    f"{trajectory_id}:{index}")
                                   for image, pose, k, index in zip(rgb, c2w, intrinsics, ids)])

    def replay_prefix(self, rgb, c2w, intrinsics, *, trajectory_id, global_frame_indices):
        """Official full-prefix replay from a fresh recurrent state per chunk."""
        ids = [int(value) for value in global_frame_indices]
        if not ids or ids[0] != 0 or ids != list(range(len(ids))):
            raise ValueError("ReCal full replay must be source global frame 0 plus contiguous unique frames")
        self.reset(preserve_alignment=True)
        frames = [(np.asarray(image, np.uint8), np.asarray(pose, np.float32), np.asarray(k, np.float32),
                   f"{trajectory_id}:{index}") for image, pose, k, index in zip(rgb, c2w, intrinsics, ids)]
        return self._update_frames(frames)

    def update(self, rgb, c2w, intrinsics, **_kwargs):
        rgb, c2w, intrinsics = np.asarray(rgb), np.asarray(c2w, np.float32), np.asarray(intrinsics, np.float32)
        if len(rgb) != len(c2w) or len(rgb) != len(intrinsics):
            raise ValueError("ReCal3R RGB/camera/intrinsics must align")
        frames = [(np.asarray(image, np.uint8), pose, k, f"legacy:{self._sequence_version}:{index}")
                  for index, (image, pose, k) in enumerate(zip(rgb, c2w, intrinsics))]
        values = self._update_frames(frames)
        return self._combine_predictions(values)

    @staticmethod
    def _combine_predictions(values):
        if not values:
            raise ValueError("ReCal3R update requires at least one frame")
        return GeometryPrediction(
            depth=np.concatenate([value.depth for value in values], 0),
            depth_confidence=np.concatenate([value.depth_confidence for value in values], 0),
            point_maps=np.concatenate([value.point_maps for value in values], 0),
            geometry_confidence=np.concatenate([value.geometry_confidence for value in values], 0),
            depth_convention=Z_DEPTH,
            scale_info=values[-1].scale_info,
            diagnostics=values[-1].diagnostics,
        )

    def _update_frames(self, frames):
        keys = [identity for *_values, identity in frames]
        additions = [(image, pose, k, identity) for image, pose, k, identity in frames if identity not in self._seen]
        self._frames.extend(_Frame(image.copy(), pose.copy(), k.copy(), identity)
                            for image, pose, k, identity in additions)
        self._seen.update(identity for *_values, identity in additions)
        if not self._frames:
            return [self._prediction_for_keys([key]) for key in keys]
        self._load()
        views = [self._view(frame, index) for index, frame in enumerate(self._frames)]
        outputs, state_args = self._inference(views, self._model, self.device, verbose=False)
        self._state_args = state_args[-1] if state_args else None
        self._last_predictions = outputs["pred"]
        self._sequence_version += 1
        self._cache_all_results()
        return [self._prediction_for_keys([key]) for key in keys]

    def _cache_all_results(self):
        recal_poses = [self._recal_c2w(prediction) for prediction in self._last_predictions]
        self._recal_poses = recal_poses
        if self._alignment is None and len(recal_poses) >= self.min_alignment_frames:
            try:
                self._alignment = self._lock_causal_recal_to_world(
                    np.stack(recal_poses), np.stack([frame.c2w for frame in self._frames]),
                )
                self._alignment_metadata = {
                    "status": "locked",
                    "anchor": "source_camera_causal_baseline_v2",
                    "scale": float(self._alignment.scale),
                    "camera_alignment_error": float(self._alignment.camera_alignment_error),
                    "camera_alignment_error_ratio": float(self._alignment.camera_alignment_error_ratio),
                    "median_rotation_error_degrees": float(self._alignment.median_rotation_error_degrees),
                    "max_rotation_error_degrees": float(self._alignment.max_rotation_error_degrees),
                    "anchor_frame": len(self._frames) - 1,
                }
            except ValueError as error:
                self._alignment_metadata = {"status": "pending", "reason": str(error)}
        for frame, prediction in zip(self._frames, self._last_predictions):
            result, raw_depth = self._geometry_for(frame, prediction)
            self._results[frame.identity] = result
            self._raw_depth[frame.identity] = raw_depth

    @staticmethod
    def _lock_causal_recal_to_world(recal_c2w, target_c2w):
        """Lock one causal ReCal reconstruction scale from camera baselines.

        ReCal camera poses are never used to place committed PointWorld points.
        They only provide a scale observable because ReCal camera translations
        and ReCal self-view points share the same arbitrary reconstruction unit.
        The source pose fixes the diagnostic global Sim(3) orientation/origin.
        """
        recal_c2w = np.asarray(recal_c2w, np.float64)
        target_c2w = np.asarray(target_c2w, np.float64)
        if recal_c2w.shape != target_c2w.shape or recal_c2w.ndim != 3 or recal_c2w.shape[1:] != (4, 4):
            raise ValueError("ReCal and commanded camera prefixes must be matching [T,4,4] arrays")
        if not np.isfinite(recal_c2w).all() or not np.isfinite(target_c2w).all():
            raise ValueError("camera prefix contains non-finite poses")
        source_recal, source_world = recal_c2w[0], target_c2w[0]
        rotation = source_world[:3, :3] @ source_recal[:3, :3].T
        recal_delta = recal_c2w[1:, :3, 3] - source_recal[:3, 3]
        world_delta = target_c2w[1:, :3, 3] - source_world[:3, 3]
        recal_baseline = np.linalg.norm(recal_delta, axis=1)
        world_baseline = np.linalg.norm(world_delta, axis=1)
        usable = (recal_baseline > 1e-5) & (world_baseline > 1e-5)
        if int(usable.sum()) < 2:
            raise ValueError("insufficient causal camera baseline for ReCal-to-PointWorld scale lock")
        ratios = world_baseline[usable] / recal_baseline[usable]
        ratios = ratios[np.isfinite(ratios) & (ratios > 1e-6)]
        if len(ratios) < 2:
            raise ValueError("invalid causal camera-baseline scale ratios")
        scale = float(np.median(ratios))
        if not np.isfinite(scale) or scale <= 1e-6:
            raise ValueError("invalid causal ReCal-to-PointWorld scale")
        translation = source_world[:3, 3] - scale * (rotation @ source_recal[:3, 3])
        aligned_centers = scale * (recal_c2w[:, :3, 3] @ rotation.T) + translation
        residual = np.linalg.norm(aligned_centers - target_c2w[:, :3, 3], axis=1)
        extent = max(float(np.median(world_baseline[usable])), 1e-8)
        rmse = float(np.sqrt(np.mean(residual * residual)))
        aligned_rotations = rotation[None] @ recal_c2w[:, :3, :3]
        relative = target_c2w[:, :3, :3].transpose(0, 2, 1) @ aligned_rotations
        cosine = np.clip((np.trace(relative, axis1=1, axis2=2) - 1.0) / 2.0, -1.0, 1.0)
        angles = np.degrees(np.arccos(cosine))
        return Sim3Alignment(
            scale, rotation.astype(np.float32), translation.astype(np.float32),
            rmse, rmse / extent, float(np.median(angles)), float(np.max(angles)),
        )

    def replay_predictions(self):
        return [self._prediction_for_keys([frame.identity]) for frame in self._frames]

    def _recal_c2w(self, prediction):
        # Official CUT3R stores poses in its compact encoding. Decode them for
        # the one-time scale lock and diagnostics only.
        from src.dust3r.utils.camera import pose_encoding_to_camera
        pose = pose_encoding_to_camera(prediction["camera_pose"].clone())
        return pose.detach().cpu().numpy()[0].astype(np.float32)

    def _geometry_for(self, frame, prediction):
        points = prediction["pts3d_in_self_view"].detach().cpu().numpy()[0]
        raw_confidence = prediction["conf_self"].detach().cpu().numpy()[0]
        transform = official_resize_crop(*frame.rgb.shape[:2], 512)
        if self._alignment is None:
            world = np.full_like(points, np.nan, dtype=np.float32)
        else:
            # ReCal owns local geometry. The commanded camera owns world pose.
            # Only the locked reconstruction scale transfers from ReCal's pose
            # stream; predicted ReCal rotations/translations never place points.
            local = np.asarray(points, np.float32) * float(self._alignment.scale)
            target_c2w = np.asarray(frame.c2w, np.float32)
            world = local @ target_c2w[:3, :3].T + target_c2w[:3, 3]
        mapped, inside = remap_model_map(world, transform, interpolation=1)
        raw_conf, _ = remap_model_map(raw_confidence.astype(np.float32), transform, interpolation=1)
        calibrated = calibrate_recal3r_confidence(
            raw_conf, self.confidence_threshold, self.confidence_temperature,
        )
        target_c2w = np.asarray(frame.c2w, np.float32)
        local_check = (mapped - target_c2w[:3, 3]) @ target_c2w[:3, :3]
        depth = local_check[..., 2].astype(np.float32)
        finite_world = np.isfinite(mapped).all(-1)
        finite_depth = np.isfinite(depth)
        positive_depth = depth > 0
        finite_confidence = np.isfinite(raw_conf)
        confidence_threshold = raw_conf >= self.confidence_threshold
        valid = (inside & finite_world & finite_depth & positive_depth
                 & np.isfinite(calibrated) & confidence_threshold)
        finite_raw_confidence = raw_conf[finite_confidence]
        self._geometry_validation[frame.identity] = {
            "pixel_count": int(raw_conf.size),
            "inside_count": int(inside.sum()),
            "finite_world_count": int(finite_world.sum()),
            "finite_depth_count": int(finite_depth.sum()),
            "positive_depth_count": int(positive_depth.sum()),
            "finite_confidence_count": int(finite_confidence.sum()),
            "confidence_threshold_count": int(confidence_threshold.sum()),
            "valid_count": int(valid.sum()),
            "raw_confidence_min": float(finite_raw_confidence.min()) if len(finite_raw_confidence) else None,
            "raw_confidence_max": float(finite_raw_confidence.max()) if len(finite_raw_confidence) else None,
            "raw_confidence_mean": float(finite_raw_confidence.mean()) if len(finite_raw_confidence) else None,
        }
        mapped[~valid] = np.nan
        depth[~valid] = np.nan
        calibrated = np.where(valid, calibrated, 0).astype(np.float32)
        return (depth, calibrated, mapped), np.asarray(points[..., 2], np.float32)

    def geometry_validation(self, trajectory_id, global_frame_index):
        identity = f"{trajectory_id}:{int(global_frame_index)}"
        value = self._geometry_validation.get(identity)
        if value is None:
            raise RuntimeError(f"missing ReCal validation metrics for {identity}")
        return dict(value)

    def raw_recal_depth(self, trajectory_id, global_frame_index):
        """Return ReCal's raw self-view z on the original frame grid."""
        identity = f"{trajectory_id}:{int(global_frame_index)}"
        value = self._raw_depth.get(identity)
        if value is None:
            raise RuntimeError(f"missing ReCal raw depth for {identity}")
        frame = next(frame for frame in self._frames if frame.identity == identity)
        depth, _ = remap_model_map(value, official_resize_crop(*frame.rgb.shape[:2], 512), interpolation=1)
        return depth.astype(np.float32)

    def _prediction_for_keys(self, keys):
        values = [self._results[key] for key in keys]
        depth, confidence, point_maps = map(np.stack, zip(*values))
        return GeometryPrediction(
            depth=depth, depth_confidence=confidence, point_maps=point_maps,
            geometry_confidence=confidence, depth_convention=Z_DEPTH,
            scale_info={"mode": "relative", "meters_per_world_unit": None,
                        "uncertainty": 1.0,
                        "anchor_source": "recal_camera_scale_commanded_pose"},
            diagnostics={"backend": "official_recal3r_recurrent", **self.get_state(),
                         "alignment": dict(self._alignment_metadata),
                         "valid_ratio": float(np.isfinite(depth).mean())},
        )

    def get_current_geometry(self, count=None):
        if not self._last_predictions:
            raise RuntimeError("ReCal3R state is empty")
        count = len(self._last_predictions) if count is None else int(count)
        keys = [frame.identity for frame in self._frames[-count:]]
        return self._prediction_for_keys(keys)

    def predict(self, view_rgb, view_c2w, intrinsics, **_kwargs):
        return self.update(view_rgb, view_c2w, intrinsics)
