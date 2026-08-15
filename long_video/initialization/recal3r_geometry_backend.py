"""Causal recurrent adapter for the official ReCal3R/CUT3R inference stack."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np

from ..data.recal3r_full_scene import (
    apply_sim3_points, calibrate_recal3r_confidence, official_resize_crop, remap_model_map,
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

    ReCal3R's public recurrent API exposes a full causal sequence interface;
    its public one-frame helper does not return the updated recurrent state.
    We therefore retain the generated-frame sequence and re-enter the official
    recurrent API on that causal prefix.  No future RGB or GT geometry is ever
    included, and the resulting state metadata is persistent across chunks.
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
        from src.dust3r.inference import inference_recurrent
        from src.dust3r.model import ARCroco3DStereo
        from src.dust3r.utils.image import ImgNorm
        model = ARCroco3DStereo.from_pretrained(self.checkpoint).to(self.device).eval()
        model.config.model_update_type = "recal3r"
        model.beta_base = model.config.beta_base = 0.1
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        self._model, self._inference, self._img_norm = model, inference_recurrent, ImgNorm

    def reset(self):
        self._frames: list[_Frame] = []
        self._state_args = None
        self._last_predictions = []
        self._results = {}
        self._raw_depth = {}
        self._seen = set()
        self._sequence_version = 0
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
        # ReCal predictions are in a recurrent reconstruction coordinate
        # system.  Lock one Sim(3) using only the prefix seen so far, then
        # transform both points and ReCal cameras into the project world.
        recal_poses = [self._recal_c2w(prediction) for prediction in self._last_predictions]
        if self._alignment is None and len(recal_poses) >= self.min_alignment_frames:
            try:
                self._alignment = self._lock_causal_recal_to_world(np.stack(recal_poses), np.stack([frame.c2w for frame in self._frames]))
                self._alignment_metadata = {
                    "status": "locked", "scale": float(self._alignment.scale),
                    "camera_alignment_error": float(self._alignment.camera_alignment_error),
                    "camera_alignment_error_ratio": float(self._alignment.camera_alignment_error_ratio),
                    "anchor_frame": len(self._frames) - 1,
                }
            except ValueError as error:
                self._alignment_metadata = {"status": "pending", "reason": str(error)}
        for frame, prediction, recal_c2w in zip(self._frames, self._last_predictions, recal_poses):
            result, raw_depth = self._geometry_for(frame, prediction, recal_c2w)
            self._results[frame.identity] = result
            self._raw_depth[frame.identity] = raw_depth

    @staticmethod
    def _lock_causal_recal_to_world(recal_c2w, target_c2w):
        """Source-anchored causal similarity transform, never global Sim(3) fitting.

        Orientation/origin come from the shared source camera; scale is a
        robust median of already-observed camera baselines.  The accumulator
        independently records depth-overlap residuals before accepting points.
        """
        from ..data.recal3r_full_scene import Sim3Alignment
        source_recal, source_world = recal_c2w[0], target_c2w[0]
        rotation = source_world[:3, :3] @ source_recal[:3, :3].T
        recal_delta = recal_c2w[1:, :3, 3] - source_recal[:3, 3]
        world_delta = target_c2w[1:, :3, 3] - source_world[:3, 3]
        a = np.linalg.norm(recal_delta, axis=1); b = np.linalg.norm(world_delta, axis=1)
        usable = (a > 1e-5) & (b > 1e-5)
        if int(usable.sum()) < 2:
            raise ValueError("insufficient causal camera baseline for ReCal-to-W0 lock")
        ratios = b[usable] / a[usable]; scale = float(np.median(ratios))
        translation = source_world[:3, 3] - scale * (rotation @ source_recal[:3, 3])
        aligned = scale * (recal_c2w[:, :3, 3] @ rotation.T) + translation
        residual = np.linalg.norm(aligned - target_c2w[:, :3, 3], axis=1)
        extent = max(float(np.median(b[usable])), 1e-8)
        return Sim3Alignment(scale, rotation, translation, float(np.sqrt(np.mean(residual * residual))),
                             float(np.sqrt(np.mean(residual * residual)) / extent), 0.0, 0.0)

    def _recal_c2w(self, prediction):
        # Official CUT3R stores poses in its compact encoding.  Decode with
        # the same helper used by the offline full-scene builder.
        from src.dust3r.utils.camera import pose_encoding_to_camera
        pose = pose_encoding_to_camera(prediction["camera_pose"].clone())
        return pose.detach().cpu().numpy()[0].astype(np.float32)

    def _geometry_for(self, frame, prediction, recal_c2w):
        points = prediction["pts3d_in_self_view"].detach().cpu().numpy()[0]
        raw_confidence = prediction["conf_self"].detach().cpu().numpy()[0]
        transform = official_resize_crop(*frame.rgb.shape[:2], 512)
        # First enter ReCal's predicted world, then the trajectory-fixed
        # causal Sim(3).  Never reinterpret self-view z as target-camera z.
        recal_world = points @ recal_c2w[:3, :3].T + recal_c2w[:3, 3]
        if self._alignment is None:
            world = np.full_like(recal_world, np.nan, dtype=np.float32)
        else:
            world = apply_sim3_points(recal_world, self._alignment).astype(np.float32)
        mapped, inside = remap_model_map(world, transform, interpolation=1)
        raw_conf, _ = remap_model_map(raw_confidence.astype(np.float32), transform, interpolation=1)
        calibrated = calibrate_recal3r_confidence(
            raw_conf, self.confidence_threshold, self.confidence_temperature,
        )
        # z-depth is measured after world alignment in the known target camera.
        local = (mapped - frame.c2w[:3, 3]) @ frame.c2w[:3, :3]
        depth = local[..., 2].astype(np.float32)
        valid = (inside & np.isfinite(mapped).all(-1) & np.isfinite(depth) & (depth > 0)
                 & np.isfinite(calibrated) & (raw_conf >= self.confidence_threshold))
        mapped[~valid] = np.nan
        depth[~valid] = np.nan
        calibrated = np.where(valid, calibrated, 0).astype(np.float32)
        return (depth, calibrated, mapped), np.asarray(points[..., 2], np.float32)

    def raw_recal_depth(self, trajectory_id, global_frame_index):
        """Causal self-view z for the accumulator's one-time overlap anchor."""
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
            # ReCal3R has arbitrary reconstruction scale until a causal
            # overlap anchor is measured by the world accumulator.
            scale_info={"mode": "relative" if self._alignment is None else "dataset_calibrated",
                        "meters_per_world_unit": None if self._alignment is None else float(self._alignment.scale),
                        "uncertainty": 1.0 if self._alignment is None else float(min(1.0, self._alignment.camera_alignment_error_ratio)),
                        "anchor_source": "causal_camera_sim3" if self._alignment is not None else "causal_overlap_pending"},
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
