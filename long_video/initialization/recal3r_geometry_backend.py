"""Causal recurrent adapter for the official ReCal3R/CUT3R inference stack."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np

from ..data.recal3r_full_scene import (
    calibrate_recal3r_confidence,
    official_resize_crop,
    remap_model_map,
)
from ..geometry.backprojection import backproject_z_depth
from ..types import Z_DEPTH
from .geometry_backend import GeometryPrediction, MultiViewGeometryBackend


@dataclass
class _Frame:
    rgb: np.ndarray
    c2w: np.ndarray
    intrinsics: np.ndarray
    identity: str


class ReCal3RGeometryBackend(MultiViewGeometryBackend):
    """Frozen official ReCal3R geometry with a causal trajectory-owned state.

    ReCal3R owns per-frame self-view geometry.  The immutable Pi3X source W0
    provides the one-time ReCal-to-PointWorld source alignment scale.  The
    commanded camera trajectory owns every generated frame's world pose.
    ReCal's predicted camera trajectory never places PointWorld geometry and
    never determines the persistent world scale.
    """

    def __init__(self, checkpoint, repo_path, device, confidence_threshold=1.5,
                 confidence_temperature=0.35, min_alignment_frames=3,
                 confidence_quantile=0.3):
        self.checkpoint = str(checkpoint)
        self.repo_path = str(repo_path)
        self.device = str(device)
        # Kept as a compatibility argument for old callers.  ReCal acceptance
        # is intentionally data-driven: each original-grid frame uses a
        # configurable raw-confidence quantile over geometrically valid pixels.
        self.confidence_threshold = float(confidence_threshold)
        self.confidence_temperature = float(confidence_temperature)
        self.confidence_quantile = float(confidence_quantile)
        if not np.isfinite(self.confidence_quantile) or not 0 <= self.confidence_quantile <= 1:
            raise ValueError("confidence_quantile must be in [0,1]")
        # Kept for API/checkpoint compatibility. Source alignment no longer
        # waits for a camera-baseline estimate.
        self.min_alignment_frames = int(min_alignment_frames)
        self._model = None
        self._inference = None
        self._img_norm = None
        self.reset()

    def _load(self):
        if self._model is not None:
            return
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
        self._model = model
        self._inference = inference_recurrent_lighter
        self._img_norm = ImgNorm

    def reset(self, *, preserve_alignment=False):
        self._frames: list[_Frame] = []
        self._state_args = None
        self._last_predictions = []
        self._results = {}
        self._raw_depth = {}
        self._raw_confidence = {}
        self._native_points = {}
        self._commanded_world = {}
        self._geometry_validation = {}
        self._seen = set()
        self._sequence_version = 0
        if not preserve_alignment:
            self._alignment_scale = None
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
            "geometry_placement": "recal_z_backprojected_with_commanded_intrinsics_at_commanded_c2w",
            "alignment": dict(self._alignment_metadata),
        }

    def _view(self, frame: _Frame, index: int):
        import torch
        from PIL import Image

        image = Image.fromarray(frame.rgb.astype(np.uint8), "RGB")
        width, height = image.size
        transform = official_resize_crop(height, width, 512)
        resized = image.resize(
            (transform["resized_width"], transform["resized_height"]),
            Image.Resampling.BICUBIC,
        )
        left, top = transform["crop_left"], transform["crop_top"]
        cropped = resized.crop(
            (left, top, left + transform["crop_width"], top + transform["crop_height"])
        )
        tensor = self._img_norm(cropped)[None]
        return {
            "img": tensor,
            "ray_map": torch.full((1, 6, tensor.shape[-2], tensor.shape[-1]), torch.nan),
            "true_shape": torch.tensor([[tensor.shape[-2], tensor.shape[-1]]], dtype=torch.int32),
            "idx": index,
            "instance": str(index),
            "camera_pose": torch.eye(4, dtype=torch.float32).unsqueeze(0),
            "img_mask": torch.tensor([True]),
            "ray_mask": torch.tensor([False]),
            "update": torch.tensor([True]),
            "reset": torch.tensor([False]),
        }

    def update_frame(self, rgb, c2w, intrinsics, *, trajectory_id, global_frame_index):
        identity = f"{trajectory_id}:{int(global_frame_index)}"
        if identity in self._seen:
            raise RuntimeError(f"ReCal3R frame was submitted twice: {identity}")
        return self._update_frames([
            (
                np.asarray(rgb, np.uint8),
                np.asarray(c2w, np.float32),
                np.asarray(intrinsics, np.float32),
                identity,
            )
        ])[0]

    def update_chunk(self, rgb, c2w, intrinsics, *, trajectory_id, global_frame_indices):
        rgb = np.asarray(rgb)
        c2w = np.asarray(c2w)
        intrinsics = np.asarray(intrinsics)
        ids = [int(value) for value in global_frame_indices]
        if len(rgb) != len(c2w) or len(rgb) != len(intrinsics) or len(rgb) != len(ids):
            raise ValueError("ReCal3R chunk RGB/cameras/indices must align")
        if ids != sorted(ids) or len(set(ids)) != len(ids):
            raise ValueError("ReCal3R chunk indices must be strictly unique and ordered")
        return self._update_frames([
            (
                np.asarray(image, np.uint8),
                np.asarray(pose, np.float32),
                np.asarray(k, np.float32),
                f"{trajectory_id}:{index}",
            )
            for image, pose, k, index in zip(rgb, c2w, intrinsics, ids)
        ])

    def replay_prefix(self, rgb, c2w, intrinsics, *, trajectory_id, global_frame_indices):
        """Official full-prefix replay from a fresh recurrent state per chunk."""
        ids = [int(value) for value in global_frame_indices]
        if not ids or ids[0] != 0 or ids != list(range(len(ids))):
            raise ValueError("ReCal full replay must be source global frame 0 plus contiguous unique frames")
        self.reset(preserve_alignment=True)
        frames = [
            (
                np.asarray(image, np.uint8),
                np.asarray(pose, np.float32),
                np.asarray(k, np.float32),
                f"{trajectory_id}:{index}",
            )
            for image, pose, k, index in zip(rgb, c2w, intrinsics, ids)
        ]
        return self._update_frames(frames)

    def update(self, rgb, c2w, intrinsics, **_kwargs):
        rgb = np.asarray(rgb)
        c2w = np.asarray(c2w, np.float32)
        intrinsics = np.asarray(intrinsics, np.float32)
        if len(rgb) != len(c2w) or len(rgb) != len(intrinsics):
            raise ValueError("ReCal3R RGB/camera/intrinsics must align")
        frames = [
            (
                np.asarray(image, np.uint8),
                pose,
                k,
                f"legacy:{self._sequence_version}:{index}",
            )
            for index, (image, pose, k) in enumerate(zip(rgb, c2w, intrinsics))
        ]
        return self._combine_predictions(self._update_frames(frames))

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
        additions = [
            (image, pose, k, identity)
            for image, pose, k, identity in frames
            if identity not in self._seen
        ]
        self._frames.extend(
            _Frame(image.copy(), pose.copy(), k.copy(), identity)
            for image, pose, k, identity in additions
        )
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
        for frame, prediction in zip(self._frames, self._last_predictions):
            result, raw_depth = self._geometry_for(frame, prediction)
            self._results[frame.identity] = result
            self._raw_depth[frame.identity] = raw_depth

    def lock_source_geometry_alignment(self, source_world_depth, *, min_overlap=1024):
        """Lock the one ReCal-to-W0 source alignment scale exactly once.

        The immutable Pi3X W0 source rendering and ReCal's source self-view
        geometry describe the same source image. Their robust depth ratio gives
        the ReCal reconstruction-unit to PointWorld-unit scale without using a
        generated-frame camera baseline. Generated source geometry is never
        fused by this method.
        """
        if self._alignment_scale is not None:
            return self.replay_predictions()
        if not self._frames or not self._last_predictions:
            raise RuntimeError("ReCal source prediction is unavailable for source alignment")

        source_depth = np.asarray(source_world_depth, np.float32)
        trajectory_id = self._frames[0].identity.rsplit(":", 1)[0]
        recal_depth = self.raw_recal_depth(trajectory_id, 0)
        if source_depth.shape != recal_depth.shape:
            raise ValueError("Pi3X W0 and ReCal source depth grids must match")

        overlap = (
            np.isfinite(source_depth)
            & (source_depth > 0)
            & np.isfinite(recal_depth)
            & (recal_depth > 0)
        )
        overlap_count = int(overlap.sum())
        if overlap_count < int(min_overlap):
            self._alignment_metadata = {
                "status": "pending",
                "reason": "insufficient_source_geometry_overlap",
                "overlap_count": overlap_count,
                "min_overlap": int(min_overlap),
            }
            return self.replay_predictions()

        ratios = source_depth[overlap] / recal_depth[overlap]
        ratios = ratios[np.isfinite(ratios) & (ratios > 1e-6)]
        if len(ratios) < int(min_overlap):
            self._alignment_metadata = {
                "status": "pending",
                "reason": "invalid_source_geometry_ratio",
                "overlap_count": int(len(ratios)),
                "min_overlap": int(min_overlap),
            }
            return self.replay_predictions()

        scale = float(np.median(ratios))
        mad = float(np.median(np.abs(ratios - scale)))
        if not np.isfinite(scale) or scale <= 1e-6:
            self._alignment_metadata = {
                "status": "pending",
                "reason": "invalid_source_geometry_scale",
            }
            return self.replay_predictions()

        self._alignment_scale = scale
        self._alignment_metadata = {
            "status": "locked",
            "anchor": "pi3x_w0_source_geometry_alignment_v2",
            "scale": scale,
            "depth_ratio_mad": mad,
            "overlap_count": int(len(ratios)),
            "anchor_frame": 0,
            "placement": "commanded_intrinsics_and_c2w",
        }
        # Rebuild all prefix outputs using the fixed scale. Pending generated
        # frames can then be backfilled by the accumulator exactly once.
        self._cache_all_results()
        return self.replay_predictions()

    def replay_predictions(self):
        return [self._prediction_for_keys([frame.identity]) for frame in self._frames]

    def _geometry_for(self, frame, prediction):
        points = prediction["pts3d_in_self_view"].detach().cpu().numpy()[0]
        raw_confidence = prediction["conf_self"].detach().cpu().numpy()[0]
        transform = official_resize_crop(*frame.rgb.shape[:2], 512)
        # ReCal's predicted X/Y coordinates live in its own implicit camera
        # model. Only its self-view Z is transferable to the commanded camera.
        raw_depth_model = np.asarray(points[..., 2], np.float32)
        raw_depth, depth_inside = remap_model_map(
            raw_depth_model, transform, interpolation=1
        )
        raw_conf, _ = remap_model_map(
            raw_confidence.astype(np.float32), transform, interpolation=1
        )
        if self._alignment_scale is None:
            depth = np.full_like(raw_depth, np.nan, dtype=np.float32)
        else:
            depth = raw_depth * float(self._alignment_scale)
        # Reconstruct local XYZ on the original RGB grid with the actual
        # commanded intrinsics, then place it with the commanded camera pose.
        native_points = backproject_z_depth(raw_depth, frame.intrinsics)
        local = backproject_z_depth(depth, frame.intrinsics)
        target_c2w = np.asarray(frame.c2w, np.float32)
        mapped = local @ target_c2w[:3, :3].T + target_c2w[:3, 3]
        inside = depth_inside
        # Keep pre-threshold, pre-fusion diagnostics separate from the returned
        # conditioning geometry, whose invalid pixels are intentionally NaN.
        # Lightweight tests intentionally call this private helper without
        # constructing a full backend/resetting its runtime caches.
        if not hasattr(self, "_raw_confidence"):
            self._raw_confidence = {}
            self._native_points = {}
            self._commanded_world = {}
        self._raw_confidence[frame.identity] = raw_conf.astype(np.float32, copy=True)
        self._native_points[frame.identity] = native_points.astype(np.float32, copy=True)
        self._commanded_world[frame.identity] = mapped.astype(np.float32, copy=True)
        finite_world = np.isfinite(mapped).all(-1)
        finite_depth = np.isfinite(depth)
        positive_depth = depth > 0
        finite_confidence = np.isfinite(raw_conf)
        threshold_support = inside & finite_depth & positive_depth & finite_confidence
        if threshold_support.any():
            confidence_quantile = float(getattr(self, "confidence_quantile", 0.3))
            effective_threshold = float(np.quantile(raw_conf[threshold_support], confidence_quantile))
            calibrated = calibrate_recal3r_confidence(
                raw_conf, effective_threshold, self.confidence_temperature,
            )
            confidence_threshold = raw_conf >= effective_threshold
        else:
            effective_threshold = float("inf")
            calibrated = np.zeros_like(raw_conf, dtype=np.float32)
            confidence_threshold = np.zeros_like(raw_conf, dtype=bool)
        valid = (
            inside
            & finite_world
            & finite_depth
            & positive_depth
            & np.isfinite(calibrated)
            & confidence_threshold
        )
        finite_raw_confidence = raw_conf[finite_confidence]
        self._geometry_validation[frame.identity] = {
            "pixel_count": int(raw_conf.size),
            "inside_count": int(inside.sum()),
            "finite_world_count": int(finite_world.sum()),
            "finite_depth_count": int(finite_depth.sum()),
            "positive_depth_count": int(positive_depth.sum()),
            "finite_confidence_count": int(finite_confidence.sum()),
            "confidence_threshold_count": int(confidence_threshold.sum()),
            "confidence_threshold_mode": (
                f"p{100 * float(getattr(self, 'confidence_quantile', 0.3)):g}_valid_grid_raw_confidence"
            ),
            "confidence_quantile": float(getattr(self, "confidence_quantile", 0.3)),
            "effective_confidence_threshold": (
                effective_threshold if np.isfinite(effective_threshold) else None
            ),
            "valid_count": int(valid.sum()),
            "raw_confidence_min": (
                float(finite_raw_confidence.min()) if len(finite_raw_confidence) else None
            ),
            "raw_confidence_max": (
                float(finite_raw_confidence.max()) if len(finite_raw_confidence) else None
            ),
            "raw_confidence_mean": (
                float(finite_raw_confidence.mean()) if len(finite_raw_confidence) else None
            ),
        }
        mapped[~valid] = np.nan
        depth[~valid] = np.nan
        calibrated = np.where(valid, calibrated, 0).astype(np.float32)
        return (depth, calibrated, mapped), raw_depth.astype(np.float32)

    def geometry_validation(self, trajectory_id, global_frame_index):
        identity = f"{trajectory_id}:{int(global_frame_index)}"
        value = self._geometry_validation.get(identity)
        if value is None:
            raise RuntimeError(f"missing ReCal validation metrics for {identity}")
        return dict(value)

    def raw_recal_depth(self, trajectory_id, global_frame_index):
        """Return ReCal self-view z on the original source-frame grid."""
        identity = f"{trajectory_id}:{int(global_frame_index)}"
        value = self._raw_depth.get(identity)
        if value is None:
            raise RuntimeError(f"missing ReCal raw depth for {identity}")
        frame = next(frame for frame in self._frames if frame.identity == identity)
        depth, _ = remap_model_map(
            value,
            official_resize_crop(*frame.rgb.shape[:2], 512),
            interpolation=1,
        )
        return depth.astype(np.float32)

    def raw_recal_debug(self, trajectory_id, global_frame_index):
        """Return original-grid ReCal maps before thresholding or voxel fusion."""
        identity = f"{trajectory_id}:{int(global_frame_index)}"
        values = {
            "raw_recal_depth": self.raw_recal_depth(trajectory_id, global_frame_index),
            "raw_recal_confidence": self._raw_confidence.get(identity),
            "native_recal_world": self._native_points.get(identity),
            "commanded_world_before_fusion": self._commanded_world.get(identity),
        }
        if any(value is None for value in values.values()):
            raise RuntimeError(f"missing ReCal debug geometry for {identity}")
        return {key: np.asarray(value).copy() for key, value in values.items()}

    def _prediction_for_keys(self, keys):
        values = [self._results[key] for key in keys]
        depth, confidence, point_maps = map(np.stack, zip(*values))
        return GeometryPrediction(
            depth=depth,
            depth_confidence=confidence,
            point_maps=point_maps,
            geometry_confidence=confidence,
            depth_convention=Z_DEPTH,
            scale_info={
                "mode": "relative",
                "meters_per_world_unit": None,
                "uncertainty": 1.0,
                "anchor_source": "pi3x_w0_source_geometry_commanded_pose",
            },
            diagnostics={
                "backend": "official_recal3r_recurrent",
                **self.get_state(),
                "alignment": dict(self._alignment_metadata),
                "valid_ratio": float(np.isfinite(depth).mean()),
            },
        )

    def get_current_geometry(self, count=None):
        if not self._last_predictions:
            raise RuntimeError("ReCal3R state is empty")
        count = len(self._last_predictions) if count is None else int(count)
        keys = [frame.identity for frame in self._frames[-count:]]
        return self._prediction_for_keys(keys)

    def predict(self, view_rgb, view_c2w, intrinsics, **_kwargs):
        return self.update(view_rgb, view_c2w, intrinsics)
