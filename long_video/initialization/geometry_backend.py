"""Unified multi-view geometry backends.

Pi3 is loaded lazily from the official Holo360D checkout so the core package
remains importable without the third-party model environment.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import sys

import numpy as np

from ..geometry.backprojection import backproject_z_depth
from ..types import Z_DEPTH


@dataclass
class GeometryPrediction:
    depth: np.ndarray
    depth_confidence: np.ndarray
    point_maps: Optional[np.ndarray] = None
    predicted_c2w: Optional[np.ndarray] = None
    geometry_confidence: Optional[np.ndarray] = None
    scale_info: dict = field(default_factory=dict)
    diagnostics: dict = field(default_factory=dict)
    depth_convention: str = Z_DEPTH


class MultiViewGeometryBackend:
    def predict(self, view_rgb, view_c2w, intrinsics, known_depth=None, known_mask=None):
        raise NotImplementedError


def _world_point_maps(depth, intrinsics, c2w):
    result = []
    for index in range(len(depth)):
        local = backproject_z_depth(depth[index], intrinsics[index])
        pose = np.asarray(c2w[index], np.float32)
        result.append(local @ pose[:3, :3].T + pose[:3, 3])
    return np.stack(result).astype(np.float32)


class GroundTruthGeometryBackend(MultiViewGeometryBackend):
    def predict(self, view_rgb, view_c2w, intrinsics, known_depth=None, known_mask=None):
        if known_depth is None:
            raise ValueError("GroundTruthGeometryBackend requires known_depth")
        depth = np.asarray(known_depth, np.float32).copy()
        valid = np.isfinite(depth) & (depth > 0)
        if known_mask is not None:
            valid &= np.asarray(known_mask, bool)
        depth[~valid] = np.nan
        confidence = valid.astype(np.float32)
        return GeometryPrediction(
            depth=depth,
            depth_confidence=confidence,
            point_maps=_world_point_maps(depth, intrinsics, view_c2w),
            geometry_confidence=confidence,
            scale_info={"scale": 1.0, "scale_is_relative": False},
            diagnostics={"backend": "ground_truth", "valid_ratio": float(valid.mean())},
        )


class Pi3GeometryBackend(MultiViewGeometryBackend):
    """Official Holo360D-finetuned Pi3 8-view model adapter."""

    def __init__(self, checkpoint, repo_path, device="cuda", input_size=518, default_median_depth=3.0):
        self.checkpoint = str(checkpoint)
        self.repo_path = str(repo_path)
        self.device = device
        self.input_size = int(input_size)
        self.default_median_depth = float(default_median_depth)
        self._model = None
        self._has_confidence_head = False

    def _load_model(self):
        import torch

        repo = Path(self.repo_path).resolve()
        if not repo.exists():
            raise FileNotFoundError(f"Pi3 repository not found: {repo}")
        if not Path(self.checkpoint).exists():
            raise FileNotFoundError(f"Pi3 checkpoint not found: {self.checkpoint}")
        if str(repo) not in sys.path:
            sys.path.insert(0, str(repo))
        from pi3.models.pi3 import Pi3

        model = Pi3().to(self.device).eval()
        weights = torch.load(self.checkpoint, map_location=self.device, weights_only=False)
        if any(key.startswith("module.") for key in weights):
            weights = {key.removeprefix("module."): value for key, value in weights.items()}
        missing, unexpected = model.load_state_dict(weights, strict=False)
        self._has_confidence_head = not any(key.startswith("conf_") for key in missing)
        self._model = model
        return {
            "missing_key_count": len(missing),
            "unexpected_key_count": len(unexpected),
            "checkpoint_has_confidence_head": self._has_confidence_head,
        }

    @staticmethod
    def _known_z_depth(known_depth, intrinsics):
        depth = np.asarray(known_depth, np.float32)
        result = np.empty_like(depth)
        for index in range(len(depth)):
            height, width = depth[index].shape
            yy, xx = np.indices((height, width), np.float32)
            k = intrinsics[index]
            ray_norm = np.sqrt(((xx-k[0,2])/k[0,0])**2 + ((yy-k[1,2])/k[1,1])**2 + 1.0)
            result[index] = depth[index] / ray_norm
        return result

    def predict(self, view_rgb, view_c2w, intrinsics, known_depth=None, known_mask=None):
        import torch
        import torch.nn.functional as functional

        if len(view_rgb) != 8:
            raise ValueError(f"The Holo360D Pi3 checkpoint expects exactly 8 views, got {len(view_rgb)}")
        load_diagnostics = self._load_model() if self._model is None else {}
        rgb = np.asarray(view_rgb)
        images = torch.from_numpy(rgb).to(self.device)
        if images.dtype == torch.uint8:
            images = images.float().div_(255.0)
        else:
            images = images.float().clamp_(0, 1)
        images = images.permute(0, 3, 1, 2)
        original_hw = tuple(images.shape[-2:])
        images = functional.interpolate(images, (self.input_size, self.input_size), mode="bilinear", align_corners=False)
        dtype = torch.bfloat16 if images.device.type == "cuda" else torch.float32
        with torch.inference_mode(), torch.amp.autocast(
            device_type=images.device.type, dtype=dtype, enabled=images.device.type == "cuda"
        ):
            result = self._model(images[None])

        predicted_z = result["local_points"][0, ..., 2].float().unsqueeze(1)
        if self._has_confidence_head:
            raw_confidence = result["conf"][0, ..., 0].float().sigmoid().unsqueeze(1)
            confidence_source = "pi3_confidence_head"
        else:
            # Holo360D's released 8views.bin omits the confidence decoder.
            # Use deterministic local depth continuity rather than random head output.
            horizontal = functional.pad(
                (predicted_z[..., 1:] - predicted_z[..., :-1]).abs(), (0, 1, 0, 0)
            )
            vertical = functional.pad(
                (predicted_z[..., 1:, :] - predicted_z[..., :-1, :]).abs(), (0, 0, 0, 1)
            )
            relative_gradient = torch.maximum(horizontal, vertical) / predicted_z.clamp_min(1e-4)
            raw_confidence = torch.exp(-4.0 * relative_gradient).clamp(0, 1)
            confidence_source = "local_depth_continuity"
        predicted_z = functional.interpolate(predicted_z, original_hw, mode="bilinear", align_corners=False)[:, 0]
        confidence = functional.interpolate(raw_confidence, original_hw, mode="bilinear", align_corners=False)[:, 0]
        predicted_z = predicted_z.cpu().numpy().astype(np.float32)
        confidence = confidence.cpu().numpy().astype(np.float32)
        predicted_c2w = result["camera_poses"][0].float().cpu().numpy().astype(np.float32)

        valid = np.isfinite(predicted_z) & (predicted_z > 0)
        if known_mask is not None:
            valid &= np.asarray(known_mask, bool)
        scale_is_relative = known_depth is None
        if known_depth is not None:
            known_z = self._known_z_depth(known_depth, np.asarray(intrinsics, np.float32))
            aligned = valid & np.isfinite(known_z) & (known_z > 0)
            if not aligned.any():
                raise ValueError("No valid overlap between Pi3 prediction and known depth")
            scale = float(np.median(known_z[aligned] / predicted_z[aligned]))
        else:
            median = float(np.median(predicted_z[valid])) if valid.any() else 0.0
            if median <= 0:
                raise RuntimeError("Pi3 returned no valid positive depth")
            scale = self.default_median_depth / median
        depth = predicted_z * scale
        depth[~valid] = np.nan
        confidence[~valid] = 0.0
        diagnostics = {
            "backend": "pi3_holo360d_8views",
            "input_size": self.input_size,
            "valid_ratio": float(valid.mean()),
            "checkpoint": self.checkpoint,
            "confidence_source": confidence_source,
            **load_diagnostics,
        }
        if known_depth is not None:
            error_valid = valid & np.isfinite(known_z) & (known_z > 0)
            absolute = np.abs(depth[error_valid] - known_z[error_valid])
            diagnostics.update(
                depth_mae_m=float(absolute.mean()),
                depth_abs_rel=float((absolute / known_z[error_valid]).mean()),
            )
        return GeometryPrediction(
            depth=depth,
            depth_confidence=confidence,
            point_maps=_world_point_maps(depth, intrinsics, view_c2w),
            predicted_c2w=predicted_c2w,
            geometry_confidence=confidence.copy(),
            scale_info={"scale": scale, "scale_is_relative": scale_is_relative},
            diagnostics=diagnostics,
        )
