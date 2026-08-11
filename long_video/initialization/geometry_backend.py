"""Unified multi-view geometry backends.

Pi3 is loaded lazily from its official 8-view checkout so the core package
remains importable without the third-party model environment.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import sys

import numpy as np

from ..geometry.backprojection import backproject_z_depth
from ..types import RAY_DISTANCE, Z_DEPTH, ScaleMetadata
from ..data.camera import resize_intrinsics


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
    def predict(self, view_rgb, view_c2w, intrinsics, known_depth=None, known_mask=None,
                known_depth_convention=None, known_scale=None):
        raise NotImplementedError


def _world_point_maps(depth, intrinsics, c2w):
    result = []
    for index in range(len(depth)):
        local = backproject_z_depth(depth[index], intrinsics[index])
        pose = np.asarray(c2w[index], np.float32)
        result.append(local @ pose[:3, :3].T + pose[:3, 3])
    return np.stack(result).astype(np.float32)
def _robust_scale(predicted_z,known_z,mask,min_pixels=32):
    ratios=(known_z[mask]/predicted_z[mask]).astype(np.float64)
    ratios=ratios[np.isfinite(ratios)&(ratios>0)]
    if len(ratios)<min_pixels:
        raise ValueError(f"Scale anchor rejected: {len(ratios)} valid pixels < {min_pixels}")
    median=float(np.median(ratios))
    absolute=np.abs(ratios-median); mad=float(np.median(absolute))
    if mad>0:
        keep=absolute<=4.5*1.4826*mad
        ratios=ratios[keep]
        if len(ratios)<min_pixels:
            raise ValueError(f"Scale anchor rejected after MAD filtering: {len(ratios)} pixels")
        median=float(np.median(ratios)); mad=float(np.median(np.abs(ratios-median)))
    return median,mad,int(len(ratios))


def _pose_consistency(predicted_c2w,control_c2w):
    predicted=np.asarray(predicted_c2w,np.float64); control=np.asarray(control_c2w,np.float64)
    if predicted.shape[-2:]==(3,4):
        bottom=np.broadcast_to(np.array([0,0,0,1],np.float64),predicted.shape[:-2]+(1,4))
        predicted=np.concatenate([predicted,bottom],axis=-2)
    if predicted.shape!=control.shape:
        return {"pose_error":float("inf"),"pose_rejection_reason":"pose shape mismatch"}
    rotation_errors=[]
    for index in range(1,len(control)):
        pr=np.linalg.inv(predicted[0])@predicted[index]
        cr=np.linalg.inv(control[0])@control[index]
        cosine=np.clip((np.trace(pr[:3,:3].T@cr[:3,:3])-1)/2,-1,1)
        rotation_errors.append(np.arccos(cosine))
    pd=np.linalg.norm(predicted[:,None,:3,3]-predicted[None,:,:3,3],axis=-1)
    cd=np.linalg.norm(control[:,None,:3,3]-control[None,:,:3,3],axis=-1)
    usable=(pd>1e-6)&(cd>1e-6)
    scale=float(np.median(cd[usable]/pd[usable])) if usable.any() else None
    dispersion=(float(np.median(np.abs(cd[usable]/pd[usable]-scale))/max(scale,1e-8))
                if usable.any() else 0.0)
    rotation=float(np.mean(rotation_errors)) if rotation_errors else 0.0
    return {"pose_error":float(rotation/np.pi+dispersion),
            "pose_rotation_error_rad":rotation,"pose_scale":scale,
            "pose_scale_dispersion":dispersion,
            "predicted_translation_baseline":float(pd.max()),
            "control_translation_baseline":float(cd.max())}




class GroundTruthGeometryBackend(MultiViewGeometryBackend):
    def predict(self, view_rgb, view_c2w, intrinsics, known_depth=None, known_mask=None,
                known_depth_convention=None, known_scale=None):
        if known_depth is None:
            raise ValueError("GroundTruthGeometryBackend requires known_depth")
        depth = np.asarray(known_depth, np.float32).copy()
        if known_depth_convention not in (RAY_DISTANCE, Z_DEPTH):
            raise ValueError("known_depth_convention must be RAY_DISTANCE or Z_DEPTH")
        valid = np.isfinite(depth) & (depth > 0)
        if known_mask is not None:
            valid &= np.asarray(known_mask, bool)
        depth[~valid] = np.nan
        confidence = valid.astype(np.float32)
        if known_depth_convention == Z_DEPTH:
            point_maps = _world_point_maps(depth, intrinsics, view_c2w)
        else:
            from ..geometry.backprojection import backproject_ray_distance
            point_maps = []
            for index in range(len(depth)):
                local = backproject_ray_distance(depth[index], intrinsics[index])
                pose = np.asarray(view_c2w[index], np.float32)
                point_maps.append(local @ pose[:3,:3].T + pose[:3,3])
            point_maps = np.stack(point_maps).astype(np.float32)
        return GeometryPrediction(
            depth=depth, depth_confidence=confidence, point_maps=point_maps,
            geometry_confidence=confidence,
            scale_info={"mode": "dataset_calibrated", "meters_per_world_unit": 1.0,
                        "uncertainty": 0.0, "anchor_source": "ground_truth_depth"},
            diagnostics={"backend": "ground_truth", "valid_ratio": float(valid.mean())},
            depth_convention=known_depth_convention,
        )


class Pi3GeometryBackend(MultiViewGeometryBackend):
    """Frozen official Pi3 8-view geometry adapter."""

    def __init__(self, checkpoint, repo_path, device, input_size=518):
        self.checkpoint = str(checkpoint)
        self.repo_path = str(repo_path)
        self.device = device
        self.input_size = int(input_size)
        self._model = None
        self._has_confidence_head = False

    @classmethod
    def from_config(cls,config):
        return cls(
            checkpoint=config["checkpoint"],repo_path=config["repo_path"],
            device=config["device"],input_size=config["input_size"],
        )


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
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        weights = torch.load(self.checkpoint, map_location=self.device, weights_only=False)
        if any(key.startswith("module.") for key in weights):
            weights = {key.removeprefix("module."): value for key, value in weights.items()}
        missing, unexpected = model.load_state_dict(weights, strict=False)
        allowed_missing=("conf_","conf_decoder","confidence")
        illegal_missing=[key for key in missing if not key.startswith(allowed_missing)]
        if illegal_missing or unexpected:
            raise RuntimeError(
                f"Pi3 checkpoint mismatch: illegal missing={illegal_missing}, "
                f"unexpected={unexpected}"
            )
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

    def predict(self, view_rgb, view_c2w, intrinsics, known_depth=None, known_mask=None,
                known_depth_convention=None, known_scale=None):
        import torch
        import torch.nn.functional as functional

        if len(view_rgb) != 8:
            raise ValueError(f"The Pi3 checkpoint expects exactly 8 views, got {len(view_rgb)}")
        load_diagnostics = self._load_model() if self._model is None else {}
        rgb = np.asarray(view_rgb)
        images = torch.from_numpy(rgb).to(self.device)
        if images.dtype == torch.uint8:
            images = images.float().div_(255.0)
        else:
            images = images.float().clamp_(0, 1)
        images = images.permute(0, 3, 1, 2)
        if known_depth is not None and known_depth_convention not in (RAY_DISTANCE, Z_DEPTH):
            raise ValueError(
                "known_depth_convention is required when known_depth is provided"
            )
        original_hw = tuple(images.shape[-2:])
        original_k = np.asarray(intrinsics, np.float32)
        image_scale = self.input_size / max(original_hw)
        resized_hw = (max(1, round(original_hw[0]*image_scale)),
                      max(1, round(original_hw[1]*image_scale)))
        images = functional.interpolate(images, resized_hw, mode="bilinear", align_corners=False)
        pad_y = self.input_size-resized_hw[0]; pad_x = self.input_size-resized_hw[1]
        top, left = pad_y//2, pad_x//2
        images = functional.pad(images, (left,pad_x-left,top,pad_y-top))
        model_k = resize_intrinsics(original_k, original_hw, resized_hw)
        model_k[...,0,2] += left; model_k[...,1,2] += top
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
            # The released 8-view checkpoint may omit the confidence decoder.
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
        predicted_z = predicted_z[...,top:top+resized_hw[0],left:left+resized_hw[1]]
        raw_confidence = raw_confidence[...,top:top+resized_hw[0],left:left+resized_hw[1]]
        predicted_z = functional.interpolate(predicted_z, original_hw, mode="bilinear", align_corners=False)[:,0]
        confidence = functional.interpolate(raw_confidence, original_hw, mode="bilinear", align_corners=False)[:,0]
        predicted_z = predicted_z.cpu().numpy().astype(np.float32)
        confidence = confidence.cpu().numpy().astype(np.float32)
        predicted_c2w = result["camera_poses"][0].float().cpu().numpy().astype(np.float32)

        valid = np.isfinite(predicted_z) & (predicted_z > 0)
        scale_is_relative = known_depth is None
        if known_depth is not None:
            known_z = (self._known_z_depth(known_depth, original_k)
                       if known_depth_convention == RAY_DISTANCE
                       else np.asarray(known_depth, np.float32))
            aligned = valid & np.isfinite(known_z) & (known_z > 0)
            if known_mask is not None:
                aligned &= np.asarray(known_mask, bool)
            scale,scale_mad,scale_pixels=_robust_scale(predicted_z,known_z,aligned)
        else:
            median = float(np.median(predicted_z[valid])) if valid.any() else 0.0
            if median <= 0:
                raise RuntimeError("Pi3 returned no valid positive depth")
            # Same-center rotations cannot determine meters. One median depth is one node unit.
            scale = 1.0 / median
        depth = predicted_z * scale
        depth[~valid] = np.nan
        confidence[~valid] = 0.0
        diagnostics = {
            "backend": "pi3_8view",
            "input_size": self.input_size,
            "valid_ratio": float(valid.mean()),
            "checkpoint": self.checkpoint,
            "confidence_source": confidence_source,
            "confidence_type": "model_head" if self._has_confidence_head else "heuristic",
            **load_diagnostics,
        }
        diagnostics.update(_pose_consistency(predicted_c2w,np.asarray(view_c2w,np.float32)))
        diagnostics.update({
            "preprocess_original_hw":list(original_hw),
            "preprocess_resized_hw":list(resized_hw),
            "preprocess_padding":[top,left,pad_y-top,pad_x-left],
            "model_intrinsics":model_k.tolist(),
        })
        if known_depth is not None:
            error_valid = aligned
            absolute = np.abs(depth[error_valid] - known_z[error_valid])
            diagnostics.update(
                depth_mae_known_units=float(absolute.mean()),
                depth_abs_rel=float((absolute / known_z[error_valid]).mean()),
            )
        if scale_is_relative:
            scale_info={
                "mode":"relative","meters_per_world_unit":None,"uncertainty":1.0,
                "anchor_source":"same_center_pi3_median_normalization",
                "normalization_scale":scale,
            }
        else:
            parent_scale=(vars(known_scale) if isinstance(known_scale,ScaleMetadata)
                          else dict(known_scale or {}))
            parent_mode=parent_scale.get("mode","relative")
            metric=parent_mode in {"metric_anchor","dataset_calibrated"}
            scale_info={
                "mode":parent_mode if metric else "relative",
                "meters_per_world_unit":parent_scale.get("meters_per_world_unit") if metric else None,
                "uncertainty":min(1.0,float(scale_mad/max(scale,1e-8))),
                "anchor_source":("known_metric_depth_overlap" if metric else "parent_relative_depth_overlap"),
                "normalization_scale":scale,"anchor_valid_pixels":scale_pixels,
                "ratio_median":scale,"ratio_mad":scale_mad,"rejection_reason":None,
            }
        if (not scale_is_relative and scale_info["mode"] in {"metric_anchor","dataset_calibrated"}):
            diagnostics["depth_mae_m"]=diagnostics["depth_mae_known_units"]
        return GeometryPrediction(
            depth=depth,
            depth_confidence=confidence,
            point_maps=_world_point_maps(depth, intrinsics, view_c2w),
            predicted_c2w=predicted_c2w,
            geometry_confidence=confidence.copy(),
            scale_info=scale_info,
            diagnostics=diagnostics,
        )
