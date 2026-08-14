"""Causal recurrent adapter for the official ReCal3R/CUT3R inference stack."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
import hashlib

import numpy as np

from ..data.recal3r_full_scene import official_resize_crop, remap_model_map
from ..geometry.backprojection import backproject_z_depth
from ..types import Z_DEPTH
from .geometry_backend import GeometryPrediction, MultiViewGeometryBackend


@dataclass
class _Frame:
    rgb: np.ndarray
    c2w: np.ndarray
    intrinsics: np.ndarray


class ReCal3RGeometryBackend(MultiViewGeometryBackend):
    """Frozen official ReCal3R geometry with a trajectory-owned causal state.

    ReCal3R's public recurrent API exposes a full causal sequence interface;
    its public one-frame helper does not return the updated recurrent state.
    We therefore retain the generated-frame sequence and re-enter the official
    recurrent API on that causal prefix.  No future RGB or GT geometry is ever
    included, and the resulting state metadata is persistent across chunks.
    """

    def __init__(self, checkpoint, repo_path, device, confidence_threshold=1.5):
        self.checkpoint = str(checkpoint)
        self.repo_path = str(repo_path)
        self.device = str(device)
        self.confidence_threshold = float(confidence_threshold)
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
        self._seen = set()
        self._sequence_version = 0

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

    def update(self, rgb, c2w, intrinsics, **_kwargs):
        rgb, c2w, intrinsics = np.asarray(rgb), np.asarray(c2w, np.float32), np.asarray(intrinsics, np.float32)
        if len(rgb) != len(c2w) or len(rgb) != len(intrinsics):
            raise ValueError("ReCal3R RGB/camera/intrinsics must align")
        keys = [hashlib.sha256(np.asarray(image, np.uint8).tobytes()).hexdigest() for image in rgb]
        self._frames.extend(_Frame(np.asarray(image, np.uint8).copy(), pose.copy(), k.copy())
                            for image, pose, k in zip(rgb, c2w, intrinsics) if
                            hashlib.sha256(np.asarray(image, np.uint8).tobytes()).hexdigest() not in self._seen)
        self._seen.update(keys)
        if not self._frames:
            return self._prediction_for_keys(keys)
        self._load()
        views = [self._view(frame, index) for index, frame in enumerate(self._frames)]
        outputs, state_args = self._inference(views, self._model, self.device, verbose=False)
        self._state_args = state_args[-1] if state_args else None
        self._last_predictions = outputs["pred"]
        self._sequence_version += 1
        self._cache_all_results()
        return self._prediction_for_keys(keys)

    def _cache_all_results(self):
        for frame, prediction in zip(self._frames, self._last_predictions):
            key = hashlib.sha256(frame.rgb.tobytes()).hexdigest()
            self._results[key] = self._geometry_for(frame, prediction)

    def _geometry_for(self, frame, prediction):
        points = prediction["pts3d_in_self_view"].detach().cpu().numpy()[0]
        confidence = prediction["conf_self"].detach().cpu().numpy()[0]
        transform = official_resize_crop(*frame.rgb.shape[:2], 512)
        local, inside = remap_model_map(points, transform, interpolation=1)
        conf, _ = remap_model_map(confidence.astype(np.float32), transform, interpolation=1)
        depth = local[..., 2].astype(np.float32)
        valid = inside & np.isfinite(depth) & (depth > 0) & np.isfinite(conf) & (conf >= self.confidence_threshold)
        depth[~valid] = np.nan; conf = np.where(valid, conf, 0).astype(np.float32)
        camera = backproject_z_depth(depth, frame.intrinsics)
        return depth, conf, camera @ frame.c2w[:3, :3].T + frame.c2w[:3, 3]

    def _prediction_for_keys(self, keys):
        values = [self._results[key] for key in keys]
        depth, confidence, point_maps = map(np.stack, zip(*values))
        return GeometryPrediction(
            depth=depth, depth_confidence=confidence, point_maps=point_maps,
            geometry_confidence=confidence, depth_convention=Z_DEPTH,
            scale_info={"mode": "dataset_calibrated", "meters_per_world_unit": 1.0,
                        "uncertainty": 0.0, "anchor_source": "target_camera_constraint"},
            diagnostics={"backend": "official_recal3r_recurrent", **self.get_state(),
                         "valid_ratio": float(np.isfinite(depth).mean())},
        )

    def get_current_geometry(self, count=None):
        if not self._last_predictions:
            raise RuntimeError("ReCal3R state is empty")
        count = len(self._last_predictions) if count is None else int(count)
        keys = [hashlib.sha256(frame.rgb.tobytes()).hexdigest() for frame in self._frames[-count:]]
        return self._prediction_for_keys(keys)

    def predict(self, view_rgb, view_c2w, intrinsics, **_kwargs):
        return self.update(view_rgb, view_c2w, intrinsics)
