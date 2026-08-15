"""Frozen official Pi3X adapter for a strictly source-only W0."""
from __future__ import annotations

from pathlib import Path
import sys
import numpy as np

from ..geometry.backprojection import backproject_z_depth
from ..types import Z_DEPTH
from .geometry_backend import GeometryPrediction, MultiViewGeometryBackend


class Pi3XGeometryBackend(MultiViewGeometryBackend):
    """Pi3X source-only geometry; no prior RGB/depth/geometry is accepted."""

    def __init__(self, checkpoint, repo_path, device):
        self.checkpoint, self.repo_path, self.device = str(checkpoint), str(repo_path), str(device)
        self._model = None

    def _load(self):
        if self._model is not None:
            return
        import torch
        repo = Path(self.repo_path).resolve()
        if not repo.is_dir() or not Path(self.checkpoint).is_file():
            raise FileNotFoundError("Pi3X repo and checkpoint are required")
        if str(repo) not in sys.path:
            sys.path.insert(0, str(repo))
        from pi3.models.pi3x import Pi3X
        from safetensors.torch import load_file
        model = Pi3X(use_multimodal=True).to(self.device).eval()
        missing, unexpected = model.load_state_dict(load_file(self.checkpoint), strict=False)
        if unexpected:
            raise RuntimeError(f"Pi3X checkpoint has unexpected keys: {unexpected[:8]}")
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        self._model = model

    def predict_source(self, rgb, c2w, intrinsics):
        import torch
        self._load()
        rgb, c2w, intrinsics = np.asarray(rgb, np.uint8), np.asarray(c2w, np.float32), np.asarray(intrinsics, np.float32)
        if rgb.ndim != 3 or c2w.shape != (4, 4) or intrinsics.shape != (3, 3):
            raise ValueError("Pi3X W0 accepts exactly one source RGB/c2w/intrinsics")
        image = torch.from_numpy(rgb).to(self.device).permute(2, 0, 1).float().div(255)[None, None]
        pose = torch.from_numpy(c2w).to(self.device)[None, None]
        k = torch.from_numpy(intrinsics).to(self.device)[None, None]
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16, enabled="cuda" in self.device):
            result = self._model(imgs=image, poses=pose, intrinsics=k, depths=None)
        local = result["local_points"][0, 0].float().cpu().numpy().astype(np.float32)
        raw_conf = result["conf"][0, 0, ..., 0].float().sigmoid().cpu().numpy().astype(np.float32)
        depth = local[..., 2]
        valid = np.isfinite(depth) & (depth > 0) & np.isfinite(raw_conf) & (raw_conf > 0)
        depth = np.where(valid, depth, np.nan).astype(np.float32)
        confidence = np.where(valid, raw_conf, 0).astype(np.float32)
        world = backproject_z_depth(depth, intrinsics) @ c2w[:3, :3].T + c2w[:3, 3]
        return GeometryPrediction(depth=depth[None], depth_confidence=confidence[None],
            point_maps=world[None].astype(np.float32), geometry_confidence=confidence[None],
            depth_convention=Z_DEPTH,
            scale_info={"mode": "relative", "meters_per_world_unit": None, "uncertainty": 1.0,
                        "anchor_source": "pi3x_source_only"},
            diagnostics={"backend": "official_pi3x_source_only", "valid_ratio": float(valid.mean()),
                         "checkpoint": self.checkpoint, "uses_only_source": True})

    def predict(self, view_rgb, view_c2w, intrinsics, **kwargs):
        if len(view_rgb) != 1 or kwargs:
            raise ValueError("Pi3X initial W0 forbids non-source views and auxiliary geometry")
        return self.predict_source(view_rgb[0], view_c2w[0], intrinsics[0])
