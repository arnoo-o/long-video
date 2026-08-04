"""File/subprocess adapter for the official MVDiffusion environment."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import tempfile

import numpy as np
from PIL import Image

from ..types import RAY_DISTANCE, ViewSet


class MVDiffusionError(RuntimeError):
    pass


class MVDiffusionCompletion:
    def __init__(
        self,
        repo_path,
        python_executable,
        checkpoint,
        runner_script=None,
        base_model_path=None,
        synthesized_confidence=0.4,
        timeout_seconds=1800,
    ):
        self.repo_path = Path(repo_path)
        self.python_executable = Path(python_executable)
        self.checkpoint = Path(checkpoint)
        self.runner_script = Path(runner_script) if runner_script else (
            Path(__file__).resolve().parents[2] / "scripts" / "run_mvdiffusion_completion.py"
        )
        self.base_model_path = None if base_model_path is None else Path(base_model_path)
        self.synthesized_confidence = float(synthesized_confidence)
        self.timeout_seconds = int(timeout_seconds)

    def _validate(self):
        missing = [
            path
            for path in (self.repo_path, self.python_executable, self.checkpoint, self.runner_script)
            if not path.exists()
        ]
        if missing:
            raise MVDiffusionError("Missing MVDiffusion paths: " + ", ".join(map(str, missing)))

    @staticmethod
    def _materialize_images(images, work_dir):
        paths = []
        for index, image in enumerate(images):
            if isinstance(image, (str, os.PathLike)):
                path = Path(image).resolve()
            else:
                path = work_dir / f"observed_{index:02d}.png"
                array = np.asarray(image)
                if array.dtype != np.uint8:
                    array = (np.clip(array, 0, 1) * 255).round().astype(np.uint8)
                Image.fromarray(array).save(path)
            if not path.exists():
                raise FileNotFoundError(f"Observed image does not exist: {path}")
            paths.append(path)
        return paths

    def complete(
        self,
        observed_images,
        observed_camera_specs,
        prompt,
        output_dir=None,
        target_yaws_degrees=(0, 45, 90, 135, 180, 225, 270, 315),
        target_pitch_degrees=0.0,
        target_fov_degrees=90.0,
        height=512,
        width=512,
    ):
        self._validate()
        if len(observed_images) != len(observed_camera_specs) or not observed_images:
            raise ValueError("observed_images and observed_camera_specs must have equal nonzero length")
        with tempfile.TemporaryDirectory(prefix="long_video_mvdiffusion_") as temporary:
            work_dir = Path(temporary)
            image_paths = self._materialize_images(observed_images, work_dir)
            result_dir = Path(output_dir).resolve() if output_dir else work_dir / "result"
            manifest = {
                "prompt": str(prompt),
                "observed_views": [
                    {
                        "image_path": str(path),
                        "yaw_degrees": float(spec["yaw_degrees"]),
                        "pitch_degrees": float(spec.get("pitch_degrees", 0.0)),
                        "fov_degrees": float(spec["fov_degrees"]),
                    }
                    for path, spec in zip(image_paths, observed_camera_specs)
                ],
                "target_yaws_degrees": list(map(float, target_yaws_degrees)),
                "target_pitch_degrees": float(target_pitch_degrees),
                "target_fov_degrees": float(target_fov_degrees),
                "height": int(height),
                "width": int(width),
                "output_dir": str(result_dir),
                "mvdiffusion_repo": str(self.repo_path.resolve()),
                "checkpoint": str(self.checkpoint.resolve()),
                "base_model_path": None if self.base_model_path is None else str(self.base_model_path.resolve()),
                "synthesized_confidence": self.synthesized_confidence,
            }
            manifest_path = work_dir / "manifest.json"
            manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            command = [str(self.python_executable), str(self.runner_script), "--manifest", str(manifest_path)]
            completed = subprocess.run(
                command, cwd=self.repo_path, capture_output=True, text=True, timeout=self.timeout_seconds
            )
            if completed.returncode:
                raise MVDiffusionError(
                    f"MVDiffusion failed: command={command}; manifest={manifest}; "
                    f"stdout={completed.stdout}; stderr={completed.stderr}"
                )
            required = ("views_rgb.npy", "view_poses.npy", "intrinsics.npy", "observed_masks.npy",
                        "source_maps.npy", "image_confidence.npy", "metadata.json")
            missing = [name for name in required if not (result_dir / name).exists()]
            if missing:
                raise MVDiffusionError(f"MVDiffusion output is incomplete in {result_dir}: {missing}")
            rgb = np.load(result_dir / "views_rgb.npy")
            c2w = np.load(result_dir / "view_poses.npy").astype(np.float32)
            intrinsics = np.load(result_dir / "intrinsics.npy").astype(np.float32)
            source = np.load(result_dir / "source_maps.npy").astype(np.int8)
            confidence = np.load(result_dir / "image_confidence.npy").astype(np.float32)
            depth = np.full(rgb.shape[:3], np.nan, np.float32)
            return ViewSet(
                rgb=rgb, depth=depth, depth_confidence=np.zeros_like(depth), c2w=c2w,
                intrinsics=intrinsics, source=source, image_confidence=confidence,
                depth_convention=RAY_DISTANCE,
            )
