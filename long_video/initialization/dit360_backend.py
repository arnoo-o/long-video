"""Subprocess adapter for official DiT360 panorama completion."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import tempfile

import numpy as np
from PIL import Image, ImageOps

from ..data.camera import rgb_to_uint8
from ..types import ViewSet, Z_DEPTH


class DiT360Error(RuntimeError):
    pass


class DiT360Completion:
    def __init__(self, repo_path, python_executable, base_model_path, lora_path,
                 runner_script=None, synthesized_confidence=0.4,
                 timeout_seconds=3600, same_center_tolerance=1e-4):
        self.repo_path=Path(repo_path)
        self.python_executable=Path(python_executable)
        self.base_model_path=str(base_model_path)
        self.lora_path=str(lora_path)
        self.runner_script=(Path(runner_script) if runner_script else
            Path(__file__).resolve().parents[2]/"scripts"/"run_dit360_completion.py")
        self.synthesized_confidence=float(synthesized_confidence)
        self.timeout_seconds=int(timeout_seconds)
        self.same_center_tolerance=float(same_center_tolerance)

    @classmethod
    def from_config(cls,config):
        return cls(
            repo_path=config["repo_path"],
            python_executable=config["python_executable"],
            base_model_path=config["base_model_path"],
            lora_path=config["lora_path"],
            runner_script=config.get("runner_script"),
            synthesized_confidence=config["synthesized_confidence"],
            timeout_seconds=config["timeout_seconds"],
            same_center_tolerance=config["same_center_tolerance"],
        )

    def _validate_paths(self):
        missing=[p for p in (self.repo_path,self.python_executable,self.runner_script)
                 if not p.exists()]
        for model in (self.base_model_path,self.lora_path):
            if os.path.sep in model and not Path(model).exists(): missing.append(Path(model))
        if missing:
            raise DiT360Error("Missing DiT360 paths: "+", ".join(map(str,missing)))

    @staticmethod
    def _validate_spec(spec,image_size):
        width,height=image_size
        if "yaw_degrees" not in spec:
            raise ValueError("Each observation requires yaw_degrees")
        if spec.get("distortion_model","none") not in (None,"none"):
            raise ValueError("Distorted input requires prior undistortion")
        coefficients=np.asarray(spec.get("distortion_coefficients",[]),np.float64)
        if coefficients.size and np.any(np.abs(coefficients)>1e-9):
            raise ValueError("Non-zero lens distortion is not supported")
        if "intrinsics" in spec:
            k=np.asarray(spec["intrinsics"],np.float64)
            if k.shape!=(3,3) or not np.isfinite(k).all():
                raise ValueError("intrinsics must be finite [3,3]")
            if min(k[0,0],k[1,1])<=0 or abs(k[0,1])>1e-6:
                raise ValueError("Only pinhole intrinsics without skew are supported")
            if not (0<=k[0,2]<width and 0<=k[1,2]<height):
                raise ValueError("principal point lies outside the EXIF-corrected image")
        elif float(spec.get("fov_degrees",0))<=0:
            raise ValueError("Each observation requires intrinsics or a positive FOV")

    def _validate_same_center(self,specs):
        centers=[]
        for spec in specs:
            if "c2w" in spec:
                pose=np.asarray(spec["c2w"],np.float64)
                if pose.shape!=(4,4): raise ValueError("c2w must be [4,4]")
                centers.append(pose[:3,3])
        if centers and np.ptp(np.stack(centers),axis=0).max()>self.same_center_tolerance:
            raise ValueError("DiT360Completion requires same-optical-center observations")

    @staticmethod
    def _materialize(images,specs,work):
        paths=[]
        sizes=[]
        for index,(item,spec) in enumerate(zip(images,specs)):
            if isinstance(item,(str,os.PathLike)):
                source=Path(item)
                if not source.exists(): raise FileNotFoundError(source)
                with Image.open(source) as raw:
                    orientation=int(raw.getexif().get(274,1) or 1)
                    if orientation!=1 and "intrinsics" in spec:
                        raise ValueError("EXIF-oriented inputs with explicit intrinsics must be normalized first")
                    image=ImageOps.exif_transpose(raw).convert("RGB")
            else:
                image=Image.fromarray(rgb_to_uint8(item))
            path=work/f"observed_{index:02d}.png"
            image.save(path)
            paths.append(path); sizes.append(image.size)
        return paths,sizes

    def complete(self,observed_images,observed_camera_specs,prompt,output_dir=None,
                 target_yaws_degrees=(0,45,90,135,180,225,270,315),
                 target_pitch_degrees=0.0,target_fov_degrees=90.0,
                 height=512,width=512,prepare_only=False):
        self._validate_paths()
        if not observed_images or len(observed_images)!=len(observed_camera_specs):
            raise ValueError("observed images/specs must have equal non-zero length")
        self._validate_same_center(observed_camera_specs)
        with tempfile.TemporaryDirectory(prefix="long_video_dit360_") as temporary:
            work=Path(temporary); paths,sizes=self._materialize(
                observed_images,observed_camera_specs,work
            )
            for spec,size in zip(observed_camera_specs,sizes): self._validate_spec(spec,size)
            result=Path(output_dir).resolve() if output_dir else work/"result"
            observations=[]
            for path,spec in zip(paths,observed_camera_specs):
                item=dict(spec); item["image_path"]=str(path)
                if "intrinsics" in item:
                    item["intrinsics"]=np.asarray(item["intrinsics"],np.float32).tolist()
                if "c2w" in item: item["c2w"]=np.asarray(item["c2w"],np.float32).tolist()
                observations.append(item)
            manifest={
                "prompt":str(prompt),"observed_views":observations,
                "erp_height":1024,"erp_width":2048,
                "target_yaws_degrees":list(map(float,target_yaws_degrees)),
                "target_pitch_degrees":float(target_pitch_degrees),
                "target_fov_degrees":float(target_fov_degrees),
                "height":int(height),"width":int(width),"output_dir":str(result),
                "dit360_repo":str(self.repo_path.resolve()),
                "base_model_path":self.base_model_path,"lora_path":self.lora_path,
                "synthesized_confidence":self.synthesized_confidence,
                "prepare_only":bool(prepare_only),
            }
            manifest_path=work/"manifest.json"
            manifest_path.write_text(json.dumps(manifest,indent=2),encoding="utf-8")
            command=[str(self.python_executable),str(self.runner_script),"--manifest",str(manifest_path)]
            completed=subprocess.run(command,cwd=self.repo_path,capture_output=True,text=True,
                                     timeout=self.timeout_seconds)
            if completed.returncode:
                raise DiT360Error(
                    f"DiT360 command failed ({completed.returncode}); manifest={manifest_path}; "
                    f"stdout={completed.stdout[-4000:]}; stderr={completed.stderr[-4000:]}"
                )
            required=("views_rgb.npy","view_poses.npy","intrinsics.npy",
                      "observed_masks.npy","source_maps.npy","image_confidence.npy",
                      "erp_rgb.npy","erp_valid_mask.npy","erp_fusion_weight.npy",
                      "erp_conflict.npy","metadata.json")
            missing=[name for name in required if not (result/name).exists()]
            if missing: raise DiT360Error(f"Incomplete DiT360 output: {missing}")
            rgb=np.load(result/"views_rgb.npy")
            depth=np.full(rgb.shape[:3],np.nan,np.float32)
            return ViewSet(
                rgb=rgb,depth=depth,depth_confidence=np.zeros_like(depth),
                c2w=np.load(result/"view_poses.npy").astype(np.float32),
                intrinsics=np.load(result/"intrinsics.npy").astype(np.float32),
                source=np.load(result/"source_maps.npy").astype(np.int8),
                image_confidence=np.load(result/"image_confidence.npy").astype(np.float32),
                depth_convention=Z_DEPTH,
            )
