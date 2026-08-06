#!/usr/bin/env python3
"""Validate the released Holo 8-view Pi3 checkpoint on translated Habitat views."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image

from long_video.initialization.geometry_backend import Pi3GeometryBackend
from long_video.types import Z_DEPTH, ScaleMetadata


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--sequence",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--repo",type=Path,required=True)
    parser.add_argument("--checkpoint",type=Path,required=True)
    parser.add_argument("--device",required=True)
    args=parser.parse_args()
    poses=np.load(args.sequence/"poses_c2w.npy")
    intrinsics=np.load(args.sequence/"intrinsics.npy")
    indices=np.linspace(0,len(poses)-1,8,dtype=int)
    rgb=np.stack([np.asarray(Image.open(args.sequence/"rgb"/f"{i:06d}.png").convert("RGB"))
                  for i in indices])
    depth=np.stack([np.load(args.sequence/"depth"/f"{i:06d}.npy") for i in indices])
    mask=np.isfinite(depth)&(depth>0)
    backend=Pi3GeometryBackend(args.checkpoint,args.repo,device=args.device,input_size=518)
    start=time.time()
    prediction=backend.predict(
        rgb,poses[indices],intrinsics[indices],depth,mask,Z_DEPTH,
        ScaleMetadata("dataset_calibrated",1.0,0.0,"Habitat_sensor_depth"),
    )
    elapsed=time.time()-start
    valid=mask&np.isfinite(prediction.depth)
    centers=poses[indices,:3,3]
    absolute=np.abs(prediction.depth[valid]-depth[valid])
    metrics={
        "indices":indices.tolist(),
        "elapsed_seconds":elapsed,
        "translation_baseline":float(np.linalg.norm(centers[:,None]-centers[None,:],axis=-1).max()),
        "depth_mae":float(absolute.mean()),
        "depth_abs_rel":float((absolute/depth[valid]).mean()),
        "ground_truth_overlap_ratio":float(valid.mean()),
        "predicted_valid_ratio":float(np.isfinite(prediction.depth).mean()),
        **prediction.diagnostics,
        "scale":prediction.scale_info,
    }
    metrics["suitable_for_translated_known_pose_views"]=bool(
        metrics["predicted_valid_ratio"]>0.9 and
        metrics["ground_truth_overlap_ratio"]>0.2 and metrics["depth_abs_rel"]<0.25 and
        metrics.get("pose_error",float("inf"))<0.25)
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/"metrics.json").write_text(json.dumps(metrics,indent=2),encoding="utf-8")
    print(json.dumps(metrics,indent=2))


if __name__=="__main__":
    main()
