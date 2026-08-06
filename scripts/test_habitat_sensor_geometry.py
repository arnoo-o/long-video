#!/usr/bin/env python3
"""Validate Habitat sensor c2w, Z-depth backprojection, and navmesh metadata."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np

from long_video.geometry.backprojection import backproject_z_depth


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--sequence",type=Path,required=True)
    parser.add_argument("--output",type=Path)
    args=parser.parse_args()
    poses=np.load(args.sequence/"poses_c2w.npy").astype(np.float32)
    intrinsics=np.load(args.sequence/"intrinsics.npy").astype(np.float32)
    depth=np.load(args.sequence/"depth"/"000000.npy").astype(np.float32)
    metadata=json.loads((args.sequence/"metadata.json").read_text(encoding="utf-8"))
    valid=np.isfinite(depth)&(depth>0)
    camera=backproject_z_depth(depth,intrinsics[0])
    world=camera@poses[0,:3,:3].T+poses[0,:3,3]
    reconstructed=(world-poses[0,:3,3])@poses[0,:3,:3]
    projected=reconstructed@intrinsics[0].T
    uv=projected[...,:2]/projected[...,2:3]
    y,x=np.indices(depth.shape,dtype=np.float32)
    pixel_error=np.linalg.norm(uv-np.stack([x,y],axis=-1),axis=-1)
    rotation_error=np.linalg.norm(poses[0,:3,:3].T@poses[0,:3,:3]-np.eye(3))
    trajectory=metadata.get("trajectory_validation",{})
    result={
        "frames":int(len(poses)),
        "valid_depth_ratio":float(valid.mean()),
        "pixel_reprojection_max":float(pixel_error[valid].max()),
        "pixel_reprojection_mean":float(pixel_error[valid].mean()),
        "z_depth_roundtrip_max":float(np.abs(reconstructed[...,2][valid]-depth[valid]).max()),
        "rotation_orthogonality_error":float(rotation_error),
        "poses_are_sensor_c2w":bool(metadata.get("poses_are_sensor_c2w")),
        "depth_convention":metadata.get("depth_convention"),
        "navmesh_loaded":bool(trajectory.get("navmesh_loaded")),
        "collision_checked":bool(trajectory.get("collision_checked")),
        "max_navmesh_correction":float(trajectory.get("max_position_correction",float("nan"))),
    }
    checks=[
        result["poses_are_sensor_c2w"],result["depth_convention"]=="Z_DEPTH",
        result["navmesh_loaded"],result["collision_checked"],
        result["valid_depth_ratio"]>0.1,result["pixel_reprojection_max"]<1e-3,
        result["z_depth_roundtrip_max"]<1e-5,result["rotation_orthogonality_error"]<1e-5,
    ]
    result["passed"]=bool(all(checks))
    text=json.dumps(result,indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(text,encoding="utf-8")
    print(text)
    raise SystemExit(0 if result["passed"] else 1)


if __name__=="__main__":
    main()