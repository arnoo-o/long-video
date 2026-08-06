#!/usr/bin/env python3
"""Indoor_016 sparse image -> DiT360 -> Pi3 -> relative-scale M0."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time

import numpy as np

from long_video.initialization.dit360_backend import DiT360Completion
class CompletedViews:
    def __init__(self,views): self.views=views
    def complete(self,*args,**kwargs): return self.views


from long_video.initialization.geometry_backend import Pi3GeometryBackend
from long_video.initialization.initial_node_pipeline import initialize_spatial_node
from long_video.memory.node_store import NodeStore


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--input",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--prepare-only",action="store_true")
    parser.add_argument("--dit-repo",type=Path,required=True)
    parser.add_argument("--dit-python",type=Path,required=True)
    parser.add_argument("--flux",type=Path,required=True)
    parser.add_argument("--lora",type=Path,required=True)
    parser.add_argument("--pi3-repo",type=Path,required=True)
    parser.add_argument("--pi3-checkpoint",type=Path,required=True)
    parser.add_argument("--device",required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    completion=DiT360Completion(
        args.dit_repo,args.dit_python,args.flux,args.lora,
        runner_script=Path(__file__).with_name("run_dit360_completion.py"))
    specs=[{"yaw_degrees":0.0,"pitch_degrees":0.0,"fov_degrees":90.0,
            "distortion_model":"none"}]
    start=time.time()
    views=completion.complete(
        [args.input],specs,
        "This is a panorama image of the same indoor room, photorealistic.",
        output_dir=args.output/"completion",prepare_only=args.prepare_only)
    completion_seconds=time.time()-start
    metrics={"prepare_only":args.prepare_only,
             "completion_seconds":completion_seconds,
             "observed_view_ratio":float((views.source==0).mean())}
    if not args.prepare_only:
        geometry=Pi3GeometryBackend(args.pi3_checkpoint,args.pi3_repo,
                                    device=args.device,input_size=518)
        start=time.time()
        node=initialize_spatial_node(
            [args.input],specs,"indoor room",CompletedViews(views),geometry,
            {"mode":"sparse_images_pi3","completion_output_dir":str(args.output/"completion"),
             "height":512,"width":512,"fov_degrees":90.0,"voxel_size":0.02,
             "node_store":NodeStore(args.output/"session")})
        metrics.update(
            geometry_seconds=time.time()-start,
            scale_mode=node.scale.mode,
            meters_per_world_unit=node.scale.meters_per_world_unit,
            points=int(len(node.points_xyz)),
            mean_point_confidence=float(node.points_confidence.mean()),
            geometry=node.quality_metrics.get("geometry_diagnostics",{}),
        )
    try:
        import torch
        metrics["peak_gpu_memory_bytes"]=int(torch.cuda.max_memory_allocated())
        metrics["gpu_name"]=torch.cuda.get_device_name()
    except Exception as error:
        metrics["gpu_diagnostics_error"]=str(error)
    (args.output/"metrics.json").write_text(json.dumps(metrics,indent=2),encoding="utf-8")
    print(json.dumps(metrics,indent=2))


if __name__=="__main__":
    main()
