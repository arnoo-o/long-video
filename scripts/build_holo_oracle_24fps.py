#!/usr/bin/env python3
"""Build disjoint 24 FPS Oracle windows from sparse Indoor_013 anchors."""
from __future__ import annotations
import argparse,json,os,subprocess,sys
from pathlib import Path
from zipfile import ZipFile

def _args():
    p=argparse.ArgumentParser(); p.add_argument("--config",default="configs/oracle_wah_training.yaml")
    p.add_argument("--set",action="append",default=[],dest="overrides"); return p.parse_args()

def _members(handle):
    return sorted([n for n in handle.namelist() if "/rgb/" in n and n.endswith(".jpg")],key=lambda n:float(Path(n).stem))

def _extract(archive,destination,start,count):
    with ZipFile(archive) as handle:
        rgb=_members(handle); root=rgb[0].split("/")[0]
        if start+count>len(rgb): raise IndexError("anchor window exceeds archive")
        for index in range(start,start+count):
            stem=Path(rgb[index]).stem
            for relative in (f"rgb/{stem}.jpg",f"depth/mesh_depth/{stem}.exr",f"mask/{stem}.jpg",f"poses/{stem}.txt"):
                handle.extract(f"{root}/{relative}",destination)
    return Path(destination)/root

def main():
    args=_args(); repo=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(repo))
    from long_video.config import load_yaml
    config=load_yaml(args.config,args.overrides)
    required=("holo_root","output_root","rife_root","rife_checkpoint","rife_python")
    missing=[key for key in required if not config.get(key)]
    if missing: raise ValueError(f"machine paths must be supplied by override: {missing}")
    physical=int(config.get("physical_gpu",1)); os.environ["CUDA_VISIBLE_DEVICES"]=str(physical)
    if physical!=1: raise ValueError("this task is restricted to physical GPU 1")
    import numpy as np
    from long_video.oracle_training.dense24 import PracticalRIFE425,allocate_disjoint_windows,continuous_runs
    from long_video.oracle_training.dense_dataset import build_dense_oracle_sequence
    archive=Path(config["holo_root"]); output=Path(config["output_root"]); output.mkdir(parents=True,exist_ok=True)
    with ZipFile(archive) as handle: names=_members(handle)
    timestamps=np.asarray([float(Path(name).stem) for name in names],np.float64)
    runs,median=continuous_runs(timestamps,float(config.get("gap_factor",2.5)))
    allocation=allocate_disjoint_windows(runs,train_count=8,diagnostic_count=2,rollout_anchors=17)
    rife=PracticalRIFE425(config["rife_root"],config["rife_checkpoint"],config["rife_python"])
    revision=subprocess.check_output(["git","-C",str(config["rife_root"]),"rev-parse","HEAD"],text=True).strip()
    records=[]
    for split,starts in allocation.items():
        for ordinal,start in enumerate(starts):
            count=17 if split=="rollout" else 5; sequence_id=f"{config['scene_id']}_{split}_{ordinal:03d}_24fps"
            scene=_extract(archive,output/"_extracted"/sequence_id,start,count)
            path,metadata=build_dense_oracle_sequence(scene,output,sequence_id=sequence_id,split=split,anchor_count=count,rife=rife,
              erp_resolution=config["erp_resolution"],perspective_resolution=config["perspective_resolution"],
              fov_degrees=float(config["fov_degrees"]),pixel_center=float(config["pixel_center"]),prompt=config["prompt"],
              voxel_size=float(config.get("oracle_voxel_size",.01)),renderer_kwargs={"device":"cuda:0",**dict(config.get("renderer") or {})},
              rife_revision=revision,rife_checkpoint=Path(config["rife_checkpoint"])/"flownet.pkl")
            records.append({"sequence_id":sequence_id,"split":split,"path":str(path),"anchor_start":start,"metadata":metadata})
    manifest={"schema_version":1,"scene_id":config["scene_id"],"gap_threshold":2.5*median,"median_timestamp_delta":median,
              "continuous_runs":[list(pair) for pair in runs],"allocation":allocation,"sequences":records}
    (output/"manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    print(json.dumps({"manifest":str(output/"manifest.json"),"allocation":allocation,"sequence_count":len(records)},indent=2))

if __name__=="__main__": main()
