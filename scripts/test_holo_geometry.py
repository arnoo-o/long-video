#!/usr/bin/env python3
import argparse,json,sys
from dataclasses import replace
from pathlib import Path
from zipfile import ZipFile
import numpy as np
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from long_video.data.holo360d import Holo360DReader
from long_video.initialization.view_completion import HoloOracleCompletion
from long_video.memory.node_builder import build_from_views
from long_video.memory.node_store import NodeStore
from long_video.geometry.point_renderer import render,render_numpy_reference
from long_video.types import CameraBatch

def depth_preview(depth):
    valid=np.isfinite(depth)&(depth>0); out=np.zeros(depth.shape,np.uint8)
    if valid.any():
        lo,hi=np.percentile(depth[valid],[2,98]); out[valid]=np.clip((depth[valid]-lo)/max(hi-lo,1e-6)*255,0,255)
    return out

def extract_first(zip_path,dest):
    with ZipFile(zip_path) as z:
        names=z.namelist(); rgb=sorted(n for n in names if "/rgb/" in n and n.endswith(".jpg"));
        if not rgb: raise RuntimeError("No RGB frames in archive")
        root=rgb[0].split("/")[0]; stem=Path(rgb[0]).stem
        members=[f"{root}/rgb/{stem}.jpg",f"{root}/depth/mesh_depth/{stem}.exr",f"{root}/mask/{stem}.jpg",f"{root}/poses/{stem}.txt"]
        for member in members: z.extract(member,dest)
    return Path(dest)/root

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--zip",required=True); ap.add_argument("--output",required=True); ap.add_argument("--height",type=int,default=256); ap.add_argument("--width",type=int,default=256); ap.add_argument("--fov",type=float,default=90); args=ap.parse_args()
    out=Path(args.output); out.mkdir(parents=True,exist_ok=True); scene=extract_first(args.zip,out/"extracted")
    frame=Holo360DReader(scene).read(0)
    completion=HoloOracleCompletion(args.fov,args.height,args.width)
    views=completion.complete(frame.rgb,frame.depth,frame.c2w,frame.mask,observed_indices=(0,))
    node=build_from_views(views,voxel_size=.01)
    store=NodeStore(out/"session"); store.save(node); loaded=store.load(node.node_id)
    for name in ("points_xyz","points_rgb","points_confidence","points_source","observation_count"): np.testing.assert_array_equal(getattr(node,name),getattr(loaded,name))
    cameras=CameraBatch(views.c2w,views.intrinsics,args.height,args.width)
    warped=render(loaded,cameras,near=.05,far=100.,point_radius=0,device="cuda",chunk_points=100000)
    yy,xx=np.indices((args.height,args.width),np.float32); ray=np.stack(((xx-views.intrinsics[0,0,2])/views.intrinsics[0,0,0],(yy-views.intrinsics[0,1,2])/views.intrinsics[0,1,1],np.ones_like(xx)),-1); ray_z=1/np.linalg.norm(ray,axis=-1)
    target_z=views.depth*ray_z[None]; valid=warped.visibility&np.isfinite(target_z)
    rgb_error=np.abs(warped.rgb-views.rgb/255.)[valid].mean(); depth_error=np.abs(warped.depth-target_z)[valid].mean()
    input_dir=out/"input"; rerender_dir=out/"rerender"; input_dir.mkdir(exist_ok=True); rerender_dir.mkdir(exist_ok=True)
    for i in range(8):
        Image.fromarray(views.rgb[i]).save(input_dir/f"view_{i:02d}.png"); Image.fromarray((warped.rgb[i]*255).astype(np.uint8)).save(rerender_dir/f"view_{i:02d}.png"); Image.fromarray(depth_preview(warped.depth[i])).save(rerender_dir/f"depth_{i:02d}.png"); Image.fromarray(warped.visibility[i].astype(np.uint8)*255).save(rerender_dir/f"visibility_{i:02d}.png"); Image.fromarray((warped.confidence[i]*255).astype(np.uint8)).save(rerender_dir/f"confidence_{i:02d}.png")
    moved=[]; names=["forward","backward","left","right","yaw"]
    for delta in ((0,0,-.1),(0,0,.1),(-.1,0,0),(.1,0,0),(0,0,0)):
        pose=views.c2w[0].copy(); pose[:3,3]+=pose[:3,:3]@np.asarray(delta,np.float32); moved.append(pose)
    angle=np.deg2rad(10); moved[-1][:3,:3]=moved[-1][:3,:3]@np.array([[np.cos(angle),0,np.sin(angle)],[0,1,0],[-np.sin(angle),0,np.cos(angle)]],np.float32)
    move_cams=CameraBatch(np.stack(moved),np.repeat(views.intrinsics[:1],5,0),args.height,args.width); move_warp=render(loaded,move_cams,near=.05,far=100.,point_radius=1,device="cuda",chunk_points=100000)
    move_dir=out/"motion"; move_dir.mkdir(exist_ok=True)
    for i,name in enumerate(names): Image.fromarray((move_warp.rgb[i]*255).astype(np.uint8)).save(move_dir/f"{name}.png")
    step=max(1,len(node.points_xyz)//1000); tiny=replace(node,points_xyz=node.points_xyz[::step],points_rgb=node.points_rgb[::step],points_confidence=node.points_confidence[::step],points_source=node.points_source[::step],observation_count=node.observation_count[::step])
    one=CameraBatch(views.c2w[:1],views.intrinsics[:1],args.height,args.width); cpu=render_numpy_reference(tiny,one,near=.05,far=100.,point_radius=0); gpu=render(tiny,one,near=.05,far=100.,point_radius=0,device="cuda",chunk_points=100)
    np.testing.assert_array_equal(cpu.visibility,gpu.visibility); np.testing.assert_allclose(cpu.depth,gpu.depth,equal_nan=True,atol=1e-5); np.testing.assert_allclose(cpu.rgb,gpu.rgb,atol=1/255)
    metrics={"frame_id":frame.frame_id,"input_points":node.quality_metrics["input_points"],"fused_points":len(node.points_xyz),"coverage":warped.coverage_per_frame.tolist(),"rgb_mae":float(rgb_error),"z_depth_mae_m":float(depth_error),"gpu_numpy_reference":"passed"}
    (out/"metrics.json").write_text(json.dumps(metrics,indent=2)); print(json.dumps(metrics,indent=2))
if __name__=="__main__": main()
