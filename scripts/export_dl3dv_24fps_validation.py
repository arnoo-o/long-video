#!/usr/bin/env python3
"""Export a causal source-only Warp comparison for a dense DL3DV sample."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
from PIL import Image

def main():
    p=argparse.ArgumentParser(); p.add_argument('--repo',type=Path,required=True)
    p.add_argument('--sample',type=Path,required=True); p.add_argument('--pi3-repo',type=Path,required=True)
    p.add_argument('--pi3-checkpoint',type=Path,required=True); p.add_argument('--device',default='cuda:0')
    a=p.parse_args(); sys.path.insert(0,str(a.repo))
    from long_video.initialization.geometry_backend import Pi3GeometryBackend
    from long_video.memory.node_builder import build_from_views
    from long_video.online.causal_renderer import CausalActiveNodeRenderer
    from long_video.types import CameraBatch, ViewSet, Z_DEPTH
    files=sorted((a.sample/'rgb_24fps').glob('*.jpg'))
    gt=np.stack([np.asarray(Image.open(x).convert('RGB'),np.uint8) for x in files])
    poses=np.load(a.sample/'target_c2w_local.npy').astype(np.float32)
    intrinsics=np.load(a.sample/'intrinsics.npy').astype(np.float32)
    fx,fy,cx,cy=intrinsics[:,0,0],intrinsics[:,1,1],intrinsics[:,0,2],intrinsics[:,1,2]
    if not (np.isfinite(intrinsics).all() and (fx>1).all() and (fy>1).all() and
            (fx<6400).all() and (fy<3840).all() and (cx>0).all() and
            (cx<640).all() and (cy>0).all() and (cy<384).all()):
        raise ValueError('renderer intrinsics are not valid for 384x640')
    source=gt[0]; views8=np.repeat(source[None],8,axis=0)
    poses8=np.repeat(np.eye(4,dtype=np.float32)[None],8,axis=0)
    k8=np.repeat(intrinsics[:1],8,axis=0)
    geo=Pi3GeometryBackend(a.pi3_checkpoint,a.pi3_repo,a.device)
    pred=geo.predict(views8,poses8,k8)
    views=ViewSet(rgb=views8[:1],depth=np.asarray(pred.depth[:1],np.float32),
        depth_confidence=np.asarray(pred.depth_confidence[:1],np.float32),c2w=poses8[:1],
        intrinsics=k8[:1],source=np.zeros((1,384,640),np.int8),
        image_confidence=np.ones((1,384,640),np.float32),depth_convention=Z_DEPTH)
    node=build_from_views(views,node_id='node_000',center_c2w=np.eye(4,dtype=np.float32),
        created_frame=0,voxel_size=.02,status='active')
    cameras=CameraBatch(c2w=poses,intrinsics=intrinsics,height=384,width=640)
    result=CausalActiveNodeRenderer(node,renderer_kwargs={'device':a.device}).render(
        cameras,frame_start=0,allow_reactivation=False)
    warp=np.rint(np.clip(result.warp.rgb,0,1)*255).astype(np.uint8)
    warp[~np.asarray(result.warp.visibility)]=0
    panel=np.concatenate([gt,warp],axis=2)
    try:
        import cv2
        def write(path, frames):
            out=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'mp4v'),24,(frames.shape[2],frames.shape[1]))
            if not out.isOpened(): raise RuntimeError('VideoWriter failed')
            for frame in frames: out.write(cv2.cvtColor(frame,cv2.COLOR_RGB2BGR))
            out.release()
        write(a.sample/'gt_24fps.mp4',gt); write(a.sample/'warp_24fps.mp4',warp)
        write(a.sample/'gt_warp_debug.mp4',panel)
    except ImportError:
        import imageio.v2 as imageio
        imageio.mimwrite(a.sample/'gt_24fps.mp4',gt,fps=24,macro_block_size=1)
        imageio.mimwrite(a.sample/'warp_24fps.mp4',warp,fps=24,macro_block_size=1)
        imageio.mimwrite(a.sample/'gt_warp_debug.mp4',panel,fps=24,macro_block_size=1)
    validation=json.loads((a.sample/'validation.json').read_text())
    validation.update({'warp_causal_source_only':True,'warp_uses_future_gt':False,
        'warp_visibility_mean':float(np.asarray(result.warp.visibility).mean()),
        'source_point_count':int(len(node.points_xyz)),
        'renderer_intrinsics_are_384x640':True})
    (a.sample/'validation.json').write_text(json.dumps(validation,indent=2))
    print(json.dumps(validation,indent=2))
if __name__=='__main__': main()
