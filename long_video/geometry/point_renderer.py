import numpy as np
from ..types import WarpBatch, CameraBatch

def render(node, cameras, point_radius=0):
    t, h, w = len(cameras.c2w), cameras.height, cameras.width
    rgb = np.zeros((t,h,w,3), np.float32); depth=np.full((t,h,w), np.nan, np.float32)
    vis=np.zeros((t,h,w), bool); conf=np.zeros((t,h,w), np.float32); src=np.full((t,h,w), 4, np.int8)
    xyz=node.points_xyz; ones=np.ones((len(xyz),1), np.float32); ph=np.c_[xyz,ones]
    for i,(pose,k) in enumerate(zip(cameras.c2w,cameras.intrinsics)):
        cam=(np.linalg.inv(pose) @ ph.T).T[:,:3]; z=cam[:,2]
        ok=z>1e-5; cam=cam[ok]; idx=np.nonzero(ok)[0]; z=z[ok]
        uv=(k @ cam.T).T; uv=uv[:,:2]/uv[:,2:3]
        px=np.rint(uv).astype(int); inside=(px[:,0]>=0)&(px[:,0]<w)&(px[:,1]>=0)&(px[:,1]<h)
        for j in np.nonzero(inside)[0]:
            x,y=px[j]; old=depth[i,y,x]
            if np.isnan(old) or z[j]<old:
                depth[i,y,x]=z[j]; rgb[i,y,x]=node.points_rgb[idx[j]]/255.; vis[i,y,x]=True; conf[i,y,x]=node.points_confidence[idx[j]]; src[i,y,x]=node.points_source[idx[j]]
    return WarpBatch(rgb,depth,vis,conf,src,vis.reshape(t,-1).mean(1).astype(np.float32))
