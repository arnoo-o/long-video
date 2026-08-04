import numpy as np
from ..types import SpatialNode, ViewSet
from ..geometry.backprojection import backproject
from ..geometry.confidence import point_confidence

def build_from_views(view_set: ViewSet, node_id='node_000', center_c2w=None, created_frame=0, voxel_size=0.01):
    center_c2w = np.eye(4, dtype=np.float32) if center_c2w is None else np.asarray(center_c2w, np.float32)
    xyzs=[]; rgbs=[]; confs=[]; srcs=[]
    for i in range(len(view_set.rgb)):
        x,c,ic,s=backproject(view_set.depth[i], view_set.rgb[i], view_set.c2w[i], view_set.intrinsics[i], view_set.image_confidence[i], view_set.source[i])
        valid=np.isfinite(view_set.depth[i]) & (view_set.depth[i]>0)
        xyzs.append(x); rgbs.append(c); confs.append(point_confidence(s,ic,view_set.depth_confidence[i][valid])); srcs.append(s)
    xyz=np.concatenate(xyzs) if xyzs else np.empty((0,3),np.float32); rgb=np.concatenate(rgbs).astype(np.uint8); cf=np.concatenate(confs); src=np.concatenate(srcs).astype(np.int8)
    if len(xyz):
        keys=np.floor(xyz/voxel_size).astype(np.int64); _, inv=np.unique(keys,axis=0,return_inverse=True); n=inv.max()+1
        wx=np.zeros((n,3)); wr=np.zeros((n,3)); wc=np.zeros(n); count=np.bincount(inv,minlength=n).astype(np.int16); best=np.zeros(n,np.int8); bestc=np.full(n,-1.)
        for j,g in enumerate(inv): wx[g]+=xyz[j]*cf[j]; wr[g]+=rgb[j]*cf[j]; wc[g]+=cf[j]
        for j,g in enumerate(inv):
            if cf[j]>bestc[g]: bestc[g]=cf[j]; best[g]=src[j]
        good=wc>1e-8; xyz=wx[good]/wc[good,None]; rgb=np.clip(wr[good]/wc[good,None],0,255).astype(np.uint8); cf=wc[good]/count[good]; src=best[good]; count=count[good]
        bmin,bmax=xyz.min(0),xyz.max(0); radius=float(np.linalg.norm(bmax-bmin)/2)
    else: bmin=bmax=np.zeros(3,np.float32); radius=0.
    return SpatialNode(node_id,'active',None,center_c2w,created_frame,radius,bmin,bmax,view_set.rgb,view_set.depth,view_set.c2w,view_set.intrinsics,xyz.astype(np.float32),rgb,cf.astype(np.float32),src,count)
