import numpy as np
from ..types import SpatialNode, ViewSet
from ..geometry.backprojection import backproject
from ..geometry.confidence import point_confidence

def build_from_views(view_set: ViewSet, node_id="node_000", center_c2w=None, created_frame=0, voxel_size=.02, status="active", parent_id=None):
    center=np.eye(4,dtype=np.float32) if center_c2w is None else np.asarray(center_c2w,np.float32)
    clouds=[]
    for i in range(len(view_set.rgb)):
        depth=np.asarray(view_set.depth[i]); valid=np.isfinite(depth)&(depth>0)
        xyz,rgb,img_conf,source=backproject(depth,view_set.rgb[i],view_set.c2w[i],view_set.intrinsics[i],view_set.image_confidence[i],view_set.source[i],view_set.depth_convention)
        conf=point_confidence(source,img_conf,np.asarray(view_set.depth_confidence[i])[valid])
        clouds.append((xyz,np.asarray(rgb,np.uint8),conf.astype(np.float32),np.asarray(source,np.int8)))
    if not clouds: raise ValueError("ViewSet contains no valid depth samples")
    xyz=np.concatenate([x[0] for x in clouds]); rgb=np.concatenate([x[1] for x in clouds]); conf=np.concatenate([x[2] for x in clouds]); source=np.concatenate([x[3] for x in clouds])
    keys=np.floor(xyz/float(voxel_size)).astype(np.int64); _,inverse=np.unique(keys,axis=0,return_inverse=True); n=int(inverse.max())+1
    weight=np.bincount(inverse,weights=conf,minlength=n).astype(np.float32); count=np.bincount(inverse,minlength=n).astype(np.int16)
    pos=np.zeros((n,3),np.float32); col=np.zeros((n,3),np.float32)
    np.add.at(pos,inverse,xyz*conf[:,None]); np.add.at(col,inverse,rgb.astype(np.float32)*conf[:,None])
    good=weight>1e-8; pos=pos[good]/weight[good,None]; col=np.clip(col[good]/weight[good,None],0,255).astype(np.uint8)
    best=np.full(n,-1.,np.float32); best_source=np.full(n,4,np.int8)
    for i,g in enumerate(inverse):
        if conf[i]>best[g]: best[g]=conf[i]; best_source[g]=source[i]
    confidence=(weight[good]/count[good]).astype(np.float32); source=best_source[good]; count=count[good]
    bmin,bmax=pos.min(0),pos.max(0); radius=float(np.linalg.norm(bmax-bmin)*.5)
    metrics={"input_points":int(len(xyz)),"fused_points":int(len(pos)),"mean_confidence":float(confidence.mean())}
    return SpatialNode(node_id,status,parent_id,center,int(created_frame),radius,bmin.astype(np.float32),bmax.astype(np.float32),view_set.rgb,view_set.depth,view_set.c2w,view_set.intrinsics,pos.astype(np.float32),col,confidence,source,count,None,view_set.depth_convention,2,metrics)
