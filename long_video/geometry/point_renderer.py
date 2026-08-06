import numpy as np
from ..types import WarpBatch

INVALID_SOURCE=4

def _coverage_metric(node,cameras,near,far,resolution=64):
    """Fixed angular occupancy, independent of output pixels and splat radius."""
    xyz=np.asarray(node.points_xyz,np.float32); result=[]
    homogeneous=np.c_[xyz,np.ones(len(xyz),np.float32)]
    for pose,k in zip(cameras.c2w,cameras.intrinsics):
        cam=(np.linalg.inv(pose)@homogeneous.T).T[:,:3]; z=cam[:,2]
        projected=cam@np.asarray(k,np.float32).T
        u=projected[:,0]/np.maximum(z,1e-8)/cameras.width
        v=projected[:,1]/np.maximum(z,1e-8)/cameras.height
        valid=(z>near)&(z<far)&(u>=0)&(u<1)&(v>=0)&(v<1)
        x=np.floor(u[valid]*resolution).astype(np.int64)
        y=np.floor(v[valid]*resolution).astype(np.int64)
        occupied=np.unique(y*resolution+x).size
        result.append(occupied/(resolution*resolution))
    return np.asarray(result,np.float32)


def render_numpy_reference(node,cameras,near=.05,far=100.,point_radius=0,depth_epsilon=1e-5):
    t,h,w=len(cameras.c2w),cameras.height,cameras.width
    rgb=np.zeros((t,h,w,3),np.float32); depth=np.full((t,h,w),np.nan,np.float32)
    vis=np.zeros((t,h,w),bool); conf=np.zeros((t,h,w),np.float32); src=np.full((t,h,w),INVALID_SOURCE,np.int8)
    xyz=np.asarray(node.points_xyz,np.float32); ph=np.c_[xyz,np.ones(len(xyz),np.float32)]
    for ti in range(t):
        cam=(np.linalg.inv(cameras.c2w[ti])@ph.T).T[:,:3]; z=cam[:,2]; k=cameras.intrinsics[ti]
        uv=cam@k.T; uv=uv[:,:2]/np.maximum(uv[:,2:3],1e-8)
        for dy in range(-point_radius,point_radius+1):
            for dx in range(-point_radius,point_radius+1):
                x=np.rint(uv[:,0]+dx).astype(int); y=np.rint(uv[:,1]+dy).astype(int)
                valid=(z>near)&(z<far)&(x>=0)&(x<w)&(y>=0)&(y<h)
                for j in np.flatnonzero(valid):
                    if not vis[ti,y[j],x[j]] or z[j]<depth[ti,y[j],x[j]]-depth_epsilon:
                        depth[ti,y[j],x[j]]=z[j]; rgb[ti,y[j],x[j]]=node.points_rgb[j]/255.; conf[ti,y[j],x[j]]=node.points_confidence[j]; src[ti,y[j],x[j]]=node.points_source[j]; vis[ti,y[j],x[j]]=True
    return WarpBatch(rgb,depth,vis,conf,src,_coverage_metric(node,cameras,near,far))

def render(node,cameras,near=.05,far=100.,point_radius=1,depth_epsilon=1e-5,device="cpu",chunk_points=1_000_000):
    try: import torch
    except ImportError: return render_numpy_reference(node,cameras,near,far,point_radius,depth_epsilon)
    if device is None: raise ValueError("renderer device must be explicitly configured")
    device=str(device)
    if device=="cpu": return render_numpy_reference(node,cameras,near,far,point_radius,depth_epsilon)
    dev=torch.device(device); t,h,w=len(cameras.c2w),cameras.height,cameras.width
    xyz=torch.as_tensor(node.points_xyz,dtype=torch.float32,device=dev)
    prgb=torch.as_tensor(node.points_rgb,dtype=torch.float32,device=dev)/255.
    pconf=torch.as_tensor(node.points_confidence,dtype=torch.float32,device=dev)
    psrc=torch.as_tensor(node.points_source,dtype=torch.int8,device=dev)
    poses=torch.as_tensor(cameras.c2w,dtype=torch.float32,device=dev)
    ks=torch.as_tensor(cameras.intrinsics,dtype=torch.float32,device=dev)
    outputs=[]
    for ti in range(t):
        inv=torch.linalg.inv(poses[ti]); zbuf=torch.full((h*w,),float("inf"),device=dev); best=torch.full((h*w,),len(xyz),dtype=torch.long,device=dev)
        for start in range(0,len(xyz),chunk_points):
            stop=min(start+chunk_points,len(xyz)); cam=xyz[start:stop]@inv[:3,:3].T+inv[:3,3]; z=cam[:,2]; uv=cam@ks[ti].T; uv=uv[:,:2]/z[:,None].clamp_min(1e-8)
            for dy in range(-point_radius,point_radius+1):
                for dx in range(-point_radius,point_radius+1):
                    x=torch.round(uv[:,0]+dx).long(); y=torch.round(uv[:,1]+dy).long(); ok=(z>near)&(z<far)&(x>=0)&(x<w)&(y>=0)&(y<h); idx=torch.nonzero(ok).flatten()
                    if len(idx): zbuf.scatter_reduce_(0,y[idx]*w+x[idx],z[idx],reduce="amin",include_self=True)
        for start in range(0,len(xyz),chunk_points):
            stop=min(start+chunk_points,len(xyz)); cam=xyz[start:stop]@inv[:3,:3].T+inv[:3,3]; z=cam[:,2]; uv=cam@ks[ti].T; uv=uv[:,:2]/z[:,None].clamp_min(1e-8)
            for dy in range(-point_radius,point_radius+1):
                for dx in range(-point_radius,point_radius+1):
                    x=torch.round(uv[:,0]+dx).long(); y=torch.round(uv[:,1]+dy).long(); ok=(z>near)&(z<far)&(x>=0)&(x<w)&(y>=0)&(y<h); idx=torch.nonzero(ok).flatten()
                    if not len(idx): continue
                    flat=y[idx]*w+x[idx]; selected=idx[z[idx]<=zbuf[flat]+depth_epsilon]
                    if len(selected):
                        flat_selected=y[selected]*w+x[selected]
                        best.scatter_reduce_(0,flat_selected,selected+start,reduce="amin",include_self=True)
        valid=(best>=0)&(best<len(xyz)); out_rgb=torch.zeros((h*w,3),device=dev); out_cf=torch.zeros(h*w,device=dev); out_src=torch.full((h*w,),INVALID_SOURCE,dtype=torch.int8,device=dev); out_z=torch.full((h*w,),float("nan"),device=dev)
        chosen=best[valid]; out_rgb[valid]=prgb[chosen]; out_cf[valid]=pconf[chosen]; out_src[valid]=psrc[chosen]; out_z[valid]=zbuf[valid]
        outputs.append((out_rgb.reshape(h,w,3),out_z.reshape(h,w),valid.reshape(h,w),out_cf.reshape(h,w),out_src.reshape(h,w)))
    rgb,depth,vis,conf,src=zip(*outputs)
    rgb=torch.stack(rgb).cpu().numpy(); depth=torch.stack(depth).cpu().numpy(); vis=torch.stack(vis).cpu().numpy(); conf=torch.stack(conf).cpu().numpy(); src=torch.stack(src).cpu().numpy()
    return WarpBatch(rgb,depth,vis,conf,src,_coverage_metric(node,cameras,near,far))
