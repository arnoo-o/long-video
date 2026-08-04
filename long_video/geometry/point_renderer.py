import numpy as np
from ..types import WarpBatch

INVALID_SOURCE = 4

def render_numpy_reference(node, cameras, near=.05, far=100., point_radius=0, depth_epsilon=1e-5):
    t,h,w = len(cameras.c2w), cameras.height, cameras.width
    rgb=np.zeros((t,h,w,3),np.float32); depth=np.full((t,h,w),np.nan,np.float32)
    vis=np.zeros((t,h,w),bool); conf=np.zeros((t,h,w),np.float32); src=np.full((t,h,w),INVALID_SOURCE,np.int8)
    xyz=np.asarray(node.points_xyz,np.float32); ph=np.c_[xyz,np.ones(len(xyz),np.float32)]
    for ti in range(t):
        cam=(np.linalg.inv(cameras.c2w[ti])@ph.T).T[:,:3]; z=cam[:,2]; k=cameras.intrinsics[ti]
        uv=(cam@k.T); uv=uv[:,:2]/np.maximum(uv[:,2:3],1e-8)
        for dy in range(-point_radius,point_radius+1):
            for dx in range(-point_radius,point_radius+1):
                x=np.rint(uv[:,0]+dx).astype(int); y=np.rint(uv[:,1]+dy).astype(int)
                valid=(z>near)&(z<far)&(x>=0)&(x<w)&(y>=0)&(y<h)
                for j in np.flatnonzero(valid):
                    if not vis[ti,y[j],x[j]] or z[j] < depth[ti,y[j],x[j]]-depth_epsilon:
                        depth[ti,y[j],x[j]]=z[j]; rgb[ti,y[j],x[j]]=node.points_rgb[j]/255.; conf[ti,y[j],x[j]]=node.points_confidence[j]; src[ti,y[j],x[j]]=node.points_source[j]; vis[ti,y[j],x[j]]=True
    return WarpBatch(rgb,depth,vis,conf,src,vis.reshape(t,-1).mean(1).astype(np.float32))

def render(node, cameras, near=.05, far=100., point_radius=1, depth_epsilon=1e-5, device=None, chunk_points=1_000_000):
    """Torch z-buffer renderer.  Returns NumPy arrays for the public WarpBatch API."""
    try:
        import torch
    except ImportError:
        return render_numpy_reference(node,cameras,near,far,point_radius,depth_epsilon)
    device=device or ("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cpu": return render_numpy_reference(node,cameras,near,far,point_radius,depth_epsilon)
    dev=torch.device(device); t,h,w=len(cameras.c2w),cameras.height,cameras.width
    xyz=torch.as_tensor(node.points_xyz,dtype=torch.float32,device=dev)
    prgb=torch.as_tensor(node.points_rgb,dtype=torch.float32,device=dev)/255.
    pconf=torch.as_tensor(node.points_confidence,dtype=torch.float32,device=dev)
    psrc=torch.as_tensor(node.points_source,dtype=torch.int8,device=dev)
    poses=torch.as_tensor(cameras.c2w,dtype=torch.float32,device=dev); ks=torch.as_tensor(cameras.intrinsics,dtype=torch.float32,device=dev)
    outputs=[]
    for ti in range(t):
        inv=torch.linalg.inv(poses[ti]); cam=xyz@inv[:3,:3].T+inv[:3,3]; z=cam[:,2]
        uv=cam@ks[ti].T; uv=uv[:,:2]/z[:,None].clamp_min(1e-8)
        zbuf=torch.full((h*w,),float("inf"),device=dev); best=torch.full((h*w,),-1,dtype=torch.long,device=dev)
        for dy in range(-point_radius,point_radius+1):
            for dx in range(-point_radius,point_radius+1):
                x=torch.round(uv[:,0]+dx).long(); y=torch.round(uv[:,1]+dy).long()
                ok=(z>near)&(z<far)&(x>=0)&(x<w)&(y>=0)&(y<h)
                idx=torch.nonzero(ok).flatten()
                if not len(idx): continue
                flat=y[idx]*w+x[idx]; zz=z[idx]
                zbuf.scatter_reduce_(0,flat,zz,reduce="amin",include_self=True)
        for dy in range(-point_radius,point_radius+1):
            for dx in range(-point_radius,point_radius+1):
                x=torch.round(uv[:,0]+dx).long(); y=torch.round(uv[:,1]+dy).long()
                ok=(z>near)&(z<far)&(x>=0)&(x<w)&(y>=0)&(y<h)
                idx=torch.nonzero(ok).flatten()
                if not len(idx): continue
                flat=y[idx]*w+x[idx]; sel=idx[z[idx] <= zbuf[flat]+depth_epsilon]
                if len(sel):
                    flat=y[sel]*w+x[sel]; best[flat]=sel
        valid=best>=0; out_rgb=torch.zeros((h*w,3),device=dev); out_cf=torch.zeros(h*w,device=dev); out_src=torch.full((h*w,),INVALID_SOURCE,dtype=torch.int8,device=dev); out_z=torch.full((h*w,),float("nan"),device=dev)
        chosen=best[valid]; out_rgb[valid]=prgb[chosen]; out_cf[valid]=pconf[chosen]; out_src[valid]=psrc[chosen]; out_z[valid]=z[chosen]
        outputs.append((out_rgb.reshape(h,w,3),out_z.reshape(h,w),valid.reshape(h,w),out_cf.reshape(h,w),out_src.reshape(h,w)))
    rgb,depth,vis,conf,src=zip(*outputs)
    rgb=torch.stack(rgb).cpu().numpy(); depth=torch.stack(depth).cpu().numpy(); vis=torch.stack(vis).cpu().numpy(); conf=torch.stack(conf).cpu().numpy(); src=torch.stack(src).cpu().numpy()
    return WarpBatch(rgb,depth,vis,conf,src,vis.reshape(t,-1).mean(1).astype(np.float32))
