"""Build sparse token correspondences from offline ReCal3R point-level teachers."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from long_video.sightline.rays import rgb_frame_latent_memberships

SCHEMA='sightline-correspondence-v3'
PAIR_OFFSETS=(-64,-33,-1,1,33,64)

def correspondence_identity(row):
    return tuple(row[key] for key in ('query_chunk','key_chunk','query_latent_temporal','key_latent_temporal','query_y','query_x','key_y','key_x'))

def aggregate_token_rows(rows):
    aggregated={}
    for row in rows:
        identity=correspondence_identity(row); previous=aggregated.get(identity)
        if previous is None or row['weight']>previous['weight']: aggregated[identity]=row
    return [aggregated[key] for key in sorted(aggregated)]

def _camera_sequence(array,name,frames=193):
    array=np.asarray(array)
    legal=((frames,4,4),(1,frames,4,4)) if name=='c2w' else ((3,3),(frames,3,3),(1,frames,3,3))
    if array.shape not in legal: raise ValueError(f'{name} has invalid shape {array.shape}')
    if array.ndim==4: array=array[0]
    if name!='c2w' and array.ndim==2: array=np.repeat(array[None],frames,axis=0)
    return array

def _nn(a,b):
    if not len(a) or not len(b): return np.empty(0,np.float32),np.empty(0,np.int64)
    try:
        from scipy.spatial import cKDTree
        return cKDTree(b).query(a,k=1)
    except ImportError:
        best_d=np.full(len(a),np.inf,np.float32); best_i=np.full(len(a),-1,np.int64)
        for i in range(0,len(a),1024):
            for j in range(0,len(b),4096):
                squared=((a[i:i+1024,None]-b[None,j:j+4096])**2).sum(-1); local=squared.argmin(1); values=squared[np.arange(len(local)),local]
                take=values<best_d[i:i+len(local)]; best_d[i:i+len(local)][take]=values[take]; best_i[i:i+len(local)][take]=local[take]+j
        return np.sqrt(best_d),best_i

def _project(points,c2w,K):
    camera=(np.c_[points,np.ones(len(points),np.float32)]@np.linalg.inv(c2w).T)[:,:3]
    uv=camera@K.T; uv=uv[:,:2]/np.maximum(uv[:,2:3],1e-8)
    return camera[:,2],uv

def _zbuffer_visible(points,scene,c2w,K,height,width):
    scene_z,scene_uv=_project(scene,c2w,K); sx=np.floor(scene_uv[:,0]).astype(int); sy=np.floor(scene_uv[:,1]).astype(int)
    inside=(scene_z>0)&(sx>=0)&(sx<width)&(sy>=0)&(sy<height); depth=np.full((height,width),np.inf,np.float32)
    np.minimum.at(depth,(sy[inside],sx[inside]),scene_z[inside])
    z,uv=_project(points,c2w,K); x=np.floor(uv[:,0]).astype(int); y=np.floor(uv[:,1]).astype(int); valid=(z>0)&(x>=0)&(x<width)&(y>=0)&(y<height)
    valid &= z<=depth[np.clip(y,0,height-1),np.clip(x,0,width-1)]+1e-4
    return valid

def _token(pixel_index,height,width,token_height,token_width):
    y,x=divmod(int(pixel_index),width)
    return min(token_height-1,y*token_height//height),min(token_width-1,x*token_width//width)

def _point_rows(xyz,valid,zbuffer_valid,confidence,frame_x,frame_y,c2w,K,token_height,token_width,distance_fraction):
    height,width=valid.shape[1:]; qmask=valid[frame_x]&np.isfinite(xyz[frame_x]).all(-1); kmask=valid[frame_y]&np.isfinite(xyz[frame_y]).all(-1)
    qi=np.flatnonzero(qmask); ki=np.flatnonzero(kmask)
    if not len(qi) or not len(ki): return []
    query=xyz[frame_x].reshape(-1,3)[qi]; key=xyz[frame_y].reshape(-1,3)[ki]
    query_z,_=_project(query,c2w[frame_x],K[frame_x]); positive_z=query_z[np.isfinite(query_z)&(query_z>0)]
    if not len(positive_z): return []
    threshold=float(distance_fraction*np.median(positive_z)); distance,q_to_k=_nn(query,key); _,k_to_q=_nn(key,query)
    keep=(q_to_k>=0)&(k_to_q[q_to_k]==np.arange(len(query)))&(distance<=threshold)
    selected=np.flatnonzero(keep)
    if not len(selected): return []
    scene_mask=zbuffer_valid[frame_y]&np.isfinite(xyz[frame_y]).all(-1)
    visible=_zbuffer_visible(key[q_to_k[selected]],xyz[frame_y][scene_mask],c2w[frame_y],K[frame_y],height,width); rows=[]
    for local,seen in zip(selected,visible):
        if not seen: continue
        cycle_consistent=bool(k_to_q[q_to_k[local]]==local)
        if not cycle_consistent: continue
        qpixel=int(qi[local]); kpixel=int(ki[q_to_k[local]])
        qy,qx=_token(qpixel,height,width,token_height,token_width); ky,kx=_token(kpixel,height,width,token_height,token_width)
        qmembers=rgb_frame_latent_memberships(frame_x,total_frames=valid.shape[0]); kmembers=rgb_frame_latent_memberships(frame_y,total_frames=valid.shape[0])
        conf=float(np.sqrt(max(0.,confidence[frame_x].reshape(-1)[qpixel])*max(0.,confidence[frame_y].reshape(-1)[kpixel])))
        weight=conf*np.exp(-float(distance[local])/max(threshold,1e-8))
        for qchunk,qt in qmembers:
            for kchunk,kt in kmembers:
                rows.append({'query_frame':frame_x,'key_frame':frame_y,'query_chunk_memberships':[x[0] for x in qmembers],'key_chunk_memberships':[x[0] for x in kmembers],
                    'query_chunk':qchunk,'key_chunk':kchunk,'query_latent_temporal':qt,'key_latent_temporal':kt,'query_y':qy,'query_x':qx,'key_y':ky,'key_x':kx,
                    'query_token_index':qy*token_width+qx,'positive_key_index':ky*token_width+kx,'stage':{'token_height':token_height,'token_width':token_width},
                    'weight':float(weight),'cycle_consistent':cycle_consistent,'cycle_type':'mutual_two_view','pair_type':'long_gap_revisit' if abs(frame_x-frame_y)>32 else ('cross_chunk' if qchunk!=kchunk else 'intra_chunk')})
    return rows

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--xyz',type=Path,required=True); parser.add_argument('--valid',type=Path,required=True); parser.add_argument('--confidence',type=Path,required=True); parser.add_argument('--out',type=Path,required=True); parser.add_argument('--c2w',type=Path,required=True); parser.add_argument('--intrinsics',type=Path,required=True); parser.add_argument('--confidence-threshold',type=float,default=0.0); parser.add_argument('--distance-fraction',type=float,default=.015); parser.add_argument('--token-height',type=int,required=True); parser.add_argument('--token-width',type=int,required=True); args=parser.parse_args()
    xyz=np.load(args.xyz,mmap_mode='r'); valid=np.load(args.valid,mmap_mode='r').astype(bool); confidence=np.load(args.confidence,mmap_mode='r')
    if xyz.shape[:3]!=valid.shape or confidence.shape!=valid.shape: raise ValueError('teacher arrays must share [F,H,W] shape')
    zbuffer_valid=valid.copy(); valid &= np.isfinite(confidence)&(confidence>=args.confidence_threshold)
    frames=valid.shape[0]; c2w=_camera_sequence(np.load(args.c2w),'c2w',frames); K=_camera_sequence(np.load(args.intrinsics),'K',frames); rows=[]
    for query_frame in range(frames):
        for offset in PAIR_OFFSETS:
            key_frame=query_frame+offset
            if 0<=key_frame<frames: rows.extend(_point_rows(xyz,valid,zbuffer_valid,confidence,query_frame,key_frame,c2w,K,args.token_height,args.token_width,args.distance_fraction))
    rows=aggregate_token_rows(rows); args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps({'schema_version':SCHEMA,'token_grid':{'source_height':valid.shape[1],'source_width':valid.shape[2],'token_height':args.token_height,'token_width':args.token_width},'rows':rows},separators=(',',':')))

if __name__=='__main__': main()
