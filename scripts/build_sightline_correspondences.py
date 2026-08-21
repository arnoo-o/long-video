"""Mine causal token correspondences from deterministic 3D revisit overlap."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import numpy as np
from long_video.sightline.rays import rgb_frame_latent_memberships

SCHEMA='sightline-correspondence-v5'

def correspondence_identity(row):
    return tuple(row[key] for key in ('query_chunk','key_chunk','query_latent_temporal','key_latent_temporal','query_y','query_x','key_y','key_x'))

def _camera_sequence(array,name,frames=193):
    array=np.asarray(array); legal=((frames,4,4),(1,frames,4,4)) if name=='c2w' else ((3,3),(frames,3,3),(1,frames,3,3))
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
                squared=((a[i:i+1024,None]-b[None,j:j+4096])**2).sum(-1); local=squared.argmin(1); values=squared[np.arange(len(local)),local]; take=values<best_d[i:i+len(local)]; best_d[i:i+len(local)][take]=values[take]; best_i[i:i+len(local)][take]=local[take]+j
        return np.sqrt(best_d),best_i

def _project(points,c2w,K):
    camera=(np.c_[points,np.ones(len(points),np.float32)]@np.linalg.inv(c2w).T)[:,:3]; uv=camera@K.T; return camera[:,2],uv[:,:2]/np.maximum(uv[:,2:3],1e-8)

def _zbuffer_visible(points,scene,c2w,K,height,width):
    scene_z,scene_uv=_project(scene,c2w,K); sx=np.floor(scene_uv[:,0]).astype(int); sy=np.floor(scene_uv[:,1]).astype(int); inside=(scene_z>0)&(sx>=0)&(sx<width)&(sy>=0)&(sy<height); depth=np.full((height,width),np.inf,np.float32); np.minimum.at(depth,(sy[inside],sx[inside]),scene_z[inside]); z,uv=_project(points,c2w,K); x=np.floor(uv[:,0]).astype(int); y=np.floor(uv[:,1]).astype(int); inside=(z>0)&(x>=0)&(x<width)&(y>=0)&(y<height); return inside&(z<=depth[np.clip(y,0,height-1),np.clip(x,0,width-1)]+1e-4)

def _token(pixel_index,height,width,token_height,token_width):
    y,x=divmod(int(pixel_index),width); return min(token_height-1,y*token_height//height),min(token_width-1,x*token_width//width)

def screen_overlap(xyz,valid,confidence,query_frame,key_frame,*,screening_stride=8,screening_distance_threshold=.05,min_overlap_count=1,min_overlap_ratio=.01):
    stride=max(1,int(screening_stride)); qvalid=valid[query_frame]&np.isfinite(xyz[query_frame]).all(-1)&np.isfinite(confidence[query_frame]); kvalid=valid[key_frame]&np.isfinite(xyz[key_frame]).all(-1)&np.isfinite(confidence[key_frame]); qmask=qvalid[::stride,::stride]; kmask=kvalid[::stride,::stride]; q=xyz[query_frame][::stride,::stride][qmask]; k=xyz[key_frame][::stride,::stride][kmask]
    if len(q)==0 or len(k)==0: return {'overlap_count':0,'overlap_ratio':0.,'query_valid_sample_count':int(len(q)),'accepted':False}
    distance,_=_nn(q,k); count=int((distance<=float(screening_distance_threshold)).sum()); ratio=float(count/len(q)); return {'overlap_count':count,'overlap_ratio':ratio,'query_valid_sample_count':int(len(q)),'accepted':bool(count>=min_overlap_count and ratio>=min_overlap_ratio)}

def _point_correspondences(xyz,valid,zbuffer_valid,confidence,query_frame,key_frame,c2w,K,token_height,token_width,distance_fraction):
    height,width=valid.shape[1:]; qmask=valid[query_frame]&np.isfinite(xyz[query_frame]).all(-1); kmask=valid[key_frame]&np.isfinite(xyz[key_frame]).all(-1); qi=np.flatnonzero(qmask); ki=np.flatnonzero(kmask)
    if not len(qi) or not len(ki): return []
    query=xyz[query_frame].reshape(-1,3)[qi]; key=xyz[key_frame].reshape(-1,3)[ki]; depth,_=_project(query,c2w[query_frame],K[query_frame]); positive=depth[np.isfinite(depth)&(depth>0)]
    if not len(positive): return []
    threshold=float(distance_fraction*np.median(positive)); distance,q_to_k=_nn(query,key); _,k_to_q=_nn(key,query); keep=(q_to_k>=0)&(k_to_q[q_to_k]==np.arange(len(query)))&(distance<=threshold); selected=np.flatnonzero(keep); scene_mask=zbuffer_valid[key_frame]&np.isfinite(xyz[key_frame]).all(-1); visible=_zbuffer_visible(key[q_to_k[selected]],xyz[key_frame][scene_mask],c2w[key_frame],K[key_frame],height,width) if len(selected) else []
    token_valid_counts={}
    for pixel in qi:
        token=_token(pixel,height,width,token_height,token_width); token_valid_counts[token]=token_valid_counts.get(token,0)+1
    rows=[]
    for local,seen in zip(selected,visible):
        if not seen: continue
        qpixel=int(qi[local]); kpixel=int(ki[q_to_k[local]]); qy,qx=_token(qpixel,height,width,token_height,token_width); ky,kx=_token(kpixel,height,width,token_height,token_width); conf=float(np.sqrt(max(0.,confidence[query_frame].reshape(-1)[qpixel])*max(0.,confidence[key_frame].reshape(-1)[kpixel])))
        rows.append({'query_frame':query_frame,'key_frame':key_frame,'query_pixel':qpixel,'key_pixel':kpixel,'query_y':qy,'query_x':qx,'key_y':ky,'key_x':kx,'valid_count':token_valid_counts[(qy,qx)],'weight':float(conf*np.exp(-float(distance[local])/max(threshold,1e-8))),'cycle_consistent':True,'cycle_type':'mutual_two_view'})
    return rows

def token_vote_rows(point_rows,*,token_height,token_width,min_count=1,min_coverage=0.,near_top_ratio=.9,total_frames=193):
    groups={}
    for row in point_rows:
        key=(row['query_frame'],row['key_frame'],row['query_y'],row['query_x']); group=groups.setdefault(key,{'pixels':set(),'valid_count':0,'votes':{}}); group['pixels'].add(row['query_pixel']); group['valid_count']=max(group['valid_count'],int(row.get('valid_count',0))); target=(row['key_y'],row['key_x']); group['votes'][target]=group['votes'].get(target,0.)+float(row['weight'])
    result=[]
    for (qf,kf,qy,qx),group in sorted(groups.items()):
        matched=len(group['pixels']); valid_count=group['valid_count']; coverage=matched/max(valid_count,1)
        if matched<min_count or coverage<min_coverage or not group['votes']: continue
        max_vote=max(group['votes'].values()); positives=[target for target,vote in sorted(group['votes'].items()) if vote>=max_vote*near_top_ratio]; qmembers=rgb_frame_latent_memberships(qf,total_frames=total_frames); kmembers=rgb_frame_latent_memberships(kf,total_frames=total_frames)
        for qchunk,qt in qmembers:
            for kchunk,kt in kmembers:
                for ky,kx in positives:
                    result.append({'query_frame':qf,'key_frame':kf,'query_chunk':qchunk,'key_chunk':kchunk,'query_latent_temporal':qt,'key_latent_temporal':kt,'query_y':qy,'query_x':qx,'key_y':ky,'key_x':kx,'query_token_index':qy*token_width+qx,'positive_key_index':ky*token_width+kx,'stage':{'token_height':token_height,'token_width':token_width},'weight':float(group['votes'][(ky,kx)]),'matched_count':matched,'valid_count':valid_count,'coverage':coverage,'vote':float(group['votes'][(ky,kx)]),'cycle_consistent':True,'cycle_type':'mutual_two_view'})
    return result

def _fingerprint(path):
    digest=hashlib.sha256(); digest.update(Path(path).read_bytes()); return digest.hexdigest()

def main():
    p=argparse.ArgumentParser(); p.add_argument('--xyz',type=Path,required=True); p.add_argument('--valid',type=Path,required=True); p.add_argument('--confidence',type=Path,required=True); p.add_argument('--out',type=Path,required=True); p.add_argument('--c2w',type=Path,required=True); p.add_argument('--intrinsics',type=Path,required=True); p.add_argument('--confidence-threshold',type=float,default=0.); p.add_argument('--distance-fraction',type=float,default=.015); p.add_argument('--token-height',type=int,required=True); p.add_argument('--token-width',type=int,required=True); p.add_argument('--min-frame-gap',type=int,default=1); p.add_argument('--screening-stride',type=int,default=8); p.add_argument('--screening-distance-threshold',type=float,default=.05); p.add_argument('--min-overlap-count',type=int,default=1); p.add_argument('--min-overlap-ratio',type=float,default=.01); p.add_argument('--min-count',type=int,default=1); p.add_argument('--min-coverage',type=float,default=0.); p.add_argument('--near-top-ratio',type=float,default=.9); p.add_argument('--trajectory-id',default=''); a=p.parse_args()
    xyz=np.load(a.xyz,mmap_mode='r'); valid=np.load(a.valid,mmap_mode='r').astype(bool); confidence=np.load(a.confidence,mmap_mode='r');
    if xyz.shape[:3]!=valid.shape or confidence.shape!=valid.shape: raise ValueError('teacher arrays must share [F,H,W] shape')
    zbuffer_valid=valid.copy(); valid &= np.isfinite(confidence)&(confidence>=a.confidence_threshold); frames=valid.shape[0]; c2w=_camera_sequence(np.load(a.c2w),'c2w',frames); K=_camera_sequence(np.load(a.intrinsics),'K',frames); points=[]; candidates=[]; accepted=0
    for query in range(frames):
        for key in range(0,max(0,query-a.min_frame_gap+1)):
            candidate=screen_overlap(xyz,valid,confidence,query,key,screening_stride=a.screening_stride,screening_distance_threshold=a.screening_distance_threshold,min_overlap_count=a.min_overlap_count,min_overlap_ratio=a.min_overlap_ratio); candidates.append({'query_frame':query,'key_frame':key,**candidate})
            if candidate['accepted']: accepted+=1; points.extend(_point_correspondences(xyz,valid,zbuffer_valid,confidence,query,key,c2w,K,a.token_height,a.token_width,a.distance_fraction))
    rows=token_vote_rows(points,token_height=a.token_height,token_width=a.token_width,min_count=a.min_count,min_coverage=a.min_coverage,near_top_ratio=a.near_top_ratio,total_frames=frames); a.out.parent.mkdir(parents=True,exist_ok=True)
    metadata={'schema_version':SCHEMA,'trajectory_id':a.trajectory_id,'source_height':int(valid.shape[1]),'source_width':int(valid.shape[2]),'token_grid':{'token_height':a.token_height,'token_width':a.token_width},'c2w_fingerprint':_fingerprint(a.c2w),'intrinsics_fingerprint':_fingerprint(a.intrinsics),'xyz_fingerprint':_fingerprint(a.xyz),'valid_fingerprint':_fingerprint(a.valid),'confidence_fingerprint':_fingerprint(a.confidence),'overlap_mining':{'min_frame_gap':a.min_frame_gap,'screening_stride':a.screening_stride,'screening_distance_threshold':a.screening_distance_threshold,'min_overlap_count':a.min_overlap_count,'min_overlap_ratio':a.min_overlap_ratio},'vote':{'min_count':a.min_count,'min_coverage':a.min_coverage,'near_top_ratio':a.near_top_ratio},'candidate_pair_count':len(candidates),'accepted_pair_count':accepted,'row_count':len(rows)}
    a.out.write_text(json.dumps({**metadata,'rows':rows},separators=(',',':')))

if __name__=='__main__': main()
