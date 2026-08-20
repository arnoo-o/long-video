"""Build bounded sparse correspondence rows from offline teacher arrays."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np

SCHEMA='sightline-correspondence-v2'
def _nn(a,b):
    try:
        from scipy.spatial import cKDTree
        tree=cKDTree(b); return tree.query(a,k=1)
    except ImportError:
        best_d=np.full(len(a),np.inf,np.float32); best_i=np.full(len(a),-1,np.int64)
        for i in range(0,len(a),1024):
            for j in range(0,len(b),4096):
                d=((a[i:i+1024,None]-b[None,j:j+4096])**2).sum(-1); idx=d.argmin(1); val=d[np.arange(len(idx)),idx]
                take=val<best_d[i:i+len(idx)]; best_d[i:i+len(idx)][take]=val[take]; best_i[i:i+len(idx)][take]=idx[take]+j
        return np.sqrt(best_d),best_i
def _view_visible(xyz,c2w,K,h,w):
    inv=np.linalg.inv(c2w); cam=(np.c_[xyz,np.ones(len(xyz))]@inv.T)[:,:3]; z=cam[:,2]; uv=cam@K.T; uv=uv[:,:2]/np.maximum(uv[:,2:3],1e-6); px=np.floor(uv[:,0]).astype(int); py=np.floor(uv[:,1]).astype(int); inside=(z>0)&(px>=0)&(px<w)&(py>=0)&(py<h); depth=np.full((h,w),np.inf,np.float32); np.minimum.at(depth,(py[inside],px[inside]),z[inside]); return inside&(z<=depth[np.clip(py,0,h-1),np.clip(px,0,w-1)]+1e-4)
def _rows(x,y,valid_x,valid_y,frame_x,frame_y,chunk_len,threshold):
    qi=np.flatnonzero(valid_x); ki=np.flatnonzero(valid_y); d,idx=_nn(x[qi],y[ki]); reverse_d,reverse_idx=_nn(y[ki],x[qi]); mutual=(reverse_idx[idx]==np.arange(len(qi)))&(d<=threshold)
    out=[]
    for q,k,dist in zip(qi[mutual],ki[idx[mutual]],d[mutual]):
        out.append({'query_frame':int(frame_x),'key_frame':int(frame_y),'query_chunk':int(frame_x//chunk_len),'key_chunk':int(frame_y//chunk_len),'query_token_index':int(q),'positive_key_index':int(k),'query_temporal':int(frame_x%33),'key_temporal':int(frame_y%33),'query_y':0,'query_x':int(q),'key_y':0,'key_x':int(k),'weight':float(np.exp(-float(dist)/max(threshold,1e-6))),'pair_type':'adjacent_cross_chunk' if abs(frame_x-frame_y)>1 else 'intra_chunk'})
    return out
def main():
    p=argparse.ArgumentParser(); p.add_argument('--xyz',type=Path,required=True); p.add_argument('--valid',type=Path,required=True); p.add_argument('--confidence',type=Path,required=True); p.add_argument('--out',type=Path,required=True); p.add_argument('--c2w',type=Path); p.add_argument('--intrinsics',type=Path); p.add_argument('--confidence-threshold',type=float,default=0.0); p.add_argument('--distance-fraction',type=float,default=.015); p.add_argument('--chunk-length',type=int,default=33); p.add_argument('--token-stride',type=int,default=16); a=p.parse_args()
    xyz=np.load(a.xyz,mmap_mode='r'); valid=np.load(a.valid,mmap_mode='r').astype(bool); conf=np.load(a.confidence,mmap_mode='r');
    if xyz.shape[:3]!=valid.shape or conf.shape!=valid.shape: raise ValueError('teacher arrays must share [F,H,W] shape')
    F,H,W=valid.shape; rows=[]
    for f in range(F):
        for g in (f+1, f+33, f+64):
            if g>=F: continue
            q=xyz[f,::a.token_stride,::a.token_stride].reshape(-1,3); k=xyz[g,::a.token_stride,::a.token_stride].reshape(-1,3)
            qv=valid[f,::a.token_stride,::token_stride].reshape(-1)&(conf[f,::a.token_stride,::token_stride].reshape(-1)>a.confidence_threshold); kv=valid[g,::a.token_stride,::token_stride].reshape(-1)&(conf[g,::a.token_stride,::token_stride].reshape(-1)>a.confidence_threshold)
            finite=np.isfinite(q).all(1)&np.isfinite(k).all(1); qv&=np.isfinite(q).all(1); kv&=np.isfinite(k).all(1); depth=np.linalg.norm(q[qv],axis=1); threshold=float(np.median(depth)*a.distance_fraction) if len(depth) else .01
            rows.extend(_rows(q,k,qv,kv,f,g,a.chunk_length,threshold))
    # Stable deduplication of query/key identities.
    unique={(r['query_frame'],r['query_token_index'],r['key_frame'],r['positive_key_index']):r for r in rows}; rows=list(unique.values()); a.out.parent.mkdir(parents=True,exist_ok=True); a.out.write_text(json.dumps({'schema_version':SCHEMA,'token_grid':{'source_height':H,'source_width':W,'stride':a.token_stride},'rows':rows},separators=(',',':')))
if __name__=='__main__': main()
