"""Build bounded sparse correspondence rows from offline teacher arrays."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np

SCHEMA='sightline-correspondence-v1'
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
def _rows(x,y,valid_x,valid_y,frame_x,frame_y,threshold, token_height, token_width, c2w=None, K=None, height=None, width=None):
    qi=np.flatnonzero(valid_x); ki=np.flatnonzero(valid_y); d,idx=_nn(x[qi],y[ki]); reverse_d,reverse_idx=_nn(y[ki],x[qi]); mutual=(reverse_idx[idx]==np.arange(len(qi)))&(d<=threshold)
    out=[]
    for q,k,dist in zip(qi[mutual],ki[idx[mutual]],d[mutual]):
        if c2w is not None and K is not None and height and width:
            if not _view_visible(y[k:k+1], c2w, K, height, width)[0]: continue
        qy,qx=divmod(int(q),token_width); ky,kx=divmod(int(k),token_width)
        qchunk,qlocal=divmod(int(frame_x),32); kchunk,klocal=divmod(int(frame_y),32)
        out.append({'query_frame':int(frame_x),'key_frame':int(frame_y),'query_chunk':qchunk,'key_chunk':kchunk,'query_token_index':int(q),'positive_key_index':int(k),'query_temporal':qlocal,'key_temporal':klocal,'query_y':qy,'query_x':qx,'key_y':ky,'key_x':kx,'stage':{'token_height':token_height,'token_width':token_width},'weight':float(np.exp(-float(dist)/max(threshold,1e-6))),'pair_type':'long_gap_revisit' if abs(frame_x-frame_y)>32 else ('cross_chunk' if qchunk!=kchunk else 'intra_chunk')})
    return out
def main():
    p=argparse.ArgumentParser(); p.add_argument('--xyz',type=Path,required=True); p.add_argument('--valid',type=Path,required=True); p.add_argument('--confidence',type=Path,required=True); p.add_argument('--out',type=Path,required=True); p.add_argument('--c2w',type=Path); p.add_argument('--intrinsics',type=Path); p.add_argument('--confidence-threshold',type=float,default=0.0); p.add_argument('--distance-fraction',type=float,default=.015); p.add_argument('--token-height',type=int,required=True); p.add_argument('--token-width',type=int,required=True); a=p.parse_args()
    xyz=np.load(a.xyz,mmap_mode='r'); valid=np.load(a.valid,mmap_mode='r').astype(bool); conf=np.load(a.confidence,mmap_mode='r');
    if xyz.shape[:3]!=valid.shape or conf.shape!=valid.shape: raise ValueError('teacher arrays must share [F,H,W] shape')
    F,H,W=valid.shape; rows=[]; c2w=np.load(a.c2w) if a.c2w else None; K=np.load(a.intrinsics) if a.intrinsics else None
    if c2w is not None and c2w.ndim==4: c2w=c2w[:,0]
    if K is not None and K.ndim==3: K=K[0]
    for f in range(F):
        for g in (f+1, f+33, f+64):
            if g>=F: continue
            def tokenize(frame):
                values=[]; flags=[]; scores=[]
                for yy in range(a.token_height):
                    y0,y1=round(yy*H/a.token_height),round((yy+1)*H/a.token_height)
                    for xx in range(a.token_width):
                        x0,x1=round(xx*W/a.token_width),round((xx+1)*W/a.token_width); block=xyz[frame,y0:y1,x0:x1].reshape(-1,3); ok=valid[frame,y0:y1,x0:x1].reshape(-1)&(conf[frame,y0:y1,x0:x1].reshape(-1)>a.confidence_threshold); ok &= np.isfinite(block).all(1)
                        values.append(np.median(block[ok],0) if ok.any() else np.zeros(3,np.float32)); flags.append(bool(ok.any())); scores.append(float(np.max(conf[frame,y0:y1,x0:x1][ok])) if ok.any() else 0.)
                return np.asarray(values,np.float32),np.asarray(flags,bool),np.asarray(scores,np.float32)
            q,qv,qc=tokenize(f); k,kv,kc=tokenize(g)
            qv&=np.isfinite(q).all(1); kv&=np.isfinite(k).all(1); depth=np.linalg.norm(q[qv],axis=1); threshold=float(np.median(depth)*a.distance_fraction) if len(depth) else .01
            rows.extend(_rows(q,k,qv,kv,f,g,threshold,a.token_height,a.token_width,c2w[f] if c2w is not None and len(c2w)>f else None, K[f] if K is not None and K.ndim==3 and len(K)>f else K, H, W))
    # Stable deduplication of query/key identities.
    unique={(r['query_frame'],r['query_token_index'],r['key_frame'],r['positive_key_index']):r for r in rows}; rows=list(unique.values()); a.out.parent.mkdir(parents=True,exist_ok=True); a.out.write_text(json.dumps({'schema_version':SCHEMA,'token_grid':{'source_height':H,'source_width':W,'token_height':a.token_height,'token_width':a.token_width},'rows':rows},separators=(',',':')))
if __name__=='__main__': main()
