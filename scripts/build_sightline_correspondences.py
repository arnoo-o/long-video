"""Build sparse token correspondences from offline teacher arrays."""
import argparse,json
from pathlib import Path
import numpy as np
from long_video.sightline.correspondence import mutual_nearest
def main():
 p=argparse.ArgumentParser(); p.add_argument('--xyz',type=Path,required=True); p.add_argument('--out',type=Path,required=True); p.add_argument('--confidence',type=Path); p.add_argument('--threshold',type=float,default=0.0); a=p.parse_args()
 xyz=np.load(a.xyz); conf=np.load(a.confidence) if a.confidence else np.ones(xyz.shape[:-1],np.float32); rows=[]
 for t in range(len(xyz)-1):
  x=xyz[t].reshape(-1,3); y=xyz[t+1].reshape(-1,3); c=conf[t].reshape(-1)>a.threshold; d=conf[t+1].reshape(-1)>a.threshold
  import torch; pairs=mutual_nearest(torch.from_numpy(x[c]).float(),torch.from_numpy(y[d]).float()).cpu().numpy(); qi=np.flatnonzero(c)[pairs[:,0]]; ki=np.flatnonzero(d)[pairs[:,1]]
  rows.extend({'query_token_index':int(q),'positive_key_index':int(k),'weight':1.0,'query_chunk':t,'key_chunk':t+1,'pair_type':'cross_chunk'} for q,k in zip(qi,ki))
 a.out.parent.mkdir(parents=True,exist_ok=True); a.out.write_text(json.dumps({'schema_version':'sightline-correspondence-v1','rows':rows}))
if __name__=='__main__': main()
