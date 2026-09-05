"""Attach deterministic per-trajectory near-depth metadata to a manifest."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from PIL import Image

def near_depth(row, root, samples=16):
    directory=Path(row['depth_dir']); directory=directory if directory.is_absolute() else root/directory
    frames=sorted(p for p in directory.iterdir() if p.suffix.lower() in {'.png','.jpg','.jpeg'})
    start=int(row.get('source_frame_start',0)); count=int(row['frame_count']); frames=frames[start:start+count]
    if len(frames)!=count: raise ValueError(f"{row.get('record_id')}: depth frame count")
    indices=np.unique(np.linspace(0,count-1,min(samples,count),dtype=int)); values=[]
    for i in indices:
        depth=np.asarray(Image.open(frames[int(i)]),dtype=np.float32)
        valid=depth[np.isfinite(depth)&(depth>0)]
        if valid.size: values.append(float(np.quantile(valid,.25)))
    value=float(np.median(values)) if values else float('nan')
    if not np.isfinite(value) or value<=0: raise ValueError(f"{row.get('record_id')}: invalid near_depth")
    return value

def main():
    p=argparse.ArgumentParser(); p.add_argument('manifest'); p.add_argument('--samples',type=int,default=16); a=p.parse_args()
    path=Path(a.manifest); data=json.loads(path.read_text()); root=path.parent
    for row in data['records']:
        value=near_depth(row,root,a.samples); row['near_depth']=value
        metadata=row.get('metadata')
        if metadata:
            mp=Path(metadata); mp=mp if mp.is_absolute() else root/mp
            if mp.is_file():
                payload=json.loads(mp.read_text()); payload['near_depth']=value; payload['near_depth_method']='median(frame_valid_depth_q25)'; mp.write_text(json.dumps(payload,indent=2))
    path.write_text(json.dumps(data,indent=2))
if __name__=='__main__': main()
