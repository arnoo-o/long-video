#!/usr/bin/env python3
"""Minimal arbitrary-timestep inference for Practical-RIFE 4.25 full."""
from __future__ import annotations
import argparse, os, sys
from pathlib import Path

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--rife-root",required=True); parser.add_argument("--checkpoint",required=True)
    parser.add_argument("--input",required=True); parser.add_argument("--output",required=True)
    parser.add_argument("--multiplier",type=int,default=8); parser.add_argument("--device",default="cuda:0")
    args=parser.parse_args()
    if args.device!="cuda:0" or os.environ.get("CUDA_VISIBLE_DEVICES")!="1":
        raise RuntimeError("RIFE requires CUDA_VISIBLE_DEVICES=1 and process device cuda:0")
    import numpy as np
    import torch
    from PIL import Image
    if torch.cuda.device_count()!=1: raise RuntimeError("RIFE must see exactly one GPU")
    root=Path(args.rife_root).resolve(); checkpoint=Path(args.checkpoint).resolve()
    sys.path.insert(0,str(root)); sys.path.insert(0,str(checkpoint.parent))
    from train_log.RIFE_HDv3 import Model
    model=Model(); model.load_model(str(Path(args.checkpoint).resolve()),-1); model.eval(); model.device()
    paths=sorted(Path(args.input).glob("*.png"))
    if len(paths)<2: raise ValueError("at least two anchor PNGs are required")
    def tensor(path):
        image=np.asarray(Image.open(path).convert("RGB"),np.float32)/255.0; h,w=image.shape[:2]
        value=torch.from_numpy(image).permute(2,0,1).unsqueeze(0).to("cuda:0")
        ph=((h+63)//64)*64; pw=((w+63)//64)*64
        return torch.nn.functional.pad(value,(0,pw-w,0,ph-h)),h,w
    dense=[]
    with torch.inference_mode():
        for i in range(len(paths)-1):
            left,h,w=tensor(paths[i]); right,rh,rw=tensor(paths[i+1])
            if (h,w)!=(rh,rw): raise ValueError("anchor resolutions differ")
            dense.append(np.asarray(Image.open(paths[i]).convert("RGB")))
            for step in range(1,args.multiplier):
                prediction=model.inference(left,right,timestep=step/args.multiplier)
                frame=prediction[0,:,:h,:w].clamp(0,1).permute(1,2,0)
                dense.append(np.rint(frame.float().cpu().numpy()*255).astype(np.uint8))
    dense.append(np.asarray(Image.open(paths[-1]).convert("RGB")))
    np.save(args.output,np.stack(dense))
    print({"frames":len(dense),"multiplier":args.multiplier,"peak_allocated_bytes":int(torch.cuda.max_memory_allocated(0)),"peak_reserved_bytes":int(torch.cuda.max_memory_reserved(0))})

if __name__=="__main__": main()
