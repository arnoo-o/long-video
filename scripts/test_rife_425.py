#!/usr/bin/env python3
"""Real Holo leave-one-anchor-out gate for Practical-RIFE 4.25 full."""
from __future__ import annotations
import argparse,io,json,os,sys,tempfile
from pathlib import Path
from zipfile import ZipFile

def main():
    p=argparse.ArgumentParser(); p.add_argument("--zip",required=True); p.add_argument("--rife-root",required=True)
    p.add_argument("--checkpoint",required=True); p.add_argument("--rife-python",required=True); p.add_argument("--output",required=True)
    p.add_argument("--physical-gpu",type=int,default=1); args=p.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"]=str(args.physical_gpu)
    if args.physical_gpu!=1: raise ValueError("only physical GPU 1 is permitted")
    repo=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(repo))
    import numpy as np
    from PIL import Image,ImageDraw
    from skimage.metrics import peak_signal_noise_ratio,structural_similarity
    from long_video.data.panorama_projection import equirectangular_to_perspective
    from long_video.oracle_training.dense24 import PracticalRIFE425,allocate_disjoint_windows,continuous_runs
    out=Path(args.output); out.mkdir(parents=True,exist_ok=True)
    with ZipFile(args.zip) as handle:
        names=sorted([n for n in handle.namelist() if "/rgb/" in n and n.endswith(".jpg")],key=lambda n:float(Path(n).stem))
        timestamps=np.asarray([float(Path(n).stem) for n in names]); runs,_=continuous_runs(timestamps)
        allocation=allocate_disjoint_windows(runs,train_count=8,diagnostic_count=2,rollout_anchors=17)
        candidates=[center for start in allocation["diagnostic"] for center in range(start+1,start+4)]
        images=[]
        for center in candidates:
            triplet=[]
            for index in (center-1,center,center+1):
                erp=np.asarray(Image.open(io.BytesIO(handle.read(names[index]))).convert("RGB"))
                triplet.append(equirectangular_to_perspective(erp,0.,0.,90.,384,640,interpolation="bilinear").astype(np.uint8))
            images.append(triplet)
    rife=PracticalRIFE425(args.rife_root,args.checkpoint,args.rife_python)
    rows=[]; metrics=[]
    for index,(left,truth,right) in enumerate(images):
        work=out/f"pair_{index}"; prediction=rife.interpolate(np.stack([left,right]),work,multiplier=2)[1]
        linear=np.rint((left.astype(np.float32)+right.astype(np.float32))*.5).astype(np.uint8)
        def score(value):
            return {"psnr":float(peak_signal_noise_ratio(truth,value,data_range=255)),
                    "ssim":float(structural_similarity(truth,value,channel_axis=2,data_range=255)),
                    "l1":float(np.abs(truth.astype(np.float32)-value.astype(np.float32)).mean()/255.)}
        record={"pair":index,"rife":score(prediction),"linear":score(linear)}; metrics.append(record)
        rows.append((left,prediction,truth,linear,right,record))
    aggregate={kind:{metric:float(np.median([r[kind][metric] for r in metrics])) for metric in ("psnr","ssim","l1")} for kind in ("rife","linear")}
    passed=(aggregate["rife"]["psnr"]>=aggregate["linear"]["psnr"] and aggregate["rife"]["ssim"]>=aggregate["linear"]["ssim"] and
            (aggregate["rife"]["psnr"]>aggregate["linear"]["psnr"] or aggregate["rife"]["ssim"]>aggregate["linear"]["ssim"]))
    canvas=Image.new("RGB",(640*5,384*len(rows)+40),"white"); draw=ImageDraw.Draw(canvas)
    draw.text((5,5),"left | RIFE 4.25 full | true middle | linear | right",fill="black")
    for row_index,row in enumerate(rows):
        for column,image in enumerate(row[:5]): canvas.paste(Image.fromarray(image),(column*640,40+row_index*384))
    canvas.save(out/"interpolation_comparison.png")
    result={"checkpoint":str(Path(args.checkpoint)/"flownet.pkl"),"pairs":metrics,"median":aggregate,"passed":passed,
            "nan":bool(any(not np.isfinite(list(r[k].values())).all() for r in metrics for k in ("rife","linear")))}
    (out/"rife_gate.json").write_text(json.dumps(result,indent=2),encoding="utf-8"); print(json.dumps(result,indent=2))
    if not passed or result["nan"]: raise SystemExit(2)

if __name__=="__main__": main()
