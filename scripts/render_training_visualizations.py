#!/usr/bin/env python3
"""Render the three static 24 FPS training/result visualizations."""
from __future__ import annotations
import argparse,json,shutil
from pathlib import Path
import numpy as np

def main():
    p=argparse.ArgumentParser(); p.add_argument("--sequence",required=True); p.add_argument("--rife-comparison",required=True)
    p.add_argument("--training-result",required=True); p.add_argument("--output",required=True); args=p.parse_args()
    import matplotlib.pyplot as plt
    root=Path(args.sequence); out=Path(args.output); out.mkdir(parents=True,exist_ok=True)
    metadata=json.loads((root/"metadata.json").read_text(encoding="utf-8")); poses=np.load(root/"target"/"target_c2w_local.npy")[:33]
    rgb_weights=np.load(root/"target"/"supervision_weights_rgb.npy")[:33]; latent=np.load(root/"primary_loss_weight_latent.npy")
    anchors=np.asarray(metadata["anchor_model_indices"][:5]); frames=np.arange(33)
    figure,axes=plt.subplots(3,1,figsize=(12,9),constrained_layout=True)
    axes[0].scatter(frames,rgb_weights,c=np.where(np.isin(frames,anchors),"tab:blue","tab:orange")); axes[0].plot(frames,rgb_weights,alpha=.4)
    axes[0].set(title="24 FPS timeline: real anchors and RIFE supervision",ylabel="RGB weight",xticks=anchors)
    axes[1].bar(np.arange(len(latent)),latent); axes[1].set(title="VAE temporal-group mean weights",ylabel="latent weight",xlabel="latent index")
    xyz=poses[:,:3,3]; axes[2].plot(xyz[:,0],xyz[:,2],"-",label="SLERP/linear dense trajectory"); axes[2].scatter(xyz[anchors,0],xyz[anchors,2],label="real Holo anchors")
    axes[2].set(title="Source-relative camera trajectory",xlabel="x (m)",ylabel="z (m)"); axes[2].axis("equal"); axes[2].legend()
    figure.savefig(out/"timeline_and_trajectory.png",dpi=150); plt.close(figure)
    shutil.copy2(args.rife_comparison,out/"interpolation_comparison.png")
    result=json.loads(Path(args.training_result).read_text(encoding="utf-8")); logs=result["logs"]; steps=[x["step"] for x in logs]
    figure,axes=plt.subplots(3,2,figsize=(14,10),constrained_layout=True)
    axes[0,0].plot(steps,[x["train_weighted_loss"] for x in logs]); axes[0,0].set_title("Train weighted loss")
    diagnostic=[x for x in logs if "fixed_diagnostic_loss" in x]; axes[0,1].plot([x["step"] for x in diagnostic],[x["fixed_diagnostic_loss"] for x in diagnostic]); axes[0,1].set_title("Fixed diagnostic loss")
    axes[1,0].plot(steps,[x["learning_rate"] for x in logs]); axes[1,0].set_title("Learning rate")
    axes[1,1].plot(steps,[x["gradient_norm"] for x in logs]); axes[1,1].set_title("Gradient norm")
    axes[2,0].plot(steps,[x["gpu_reserved_bytes"]/2**30 for x in logs],label="reserved"); axes[2,0].plot(steps,[x["gpu_allocated_bytes"]/2**30 for x in logs],label="allocated"); axes[2,0].set_title("GPU memory (GiB)"); axes[2,0].legend()
    names=("correct","shuffled","empty"); before=result.get("diagnostics_before",{}); after=result.get("diagnostics_after",{})
    x=np.arange(3); axes[2,1].bar(x-.18,[before.get(k,np.nan) for k in names],.36,label="before"); axes[2,1].bar(x+.18,[after.get(k,np.nan) for k in names],.36,label="after")
    axes[2,1].set_xticks(x,names); axes[2,1].set_title("Warp diagnostics"); axes[2,1].legend()
    figure.savefig(out/"training_curves.png",dpi=150); plt.close(figure)
    print(json.dumps({"files":[str(out/name) for name in ("timeline_and_trajectory.png","interpolation_comparison.png","training_curves.png")]},indent=2))

if __name__=="__main__": main()
