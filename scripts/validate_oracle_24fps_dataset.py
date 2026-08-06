#!/usr/bin/env python3
"""Validate the complete sparse-anchor to dense-24-FPS Oracle manifest."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np

def main():
    p=argparse.ArgumentParser(); p.add_argument("--manifest",required=True); args=p.parse_args()
    manifest=json.loads(Path(args.manifest).read_text(encoding="utf-8")); counts={"train":0,"diagnostic":0,"rollout":0}; anchors=set(); results=[]
    for item in manifest["sequences"]:
        counts[item["split"]]+=1; root=Path(item["path"]); metadata=json.loads((root/"metadata.json").read_text(encoding="utf-8"))
        expected=129 if item["split"]=="rollout" else 33; anchor_count=17 if item["split"]=="rollout" else 5
        rgb=list((root/"target"/"target_rgb_for_loss").glob("*.png")); poses=np.load(root/"target"/"target_c2w_local.npy")
        depth=np.load(root/"target"/"target_z_depth_for_eval.npy"); weights=np.load(root/"target"/"supervision_weights_rgb.npy")
        indices=np.arange(anchor_count)*8; interpolated=np.ones(expected,bool); interpolated[indices]=False
        if not(len(rgb)==len(poses)==len(depth)==len(weights)==expected): raise ValueError(f"dense length mismatch: {root}")
        if not np.isnan(depth[interpolated]).all(): raise ValueError(f"interpolated GT depth is not NaN: {root}")
        if not np.isfinite(depth[indices]).any(axis=(1,2)).all(): raise ValueError(f"anchor GT depth missing: {root}")
        if weights[0]!=0 or not np.all(weights[indices[1:]]==1) or not np.all(weights[interpolated]==.25): raise ValueError(f"supervision weights invalid: {root}")
        overlap=anchors.intersection(metadata["anchor_frame_ids"])
        if overlap: raise ValueError(f"real anchor leakage across splits/windows: {sorted(overlap)}")
        anchors.update(metadata["anchor_frame_ids"]); results.append({"sequence_id":metadata["sequence_id"],"frames":expected,"anchors":anchor_count})
    if counts!={"train":8,"diagnostic":2,"rollout":1}: raise ValueError(f"split counts invalid: {counts}")
    result={"passed":True,"split_counts":counts,"unique_real_anchors":len(anchors),"sequences":results}; print(json.dumps(result,indent=2))

if __name__=="__main__": main()
