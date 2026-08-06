#!/usr/bin/env python3
"""Report continuous timestamp runs in a Holo360D archive."""
import argparse
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--zip",required=True)
    parser.add_argument("--max-gap-factor",type=float,default=2.5)
    args=parser.parse_args()
    with ZipFile(args.zip) as handle:
        names=sorted(
            (name for name in handle.namelist() if "/rgb/" in name and name.endswith(".jpg")),
            key=lambda name:float(Path(name).stem),
        )
    timestamps=np.asarray([float(Path(name).stem) for name in names],np.float64)
    delta=np.diff(timestamps)
    base=float(np.median(delta[delta<np.percentile(delta,75)]))
    breaks=np.flatnonzero(delta>args.max_gap_factor*base)+1
    starts=np.r_[0,breaks]; ends=np.r_[breaks,len(names)]
    runs=sorted(
        ({"start":int(start),"end_exclusive":int(end),"length":int(end-start),
          "first_frame_id":Path(names[start]).stem,"last_frame_id":Path(names[end-1]).stem}
         for start,end in zip(starts,ends)),
        key=lambda item:item["length"],reverse=True,
    )
    print(json.dumps({"frames":len(names),"base_delta":base,"longest_runs":runs[:20]},indent=2))


if __name__=="__main__": main()
