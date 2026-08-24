#!/usr/bin/env python3
"""Rebuild sparse-LiDAR DDAD causal caches without grid aliasing."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from long_video.data.rgbd_memory import build_causal_correspondence_cache


def rebuild(root: Path) -> tuple[str, int]:
    result=build_causal_correspondence_cache(sorted((root/'depth').glob('*.png')),np.load(root/'c2w_abs.npy'),np.load(root/'intrinsics.npy'),root/'correspondence_cache.npz',chunk_count=3,pixel_stride=1)
    metadata=root/'metadata.json'; value=json.loads(metadata.read_text(encoding='utf-8'));value['correspondence']=result
    temporary=metadata.with_suffix('.json.tmp');temporary.write_text(json.dumps(value,indent=2),encoding='utf-8');os.replace(temporary,metadata)
    return root.name,int(result['row_count'])


def main() -> None:
    parser=argparse.ArgumentParser();parser.add_argument('--records-root',type=Path,required=True);parser.add_argument('--workers',type=int,default=8);args=parser.parse_args()
    roots=sorted(path for path in args.records_root.iterdir() if (path/'metadata.json').is_file())
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        results=list(pool.map(rebuild,roots))
    zero=[record_id for record_id,count in results if count==0]
    print(json.dumps({'records':len(results),'rows':sum(count for _,count in results),'zero_records':zero},indent=2))
    if zero: raise RuntimeError(f'DDAD still has empty correspondence caches: {zero}')


if __name__=='__main__': main()
