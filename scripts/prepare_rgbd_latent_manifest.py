#!/usr/bin/env python3
"""Create the cache-only manifest before final latent validation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument('--existing-manifest',type=Path,required=True)
    parser.add_argument('--additions-root',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    payload=json.loads(args.existing_manifest.read_text(encoding='utf-8'))
    rows=[]
    for row in payload['records']:
        rgb=Path(row['rgb_dir'])
        if not rgb.is_absolute(): rgb=args.existing_manifest.parent/rgb
        rows.append({'record_id':row['record_id'],'frame_count':int(row['frame_count']),'rgb_dir':str(rgb.resolve())})
    for metadata in sorted(args.additions_root.glob('records/*/*/metadata.json')):
        row=json.loads(metadata.read_text(encoding='utf-8'))
        rows.append({'record_id':row['record_id'],'frame_count':int(row['frame_count']),'rgb_dir':str(metadata.parent/'rgb')})
    if len({row['record_id'] for row in rows})!=len(rows):
        raise ValueError('duplicate record ids in latent manifest')
    temporary=args.out.with_suffix(args.out.suffix+'.tmp');args.out.parent.mkdir(parents=True,exist_ok=True)
    temporary.write_text(json.dumps({'schema_version':'rgbd-latent-cache-input-v1','records':rows},indent=2),encoding='utf-8');temporary.replace(args.out)
    print(json.dumps({'records':len(rows),'out':str(args.out)},indent=2))


if __name__=='__main__': main()
