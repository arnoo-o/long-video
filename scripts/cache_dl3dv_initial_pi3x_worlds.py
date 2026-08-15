#!/usr/bin/env python3
"""Rebuild source-only Pi3X W0 caches; old geometry caches are rejected."""
from __future__ import annotations
import argparse, hashlib, json, subprocess
from pathlib import Path
import numpy as np
from PIL import Image

GEOMETRY_SCHEMA_VERSION = 3
GEOMETRY_IMPLEMENTATION_VERSION = "pi3x-w0-recal-prefix-replay-v2"

def identity(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def main():
    p=argparse.ArgumentParser(); p.add_argument('--dataset-root',type=Path,required=True); p.add_argument('--cache-root',type=Path,required=True)
    p.add_argument('--pi3x-repo',type=Path,required=True); p.add_argument('--pi3x-checkpoint',type=Path,required=True); p.add_argument('--device',default='cuda:0'); p.add_argument('--record-count',type=int,default=100); a=p.parse_args()
    from long_video.initialization.pi3x_geometry_backend import Pi3XGeometryBackend
    from long_video.initialization.pi3x_initial_world import build_pi3x_source_world
    from long_video.memory.node_store import NodeStore
    from long_video.training.wpf_adaptation import select_balanced_training_records
    records=select_balanced_training_records(json.loads((a.dataset_root/'dl3dv_24fps_manifest.json').read_text())['records'],a.record_count)
    backend=Pi3XGeometryBackend(a.pi3x_checkpoint,a.pi3x_repo,a.device); a.cache_root.mkdir(parents=True,exist_ok=True)
    commit=subprocess.check_output(['git','-C',str(a.pi3x_repo),'rev-parse','HEAD'],text=True).strip()
    for record in records:
        paths=sorted((a.dataset_root/record['rgb_dir']).glob('*'))
        source_index=int(record.get('source_frame_index',0)); source_path=paths[source_index]
        c2w=np.load(a.dataset_root/record['target_c2w_local']).astype(np.float32)[source_index]
        k=np.load(a.dataset_root/record['intrinsics']).astype(np.float32); k=k[source_index] if k.ndim==3 else k
        rgb=np.asarray(Image.open(source_path).convert('RGB'),np.uint8)
        target=a.cache_root/record['trajectory_id']; target.mkdir(parents=True,exist_ok=True)
        node=build_pi3x_source_world(rgb,c2w,k,backend); NodeStore(target).save(node)
        meta={'schema_version':GEOMETRY_SCHEMA_VERSION,'geometry_implementation_version':GEOMETRY_IMPLEMENTATION_VERSION,
              'trajectory_id':record['trajectory_id'],'source_frame_index':source_index,'source_rgb_sha256':identity(source_path),
              'pi3x_repo_commit':commit,'pi3x_checkpoint_sha256':identity(a.pi3x_checkpoint),'recal3r_checkpoint_identity':None,
              'confidence_calibration':{'pi3x':'native_sigmoid','recal3r':'sigmoid_threshold_1.5_temperature_0.35'},
              'voxel_size':0.02,'alignment_version':'recal_to_pi3x_w0_source-anchored-v2','uses_only_source':True}
        (target/'cache_metadata.json').write_text(json.dumps(meta,indent=2)); print(json.dumps({'trajectory_id':record['trajectory_id'],'status':'rebuilt'}),flush=True)
if __name__=='__main__': main()
