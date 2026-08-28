#!/usr/bin/env python3
"""Validate, split, report, and merge strict ARKitScenes RRD records."""
from __future__ import annotations
import argparse, json, os, shutil
from collections import Counter
from pathlib import Path
import numpy as np
from long_video.training.rgbd_memory_data import RGBDMemoryRecord

def atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(payload,indent=2,sort_keys=True),encoding='utf-8'); os.replace(tmp,path)

def row_for(root: Path, metadata: dict, latent_root: Path, split: str) -> dict:
    rid=metadata['record_id']; return {
        'record_id':rid,'dataset':'arkitscenes','scene_id':metadata['scene_id'],'sequence_id':metadata['sequence_id'],'split':split,
        'rgb_dir':str(root/'rgb'),'depth_dir':str(root/'depth'),'c2w_abs':str(root/'c2w_abs.npy'),'c2w_local':str(root/'c2w_local.npy'),
        'intrinsics':str(root/'intrinsics.npy'),'timestamps':str(root/'timestamps.npy'),'source_timestamps':str(root/'source_timestamps.npy'),
        'source_frame_indices':str(root/'source_frame_indices.npy'),'pointcloud':str(root/'pointcloud.npz'),'correspondence_cache':str(root/'correspondence_cache.npz'),
        'metadata':str(root/'metadata.json'),'latent_cache':str(latent_root/rid/'continuous_49.pt'),'frame_count':193,'chunk_count':6,
        'height':480,'width':832,'fps':24.0,'training_scope':'rgbd_memory','memory_eligible':True,'intrinsics_quality':'calibrated',
    }

def main() -> None:
    p=argparse.ArgumentParser(); p.add_argument('--records-root',type=Path,required=True); p.add_argument('--manifest-root',type=Path,required=True)
    p.add_argument('--latent-root',type=Path,required=True); p.add_argument('--target',type=int,default=600); p.add_argument('--val-target',type=int,default=60)
    p.add_argument('--report',type=Path,required=True); p.add_argument('--qa-root',type=Path,required=True); a=p.parse_args()
    valid=[]; rejected=[]
    for mp in sorted(a.records_root.glob('*/metadata.json')):
        try:
            m=json.loads(mp.read_text(encoding='utf-8')); row=row_for(mp.parent,m,a.latent_root,'candidate'); RGBDMemoryRecord(row,a.manifest_root).validate(); valid.append((mp,m))
        except Exception as exc: rejected.append({'metadata':str(mp),'reason':str(exc)})
    selected=valid[:a.target]; groups={}
    for item in selected: groups.setdefault(str(item[1]['visit_id']),[]).append(item)
    val=[]
    if len(selected)>=a.target:
        for _,items in sorted(groups.items(),key=lambda x:(len(x[1]),x[0])):
            if len(val)+len(items)<=a.val_target: val.extend(items)
            if len(val)==a.val_target: break
    val_ids={item[1]['record_id'] for item in val}; train=[x for x in selected if x[1]['record_id'] not in val_ids]
    if len(selected)<a.target: train=selected; val=[]
    train_rows=[row_for(mp.parent,m,a.latent_root,'train') for mp,m in train]; val_rows=[row_for(mp.parent,m,a.latent_root,'val') for mp,m in val]
    def merge(name, additions, split=None):
        path=a.manifest_root/name; payload=json.loads(path.read_text(encoding='utf-8')) if path.is_file() else {'schema_version':'rgbd-memory-manifest-v3','height':480,'width':832,'records':[]}
        payload['records']=[r for r in payload.get('records',[]) if r.get('dataset')!='arkitscenes']+additions
        if split: payload['split']=split
        atomic(path,payload)
    merge(Path('manifest_all.json'),train_rows+val_rows); merge(Path('manifest_train.json'),train_rows,'train'); merge(Path('manifest_val.json'),val_rows,'val'); merge(Path('manifest_train_p3.json'),train_rows,'train')
    a.qa_root.mkdir(parents=True,exist_ok=True)
    for mp,m in selected:
        src=mp.parent/'qa_7frames.jpg'
        if src.is_file(): shutil.copy2(src,a.qa_root/f"{m['record_id']}.jpg")
    corr=[]; points=[]; depth=[]; timing=[]; scenes=set(); sequences=set()
    for _,m in selected:
        corr.append(int(m['correspondence']['row_count'])); points.append(int(m['pointcloud_points'])); depth.append(float(m['depth_valid_fraction_median']))
        timing.append(max(float(x) for x in m['timing_validation'].values())); scenes.add(m['scene_id']); sequences.add(m['sequence_id'])
    report={'requested':a.target,'strict_valid_available':len(valid),'selected':len(selected),'train':len(train_rows),'val':len(val_rows),'scene_count':len(scenes),'sequence_count':len(sequences),'rejected_count':len(rejected),'rejected_reasons':dict(Counter(x['reason'] for x in rejected)),'correspondence_rows_total':sum(corr),'correspondence_rows_median':float(np.median(corr)) if corr else 0,'pointcloud_points_total':sum(points),'depth_valid_median':float(np.median(depth)) if depth else 0,'max_cross_stream_timing_error_seconds':max(timing,default=0),'qa_count':len(list(a.qa_root.glob('*.jpg'))),'records_bytes':sum(f.stat().st_size for mp,_ in selected for f in mp.parent.rglob('*') if f.is_file())}
    atomic(a.report,report); print(json.dumps(report,indent=2))

if __name__=='__main__': main()
