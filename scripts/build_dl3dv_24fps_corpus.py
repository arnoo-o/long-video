#!/usr/bin/env python3
"""Build a resumable balanced 180-trajectory DL3DV 24fps RIFE corpus."""
from __future__ import annotations
import argparse, json, shutil, subprocess, sys, time
from collections import defaultdict
from pathlib import Path
import numpy as np

CHUNKS=6; FRAME_COUNT=CHUNKS*32+1

def parse_args():
 p=argparse.ArgumentParser(); p.add_argument('--old-manifest',type=Path,required=True)
 p.add_argument('--selection-state',type=Path,required=True); p.add_argument('--output-root',type=Path,required=True)
 p.add_argument('--single-script',type=Path,required=True); p.add_argument('--python',required=True)
 p.add_argument('--rife-root',type=Path,required=True); p.add_argument('--rife-checkpoint',type=Path,required=True)
 p.add_argument('--device',default='cuda:0'); p.add_argument('--train-count',type=int,default=150)
 p.add_argument('--val-count',type=int,default=30); p.add_argument('--jpeg-quality',type=int,default=95)
 return p.parse_args()

def load_candidates(args):
 m=json.loads(args.old_manifest.read_text()); state=json.loads(args.selection_state.read_text())
 meta={x['scene_hash']:x for x in state['qualified']}
 env={x['scene_hash']:x['environment'] for x in m['scenes']}
 records=defaultdict(list)
 for r in m['records']: records[r['scene_hash']].append(r)
 pools={'train':{'indoor':[],'outdoor':[]},'val':{'indoor':[],'outdoor':[]}}
 for scene_hash, rows in records.items():
  if scene_hash not in meta or env.get(scene_hash) not in ('indoor','outdoor'): continue
  old_split=rows[0]['split']; split='train' if old_split=='train' else 'val'
  rows.sort(key=lambda r:(r['sample_type']!='source_revisit',-int(r['chunk_count']),r['trajectory_id']))
  r=rows[0]; item={**meta[scene_hash], 'environment':env[scene_hash], 'split':split,
      'start_real_index':int(r['source_global_frame']), 'old_trajectory_id':r['trajectory_id'],
      'sample_type':r['sample_type']}
  pools[split][env[scene_hash]].append(item)
 for split in pools:
  for e in pools[split]: pools[split][e].sort(key=lambda x:x['scene_hash'])
 selected=[]
 for split,count in [('train',args.train_count),('val',args.val_count)]:
  left=count//2; right=count-left
  selected.extend(pools[split]['indoor'][:left]); selected.extend(pools[split]['outdoor'][:right])
 if len(selected)!=args.train_count+args.val_count: raise RuntimeError('insufficient balanced scenes')
 return selected

def valid_output(path):
 try:
  v=json.loads((path/'validation.json').read_text()); p=np.load(path/'target_c2w_local.npy')
  return v.get('valid') and v.get('frame_count')==FRAME_COUNT and p.shape==(FRAME_COUNT,4,4) and len(list((path/'rgb_24fps').glob('*.jpg')))==FRAME_COUNT
 except Exception: return False

def write_manifest(root, records, failures):
 manifest={'schema_version':2,'dataset':'DL3DV-10K official 480P images+poses + Practical-RIFE 4.25',
  'fps':24.0,'resolution':[384,640],'chunk_frames':33,'chunk_stride':32,'chunk_count':CHUNKS,
  'trajectory_count':len(records),'split_counts':{'train':sum(x['split']=='train' for x in records),'val':sum(x['split']=='val' for x in records)},
  'rgb_pose_timestamp_aligned':True,'uses_future_gt':False,'records':records,'failures':failures}
 tmp=root/'dl3dv_24fps_manifest.json.tmp'; tmp.write_text(json.dumps(manifest,indent=2)); tmp.replace(root/'dl3dv_24fps_manifest.json')

def main():
 a=parse_args(); a.output_root.mkdir(parents=True,exist_ok=True); selected=load_candidates(a)
 records=[]; failures=[]; started=time.time()
 for ordinal,item in enumerate(selected):
  tid=f"{item['scene_hash']}_rife24_6chunk"; final=a.output_root/item['split']/item['scene_hash']/tid
  if valid_output(final): status='cached'
  else:
   temp=final.with_name(tid+'.building'); shutil.rmtree(temp,ignore_errors=True); temp.parent.mkdir(parents=True,exist_ok=True)
   cmd=[a.python,str(a.single_script),'--scene-root',item['raw_path'],'--duration',str(item['duration']),
    '--output',str(temp),'--start-real-index',str(item['start_real_index']),'--frame-count',str(FRAME_COUNT),
    '--rife-root',str(a.rife_root),'--rife-checkpoint',str(a.rife_checkpoint),'--device',a.device,
    '--jpeg-quality',str(a.jpeg_quality)]
   result=subprocess.run(cmd,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
   if result.returncode or not valid_output(temp):
    failures.append({'scene_hash':item['scene_hash'],'returncode':result.returncode,'output':result.stdout[-4000:]}); shutil.rmtree(temp,ignore_errors=True)
    print(json.dumps({'index':ordinal,'status':'failed','scene_hash':item['scene_hash']}),flush=True); continue
   shutil.rmtree(final,ignore_errors=True); temp.replace(final); status='built'
  rel=lambda p:str(p.relative_to(a.output_root)).replace('\\','/')
  record={'trajectory_id':tid,'scene_hash':item['scene_hash'],'split':item['split'],'environment':item['environment'],
   'sample_type':item['sample_type'],'chunk_count':CHUNKS,'frame_count':FRAME_COUNT,'fps':24.0,
   'source_global_frame':item['start_real_index'],'source':rel(final/'source/source.png'),'rgb_dir':rel(final/'rgb_24fps'),
   'target_c2w_local':rel(final/'target_c2w_local.npy'),'intrinsics':rel(final/'intrinsics.npy'),
   'timestamps':rel(final/'timestamps.npy'),'frame_sources':rel(final/'frame_sources.json'),
   'source_frame_indices':rel(final/'source_frame_indices.npy'),'real_keyframe_indices':rel(final/'real_keyframe_indices.npy'),
   'pi3_initial_rgb_dir':rel(final/'pi3_initial_real'),'pi3_initial_c2w_local':rel(final/'pi3_initial_c2w_local.npy'),
   'pi3_initial_intrinsics':rel(final/'pi3_initial_intrinsics.npy'),'pi3_initial_real_frame_indices':rel(final/'pi3_initial_real_frame_indices.npy'),
   'chunk_frame_indices':[list(range(k*32,k*32+33)) for k in range(CHUNKS)],'trainable_chunk_indices':list(range(CHUNKS)),'uses_future_gt':False,'old_trajectory_id':item['old_trajectory_id']}
  records.append(record); write_manifest(a.output_root,records,failures)
  print(json.dumps({'index':ordinal+1,'total':len(selected),'status':status,'trajectory_id':tid,'elapsed_sec':round(time.time()-started,1)}),flush=True)
 if len(records)!=len(selected): raise SystemExit(f'built {len(records)}/{len(selected)}; failures={len(failures)}')
 print(json.dumps({'complete':True,'records':len(records),'elapsed_sec':time.time()-started},indent=2))
if __name__=='__main__': main()
