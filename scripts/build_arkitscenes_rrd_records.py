#!/usr/bin/env python3
"""Build 193-frame ARKitScenes records from Rerun RRD layers."""
from __future__ import annotations
import argparse, json, os, shutil
from pathlib import Path
from collections import Counter, defaultdict
import av, cv2, numpy as np
from scipy.spatial.transform import Rotation
from rerun.experimental import RrdReader
from long_video.data.rgbd_memory import (build_causal_correspondence_cache, center_crop_resize_geometry, localize_c2w, HEIGHT, WIDTH)

N, CHUNKS, FPS, SRC_FPS = 193, 6, 24.0, 60.0

def atomic(path, value):
    path.parent.mkdir(parents=True, exist_ok=True); tmp=path.with_suffix(path.suffix+'.tmp'); tmp.write_text(json.dumps(value,indent=2,sort_keys=True)); os.replace(tmp,path)

def props(path):
    out={}
    for c in RrdReader(path).store().stream():
        if c.entity_path.startswith('/__properties/'):
            d=c.to_record_batch().to_pydict()
            if 'value' in d and d['value']: out[c.entity_path.rsplit('/',1)[-1]]=d['value'][0][0]
    return out

def table(path, entity, required):
    rows=[]
    for c in RrdReader(path).store().stream():
        if c.entity_path == entity:
            d=c.to_record_batch().to_pydict()
            if all(k in d for k in required): rows.append(d)
    out={k:[] for k in required+['video_time']}
    for d in rows:
        for k in out: out[k].extend(d.get(k,[]))
    return out

def decode_video(path, max_frames=600):
    packets=[]
    for c in RrdReader(path).store().stream():
        if c.entity_path.endswith('/pinhole/video'):
            d=c.to_record_batch().to_pydict()
            if 'VideoStream:sample' in d: packets.extend(bytes(x[0]) for x in d['VideoStream:sample'])
    ctx=av.CodecContext.create('libdav1d','r'); frames=[]
    for packet in packets:
        frames.extend(ctx.decode(av.Packet(packet)))
        if len(frames) >= max_frames:
            break
    if len(frames) < max_frames:
        frames.extend(ctx.decode(None))
    return [f.to_ndarray(format='bgr24') for f in frames[:max_frames]]

def decode_depth(path, max_frames=600):
    blobs=[]; times=[]; conf=[]
    for c in RrdReader(path).store().stream():
        d=c.to_record_batch().to_pydict()
        if c.entity_path.endswith('/depth') and 'EncodedDepthImage:blob' in d:
            blobs.extend(bytes(x[0]) for x in d['EncodedDepthImage:blob']); times.extend(d['video_time'])
        if c.entity_path.endswith('/confidence') and 'SegmentationImage:buffer' in d:
            conf.extend(bytes(x[0]) for x in d['SegmentationImage:buffer'])
    depths=[cv2.imdecode(np.frombuffer(b,np.uint8),cv2.IMREAD_UNCHANGED) for b in blobs[:max_frames]]
    return depths, times[:max_frames], conf[:max_frames]

def calibration(path):
    t=table(path,'/world/rig_00',['Transform3D:quaternion','Transform3D:translation']); p=table(path,'/world/rig_00/cam_00/pinhole',['Pinhole:image_from_camera'])
    times=np.asarray([x.total_seconds() for x in t['video_time']],np.float64)
    q=np.asarray([x[0] for x in t['Transform3D:quaternion']],np.float64); tr=np.asarray([x[0] for x in t['Transform3D:translation']],np.float64)
    c2w=np.repeat(np.eye(4)[None],len(q),0); c2w[:,:3,:3]=Rotation.from_quat(q).as_matrix(); c2w[:,:3,3]=tr
    K=np.asarray([np.asarray(x[0],np.float64).reshape(3,3) for x in p['Pinhole:image_from_camera']],np.float64)
    return times,c2w,K

def select_indices(times, start):
    target=times[start]+np.arange(N)/FPS; pos=np.searchsorted(times,target); left=np.clip(pos-1,0,len(times)-1); right=np.clip(pos,0,len(times)-1); use=np.where(np.abs(times[right]-target)<np.abs(times[left]-target),right,left)
    if use[0]!=start or len(np.unique(use))!=N or np.any(np.diff(use)<=0) or np.max(np.abs(times[use]-target))>0.010: return None
    return use

def pointcloud(depths,K,c2w,out):
    xyz=[]; offs=[0]; yy,xx=np.mgrid[0:HEIGHT:4,0:WIDTH:4]
    for i,d in enumerate(depths):
        z=d.astype(np.float32)/1000; valid=z[::4,::4]>0; zz=z[::4,::4][valid]; x=xx[valid]; y=yy[valid]; cam=(np.linalg.inv(K[i])@np.stack((x*zz,y*zz,zz))).T; xyz.append((c2w[i,:3]@np.c_[cam,np.ones(len(cam))].T).T.astype(np.float32)); offs.append(offs[-1]+len(xyz[-1]))
    np.savez_compressed(out,xyz_world=np.concatenate(xyz),offsets=np.asarray(offs,np.int64),source_frame_indices=np.arange(N,dtype=np.int32),timestamps=np.arange(N,dtype=np.float64)/FPS); return int(offs[-1])

def build_one(video_id, root, out, ordinal, split):
    base=root/'base'/f'{video_id}.rrd'; props0=props(base)
    if props0.get('pose_source')!='mebx_stream_4_vision_transform' or props0.get('orientation_source')!='measured_gravity': return False,'provenance'
    if not all((root/layer/f'{video_id}.rrd').is_file() for layer in ('calibration','video_wide','depth')):
        return False,'missing_layers'
    times,c2w,K=calibration(root/'calibration'/f'{video_id}.rrd'); frames=decode_video(root/'video_wide'/f'{video_id}.rrd'); depths,dtimes,conf=decode_depth(root/'depth'/f'{video_id}.rrd')
    if not frames or not depths or len(frames) < N or len(times) < N: return False,'missing_stream'
    # Streams may contain different tail lengths; use the common decoded prefix.
    common = min(len(frames), len(times), len(depths))
    frames, times, c2w, K, depths = frames[:common], times[:common], c2w[:common], K[:common], depths[:common]
    for start in range(0,len(times)-N+1,N+60):
        ids=select_indices(times,start)
        if ids is None: continue
        rid=f'arkitscenes__{video_id}__{ordinal:04d}'; dest=out/'records'/'arkitscenes'/rid
        if (dest/'metadata.json').is_file(): return True,rid
        tmp=dest.with_name(dest.name+'.tmp'); shutil.rmtree(tmp,ignore_errors=True); (tmp/'rgb').mkdir(parents=True); (tmp/'depth').mkdir()
        Ks=[]; ds=[]
        try:
            for j,src in enumerate(ids):
                rgb=frames[src]; dep=depths[min(src,len(depths)-1)]
                crop,K0=center_crop_resize_geometry(rgb.shape[0],rgb.shape[1],K[src]); l,t,r,b=crop; rgb=cv2.resize(rgb[t:b,l:r],(WIDTH,HEIGHT),interpolation=cv2.INTER_AREA); dep=cv2.resize(dep[t:b,l:r],(WIDTH,HEIGHT),interpolation=cv2.INTER_NEAREST); cv2.imwrite(str(tmp/'rgb'/f'{j:06d}.png'),rgb); cv2.imwrite(str(tmp/'depth'/f'{j:06d}.png'),dep); Ks.append(K0); ds.append(dep)
            Ks=np.stack(Ks); pose=c2w[ids]; target=np.arange(N,dtype=np.float64)/FPS; np.save(tmp/'c2w_abs.npy',pose); np.save(tmp/'c2w_local.npy',localize_c2w(pose)); np.save(tmp/'intrinsics.npy',Ks); np.save(tmp/'timestamps.npy',target); np.save(tmp/'source_timestamps.npy',times[ids]); np.save(tmp/'source_frame_indices.npy',ids.astype(np.int32)); points=pointcloud(ds,Ks,pose,tmp/'pointcloud.npz'); corr=build_causal_correspondence_cache(sorted((tmp/'depth').glob('*.png')),pose,Ks,tmp/'correspondence_cache.npz',chunk_count=CHUNKS,pixel_stride=8)
            if corr['row_count']==0: raise ValueError('empty_correspondence')
            meta={'schema_version':'rgbd-memory-record-v3','record_id':rid,'dataset':'arkitscenes','scene_id':props0.get('visit_id',video_id),'sequence_id':video_id,'visit_id':props0.get('visit_id',video_id),'split':split,'frame_count':N,'chunk_count':CHUNKS,'chunk_stride':32,'fps':FPS,'duration_seconds':8.0,'pose_source':props0.get('pose_source'),'orientation_source':props0.get('orientation_source'),'source_frame_indices':ids.tolist(),'source_timing':'RRD video_time 60Hz nearest mapping','pointcloud_points':points,'correspondence':corr,'depth_confidence':'RRD EncodedDepthImage PNG + confidence'}; atomic(tmp/'metadata.json',meta); dest.parent.mkdir(parents=True,exist_ok=True); os.replace(tmp,dest); return True,rid
        except Exception as e: shutil.rmtree(tmp,ignore_errors=True); return False,str(e)
    return False,'no_valid_window'

def main():
    p=argparse.ArgumentParser(); p.add_argument('--rrd-root',type=Path,required=True); p.add_argument('--output-root',type=Path,required=True); p.add_argument('--max-records',type=int,default=600); p.add_argument('--start',type=int,default=0); a=p.parse_args(); ids=sorted(x.stem for x in (a.rrd_root/'base').glob('*.rrd')); stats=Counter(); rows=[]
    for vid in ids[a.start:]:
        ok,r=build_one(vid,a.rrd_root,a.output_root,len(rows),'candidate'); stats['built' if ok else r]+=1
        if ok: rows.append(r)
        if len(rows)>=a.max_records: break
    print(json.dumps({'records':len(rows),'stats':dict(stats)},indent=2))
if __name__=='__main__': main()
