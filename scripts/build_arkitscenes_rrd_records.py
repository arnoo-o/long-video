#!/usr/bin/env python3
"""Build 193-frame ARKitScenes records from Rerun RRD layers."""
from __future__ import annotations
import argparse, json, os, shutil
from pathlib import Path
from collections import Counter, defaultdict, deque
import av, cv2, numpy as np
from scipy.spatial.transform import Rotation
from rerun.experimental import RrdReader
from long_video.data.rgbd_memory import (build_causal_correspondence_cache, center_crop_resize_geometry, localize_c2w, HEIGHT, WIDTH)

N, CHUNKS, FPS, SRC_FPS = 193, 6, 24.0, 60.0
cv2.setNumThreads(1)

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

def _seconds(value):
    return float(value.total_seconds()) if hasattr(value, 'total_seconds') else float(value)

def decode_video(path, max_frames=600):
    ctx=av.CodecContext.create('libdav1d','r'); frames=[]; frame_times=[]; pending=deque()
    for c in RrdReader(path).store().stream():
        if c.entity_path.endswith('/pinhole/video'):
            d=c.to_record_batch().to_pydict()
            for sample, timestamp in zip(d.get('VideoStream:sample', ()), d.get('video_time', ())):
                pending.append(_seconds(timestamp))
                for frame in ctx.decode(av.Packet(bytes(sample[0]))):
                    frames.append(frame.to_ndarray(format='bgr24')); frame_times.append(pending.popleft())
                if len(frames) >= max_frames:
                    return frames[:max_frames], np.asarray(frame_times[:max_frames], np.float64)
    if len(frames) < max_frames:
        for frame in ctx.decode(None):
            if not pending: break
            frames.append(frame.to_ndarray(format='bgr24')); frame_times.append(pending.popleft())
    return frames[:max_frames], np.asarray(frame_times[:max_frames], np.float64)

def decode_depth(path, max_frames=600):
    blobs=[]; times=[]; conf=[]; conf_times=[]
    for c in RrdReader(path).store().stream():
        d=c.to_record_batch().to_pydict()
        if c.entity_path.endswith('/depth') and 'EncodedDepthImage:blob' in d:
            remaining=max_frames-len(blobs)
            if remaining>0:
                blobs.extend(bytes(x[0]) for x in d['EncodedDepthImage:blob'][:remaining]); times.extend(d['video_time'][:remaining])
        if c.entity_path.endswith('/confidence') and 'SegmentationImage:buffer' in d:
            remaining=max_frames-len(conf)
            if remaining>0:
                conf.extend(bytes(x[0]) for x in d['SegmentationImage:buffer'][:remaining]); conf_times.extend(d['video_time'][:remaining])
        if len(blobs)>=max_frames and len(conf)>=max_frames: break
    return blobs[:max_frames], np.asarray([_seconds(x) for x in times[:max_frames]], np.float64), conf[:max_frames], np.asarray([_seconds(x) for x in conf_times[:max_frames]],np.float64)

def calibration(path):
    t=table(path,'/world/rig_00',['Transform3D:quaternion','Transform3D:translation']); p=table(path,'/world/rig_00/cam_00/pinhole',['Pinhole:image_from_camera']); pl=table(path,'/world/rig_00/cam_00/pinhole_lowres',['Pinhole:image_from_camera'])
    times=np.asarray([x.total_seconds() for x in t['video_time']],np.float64)
    q=np.asarray([x[0] for x in t['Transform3D:quaternion']],np.float64); tr=np.asarray([x[0] for x in t['Transform3D:translation']],np.float64)
    rig2w=np.repeat(np.eye(4)[None],len(q),0); rig2w[:,:3,:3]=Rotation.from_quat(q).as_matrix(); rig2w[:,:3,3]=tr
    cam2rig=np.eye(4)
    for chunk in RrdReader(path).store().stream():
        if chunk.entity_path=='/world/rig_00/cam_00':
            values=chunk.to_record_batch().to_pydict()
            if 'Transform3D:quaternion' in values:
                cam2rig[:3,:3]=Rotation.from_quat(np.asarray(values['Transform3D:quaternion'][0][0],np.float64)).as_matrix()
                cam2rig[:3,3]=np.asarray(values.get('Transform3D:translation',[[[0,0,0]]])[0][0],np.float64); break
    c2w=rig2w@cam2rig
    ktimes=np.asarray([_seconds(x) for x in p['video_time']],np.float64)
    K=np.asarray([np.asarray(x[0],np.float64).reshape(3,3) for x in p['Pinhole:image_from_camera']],np.float64)
    # RRD stores the pinhole matrix in column-vector layout; normalize to OpenCV row layout.
    if np.mean(np.abs(K[:,2,2]-1.0)) < 1e-3 and np.mean(np.abs(K[:,2,:2])) < 1e-3:
        pass
    elif np.mean(np.abs(K[:,:,2][:,:2])) < 1e-3:
        K = np.transpose(K, (0,2,1))
    lktimes=np.asarray([_seconds(x) for x in pl['video_time']],np.float64); lowK=np.asarray([np.asarray(x[0],np.float64).reshape(3,3).T for x in pl['Pinhole:image_from_camera']],np.float64)
    return times,c2w,ktimes,K,lktimes,lowK,cam2rig

def select_indices(times, start):
    target=times[start]+np.arange(N)/FPS; pos=np.searchsorted(times,target); left=np.clip(pos-1,0,len(times)-1); right=np.clip(pos,0,len(times)-1); use=np.where(np.abs(times[right]-target)<np.abs(times[left]-target),right,left)
    if use[0]!=start or len(np.unique(use))!=N or np.any(np.diff(use)<=0) or np.max(np.abs(times[use]-target))>0.010: return None
    return use

def match_times(times, targets, tolerance=0.010):
    pos=np.searchsorted(times,targets); left=np.clip(pos-1,0,len(times)-1); right=np.clip(pos,0,len(times)-1)
    use=np.where(np.abs(times[right]-targets)<np.abs(times[left]-targets),right,left)
    if len(np.unique(use))!=len(use) or np.any(np.diff(use)<=0) or np.max(np.abs(times[use]-targets))>tolerance: return None
    return use

def pointcloud(depths,K,c2w,source_indices,timestamps,out):
    xyz=[]; offs=[0]; yy,xx=np.mgrid[0:HEIGHT:4,0:WIDTH:4]
    for i,d in enumerate(depths):
        z=d.astype(np.float32)/1000; valid=z[::4,::4]>0; zz=z[::4,::4][valid]; x=xx[valid]; y=yy[valid]; cam=(np.linalg.inv(K[i])@np.stack((x*zz,y*zz,zz))).T; xyz.append((c2w[i,:3]@np.c_[cam,np.ones(len(cam))].T).T.astype(np.float32)); offs.append(offs[-1]+len(xyz[-1]))
    np.savez(out,xyz_world=np.concatenate(xyz),offsets=np.asarray(offs,np.int64),source_frame_indices=np.asarray(source_indices,np.int32),timestamps=np.asarray(timestamps,np.float64)); return int(offs[-1])

def pose_metrics(c2w):
    trans=np.linalg.norm(np.diff(c2w[:,:3,3],axis=0),axis=1); rel=c2w[:-1,:3,:3].transpose(0,2,1)@c2w[1:,:3,:3]
    angle=np.rad2deg(np.arccos(np.clip((np.trace(rel,axis1=1,axis2=2)-1)/2,-1,1)))
    return {'max_translation_m':float(trans.max()),'max_rotation_degrees':float(angle.max())}

def render_qa(paths,out):
    panels=[]
    for index in (0,32,64,96,128,160,192):
        image=cv2.imread(str(paths[index]),cv2.IMREAD_COLOR); panel=cv2.resize(image,(416,240),interpolation=cv2.INTER_AREA)
        cv2.putText(panel,f'frame {index}',(10,25),cv2.FONT_HERSHEY_SIMPLEX,.7,(255,255,255),2,cv2.LINE_AA); panels.append(panel)
    cv2.imwrite(str(out),np.concatenate(panels,axis=1))

def build_one(video_id, root, out, ordinal, split):
    base=root/'base'/f'{video_id}.rrd'; props0=props(base)
    if props0.get('pose_source')!='mebx_stream_4_vision_transform' or props0.get('orientation_source')!='measured_gravity': return False,'provenance'
    if not all((root/layer/f'{video_id}.rrd').is_file() for layer in ('calibration','video_wide','depth')):
        return False,'missing_layers'
    pose_times,c2w,k_times,K,low_k_times,lowK,cam2rig=calibration(root/'calibration'/f'{video_id}.rrd'); frames,rgb_times=decode_video(root/'video_wide'/f'{video_id}.rrd'); depth_blobs,dtimes,conf,ctimes=decode_depth(root/'depth'/f'{video_id}.rrd')
    if not frames or not depth_blobs or len(frames) < N or len(pose_times) < N: return False,'missing_stream'
    limit=len(rgb_times)-N+1
    candidate_starts=list(range(0,min(60,max(0,limit))))+list(range(N+60,max(0,limit),N+60))
    for start in candidate_starts:
        ids=select_indices(rgb_times,start)
        if ids is None: continue
        selected_times=rgb_times[ids]
        pose_ids=match_times(pose_times,selected_times); depth_ids=match_times(dtimes,selected_times); k_ids=match_times(k_times,selected_times); low_k_ids=match_times(low_k_times,selected_times); conf_ids=match_times(ctimes,selected_times) if len(ctimes) else None
        if pose_ids is None or depth_ids is None or k_ids is None or low_k_ids is None or conf_ids is None: continue
        pose=c2w[pose_ids]; rotations=pose[:,:3,:3]
        if not np.allclose(rotations.transpose(0,2,1)@rotations,np.eye(3),atol=1e-4) or not np.allclose(np.linalg.det(rotations),1,atol=1e-4): continue
        pm=pose_metrics(pose)
        if pm['max_translation_m']>.50 or pm['max_rotation_degrees']>45: continue
        # Stable across sharding/restarts; one non-overlapping 8-second window per video.
        rid=f'arkitscenes__{video_id}__{int(ids[0]):06d}'; dest=out/'records'/'arkitscenes'/rid
        existing = list((out/'records'/'arkitscenes').glob(f'arkitscenes__{video_id}__*/metadata.json'))
        if existing:
            return True, existing[0].parent.name
        if (dest/'metadata.json').is_file(): return True,rid
        tmp=dest.with_name(dest.name+'.tmp'); shutil.rmtree(tmp,ignore_errors=True); (tmp/'rgb').mkdir(parents=True); (tmp/'depth').mkdir()
        Ks=[]; ds=[]; means=[]; stds=[]; changes=[]; previous=None; depth_valid=[]; confidence_valid=[]
        try:
            for j,src in enumerate(ids):
                rgb=frames[src]; dep=cv2.imdecode(np.frombuffer(depth_blobs[depth_ids[j]],np.uint8),cv2.IMREAD_UNCHANGED)
                if dep is None or dep.ndim!=2: raise ValueError('depth_decode_failure')
                confidence=np.frombuffer(conf[conf_ids[j]],np.uint8).reshape(dep.shape)
                rgb_h,rgb_w=rgb.shape[:2]; dep_h,dep_w=dep.shape
                scaled_low=lowK[low_k_ids[j]].copy(); scaled_low[0]*=rgb_w/dep_w; scaled_low[1]*=rgb_h/dep_h
                if not np.allclose(K[k_ids[j]],scaled_low,atol=.05): raise ValueError('rgb_depth_intrinsics_mismatch')
                crop,K0=center_crop_resize_geometry(rgb_h,rgb_w,K[k_ids[j]]); l,t,r,b=crop
                dl=int(round(l*dep_w/rgb_w)); dr=int(round(r*dep_w/rgb_w)); dt=int(round(t*dep_h/rgb_h)); db=int(round(b*dep_h/rgb_h))
                rgb=cv2.resize(rgb[t:b,l:r],(WIDTH,HEIGHT),interpolation=cv2.INTER_AREA)
                dep=cv2.resize(dep[dt:db,dl:dr],(WIDTH,HEIGHT),interpolation=cv2.INTER_NEAREST)
                confidence_roi=confidence[dt:db,dl:dr]
                gray=cv2.resize(cv2.cvtColor(rgb,cv2.COLOR_BGR2GRAY),(160,120),interpolation=cv2.INTER_AREA); means.append(float(gray.mean())); stds.append(float(gray.std())); depth_valid.append(float(np.mean(dep>0))); confidence_valid.append(float(np.mean(confidence_roi>=1)))
                if previous is not None: changes.append(float(np.mean(np.abs(gray.astype(np.float32)-previous.astype(np.float32)))))
                previous=gray
                if not cv2.imwrite(str(tmp/'rgb'/f'{j:06d}.png'),rgb) or not cv2.imwrite(str(tmp/'depth'/f'{j:06d}.png'),dep): raise OSError('image_write_failure')
                Ks.append(K0); ds.append(dep)
            if min(means)<5 or min(stds)<3 or max(changes)>80: raise ValueError('rgb_quality_failure')
            if min(depth_valid)<.02: raise ValueError('insufficient_valid_depth')
            if np.median(confidence_valid)<.20: raise ValueError('insufficient_depth_confidence')
            Ks=np.stack(Ks)
            target=np.arange(N,dtype=np.float64)/FPS; np.save(tmp/'c2w_abs.npy',pose); np.save(tmp/'c2w_local.npy',localize_c2w(pose)); np.save(tmp/'intrinsics.npy',Ks); np.save(tmp/'timestamps.npy',target); np.save(tmp/'source_timestamps.npy',selected_times); np.save(tmp/'source_frame_indices.npy',ids.astype(np.int32)); np.save(tmp/'source_depth_timestamps.npy',dtimes[depth_ids]); np.save(tmp/'source_pose_timestamps.npy',pose_times[pose_ids]); np.save(tmp/'source_intrinsics_timestamps.npy',k_times[k_ids]); np.save(tmp/'source_confidence_timestamps.npy',ctimes[conf_ids]); points=pointcloud(ds,Ks,pose,ids,target,tmp/'pointcloud.npz'); corr=build_causal_correspondence_cache(sorted((tmp/'depth').glob('*.png')),pose,Ks,tmp/'correspondence_cache.npz',chunk_count=CHUNKS,pixel_stride=8,compressed=False)
            if corr['row_count']==0: raise ValueError('empty_correspondence')
            render_qa(sorted((tmp/'rgb').glob('*.png')),tmp/'qa_7frames.jpg')
            visit=str(props0.get('visit_id') or video_id); visit=video_id if visit.upper() in {'NA','NONE','UNKNOWN'} else visit
            timing={'rgb_target_max_error_seconds':float(np.max(np.abs(selected_times-(selected_times[0]+target)))),'depth_rgb_max_error_seconds':float(np.max(np.abs(dtimes[depth_ids]-selected_times))),'pose_rgb_max_error_seconds':float(np.max(np.abs(pose_times[pose_ids]-selected_times))),'intrinsics_rgb_max_error_seconds':float(np.max(np.abs(k_times[k_ids]-selected_times))),'depth_intrinsics_rgb_max_error_seconds':float(np.max(np.abs(low_k_times[low_k_ids]-selected_times))),'confidence_rgb_max_error_seconds':float(np.max(np.abs(ctimes[conf_ids]-selected_times)))}
            meta={'schema_version':'rgbd-memory-record-v3','record_id':rid,'dataset':'arkitscenes','scene_id':visit,'sequence_id':video_id,'visit_id':visit,'split':split,'frame_count':N,'chunk_count':CHUNKS,'chunk_stride':32,'fps':FPS,'duration_seconds':8.0,'pose_source':props0.get('pose_source'),'orientation_source':props0.get('orientation_source'),'pose_convention':'OpenCV camera-to-world = RRD rig-to-world @ cam_00-to-rig','cam_00_to_rig':cam2rig.tolist(),'source_frame_indices':ids.tolist(),'source_timing':'RRD video_time 60Hz nearest mapping','timing_validation':timing,'pose_validation':pm,'depth_valid_fraction_min':min(depth_valid),'depth_valid_fraction_median':float(np.median(depth_valid)),'confidence_valid_fraction_median':float(np.median(confidence_valid)),'pointcloud_points':points,'correspondence':corr,'depth_confidence':'RRD ARKit confidence >= 1'}; atomic(tmp/'metadata.json',meta); dest.parent.mkdir(parents=True,exist_ok=True); os.replace(tmp,dest); return True,rid
        except Exception as e: shutil.rmtree(tmp,ignore_errors=True); return False,str(e)
    return False,'no_valid_window'

def main():
    p=argparse.ArgumentParser(); p.add_argument('--rrd-root',type=Path,required=True); p.add_argument('--output-root',type=Path,required=True); p.add_argument('--max-records',type=int,default=600); p.add_argument('--start',type=int,default=0)
    p.add_argument('--shard-index',type=int,default=0); p.add_argument('--shard-count',type=int,default=1); a=p.parse_args()
    if not 0 <= a.shard_index < a.shard_count: raise ValueError('shard-index must be in [0, shard-count)')
    ids=sorted((x.stem for x in (a.rrd_root/'base').glob('*.rrd')),key=lambda v:sum((a.rrd_root/l/f'{v}.rrd').stat().st_size if (a.rrd_root/l/f'{v}.rrd').is_file() else 1<<60 for l in ('video_wide','depth')))[a.start:]
    ids=ids[a.shard_index::a.shard_count]; stats=Counter(); rows=[]
    for vid in ids:
        ok,r=build_one(vid,a.rrd_root,a.output_root,len(rows),'candidate'); stats['built' if ok else r]+=1
        if ok: rows.append(r)
        print(json.dumps({'video_id':vid,'ok':ok,'result':r,'accepted_in_shard':len(rows)}),flush=True)
        if len(rows)>=a.max_records: break
    print(json.dumps({'records':len(rows),'stats':dict(stats)},indent=2))
if __name__=='__main__': main()
