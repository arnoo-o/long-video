#!/usr/bin/env python3
"""Build DDAD and ARKitScenes records using their official geometry semantics."""
from __future__ import annotations
import argparse, json, os
from datetime import datetime
from pathlib import Path
import cv2, numpy as np
from scipy.spatial.transform import Rotation
from long_video.data.rgbd_memory import build_causal_correspondence_cache, localize_c2w, transform_rgb_depth

H, W = 480, 832

def atomic_json(path, value):
    tmp=path.with_suffix(path.suffix+'.tmp'); tmp.write_text(json.dumps(value,indent=2)); os.replace(tmp,path)

def pose_matrix(value):
    q=value['rotation']; t=value['translation']; m=np.eye(4)
    m[:3,:3]=Rotation.from_quat([q['qx'],q['qy'],q['qz'],q['qw']]).as_matrix(); m[:3,3]=[t['x'],t['y'],t['z']]
    return m

def pointcloud_from_depth(depth, K, c2w, stride=4):
    yy,xx=np.mgrid[0:H:stride,0:W:stride]; z=depth[::stride,::stride].astype(np.float32)/1000
    valid=np.isfinite(z)&(z>0); x=xx[valid]; y=yy[valid]; z=z[valid]
    cam=(np.linalg.inv(K)@np.stack((x*z,y*z,z))).T
    return (c2w[:3]@np.c_[cam,np.ones(len(cam))].T).T.astype(np.float32)

def write_record(out, dataset, scene, sequence, observations, chunk_count, split, source):
    count=1+32*chunk_count
    if len(observations)!=count: raise ValueError('window length mismatch')
    rid=f'{dataset}__{sequence.replace("/","_")}' ; root=out/'records'/dataset/rid
    if (root/'metadata.json').is_file(): return
    rgbdir=root/'rgb'; depdir=root/'depth'; rgbdir.mkdir(parents=True,exist_ok=True); depdir.mkdir()
    poses=[]; Ks=[]; times=[]; points=[]; offsets=[0]
    for i,row in enumerate(observations):
        rgb=cv2.imread(str(row['rgb']),cv2.IMREAD_COLOR); depth=row['depth']() if callable(row['depth']) else cv2.imread(str(row['depth']),cv2.IMREAD_UNCHANGED)
        if rgb is None or depth is None: raise ValueError(f'missing observation {i}')
        if depth.dtype==np.uint16: depth_m=depth.astype(np.float32)/1000
        else: depth_m=depth.astype(np.float32)
        rgb2,d2,K2,_=transform_rgb_depth(rgb,depth_m,row['K'])
        cv2.imwrite(str(rgbdir/f'{i:06d}.png'),rgb2); cv2.imwrite(str(depdir/f'{i:06d}.png'),d2)
        poses.append(row['c2w']); Ks.append(K2); times.append(row['timestamp'])
        xyz=pointcloud_from_depth(d2,K2,row['c2w']); points.append(xyz); offsets.append(offsets[-1]+len(xyz))
    poses=np.stack(poses); Ks=np.stack(Ks); times=np.asarray(times,np.float64)
    np.save(root/'c2w_abs.npy',poses); np.save(root/'c2w_local.npy',localize_c2w(poses)); np.save(root/'intrinsics.npy',Ks); np.save(root/'timestamps.npy',times)
    np.savez_compressed(root/'pointcloud.npz',xyz_world=np.concatenate(points),offsets=np.asarray(offsets,np.int64))
    corr=build_causal_correspondence_cache(sorted(depdir.glob('*.png')),poses,Ks,root/'correspondence_cache.npz',chunk_count=chunk_count,pixel_stride=8)
    meta={'schema_version':'rgbd-memory-record-v2','record_id':rid,'dataset':dataset,'scene_id':scene,'sequence_id':sequence,'split':split,'frame_count':count,'chunk_count':chunk_count,'height':H,'width':W,'stride':1,'source':source,'pose_convention':'OpenCV camera-to-world','correspondence':corr}
    atomic_json(root/'metadata.json',meta)

def arkit(root,out):
    for split_dir,split in ((root/'Training','train'),(root/'Validation','val')):
      if not split_dir.is_dir(): continue
      for video in sorted(x for x in split_dir.iterdir() if x.is_dir()):
        traj={}
        for line in (video/'lowres_wide.traj').read_text().splitlines():
            a=line.split(); R=Rotation.from_rotvec(np.asarray(a[1:4],float)).as_matrix(); ext=np.eye(4); ext[:3,:3]=R; ext[:3,3]=np.asarray(a[4:7],float); traj[f'{float(a[0]):.3f}']=np.linalg.inv(ext)
        rgb={p.stem.split('_')[-1]:p for p in (video/'lowres_wide').glob('*.png')}; dep={p.stem.split('_')[-1]:p for p in (video/'lowres_depth').glob('*.png')}; intr={p.stem.split('_')[-1]:p for p in (video/'lowres_wide_intrinsics').glob('*.pincam')}
        ids=sorted(set(rgb)&set(dep),key=float); obs=[]
        for fid in ids:
            key=f'{float(fid):.3f}'; ikey=min(intr,key=lambda x:abs(float(x)-float(fid))) if intr else None
            if key not in traj or ikey is None or abs(float(ikey)-float(fid))>.0011: continue
            w,h,fx,fy,cx,cy=np.loadtxt(intr[ikey]); K=np.array([[fx,0,cx],[0,fy,cy],[0,0,1.]])
            obs.append({'rgb':rgb[fid],'depth':dep[fid],'K':K,'c2w':traj[key],'timestamp':float(fid)})
        for n,start in enumerate(range(0,len(obs)-192,193)):
            write_record(out,'arkitscenes',video.name,f'{video.name}/{n:04d}',obs[start:start+193],6,split,{'official_asset':'raw lowres_wide/depth/traj/intrinsics'})

def ddad_depth(scene, datum, camera_pose, K):
    pc=np.load(scene/datum['datum']['point_cloud']['filename'])['data'][:,:3]
    world=(pose_matrix(datum['datum']['point_cloud']['pose'])[:3]@np.c_[pc,np.ones(len(pc))].T).T
    cam=(np.linalg.inv(camera_pose)[:3]@np.c_[world,np.ones(len(world))].T).T
    uv=(K@cam.T).T; uv=(uv[:,:2]/np.maximum(uv[:,2:3],1e-12)).astype(int); z=cam[:,2]
    depth=np.zeros((1216,1936),np.float32); valid=(z>0)&(uv[:,0]>=0)&(uv[:,0]<1936)&(uv[:,1]>=0)&(uv[:,1]<1216)
    depth[uv[valid,1],uv[valid,0]]=z[valid]
    return depth

def ddad(root,out):
    candidates=[]
    for scene in sorted(x for x in root.iterdir() if x.is_dir()):
        payload=json.loads(next(scene.glob('scene_*.json')).read_text()); data={x['key']:x for x in payload['data']}
        calibration=json.loads(next((scene/'calibration').glob('*.json')).read_text()); ci={n:i for i,n in enumerate(calibration['names'])}
        for camera_name in ('CAMERA_01','CAMERA_05'):
            rows=[]; kidx=ci[camera_name]; intr=calibration['intrinsics'][kidx]; K=np.array([[intr['fx'],0,intr['cx']],[0,intr['fy'],intr['cy']],[0,0,1.]])
            for sample in payload['samples']:
                datums=[data[k] for k in sample['datum_keys']]; cam=next((d for d in datums if d['id']['name']==camera_name),None); lidar=next((d for d in datums if d['id']['name']=='LIDAR'),None)
                if cam is None or lidar is None: continue
                image=cam['datum']['image']; c2w=pose_matrix(image['pose']); rgb=scene/image['filename']; pc_path=scene/lidar['datum']['point_cloud']['filename']
                if not rgb.is_file() or not pc_path.is_file(): continue
                timestamp=datetime.fromisoformat(cam['id']['timestamp'].replace('Z','+00:00')).timestamp()
                rows.append({'rgb':rgb,'depth':lambda s=scene,d=lidar,p=c2w,k=K:ddad_depth(s,d,p,k),'K':K,'c2w':c2w,'timestamp':timestamp})
            if len(rows)>=97: candidates.append((scene.name,camera_name,rows[:97]))
    # One record per CAMERA_01 scene, then CAMERA_05 from distinct valid source streams.
    selected=[x for x in candidates if x[1]=='CAMERA_01']
    selected.extend(x for x in candidates if x[1]=='CAMERA_05')
    for scene,camera,rows in selected[:200]:
        write_record(out,'ddad',scene,f'{scene}/{camera}',rows,3,'train',{'official_loader_semantics':'DGP pose_WC + generate_depth_from_datum=LIDAR','camera':camera})

def main():
    p=argparse.ArgumentParser(); p.add_argument('--dataset',choices=('arkit','ddad'),required=True); p.add_argument('--source',type=Path,required=True); p.add_argument('--output',type=Path,required=True); a=p.parse_args()
    if a.dataset=='arkit': arkit(a.source,a.output)
    else: ddad(a.source,a.output)
if __name__=='__main__': main()
