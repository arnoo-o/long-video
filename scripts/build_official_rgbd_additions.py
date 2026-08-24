#!/usr/bin/env python3
"""Build DDAD and ARKitScenes records using their official geometry semantics."""
from __future__ import annotations
import argparse, json, os, shutil, sys
from datetime import datetime
from pathlib import Path
import cv2, numpy as np
from scipy.spatial.transform import Rotation
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from long_video.data.rgbd_memory import build_causal_correspondence_cache, localize_c2w, transform_rgb_depth

H, W = 480, 832
TARTAN_NED_R_OPENCV = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=np.float64)
TARTAN_K = np.array([[320, 0, 319.5], [0, 320, 319.5], [0, 0, 1]], dtype=np.float64)

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
    if (root/'metadata.json').is_file(): return False
    if root.exists():
        # A terminated builder may leave a metadata-less record.  Such a
        # directory is never a committed record and is safe to reconstruct.
        shutil.rmtree(root)
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
    return True

def arkit(root,out):
    for split_dir,split in ((root/'Training','train'),(root/'Validation','val')):
      if not split_dir.is_dir(): continue
      target={'train':350,'val':50}[split]
      built=sum(1 for path in (out/'records'/'arkitscenes').glob('*/metadata.json')
                if json.loads(path.read_text()).get('split')==split)
      for video in sorted(x for x in split_dir.iterdir() if x.is_dir()):
        if built >= target: break
        traj={}
        for line in (video/'lowres_wide.traj').read_text().splitlines():
            a=line.split(); R=Rotation.from_rotvec(np.asarray(a[1:4],float)).as_matrix(); ext=np.eye(4); ext[:3,:3]=R; ext[:3,3]=np.asarray(a[4:7],float); traj[f'{float(a[0]):.3f}']=np.linalg.inv(ext)
        rgb={p.stem.split('_')[-1]:p for p in (video/'lowres_wide').glob('*.png')}; dep={p.stem.split('_')[-1]:p for p in (video/'lowres_depth').glob('*.png')}; intr={p.stem.split('_')[-1]:p for p in (video/'lowres_wide_intrinsics').glob('*.pincam')}
        ids=sorted(set(rgb)&set(dep),key=float); obs=[]
        for fid in ids:
            value=float(fid); key=f'{value:.3f}'
            ikey=next((candidate for candidate in (key,f'{value-.001:.3f}',f'{value+.001:.3f}') if candidate in intr),None)
            if key not in traj or ikey is None: continue
            w,h,fx,fy,cx,cy=np.loadtxt(intr[ikey]); K=np.array([[fx,0,cx],[0,fy,cy],[0,0,1.]])
            obs.append({'rgb':rgb[fid],'depth':dep[fid],'K':K,'c2w':traj[key],'timestamp':float(fid)})
        for n,start in enumerate(range(0,len(obs)-192,193)):
            if built >= target: break
            built += int(write_record(out,'arkitscenes',video.name,f'{video.name}/{n:04d}',obs[start:start+193],6,split,{'official_asset':'raw lowres_wide/depth/traj/intrinsics'}))

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
        for camera_name in ('CAMERA_01',):
            rows=[]; kidx=ci[camera_name]; intr=calibration['intrinsics'][kidx]; K=np.array([[intr['fx'],0,intr['cx']],[0,intr['fy'],intr['cy']],[0,0,1.]])
            for sample in payload['samples']:
                datums=[data[k] for k in sample['datum_keys']]; cam=next((d for d in datums if d['id']['name']==camera_name),None); lidar=next((d for d in datums if d['id']['name']=='LIDAR'),None)
                if cam is None or lidar is None: continue
                image=cam['datum']['image']; c2w=pose_matrix(image['pose']); rgb=scene/image['filename']; pc_path=scene/lidar['datum']['point_cloud']['filename']
                if not rgb.is_file() or not pc_path.is_file(): continue
                timestamp=datetime.fromisoformat(cam['id']['timestamp'].replace('Z','+00:00')).timestamp()
                rows.append({'rgb':rgb,'depth':lambda s=scene,d=lidar,p=c2w,k=K:ddad_depth(s,d,p,k),'K':K,'c2w':c2w,'timestamp':timestamp})
            if len(rows)>=97: candidates.append((scene.name,camera_name,rows[:97]))
    # The selected official subset contains exactly 132 scenes with a complete
    # 97-frame CAMERA_01 stream.  Do not synthesize extra records from another
    # camera on the same drive.
    selected=[x for x in candidates if x[1]=='CAMERA_01']
    if len(selected) != 132:
        raise ValueError(f'expected 132 complete DDAD CAMERA_01 scenes, found {len(selected)}')
    for scene,camera,rows in selected:
        write_record(out,'ddad',scene,f'{scene}/{camera}',rows,3,'train',{'official_loader_semantics':'DGP pose_WC + generate_depth_from_datum=LIDAR','camera':camera})

def tartan_depth(path):
    """Decode the official TartanAir/TartanGround float32 RGBA PNG."""
    rgba=cv2.imread(str(path),cv2.IMREAD_UNCHANGED)
    if rgba is None or rgba.dtype != np.uint8 or rgba.ndim != 3 or rgba.shape[2] != 4:
        raise ValueError(f'invalid TartanGround depth image: {path}')
    return np.ascontiguousarray(rgba).view('<f4').reshape(rgba.shape[:2])

def tartanground(root,out):
    built=sum(1 for _ in (out/'records'/'tartanground').glob('*/metadata.json'))
    for trajectory in sorted(root.glob('*/Data_*/P*')):
        if built >= 400: break
        rgbdir=trajectory/'image_lcam_front'; depdir=trajectory/'depth_lcam_front'
        posefile=trajectory/'pose_lcam_front.txt'; metafile=trajectory/f'{trajectory.name}_metadata.json'
        if not (rgbdir.is_dir() and depdir.is_dir() and posefile.is_file() and metafile.is_file()):
            continue
        metadata=json.loads(metafile.read_text()); dt=float(metadata['time_step'])
        poses=np.loadtxt(posefile,dtype=np.float64)
        if poses.ndim != 2 or poses.shape[1] != 7:
            raise ValueError(f'invalid TartanGround poses: {posefile}')
        rgb={int(p.name.split('_',1)[0]):p for p in rgbdir.glob('*.png')}
        depth={int(p.name.split('_',1)[0]):p for p in depdir.glob('*.png')}
        ids=sorted(set(rgb)&set(depth)&set(range(len(poses))))
        obs=[]
        for fid in ids:
            p=poses[fid]; c2w=np.eye(4,dtype=np.float64)
            # Exact convention used by the official TartanAir customizer:
            # camera pose is camera-to-world in NED, converted to OpenCV axes.
            c2w[:3,:3]=Rotation.from_quat(p[3:]).as_matrix()@TARTAN_NED_R_OPENCV
            c2w[:3,3]=p[:3]
            obs.append({'rgb':rgb[fid], 'depth':lambda p=depth[fid]:tartan_depth(p),
                        'K':TARTAN_K, 'c2w':c2w, 'timestamp':fid*dt, 'frame_id':fid})
        # Identity must be explicit: no gaps and no filename-order association.
        runs=[]; start=0
        for i in range(1,len(obs)+1):
            if i == len(obs) or obs[i]['frame_id'] != obs[i-1]['frame_id']+1:
                runs.append(obs[start:i]); start=i
        env=trajectory.parents[1].name; difficulty=trajectory.parent.name
        sequence=f'{env}/{difficulty}/{trajectory.name}'
        clip=0
        for run in runs:
            for start in range(0,len(run)-192,193):
                if built >= 400: return
                built += int(write_record(out,'tartanground',env,f'{sequence}/{clip:04d}',run[start:start+193],6,'candidate',
                                          {'official_asset':'TartanGround pinhole image/depth/pose/metadata',
                                           'official_camera':'lcam_front','time_step':dt}))
                clip+=1

def main():
    p=argparse.ArgumentParser(); p.add_argument('--dataset',choices=('arkit','ddad','tartanground'),required=True); p.add_argument('--source',type=Path,required=True); p.add_argument('--output',type=Path,required=True); a=p.parse_args()
    if a.dataset=='arkit': arkit(a.source,a.output)
    elif a.dataset=='ddad': ddad(a.source,a.output)
    else: tartanground(a.source,a.output)
if __name__=='__main__': main()
