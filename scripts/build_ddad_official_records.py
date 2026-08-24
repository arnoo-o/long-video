#!/usr/bin/env python3
"""Build DDAD 97-frame records using the official DGP projection semantics.

The implementation follows DGP ``get_depth_from_point_cloud`` exactly:
world points are ``p_WS * X_S`` and image datum ``pose`` is ``p_WC``.
"""
from __future__ import annotations
import argparse, json, os
from pathlib import Path
import cv2, numpy as np
from scipy.spatial.transform import Rotation
from long_video.data.rgbd_memory import transform_rgb_depth, localize_c2w, build_causal_correspondence_cache

def atomic_json(path, value):
    tmp=path.with_suffix(path.suffix+'.tmp'); tmp.write_text(json.dumps(value,indent=2),encoding='utf-8'); os.replace(tmp,path)

def pose(value):
    r=value['rotation']; t=value['translation']; out=np.eye(4,dtype=np.float64)
    out[:3,:3]=Rotation.from_quat((r['qx'],r['qy'],r['qz'],r['qw'])).as_matrix(); out[:3,3]=(t['x'],t['y'],t['z']); return out

def depth_from_lidar(points, lidar_c2w, camera_c2w, K, height, width):
    # This is the same p_WS * X_S -> Camera(K, p_cw=p_WC.inverse()) projection
    # used by the official DGP get_depth_from_point_cloud implementation.
    world=(lidar_c2w[:3,:3]@points[:,:3].T).T+lidar_c2w[:3,3]
    camera=(np.linalg.inv(camera_c2w)[:3,:3]@(world-camera_c2w[:3,3]).T).T
    z=camera[:,2]; uv=(K@camera.T).T; uv=uv[:,:2]/np.maximum(z[:,None],1e-12)
    xy=np.rint(uv).astype(np.int32); valid=np.isfinite(uv).all(1)&(z>0)&(xy[:,0]>=0)&(xy[:,0]<width)&(xy[:,1]>=0)&(xy[:,1]<height)
    result=np.full((height,width),np.inf,np.float32); flat=xy[valid,1]*width+xy[valid,0]
    np.minimum.at(result.ravel(),flat,z[valid].astype(np.float32)); result[~np.isfinite(result)]=0
    return result

def build_scene(scene_root, output_root, corr_stride):
    scene=json.loads(next(scene_root.glob('scene_*.json')).read_text()); calibration=json.loads(next((scene_root/'calibration').glob('*.json')).read_text())
    index={name:i for i,name in enumerate(calibration['names'])}; cam_i=index['CAMERA_01']; K=np.array(((calibration['intrinsics'][cam_i]['fx'],0,calibration['intrinsics'][cam_i]['cx']),(0,calibration['intrinsics'][cam_i]['fy'],calibration['intrinsics'][cam_i]['cy']),(0,0,1)),dtype=np.float64)
    datums={row['key']:row for row in scene['data']}; rows=[]
    for sample in scene['samples']:
        selected=[datums[key] for key in sample['datum_keys']]
        image=next(row for row in selected if row['id']['name']=='CAMERA_01')['datum']['image']; lidar=next(row for row in selected if row['id']['name']=='LIDAR')['datum']['point_cloud']
        rows.append((float(sample['id']['timestamp'].replace('T',' ').replace('Z','').split()[-1].split(':')[-1]) if False else len(rows), image, lidar))
    if len(rows)<97: return None
    rows=rows[:97]; rid=f"ddad__{scene_root.name}__000000"; root=output_root/'records'/'ddad'/rid; meta=root/'metadata.json'
    if meta.is_file(): return json.loads(meta.read_text())
    (root/'rgb').mkdir(parents=True,exist_ok=True); (root/'depth').mkdir(exist_ok=True); c2ws=[]; Ks=[]; times=[]
    for i,(timestamp,image,lidar) in enumerate(rows):
        rgb=cv2.imread(str(scene_root/image['filename']),cv2.IMREAD_COLOR); pts=np.load(scene_root/lidar['filename'])['data']; c2w=pose(image['pose']); depth=depth_from_lidar(pts,pose(lidar['pose']),c2w,K,rgb.shape[0],rgb.shape[1]); rgb_out,depth_mm,K_out,_=transform_rgb_depth(rgb,depth,K)
        cv2.imwrite(str(root/'rgb'/f'{i:06d}.png'),rgb_out,(cv2.IMWRITE_PNG_COMPRESSION,3)); cv2.imwrite(str(root/'depth'/f'{i:06d}.png'),depth_mm,(cv2.IMWRITE_PNG_COMPRESSION,3)); c2ws.append(c2w); Ks.append(K_out); times.append(timestamp)
    c2w=np.stack(c2ws); Kall=np.stack(Ks); np.save(root/'c2w_abs.npy',c2w); np.save(root/'c2w_local.npy',localize_c2w(c2w)); np.save(root/'intrinsics.npy',Kall); np.save(root/'timestamps.npy',np.asarray(times,dtype=np.float64))
    corr=build_causal_correspondence_cache(sorted((root/'depth').glob('*.png')),c2w,Kall,root/'correspondence_cache.npz',chunk_count=3,pixel_stride=corr_stride)
    value={'record_id':rid,'dataset':'ddad','scene_id':scene_root.name,'sequence_id':scene_root.name,'rgb_dir':str((root/'rgb').relative_to(output_root)),'depth_dir':str((root/'depth').relative_to(output_root)),'c2w_abs':str((root/'c2w_abs.npy').relative_to(output_root)),'c2w_local':str((root/'c2w_local.npy').relative_to(output_root)),'intrinsics':str((root/'intrinsics.npy').relative_to(output_root)),'timestamps':str((root/'timestamps.npy').relative_to(output_root)),'correspondence_cache':str((root/'correspondence_cache.npz').relative_to(output_root)),'frame_count':97,'chunk_count':3,'height':480,'width':832,'split':'train','memory_eligible':True,'source':'DDAD official DGP p_WS*X_S projection'}
    atomic_json(meta,value); return value

def main():
    p=argparse.ArgumentParser(); p.add_argument('--source',type=Path,required=True); p.add_argument('--output',type=Path,required=True); p.add_argument('--scene-index',type=int); p.add_argument('--corr-stride',type=int,default=8); a=p.parse_args()
    scenes=sorted(x for x in a.source.iterdir() if x.is_dir()); scenes=[scenes[a.scene_index]] if a.scene_index is not None else scenes; records=[]
    for scene in scenes:
        try:
            value=build_scene(scene,a.output,a.corr_stride)
            if value: records.append(value); print(json.dumps({'scene':scene.name,'ok':True}))
        except Exception as exc: print(json.dumps({'scene':scene.name,'ok':False,'error':str(exc)}))
    if a.scene_index is None: atomic_json(a.output/'manifest_ddad.json',{'schema_version':'rgbd-memory-manifest-v2','records':records})
if __name__=='__main__': main()
