import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from PIL import Image
from ..types import RAY_DISTANCE
@dataclass
class HoloFrame:
    frame_id:str; rgb:np.ndarray; depth:np.ndarray; mask:np.ndarray; c2w:np.ndarray; raw_c2w:np.ndarray
class Holo360DReader:
    depth_convention=RAY_DISTANCE
    def __init__(self,scene_root,normalize_first_pose=True):
        self.root=Path(scene_root); self.normalize_first_pose=normalize_first_pose; self._ids=self._match_ids()
    def _match_ids(self):
        groups={"rgb":{p.stem for p in (self.root/"rgb").glob("*.jpg")},"depth":{p.stem for p in (self.root/"depth"/"mesh_depth").glob("*.exr")},"mask":{p.stem for p in (self.root/"mask").glob("*.jpg")},"poses":{p.stem for p in (self.root/"poses").glob("*.txt")}}
        common=set.intersection(*groups.values()); self.missing={k:sorted(set.union(*groups.values())-v)[:10] for k,v in groups.items()}
        if not common: raise FileNotFoundError(f"No matched Holo360D frames under {self.root}; missing={self.missing}")
        return sorted(common,key=float)
    @property
    def frame_ids(self): return self._ids
    def _read_depth(self,path):
        import cv2
        arr=cv2.imread(str(path),cv2.IMREAD_ANYDEPTH|cv2.IMREAD_ANYCOLOR)
        if arr is None: raise ValueError(f"Could not decode EXR: {path}")
        arr=np.asarray(arr,np.float32); return arr[...,0] if arr.ndim==3 else arr
    def _read_pose(self,path):
        values=np.loadtxt(path,dtype=np.float32).reshape(-1)
        if values.size!=12: raise ValueError(f"Expected 12 c2w values, got {values.size}: {path}")
        pose=np.eye(4,dtype=np.float32); pose[:3,3]=values[:3]; pose[:3,:3]=values[3:].reshape(3,3); return pose
    def read(self,index):
        fid=self._ids[index] if isinstance(index,int) else str(index); rgb=np.asarray(Image.open(self.root/"rgb"/f"{fid}.jpg").convert("RGB")); mask=np.asarray(Image.open(self.root/"mask"/f"{fid}.jpg").convert("L"))>0; depth=self._read_depth(self.root/"depth"/"mesh_depth"/f"{fid}.exr")
        if depth.shape!=mask.shape: raise ValueError(f"Depth/mask mismatch {fid}: {depth.shape} vs {mask.shape}")
        depth[~np.isfinite(depth)|(depth<=0)|~mask]=np.nan; raw=self._read_pose(self.root/"poses"/f"{fid}.txt"); first=self._read_pose(self.root/"poses"/f"{self._ids[0]}.txt"); c2w=np.linalg.inv(first)@raw if self.normalize_first_pose else raw.copy()
        return HoloFrame(fid,rgb,depth.astype(np.float32),mask,c2w.astype(np.float32),raw)
