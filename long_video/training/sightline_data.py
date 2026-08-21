"""Load and validate the already-built DL3DV/latent/teacher cache manifest."""
from __future__ import annotations
from dataclasses import dataclass
import json
from pathlib import Path
import numpy as np

REQUIRED_RECORD_KEYS=("trajectory_id","rgb_dir","target_c2w_local","intrinsics")
OPTIONAL_CACHE_KEYS=("latent_cache","gt_latent_cache","recal_xyz","recal_valid","recal_confidence","recal_pointmap")
@dataclass(frozen=True)
class SightlineRecord:
    raw: dict; root: Path
    @property
    def trajectory_id(self): return str(self.raw["trajectory_id"])
    def path(self,key): return self.root / self.raw[key]
    def load_cameras(self):
        c2w=np.load(self.path("target_c2w_local"),mmap_mode="r"); K=np.load(self.path("intrinsics"),mmap_mode="r")
        if c2w.shape != (193,4,4) or K.shape not in ((3,3),(193,3,3)): raise ValueError(f"{self.trajectory_id}: camera cache must contain 193 frames")
        if K.ndim==2: K=np.repeat(K[None],193,axis=0)
        return c2w,K
    def rgb_paths(self):
        paths=sorted((self.root/self.raw["rgb_dir"]).glob("*"))
        if len(paths)!=193: raise ValueError(f"{self.trajectory_id}: expected 193 RGB frames, found {len(paths)}")
        return paths
    def validate_teacher_and_latent_caches(self):
        checked=[]
        for key in OPTIONAL_CACHE_KEYS:
            if key not in self.raw:
                continue
            path=self.path(key)
            if not path.exists():
                raise FileNotFoundError(f"{self.trajectory_id}: missing {key}: {path}")
            checked.append(path)
        for key in ("latent_cache", "gt_latent_cache"):
            if key in self.raw:
                validate_latent_cache(self.path(key))
        return checked

def load_sightline_manifest(path: str|Path, *, expected_count: int|None=None) -> list[SightlineRecord]:
    path=Path(path); payload=json.loads(path.read_text()); records=payload.get("records")
    if not isinstance(records,list) or (expected_count is not None and len(records)!=expected_count): raise ValueError("invalid Sightline dataset manifest record count")
    root=path.parent; result=[]
    for row in records:
        missing=[key for key in REQUIRED_RECORD_KEYS if key not in row]
        if missing: raise ValueError(f"record missing keys: {missing}")
        item=SightlineRecord(row,root); item.load_cameras(); item.rgb_paths(); item.validate_teacher_and_latent_caches(); result.append(item)
    return result

def validate_latent_cache(path: str|Path, *, expected_frames=193):
    path=Path(path)
    if not path.exists(): raise FileNotFoundError(path)
    files=sorted(path.glob("**/*")) if path.is_dir() else [path]
    if not files: raise ValueError(f"empty latent cache: {path}")
    for file in files:
        if file.suffix not in (".npy",".npz",".pt",".pth"): continue
        if file.suffix==".npy":
            arr=np.load(file,mmap_mode="r")
            if arr.ndim < 3: continue
            temporal_axes=[axis for axis,size in enumerate(arr.shape) if size in (expected_frames, expected_frames-1, 9, 33)]
            if not temporal_axes:
                raise ValueError(f"cannot identify temporal axis in latent cache {file} shape={arr.shape}")
            if len(temporal_axes)>1 and expected_frames in arr.shape:
                temporal_axes=[axis for axis in temporal_axes if arr.shape[axis] == expected_frames]
            # Record the detected axis for callers without imposing C/T order.
            if arr.shape[temporal_axes[0]] not in (expected_frames, expected_frames-1,9,33):
                raise ValueError(f"invalid temporal axis in {file}")
    return files

def load_latent_tensor(path: str|Path, *, expected_frames=193):
    """Load an existing latent cache as canonical [B,C,T,H,W]."""
    import torch
    path=Path(path); files=validate_latent_cache(path,expected_frames=expected_frames)
    file=next((f for f in files if f.suffix in ('.npy','.pt','.pth')),None)
    if file is None: raise ValueError(f"no supported latent tensor in {path}")
    value=torch.load(file,map_location='cpu') if file.suffix in ('.pt','.pth') else torch.from_numpy(np.asarray(np.load(file)))
    if isinstance(value,dict):
        value=next((value[k] for k in ('latents','video_latents','target_latents') if k in value),None)
    if not isinstance(value,torch.Tensor) or value.ndim not in (4,5): raise ValueError(f"unsupported latent payload in {file}")
    if value.ndim==4: value=value.unsqueeze(0)
    candidates=[axis for axis,size in enumerate(value.shape) if axis>0 and size in (expected_frames,expected_frames-1,49,9)]
    if not candidates: raise ValueError(f"cannot identify latent temporal axis: {tuple(value.shape)}")
    temporal=2 if value.shape[2] in (expected_frames,expected_frames-1,49,9) else candidates[0]
    if temporal!=2: value=value.movedim(temporal,2)
    return value.contiguous()
