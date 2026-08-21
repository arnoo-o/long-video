"""Load and validate the already-built DL3DV/latent/teacher cache manifest."""
from __future__ import annotations
from dataclasses import dataclass
import json
from pathlib import Path
import numpy as np

REQUIRED_RECORD_KEYS=("trajectory_id","rgb_dir","target_c2w_local","intrinsics")
OPTIONAL_CACHE_KEYS=("latent_cache","gt_latent_cache","recal_xyz","recal_valid","recal_confidence","recal_pointmap","correspondence_cache")
LATENT_SCHEMAS=("continuous_49","overlap_chunks_6x9")

def identify_latent_temporal_axis(shape, *, temporal_size):
    candidates=[axis for axis,size in enumerate(shape) if size == int(temporal_size)]
    if len(shape)==5: candidates=[axis for axis in candidates if axis not in (0,1)]
    if len(candidates)!=1: raise ValueError(f"cannot uniquely identify temporal axis {temporal_size} in latent shape {tuple(shape)}: {candidates}")
    return candidates[0]

def _payload_tensor(file):
    import torch
    if file.suffix in ('.pt','.pth'):
        value=torch.load(file,map_location='cpu')
    elif file.suffix=='.npy': value=torch.from_numpy(np.asarray(np.load(file)))
    elif file.suffix=='.npz':
        archive=np.load(file); keys=[key for key in ('latent','latents','video_latents','target_latents') if key in archive]
        if len(keys)!=1: raise ValueError(f"npz latent payload must contain exactly one supported key: {file}")
        value=torch.from_numpy(np.asarray(archive[keys[0]]))
    else: raise ValueError(f"unsupported latent cache file: {file}")
    if isinstance(value,dict):
        keys=[key for key in ('latent','latents','video_latents','target_latents') if key in value]
        if len(keys)!=1: raise ValueError(f"latent payload must contain exactly one supported key: {file}")
        value=value[keys[0]]
    if not isinstance(value,torch.Tensor) or value.ndim not in (4,5): raise ValueError(f"unsupported latent payload in {file}")
    return value

def _canonical_tensor(value, temporal_size):
    if value.ndim==4:
        temporal=identify_latent_temporal_axis(value.shape,temporal_size=temporal_size)+1; value=value.unsqueeze(0)
    else: temporal=identify_latent_temporal_axis(value.shape,temporal_size=temporal_size)
    if temporal!=2: value=value.movedim(temporal,2)
    if value.shape[2]!=temporal_size: raise ValueError('latent temporal canonicalization failed')
    return value.contiguous()

def _latent_files(path):
    path=Path(path)
    if path.is_dir():
        chunks=[path/f'chunk_{index:02d}.pt' for index in range(6)]
        if all(file.is_file() for file in chunks): return 'overlap_chunks_6x9',chunks
        files=sorted(file for file in path.iterdir() if file.suffix in ('.npy','.npz','.pt','.pth'))
        if len(files)!=1: raise ValueError(f"latent cache directory must contain chunk_00..05.pt or one continuous tensor: {path}")
        return 'continuous_49',files
    return 'continuous_49',[path]
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

def validate_latent_cache(path: str|Path, *, schema=None):
    path=Path(path)
    if not path.exists(): raise FileNotFoundError(path)
    detected,files=_latent_files(path)
    if schema is not None and schema!=detected: raise ValueError(f"latent schema mismatch: expected {schema}, found {detected}")
    expected=9 if detected=='overlap_chunks_6x9' else 49
    for file in files: _canonical_tensor(_payload_tensor(file),expected)
    return detected,files

def load_latent_tensor(path: str|Path, *, schema=None):
    """Load an existing latent cache as canonical [B,C,T,H,W]."""
    import torch
    detected,files=validate_latent_cache(path,schema=schema)
    values=[_canonical_tensor(_payload_tensor(file),9 if detected=='overlap_chunks_6x9' else 49) for file in files]
    if detected=='continuous_49': return values[0]
    reference=values[0].shape[:2]+values[0].shape[3:]
    if any(value.shape[:2]+value.shape[3:]!=reference for value in values): raise ValueError('overlap chunk latent shapes differ')
    result=torch.cat([values[0]]+[value[:,:,1:] for value in values[1:]],dim=2)
    if result.shape[2]!=49: raise RuntimeError('6x9 overlap latent cache did not produce 49 latents')
    return result.contiguous()

def require_overlap_validation(path: str|Path, *, expected_provenance: str) -> dict:
    """Reject overlap caches unless a matching continuous-VAE validation passed."""
    path=Path(path); candidate=path/'latent_validation.json' if path.is_dir() else path.with_suffix('.validation.json')
    if not candidate.is_file(): raise RuntimeError(f'overlap_chunks_6x9 requires a passed validation file: {candidate}')
    payload=json.loads(candidate.read_text())
    if payload.get('passed') is not True or payload.get('model_provenance')!=expected_provenance:
        raise RuntimeError('overlap latent validation is missing, failed, or belongs to another VAE/model provenance')
    return payload
