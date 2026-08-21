"""Strictly causal source + long/mid/short latent history."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Iterable
import torch
from .rays import latent_camera_indices

def native_helios_indices(device=None, batch_size=1):
    """Pinned Helios source/history/current RoPE slots, independent of global identity."""
    def row(values): return torch.tensor(values,device=device,dtype=torch.long).view(1,-1).expand(batch_size,-1)
    return {'long':row(range(1,17)),'mid':row((17,18)),'short':row((0,19)),'current':row(range(20,29))}

class NativeHistoryState:
    """One causal FIFO shared by training, rollout and inference."""
    def __init__(self, source_latent, fake_image_latent=None):
        if source_latent.ndim!=5 or source_latent.shape[2]!=1: raise ValueError('source latent must be [B,C,1,H,W]')
        self.source=source_latent; zero=torch.zeros_like(source_latent); tail=zero if fake_image_latent is None else fake_image_latent
        if tail.shape!=source_latent.shape: raise ValueError('fake image latent must match source latent')
        self._entries=[(None,zero) for _ in range(18)]+[(None,tail)]
    def groups(self):
        long=self._entries[:16]; mid=self._entries[16:18]; short=[(0,self.source),self._entries[18]]
        indices=native_helios_indices(self.source.device,self.source.shape[0])
        def pack(entries,name): return (torch.cat([value for _,value in entries],2),indices[name])
        return {'long':pack(long,'long'),'mid':pack(mid,'mid'),'short':pack(short,'short')}
    def coverage(self):
        ids=[identity for identity,_ in self._entries]
        return {'long':tuple(tuple(x for x in ids[i:i+4] if x is not None) for i in range(0,16,4)),
                'mid':(tuple(x for x in ids[16:18] if x is not None),),
                'short':((0,),tuple(x for x in ids[18:19] if x is not None))}
    def append_chunk(self, chunk, chunk_index):
        if chunk.ndim!=5 or chunk.shape[2]!=9: raise ValueError('history chunk must have 9 latents')
        start=0 if chunk_index==0 else 1
        for temporal in range(start,9): self._entries.append((chunk_index*8+temporal,chunk[:,:,temporal:temporal+1].detach()))
        self._entries=self._entries[-19:]
    def global_ids(self): return tuple(identity for identity,_ in self._entries)

@dataclass(frozen=True)
class HistoryLayout:
    source: int
    long: tuple[int, ...]
    mid: tuple[int, ...]
    short: tuple[int, ...]

class HistoryManager:
    """Store the shared 32-frame boundary once and reject inconsistent replay."""
    def __init__(self, sizes: tuple[int,int,int] = (16,2,1), chunk_length: int = 33, stride: int = 32):
        if sizes != (16,2,1) or (chunk_length,stride) != (33,32):
            raise ValueError("Sightline requires source + 16/2/1 and 33-frame stride-32 chunks")
        self._source: torch.Tensor | None = None; self._frames: dict[int, Any] = {}; self.chunk_index = 0
    def set_source(self, latent: torch.Tensor) -> None:
        if self._source is None: self._source = latent
        elif not torch.equal(self._source, latent): raise RuntimeError("source prefix changed")
    @staticmethod
    def _same(a: Any, b: Any) -> bool:
        return a is b or (isinstance(a,torch.Tensor) and isinstance(b,torch.Tensor) and torch.equal(a,b))
    def append_chunk(self, frames: Iterable[Any]) -> None:
        frames=list(frames)
        if len(frames) != 33: raise ValueError("each generated chunk must contain 33 frames")
        start=self.chunk_index*32
        for local,latent in enumerate(frames):
            global_frame=start+local; old=self._frames.get(global_frame)
            if old is not None:
                if local != 0 or not self._same(old,latent): raise RuntimeError(f"inconsistent overlapping frame {global_frame}")
                continue
            self._frames[global_frame]=latent
        self.chunk_index += 1
    def layout(self) -> HistoryLayout:
        if self._source is None: raise RuntimeError("source prefix is not initialized")
        selected=sorted(self._frames)[-19:]; padded=[None]*(19-len(selected))+selected
        return HistoryLayout(0,tuple(padded[:16]),tuple(padded[16:18]),tuple(padded[18:]))
    def slots(self) -> list[Any]:
        layout=self.layout(); return [self._source]+[self._source if f is None else self._frames[f] for f in layout.long+layout.mid+layout.short]
    def seen_frames(self) -> tuple[int,...]: return tuple(sorted(self._frames))

class CameraHistoryState:
    """Camera representatives kept in lockstep with native latent history."""
    def __init__(self): self._items={}; self._chunk_index=0
    def append_chunk(self, representatives, frame_ids=None, intrinsics=None):
        if len(representatives)!=9: raise ValueError("a 33-RGB-frame chunk must have 9 latent cameras")
        if frame_ids is None: raise ValueError("camera frame identities are required")
        if len(frame_ids)!=9: raise ValueError("camera frame identity count must be 9")
        if intrinsics is not None and len(intrinsics) != 9: raise ValueError("camera intrinsics count must be 9")
        start=self._chunk_index*8
        for i,item in enumerate(representatives):
            index=start+i
            K = None if intrinsics is None else intrinsics[i]
            if index in self._items:
                old_frame,old,old_K=self._items[index]
                if old_frame != int(frame_ids[i]): raise RuntimeError(f"inconsistent camera boundary identity {index}")
                if old is not item and not (isinstance(old,torch.Tensor) and isinstance(item,torch.Tensor) and torch.equal(old,item)):
                    raise RuntimeError(f"inconsistent camera boundary {index}")
                if K is not None and old_K is not None and not torch.equal(old_K,K): raise RuntimeError(f"inconsistent intrinsics boundary {index}")
            else: self._items[index]=(int(frame_ids[i]),item,K)
        self._chunk_index+=1
    def slots(self, source_camera, source_intrinsics=None):
        if not self._items:
            return [(source_camera, source_intrinsics)] * 19
        last=max(self._items)
        keys=list(range(max(0,last-18),last+1))
        if len(keys)<19: keys=[None]*(19-len(keys))+keys
        return [(source_camera, source_intrinsics) if key is None or key not in self._items else (self._items[key][1], self._items[key][2] if self._items[key][2] is not None else source_intrinsics) for key in keys]
    def slot_frame_ids(self):
        return tuple(self._items[key][0] for key in sorted(self._items)[-19:])
    def slot_latent_ids(self):
        if not self._items: return (0,)*19
        last=max(self._items); keys=list(range(max(0,last-18),last+1))
        return tuple([0]*(19-len(keys))+keys)
    def indices(self): return tuple(sorted(self._items))
    def camera_for(self, global_id, source_camera, source_intrinsics=None):
        if global_id is None or global_id not in self._items: return source_camera,source_intrinsics
        _,camera,K=self._items[global_id]; return camera,K if K is not None else source_intrinsics
