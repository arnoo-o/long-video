"""Strictly causal source + long/mid/short latent history."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Iterable
import torch
from .rays import latent_camera_indices

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
