"""Strictly causal source + long/mid/short latent history."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Iterable
import torch

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
        selected=sorted(self._frames)[-19:]; padded=[0]*(19-len(selected))+selected
        return HistoryLayout(0,tuple(padded[:16]),tuple(padded[16:18]),tuple(padded[18:]))
    def slots(self) -> list[Any]:
        layout=self.layout(); return [self._source]+[self._source if f==0 else self._frames[f] for f in layout.long+layout.mid+layout.short]
    def seen_frames(self) -> tuple[int,...]: return tuple(sorted(self._frames))
