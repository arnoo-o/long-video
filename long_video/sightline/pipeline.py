"""Geometry-free Helios autoregressive pipeline contract."""
from dataclasses import dataclass
import torch
from .history import HistoryManager
from .memory import LongTermKVMemory
from .rays import token_rays_for_shape
@dataclass
class SightlineConfig:
    history_sizes: tuple=(16,2,1); chunk_length:int=33; stride:int=32
    memory_budget:int=12960; memory_layers:tuple=(); sightline_enabled:bool=True
class SightlinePipeline:
    def __init__(self, helios, *, config=None, conditioner=None):
        self.helios=helios; self.config=config or SightlineConfig(); self.conditioner=conditioner
        self.history=HistoryManager(self.config.history_sizes,self.config.chunk_length,self.config.stride); self.memory=LongTermKVMemory(self.config.memory_budget)
    @torch.no_grad()
    def generate_chunk(self, source_latent, c2w, intrinsics, controls=None):
        if self.history._source is None: self.history.set_source(source_latent)
        if not hasattr(self.helios,'generate_chunk'): raise RuntimeError('Helios adapter must expose generate_chunk; no WAH fallback is allowed')
        latent_shape=(c2w.shape[0],self.config.chunk_length,getattr(self.helios,'token_height',48),getattr(self.helios,'token_width',80),1)
        rays=token_rays_for_shape(c2w,intrinsics,latent_shape,getattr(self.helios,'patch_size',(1,1)))
        out=self.helios.generate_chunk(source_latent,self.history.slots(),rays,controls=controls)
        if not isinstance(out,dict) or 'latents' not in out: raise RuntimeError('Helios adapter returned invalid chunk')
        self.history.append_chunk(out['latents'])
        for layer,hidden in out.get('memory_features',{}).items(): self.memory.capture(hidden,out['memory_rays'][layer],self.history.chunk_index-1)
        return out
def assert_geometry_free_imports():
    import sys
    forbidden=('warp_as_history','long_video.wah','long_video.geometry.point_renderer','long_video.initialization.recal3r','pi3')
    bad=[x for x in sys.modules if any(x==f or x.startswith(f+'.') for f in forbidden)]
    if bad: raise RuntimeError('geometry/WAH modules loaded in Sightline process: '+','.join(bad))
