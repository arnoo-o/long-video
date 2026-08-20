"""Geometry-free Helios autoregressive pipeline contract."""
import torch
from ..config import SightlineConfig
from .history import HistoryManager
from .memory import LongTermKVMemory
from .rays import token_rays_for_shape
class SightlinePipeline:
    def __init__(self, helios, *, config=None, conditioner=None):
        if config is None: raise ValueError("SightlinePipeline requires validated SightlineConfig")
        self.helios=helios; self.config=config; self.conditioner=conditioner
        self.history=HistoryManager(self.config.history_sizes,self.config.chunk_length,self.config.chunk_stride); self.memory=LongTermKVMemory(self.config.memory_budget)
    @torch.no_grad()
    def generate_chunk(self, source_latent, c2w, intrinsics, controls=None, *, source_height: int, source_width: int):
        if self.history._source is None: self.history.set_source(source_latent)
        if not hasattr(self.helios,'generate_chunk'): raise RuntimeError('Helios adapter must expose generate_chunk; no WAH fallback is allowed')
        token_shape=getattr(self.helios, 'token_shape', None)
        if token_shape is None or len(token_shape) != 3: raise RuntimeError('Helios adapter must expose token_shape=(T,H,W) from its real patch embedding')
        latent_shape=(c2w.shape[0],token_shape[0],token_shape[1],token_shape[2],1)
        rays=token_rays_for_shape(c2w,intrinsics,latent_shape,source_height=source_height,source_width=source_width)
        out=self.helios.generate_chunk(source_latent,self.history.slots(),rays,controls=controls)
        if not isinstance(out,dict) or 'latents' not in out: raise RuntimeError('Helios adapter returned invalid chunk')
        self.history.append_chunk(out['latents'])
        for layer,hidden in out.get('memory_features',{}).items():
            grid=out.get('memory_grid_shapes',{}).get(layer)
            self.memory.capture(hidden,out['memory_rays'][layer],self.history.chunk_index-1,grid_shape=grid)
        return out
def assert_geometry_free_imports():
    import sys
    forbidden=('warp_as_history','long_video.wah','long_video.geometry.point_renderer','long_video.initialization.recal3r','pi3')
    bad=[x for x in sys.modules if any(x==f or x.startswith(f+'.') for f in forbidden)]
    if bad: raise RuntimeError('geometry/WAH modules loaded in Sightline process: '+','.join(bad))
