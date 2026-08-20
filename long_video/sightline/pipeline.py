"""Formal geometry-free Sightline runtime around the native Helios pipeline."""
from __future__ import annotations
from dataclasses import dataclass
import torch
from ..config import SightlineConfig
from .history import CameraHistoryState
from .memory import LayerKVMemoryBank

@dataclass
class SightlineRuntimeContext:
    chunk_index:int=0; stage:int=0; sigma:float=0.0; memory_write_allowed:bool=False

class SightlinePipeline:
    """Owns runtime state while delegating denoising to pinned Helios itself."""
    def __init__(self, helios_pipeline, *, config:SightlineConfig, conditioner=None):
        self.helios=helios_pipeline; self.config=config; self.conditioner=conditioner
        self.camera_history=CameraHistoryState(); self.memory=LayerKVMemoryBank(config.memory_layers,config.memory_budget,config.memory_pool)
        self.runtime=SightlineRuntimeContext(); self._source_initialized=False
    def generate(self, *, image, prompt, negative_prompt, height, width, num_frames, steps, attention_kwargs=None):
        if not self._source_initialized: self._source_initialized=True
        if num_frames < 33 or (num_frames-1)%32: raise ValueError('Sightline inference requires 1+32*N frames')
        kwargs=dict(prompt=prompt,negative_prompt=negative_prompt,image=image,height=height,width=width,num_frames=num_frames,num_inference_steps=steps,history_sizes=list(self.config.history_sizes),num_latent_frames_per_chunk=9,is_enable_stage2=True,pyramid_num_inference_steps_list=list(self.config.pyramid_steps),attention_kwargs=attention_kwargs or {},output_type='np')
        return self.helios(**kwargs)
    @staticmethod
    def assert_geometry_free_imports():
        import sys
        forbidden=('warp_as_history','long_video.wah','long_video.geometry.point_renderer','long_video.initialization.recal3r','pi3')
        bad=[x for x in sys.modules if any(x==f or x.startswith(f+'.') for f in forbidden)]
        if bad: raise RuntimeError('geometry/legacy modules loaded: '+','.join(bad))
