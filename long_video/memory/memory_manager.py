import numpy as np
from .node_builder import build_from_views
class MemoryManager:
    def __init__(self, coverage_threshold=.35, min_transition_frames=3): self.coverage_threshold=coverage_threshold; self.min_transition_frames=min_transition_frames; self.low_count=0
    def observe(self, coverage): self.low_count=self.low_count+1 if coverage<self.coverage_threshold else 0; return self.low_count
    def should_build(self, buffer): return len(buffer)>=self.min_transition_frames
    def promote(self, active, candidate): active.status='archived'; candidate.status='active'; candidate.parent_id=active.node_id; return candidate
