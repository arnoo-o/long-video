from dataclasses import dataclass
import numpy as np
@dataclass
class TransitionFrame:
    generated_rgb: np.ndarray; camera_c2w: np.ndarray; old_node_warp: object; warp_visibility: np.ndarray; warp_confidence: np.ndarray; coverage: float; global_frame_index: int
class TransitionBuffer:
    def __init__(self): self.frames=[]
    def append(self, **kwargs): self.frames.append(TransitionFrame(**kwargs))
    def clear(self): self.frames.clear()
    def __len__(self): return len(self.frames)
