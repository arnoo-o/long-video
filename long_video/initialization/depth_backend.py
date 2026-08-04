import numpy as np
from ..types import RAY_DISTANCE

class GroundTruthDepthBackend:
    depth_convention=RAY_DISTANCE
    def predict(self, view_rgb, view_c2w=None, intrinsics=None, known_depth=None, known_mask=None):
        if known_depth is None: raise ValueError("GroundTruthDepthBackend requires known_depth")
        depth=np.asarray(known_depth,np.float32).copy()
        valid=np.isfinite(depth)&(depth>0)
        if known_mask is not None: valid &= np.asarray(known_mask).astype(bool)
        depth[~valid]=np.nan
        return depth,valid.astype(np.float32),None,{"backend":"ground_truth","depth_convention":self.depth_convention}

class Pi3XDepthBackend:
    def __init__(self, checkpoint, repo_path=None, device="cuda"):
        self.checkpoint=checkpoint; self.repo_path=repo_path; self.device=device
    def predict(self, view_rgb, view_c2w, intrinsics, known_depth=None, known_mask=None):
        if known_depth is not None: return GroundTruthDepthBackend().predict(view_rgb,view_c2w,intrinsics,known_depth,known_mask)
        raise RuntimeError("Pi3X invocation is not installed. Configure pi3x.repo_path and checkpoint; this backend intentionally fails rather than returning fabricated geometry.")
