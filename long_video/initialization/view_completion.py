import numpy as np
from ..types import ViewSet, RAY_DISTANCE
from ..data.panorama_projection import equirectangular_to_perspective, build_canonical_view_cameras
from .mvdiffusion_backend import MVDiffusionCompletion

class HoloOracleCompletion:
    """Geometry upper bound: unobserved views are labeled synthesized but sampled from held-out panorama."""
    def __init__(self,fov_degrees=90.,height=512,width=512,observed_confidence=1.,synthesized_confidence=.4):
        self.fov_degrees=fov_degrees; self.height=height; self.width=width; self.observed_confidence=observed_confidence; self.synthesized_confidence=synthesized_confidence
    def complete(self,panorama,depth,panorama_c2w=None,mask=None,observed_indices=(0,)):
        center=np.eye(4,dtype=np.float32) if panorama_c2w is None else np.asarray(panorama_c2w,np.float32)
        c2w,k=build_canonical_view_cameras(center,self.fov_degrees,self.width,self.height)
        yaws=np.deg2rad(np.arange(8)*45.)
        rgb=np.stack([equirectangular_to_perspective(panorama,y,0,self.fov_degrees,self.height,self.width,"bilinear") for y in yaws])
        dep=np.stack([equirectangular_to_perspective(depth,y,0,self.fov_degrees,self.height,self.width,"bilinear") for y in yaws]).astype(np.float32)
        valid=np.isfinite(dep)&(dep>0)
        if mask is not None:
            p_mask=np.stack([equirectangular_to_perspective(np.asarray(mask,dtype=np.uint8),y,0,self.fov_degrees,self.height,self.width,"nearest") for y in yaws])>0
            valid &= p_mask
        dep[~valid]=np.nan
        source=np.ones((8,self.height,self.width),np.int8); source[list(observed_indices)]=0
        confidence=np.where(source==0,self.observed_confidence,self.synthesized_confidence).astype(np.float32)
        return ViewSet(rgb,dep,valid.astype(np.float32),c2w,k,source,confidence,RAY_DISTANCE)

class PrecomputedCompletion:
    def __init__(self,root): self.root=__import__("pathlib").Path(root)
    def complete(self,*_args,**_kwargs):
        p=self.root; rgb=np.load(p/"views_rgb.npy"); depth=np.load(p/"views_depth.npy"); c2w=np.load(p/"view_poses.npy"); k=np.load(p/"intrinsics.npy")
        source=np.load(p/"source_maps.npy"); confidence=np.load(p/"image_confidence.npy")
        return ViewSet(rgb,depth,np.isfinite(depth).astype(np.float32),c2w,k,source,confidence,RAY_DISTANCE)
