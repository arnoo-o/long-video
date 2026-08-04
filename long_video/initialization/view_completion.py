import numpy as np
from ..types import ViewSet
from ..data.panorama_projection import panorama_to_perspective
class HoloOracleCompletion:
    def __init__(self, fov=90., height=64, width=64): self.fov=fov; self.height=height; self.width=width
    def complete(self, panorama, depth, pose=None, observed_indices=(0,)):
        yaws=np.arange(8)*np.pi/4; rgb=np.stack([panorama_to_perspective(panorama,y,self.fov and 0,self.fov,self.height,self.width) for y in yaws]); dep=np.stack([panorama_to_perspective(depth,y,0,self.fov,self.height,self.width) for y in yaws]); dep=dep.astype(np.float32)
        src=np.ones((8,self.height,self.width),np.int8); src[list(observed_indices)]=0; conf=np.where(src==0,1.,.4).astype(np.float32); c2w=np.repeat(np.eye(4,dtype=np.float32)[None],8,0); intr=np.repeat(np.array([[[self.width/2,0,self.width/2],[0,self.width/2,self.height/2],[0,0,1]]],np.float32),8,0); return ViewSet(rgb,dep,np.isfinite(dep).astype(np.float32),c2w,intr,src,conf)
