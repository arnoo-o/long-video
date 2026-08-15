"""Regression: Pi3X and ReCal inputs share the canonical voxel formula."""
import numpy as np
from long_video.geometry.voxel_fusion import fuse_voxels

def main():
    pi3_xyz=np.array([[0.001,0.001,0.001]],np.float32); recal_xyz=np.array([[0.019,0.001,0.001]],np.float32)
    rgb=np.array([[10,20,30],[110,120,130]],np.uint8); conf=np.array([.5,.8],np.float32)
    xyz, color, confidence, count, keys=fuse_voxels(np.r_[pi3_xyz,recal_xyz],rgb,conf,[1,1],.02)
    assert len(xyz)==1 and tuple(keys[0])==(0,0,0) and count[0]==2
    expected=(pi3_xyz[0]*.5+recal_xyz[0]*.8)/1.3
    assert np.allclose(xyz[0],expected) and np.isclose(confidence[0],.65)
    print('pi3x-recal-voxel-fusion-ok')
if __name__=='__main__': main()
