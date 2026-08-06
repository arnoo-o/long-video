import numpy as np
import pytest
from long_video.oracle_training.dense24 import DenseTiming,allocate_disjoint_windows,continuous_runs,dense_rgb_weights,interpolate_c2w,temporal_weights_to_latent,validate_window

def test_anchor_dense_contracts():
    t=DenseTiming(); assert t.dense_count(5)==33; assert t.dense_count(17)==129
    np.testing.assert_array_equal(t.anchor_indices(5),[0,8,16,24,32]); np.testing.assert_allclose(t.alphas(),np.arange(1,8)/8)

def test_gap_rejected_and_windows_disjoint():
    runs,median=continuous_runs([0,1,2,3,4,20,21,22,23,24]); assert median==1 and runs==[(0,5),(5,10)]
    validate_window(0,5,runs)
    with pytest.raises(ValueError,match="acquisition gap"): validate_window(3,5,runs)
    result=allocate_disjoint_windows([(0,33),(40,71),(80,109)])
    used=[set(range(result["rollout"][0],result["rollout"][0]+17))]+[set(range(s,s+5)) for s in result["train"]+result["diagnostic"]]
    assert all(not(a&b) for i,a in enumerate(used) for b in used[i+1:])

def test_c2w_slerp_and_endpoints():
    a=np.eye(4); b=np.eye(4); b[:3,:3]=[[0,0,1],[0,1,0],[-1,0,0]]; b[:3,3]=[8,4,-2]
    dense=interpolate_c2w(np.stack([a,b])); np.testing.assert_allclose(dense[0],a,atol=1e-6); np.testing.assert_allclose(dense[-1],b,atol=1e-6)
    np.testing.assert_allclose(dense[4,:3,3],[4,2,-1],atol=1e-6); np.testing.assert_allclose(np.linalg.det(dense[:,:3,:3]),1,atol=1e-5)

def test_float_temporal_weights_use_real_vae_groups():
    weights=dense_rgb_weights(5); assert weights[0]==0 and np.all(weights[[8,16,24,32]]==1); assert np.all(weights[[1,7,9,31]]==.25)
    latent=temporal_weights_to_latent(weights,4); assert latent.dtype==np.float32 and len(latent)==9
    np.testing.assert_allclose(latent,[0,.25,.4375,.25,.4375,.25,.4375,.25,.4375])
    with pytest.raises(ValueError,match="all zero"): temporal_weights_to_latent(np.zeros(33,np.float32))
