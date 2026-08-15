import numpy as np
import pytest
torch = pytest.importorskip("torch")

from long_video.geometry.geotoken import (GEOTOKEN_BLOCKS, GeometryTokenizer, GeoTokenConditioner,
    GeometryTokenBatch, TEMPORAL_GROUPS, camera_channels_from_cameras, world_channels_from_cuda_render)
from long_video.geometry.voxel_fusion import fuse_voxels
from long_video.training.geotoken import BalancedRolloutSampler, checkpoint_names, max_chunks_for_step, phase_for_step


def test_camera_ray_survives_unknown_world():
    camera = camera_channels_from_cameras(np.repeat(np.eye(4, dtype=np.float32)[None], 33, 0),
        np.repeat(np.eye(3, dtype=np.float32)[None], 33, 0), np.zeros(3), 1., 3, 5, device="cpu")
    world = world_channels_from_cuda_render(torch.zeros(33,3,5,3), torch.zeros(33,3,5), torch.zeros(33,3,5,dtype=torch.bool), torch.zeros(33,3,5), np.repeat(np.eye(4,dtype=np.float32)[None],33,0), np.zeros(3), 1.)
    tokenizer=GeometryTokenizer(16); c,w,s=tokenizer(camera.permute(3,0,1,2).unsqueeze(0),world.permute(3,0,1,2).unsqueeze(0))
    assert c.shape == (1,256,9,3,5) and torch.count_nonzero(c)
    assert not torch.count_nonzero(w) and not torch.count_nonzero(s)


def test_qk_binding_keeps_value_and_history_query_frozen():
    module=GeoTokenConditioner(16); assert module.block_indices == GEOTOKEN_BLOCKS
    current=GeometryTokenBatch(torch.randn(1,3,256),torch.randn(1,3,256),torch.ones(1,3,1))
    history=GeometryTokenBatch(torch.randn(1,2,256),torch.randn(1,2,256),torch.ones(1,2,1))
    module.set_active(current, history); q=torch.randn(1,5,16); k=torch.randn(1,5,16); v=torch.randn(1,5,16)
    outq,outk,outv=module._make_qk_binding(8)(q,k,v,original_context_length=3)
    assert outv is v and torch.equal(outq,q) and torch.equal(outk,k)  # tanh(0) strict no-op
    with torch.no_grad(): module.gates["8_camera_q"].fill_(1.)
    outq,_,_=module._make_qk_binding(8)(q,k,v,original_context_length=3)
    assert torch.equal(outq[:,:2],q[:,:2])


def test_rgb_anchor_is_source_locked_and_recal_formula_is_shared():
    points=np.array([[.001,.001,.001],[.019,.001,.001]],np.float32); rgb=np.array([[10,20,30],[110,120,130]],np.uint8)
    xyz,color,confidence,count,keys,anchors=fuse_voxels(points,rgb,np.array([.5,.9],np.float32),[1,1],.02,source_locked=[True,False],return_anchors=True)
    assert len(xyz)==1 and count[0]==2 and tuple(color[0]) == (10,20,30) and anchors["source_locked"][0]
    assert np.allclose(xyz[0],(points[0]*.5+points[1]*.9)/1.4) and np.isclose(confidence[0],.7)


def test_curriculum_is_unchanged():
    assert TEMPORAL_GROUPS[0] == (0,) and phase_for_step(1101)=="C" and max_chunks_for_step(1861)==6
    sampler=BalancedRolloutSampler(7); values=[sampler.choose_length(1701+i) for i in range(40)]
    assert max(values.count(i) for i in range(1,5))-min(values.count(i) for i in range(1,5))<=1
    assert checkpoint_names(2000)==("checkpoint_step_2000.pt","phase_c_final_step_2000.pt")
