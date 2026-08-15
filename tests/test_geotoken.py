import numpy as np
import pytest
from pathlib import Path
torch = pytest.importorskip("torch")

from long_video.geometry.geotoken import (
    GEOTOKEN_BLOCKS, STAGE_RMS_CAPS, STAGE_SCALES, GeometryTokenizer,
    GeoTokenConditioner, GeometryTokenBatch, TEMPORAL_GROUPS,
    camera_channels_from_cameras, effective_strengths, progress_from_sigma,
    scheduler_progress_from_timestep, time_scale_from_progress,
    world_channels_from_cuda_render,
)
from long_video.geometry.geotoken_runtime import stage_for_grid
from long_video.geometry.voxel_fusion import fuse_voxels
from long_video.training.geotoken import BalancedRolloutSampler, checkpoint_names, max_chunks_for_step, phase_for_step


def _tokens():
    current=GeometryTokenBatch(torch.randn(1,3,256),torch.randn(1,3,256),torch.ones(1,3,1))
    history=GeometryTokenBatch(torch.randn(1,2,256),torch.randn(1,2,256),torch.ones(1,2,1))
    return current, history


def test_world_support_is_visibility_times_confidence():
    tokenizer=GeometryTokenizer(16)
    camera=torch.zeros(3,6,33,1,1)
    world=torch.zeros(3,9,33,1,1)
    world[0,7]=1; world[0,8]=0
    world[1,7]=0; world[1,8]=1
    world[2,7]=.5; world[2,8]=.8
    _,_,support=tokenizer(camera,world)
    assert torch.allclose(support[:,0,:,0,0],torch.tensor([[0.]*9,[0.]*9,[.4]*9]))


def test_camera_ray_survives_unknown_world():
    c2w=np.repeat(np.eye(4,dtype=np.float32)[None],33,0); k=np.repeat(np.eye(3,dtype=np.float32)[None],33,0)
    camera=camera_channels_from_cameras(c2w,k,np.zeros(3),1.,3,5,device="cpu")
    world=world_channels_from_cuda_render(torch.zeros(33,3,5,3),torch.zeros(33,3,5),torch.zeros(33,3,5,dtype=torch.bool),torch.zeros(33,3,5),c2w,np.zeros(3),1.)
    c,w,s=GeometryTokenizer(16)(camera.permute(3,0,1,2)[None],world.permute(3,0,1,2)[None])
    assert c.shape==(1,256,9,3,5) and torch.count_nonzero(c)
    assert not torch.count_nonzero(w) and not torch.count_nonzero(s)


def test_stage_mapping_scales_and_caps():
    assert stage_for_grid(12,20)==0 and stage_for_grid(24,40)==1 and stage_for_grid(48,80)==2
    assert STAGE_SCALES==(1.,.7,.4) and STAGE_RMS_CAPS==(.15,.10,.06)
    with pytest.raises(RuntimeError): stage_for_grid(16,24)


def test_qk_rms_cap_for_every_stage():
    module=GeoTokenConditioner(16); current,history=_tokens(); module.set_active(current,history)
    module.configure_strengths(geotoken=1,camera=1,world=1)
    with torch.no_grad():
        for gate in module.gates.values(): gate.fill_(5)
        for adapters in module.adapters.values():
            for adapter in adapters.values(): adapter.up.weight.fill_(10)
    q=torch.randn(1,5,16); k=torch.randn(1,5,16); v=torch.randn(1,5,16)
    for stage,cap in enumerate(STAGE_RMS_CAPS):
        module.set_timing(stage_index=stage,denoise_progress=0)
        oq,ok,_=module._make_qk_binding(8)(q,k,v,original_context_length=3)
        qratio=(oq-q).float().square().mean(-1).sqrt()/q.float().square().mean(-1).sqrt().clamp_min(1e-8)
        kratio=(ok-k).float().square().mean(-1).sqrt()/k.float().square().mean(-1).sqrt().clamp_min(1e-8)
        assert float(qratio.max())<=cap+1e-5 and float(kratio.max())<=cap+1e-5


def test_sigma_progress_and_time_scale():
    assert progress_from_sigma(torch.tensor([1.]))==0 and time_scale_from_progress(0)==1
    assert progress_from_sigma(torch.tensor([.4]))==pytest.approx(.6) and time_scale_from_progress(.6)==1
    assert progress_from_sigma(torch.tensor([0.]))==1 and time_scale_from_progress(1)==pytest.approx(.25)
    class Scheduler:
        timesteps=torch.tensor([900.,400.,0.]); sigmas=torch.tensor([1.,.4,0.])
    assert scheduler_progress_from_timestep(Scheduler(),torch.tensor([400.]))==pytest.approx(.6)


def test_qk_timing_history_and_value_semantics():
    torch.manual_seed(4); module=GeoTokenConditioner(16); current,history=_tokens(); module.set_active(current,history)
    module.configure_strengths(geotoken=1,camera=1,world=1)
    with torch.no_grad():
        module.gates["8_camera_q"].fill_(.2); module.gates["8_camera_k"].fill_(.2)
    q=torch.ones(1,5,16)*100; k=torch.ones(1,5,16)*100; v=torch.randn(1,5,16)
    module.set_timing(stage_index=0,denoise_progress=0)
    q0,k0,v0=module._make_qk_binding(8)(q,k,v,original_context_length=3)
    module.set_timing(stage_index=0,denoise_progress=1)
    q1,k1,v1=module._make_qk_binding(8)(q,k,v,original_context_length=3)
    assert torch.equal(q0[:,:2],q[:,:2]) and torch.equal(q1[:,:2],q[:,:2])
    assert not torch.equal(k0[:,:2],k1[:,:2])
    assert v0 is v and v1 is v


def test_wah_patch_disables_cache_only_when_qk_binding_is_installed():
    patch=(Path(__file__).parents[1]/"patches/wah_geotoken_qk_binding.patch").read_text()
    assert 'getattr(attn, "geotoken_qk_binding", None) is None' in patch


def test_zero_strength_is_strict_noop_and_strength_formula_matches():
    assert effective_strengths(.5,.25,.8)==(.125,.4)
    module=GeoTokenConditioner(16); current,history=_tokens(); module.set_active(current,history)
    with torch.no_grad():
        for gate in module.gates.values(): gate.fill_(1)
    module.configure_strengths(geotoken=0,camera=1,world=1)
    q=torch.randn(1,5,16); k=torch.randn(1,5,16); v=torch.randn(1,5,16)
    oq,ok,ov=module._make_qk_binding(8)(q,k,v,original_context_length=3)
    assert torch.equal(oq,q) and torch.equal(ok,k) and ov is v


def test_rgb_anchor_is_source_locked_and_recal_formula_is_shared():
    points=np.array([[.001,.001,.001],[.019,.001,.001]],np.float32); rgb=np.array([[10,20,30],[110,120,130]],np.uint8)
    xyz,color,confidence,count,_,anchors=fuse_voxels(points,rgb,np.array([.5,.9],np.float32),[1,1],.02,source_locked=[True,False],return_anchors=True)
    assert len(xyz)==1 and count[0]==2 and tuple(color[0])==(10,20,30) and anchors["source_locked"][0]
    assert np.allclose(xyz[0],(points[0]*.5+points[1]*.9)/1.4) and np.isclose(confidence[0],.7)


def test_curriculum_is_unchanged():
    assert TEMPORAL_GROUPS[0]==(0,) and phase_for_step(1101)=="C" and max_chunks_for_step(1861)==6
    sampler=BalancedRolloutSampler(7); values=[sampler.choose_length(1701+i) for i in range(40)]
    assert max(values.count(i) for i in range(1,5))-min(values.count(i) for i in range(1,5))<=1
    assert checkpoint_names(2000)==("checkpoint_step_2000.pt","phase_c_final_step_2000.pt")
