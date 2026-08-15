import numpy as np
import pytest
from pathlib import Path
from types import SimpleNamespace
torch = pytest.importorskip("torch")

from long_video.geometry.geotoken import (
    GEOTOKEN_BLOCKS, STAGE_RMS_CAPS, STAGE_SCALES, GeometryTokenizer,
    GeoTokenConditioner, GeometryTokenBatch, TEMPORAL_GROUPS,
    camera_channels_from_cameras, effective_strengths, progress_from_sigma,
    scheduler_progress_from_timestep, time_scale_from_progress,
    world_channels_from_cuda_render,
)
from long_video.geometry.geotoken_runtime import (
    stage_for_grid, stage_for_hidden_states, token_grid_for_hidden_states,
)
from long_video.geometry.voxel_fusion import _select_anchor_indices, fuse_voxels
from long_video.initialization.pi3x_initial_world import build_pi3x_source_world
from long_video.online.pipeline import validate_conditioning_world_identities
from long_video.training.geotoken import (BalancedRolloutSampler, assert_causal_world_cutoff,
    checkpoint_names, max_chunks_for_step, phase_for_step, split_phase_a_conditioning)


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
    assert stage_for_hidden_states(torch.zeros(1,16,9,12,20))==0
    assert stage_for_hidden_states(torch.zeros(1,16,9,24,40))==1
    assert stage_for_hidden_states(torch.zeros(1,16,9,48,80))==2
    assert token_grid_for_hidden_states(torch.zeros(1,16,9,12,20),(1,2,2))==(6,10)
    assert token_grid_for_hidden_states(torch.zeros(1,16,9,24,40),(1,2,2))==(12,20)
    assert token_grid_for_hidden_states(torch.zeros(1,16,9,48,80),(1,2,2))==(24,40)
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
        diagnostics=module.diagnostics[8]
        assert diagnostics["q_delta_ratio_pre_cap"]>cap
        assert diagnostics["k_delta_ratio_pre_cap"]>cap
        assert diagnostics["q_delta_ratio_post_cap"]<=cap+1e-5
        assert diagnostics["k_delta_ratio_post_cap"]<=cap+1e-5
        assert "q_delta_ratio" not in diagnostics and "k_delta_ratio" not in diagnostics


def test_sigma_progress_and_time_scale():
    assert progress_from_sigma(torch.tensor([1.]))==0 and time_scale_from_progress(0)==1
    assert progress_from_sigma(torch.tensor([.4]))==pytest.approx(.6) and time_scale_from_progress(.6)==1
    assert progress_from_sigma(torch.tensor([0.]))==1 and time_scale_from_progress(1)==pytest.approx(.25)
    class Scheduler:
        timesteps=torch.tensor([999.,499.5,0.]); sigmas=torch.tensor([1.,.4,0.,0.])
    assert scheduler_progress_from_timestep(Scheduler(),torch.tensor([499]))==pytest.approx(.6)
    assert scheduler_progress_from_timestep(Scheduler(),torch.tensor([999]))==0
    assert scheduler_progress_from_timestep(Scheduler(),torch.tensor([0]))==1
    with pytest.raises(RuntimeError): scheduler_progress_from_timestep(Scheduler(),torch.tensor([123]))
    class Ambiguous:
        timesteps=torch.tensor([499.2,499.8,0.]); sigmas=torch.tensor([1.,.4,0.,0.])
    with pytest.raises(RuntimeError): scheduler_progress_from_timestep(Ambiguous(),torch.tensor([499]))


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
    assert not (Path(__file__).parents[1]/"patches/wah_geotoken_kv_cache.patch").exists()
    apply_script=(Path(__file__).parents[1]/"scripts/apply_wah_patch.sh").read_text()
    assert "wah_geotoken_kv_cache.patch" not in apply_script


def test_phase_a_may_split_geometry_and_appearance_but_other_phases_may_not():
    validate_conditioning_world_identities(
        {"world_identity":"full-geometry","wah_world_identity":"causal-rgb","allow_distinct_worlds":True},
        "causal-rgb","causal-rgb")
    with pytest.raises(RuntimeError):
        validate_conditioning_world_identities(
            {"world_identity":"full-geometry","wah_world_identity":"causal-rgb"},
            "causal-rgb","causal-rgb")
    with pytest.raises(RuntimeError):
        validate_conditioning_world_identities(
            {"world_identity":"full-geometry","wah_world_identity":"causal-rgb","allow_distinct_worlds":True},
            "causal-rgb","mutated-rgb")
    assert_causal_world_cutoff(np.array([0,32]),32,label="Phase A WAH appearance world")
    with pytest.raises(RuntimeError):
        assert_causal_world_cutoff(np.array([0,33]),32,label="Phase A WAH appearance world")
    full=(np.array([[1,2,3]],np.float32),np.array([[255,0,0]],np.uint8),np.array([.9]),np.array([5]))
    causal=(np.array([[4,5,6]],np.float32),np.array([[0,255,0]],np.uint8),np.array([.7]),np.array([1]),np.array([[0,0,0]]),np.array([32]))
    geometry,appearance=split_phase_a_conditioning(full,causal,32)
    assert np.array_equal(geometry[0],full[0]) and np.array_equal(geometry[1],full[2])
    assert np.array_equal(appearance[1],causal[1])
    assert not np.array_equal(appearance[1],full[1])


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


def test_fast_anchor_selection_matches_original_causal_rule():
    rng = np.random.default_rng(5)
    inverse = np.repeat(np.arange(200), rng.integers(1, 12, size=200))
    rng.shuffle(inverse)
    confidence = rng.uniform(.01, 1., size=len(inverse)).astype(np.float32)
    locked = rng.random(len(inverse)) < .08
    expected = np.empty(200, np.int64)
    for voxel_index in range(200):
        members = np.flatnonzero(inverse == voxel_index)
        locked_members = members[locked[members]]
        if len(locked_members):
            expected[voxel_index] = locked_members[0]
            continue
        best = members[0]
        for candidate in members[1:]:
            if confidence[candidate] > 1.1 * confidence[best]:
                best = candidate
        expected[voxel_index] = best
    actual = _select_anchor_indices(inverse, confidence, locked, 200)
    assert np.array_equal(actual, expected)


def test_pi3x_w0_keeps_v3_weighted_rgb_then_locks_voxel_anchor():
    class Backend:
        def predict_source(self, _rgb, _c2w, _intrinsics):
            return SimpleNamespace(
                point_maps=np.array([[[[.001,0,1],[.019,0,1]]]],np.float32),
                geometry_confidence=np.array([[[.25,.75]]],np.float32),
                depth=np.ones((1,1,2),np.float32), depth_convention="Z_DEPTH",
                diagnostics={"source_rgb_resized":np.array([[[10,20,30],[110,120,130]]],np.uint8)},
            )
    node=build_pi3x_source_world(np.zeros((1,2,3),np.uint8),np.eye(4,dtype=np.float32),np.eye(3,dtype=np.float32),Backend())
    expected=np.rint((np.array([10,20,30])*.25+np.array([110,120,130])*.75)/1.).astype(np.uint8)
    assert np.array_equal(node.points_rgb[0],expected)
    assert np.array_equal(node.appearance_anchors["anchor_rgb"],node.points_rgb)
    assert bool(node.appearance_anchors["source_locked"][0])
    new_xyz=node.points_xyz.copy(); new_rgb=np.array([[255,0,0]],np.uint8); new_conf=np.array([10.],np.float32)
    _,rgb,_,_,_,anchors=fuse_voxels(
        np.concatenate([node.points_xyz,new_xyz]),np.concatenate([node.points_rgb,new_rgb]),
        np.concatenate([node.points_confidence,new_conf]),np.concatenate([node.observation_count,[1]]),.02,
        anchor_confidence=np.concatenate([node.appearance_anchors["anchor_confidence"],new_conf]),
        anchor_frame=np.array([0,1],np.int32),source_locked=np.array([True,False]),return_anchors=True)
    assert np.array_equal(rgb[0],expected) and np.array_equal(anchors["anchor_rgb"][0],expected)


def test_curriculum_is_unchanged():
    assert TEMPORAL_GROUPS[0]==(0,) and phase_for_step(1101)=="C" and max_chunks_for_step(1861)==6
    sampler=BalancedRolloutSampler(7); values=[sampler.choose_length(1701+i) for i in range(40)]
    assert max(values.count(i) for i in range(1,5))-min(values.count(i) for i in range(1,5))<=1
    assert checkpoint_names(2000)==("checkpoint_step_2000.pt","phase_c_final_step_2000.pt")
