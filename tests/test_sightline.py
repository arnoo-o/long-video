import pytest
import numpy as np
import os, sys
torch=pytest.importorskip('torch')
from long_video.sightline.rays import plucker_rays, temporal_group_cameras
from long_video.sightline.conditioning import SightlineConditioner
from long_video.sightline.history import HistoryManager, CameraHistoryState
from long_video.sightline.memory import LongTermKVMemory
from long_video.sightline.rays import temporal_group_cameras
from long_video.sightline.helios_integration import SightlineHeliosAttnProcessor, SightlineRayProvider

def test_plucker_ray_geometry():
    K=torch.tensor([[[100.,0,2],[0,100,2],[0,0,1]]]); c=torch.eye(4).unsqueeze(0)
    r=plucker_rays(c,K,4,4,source_height=4,source_width=4); assert r.shape==(1,1,4,4,7)
    assert torch.allclose(r[0,0,2,2,:3],torch.tensor([0.,0.,1.]),atol=1e-5)

def test_scale_augmentation_gate_only_and_zero_alpha():
    torch.manual_seed(1); m=SightlineConditioner(16); r=torch.randn(2,3,7); q,k=m(r,training=False); assert q.shape==k.shape==(2,3,16)
    m.alpha_q.data.zero_(); m.alpha_k.data.zero_(); q,k=m(r,training=True); assert torch.count_nonzero(q)==0 and torch.count_nonzero(k)==0

def test_history_six_chunks_causal_and_shared_boundary():
    h=HistoryManager(); src=torch.zeros(1); h.set_source(src)
    chunks=[[torch.tensor(float(c*32+i)) for i in range(33)] for c in range(6)]
    for c,chunk in enumerate(chunks): h.append_chunk(chunk); assert max(h.seen_frames())==c*32+32
    assert len(h.slots())==20 and max(h.layout().long+h.layout().mid+h.layout().short)<=192

def test_camera_history_33_to_9_and_exact_shared_boundary():
    state=CameraHistoryState(); source=torch.eye(4); K=torch.eye(3)
    reps=[torch.full((4,4),float(i)) for i in range(9)]
    Ks=[torch.eye(3)*(i+1) for i in range(9)]
    state.append_chunk(reps, [0,4,8,12,16,20,24,28,32], Ks)
    reps2=[reps[-1]]+[torch.full((4,4),float(i)) for i in range(9,17)]
    Ks2=[Ks[-1]]+[torch.eye(3)*(i+1) for i in range(9,17)]
    state.append_chunk(reps2, [32,36,40,44,48,52,56,60,64], Ks2)
    slots=state.slots(source,K)
    assert len(slots)==19 and torch.equal(slots[0][0], source) and torch.equal(slots[2][0], reps[0])
    assert state.slot_frame_ids()[-1] == 64

def test_history_ray_count_must_be_exact():
    from long_video.sightline.helios_integration import SightlineRayProvider
    provider=SightlineRayProvider(source_height=4,source_width=4)
    c=torch.eye(4).view(1,1,4,4); k=torch.eye(3).view(1,1,3,3)
    provider.set_context(chunk_index=0,c2w=c,intrinsics=k,history_cameras=c[:,:1],history_intrinsics=k[:,:1],token_shape=(1,2,2),stage_shapes=((1,2,2),))
    with pytest.raises(RuntimeError,match="exactly match"):
        provider(torch.zeros(1,4,8),key_length=9,current_length=4)

def test_train_chunk_policy_is_single_and_causal():
    from long_video.training.sightline import chunk_grad_policy, assert_single_backward_chunk, causal_chunk_plan, run_single_graph_chunks, run_causal_prefix_chunks
    train_chunk=2; policies=[chunk_grad_policy(i,train_chunk) for i in range(6)]
    assert policies[:2]==["forward_detached"]*2 and policies[2]=="backward"
    assert policies[3:]==["rollout_detached"]*3
    assert_single_backward_chunk(policies,train_chunk)
    assert causal_chunk_plan(6,2)==tuple(policies)
    weight=torch.nn.Parameter(torch.tensor(2.)); seen=[]
    outputs,_=run_single_graph_chunks(4,2,lambda chunk,grad: seen.append((chunk,grad)) or weight*(chunk+1))
    assert [out.requires_grad for out in outputs]==[False,False,True,False]
    seen=[]; outputs,policies=run_causal_prefix_chunks(3,2,lambda chunk,grad: seen.append((chunk,grad)) or weight*(chunk+1))
    assert seen==[(0,False),(1,False),(2,True)] and len(outputs)==3 and policies==('forward_detached','forward_detached','backward')

def test_camera_first_curriculum_uses_random_two_chunk_window():
    from long_video.training.sightline import curriculum_phase, select_chunk_window
    assert curriculum_phase(0)=={'name':'P1','max_chunks':1,'lora':False,'correspondence':False,'memory':False}
    assert curriculum_phase(299)['max_chunks']==1
    assert not curriculum_phase(300)['lora'] and curriculum_phase(300)['max_chunks']==2
    assert curriculum_phase(400)['lora'] and curriculum_phase(400)['max_chunks']==2
    assert curriculum_phase(999)['max_chunks']==2
    assert curriculum_phase(1000)['memory'] and curriculum_phase(1000)['correspondence'] and curriculum_phase(1000)['max_chunks']==1
    assert curriculum_phase(2499)['max_chunks']==6
    generator=torch.Generator().manual_seed(7)
    assert 0 <= select_chunk_window(2,generator=generator) <= 4

def test_memory_is_kv_only_and_eviction():
    m=LongTermKVMemory(budget=2,pool=1); x=torch.randn(1,4,4); r=torch.randn(1,4,7); m.capture(x,r,0,grid_shape=(1,2,2)); assert len(m)==2; k,v=m.get(); assert k.shape[-1]==4 and v.shape[-1]==7

def test_selected_layers_have_independent_qk_geometry_and_alphas():
    from long_video.training.sightline import SightlineTrainable
    trainable=SightlineTrainable(8,layers=(16,20,24),heads=2)
    conditioners=[trainable.conditioner.for_layer(layer) for layer in (16,20,24)]
    for name in ('q_proj','k_proj','gate','rms_norm_q','rms_norm_k'):
        assert len({id(getattr(layer,name).weight) for layer in conditioners})==3
    assert all(float(layer.alpha_q.detach())==1.0 and float(layer.alpha_k.detach())==1.0 for layer in conditioners)
    assert len([name for name,_ in trainable.named_parameters() if name.endswith(('alpha_q','alpha_k'))])==6
    assert all(torch.count_nonzero(layer.q_proj.weight)==0 and torch.count_nonzero(layer.k_proj.weight)==0 for layer in conditioners)

def test_scheduler_provenance_ignores_default_field_order():
    from long_video.training.sightline_checkpoint import scheduler_config_fingerprint
    left={'stages':3,'_use_default_values':['solver_order','predict_x0','thresholding']}
    right={'_use_default_values':['thresholding','solver_order','predict_x0'],'stages':3}
    assert scheduler_config_fingerprint(left)==scheduler_config_fingerprint(right)

def test_native_history_zero_fake_initialization_and_fixed_rope_slots():
    from long_video.sightline.history import NativeHistoryState,native_helios_indices
    source=torch.ones(1,1,1,1,1); fake=torch.full_like(source,2); state=NativeHistoryState(source,fake); packed=state.groups(); indices=native_helios_indices()
    assert torch.count_nonzero(packed['long'][0])==0 and torch.count_nonzero(packed['mid'][0])==0
    assert torch.equal(packed['short'][0][:,:,:1],source) and torch.equal(packed['short'][0][:,:,1:],fake)
    assert indices['long'].tolist()==[list(range(1,17))] and indices['mid'].tolist()==[[17,18]] and indices['short'].tolist()==[[0,19]] and indices['current'].tolist()==[list(range(20,29))]
    chunk=torch.arange(9.).view(1,1,9,1,1); state.append_chunk(chunk,0)
    assert state.global_ids()[-9:]==tuple(range(9))

def test_memory_rope_uses_saved_metadata_and_second_chunk_reads():
    class Rope:
        def forward_with_positions(self,t,y,x,device): return torch.stack((t,y,x,t,y,x),1)
    m=LongTermKVMemory(budget=8,pool=1); m.rope=Rope()
    x=torch.randn(1,2,4); r=torch.randn(1,2,7); m.capture(x,r,0,grid_shape=(1,1,2))
    rope=m.memory_rotary_emb(x.device,current_global_start=8)
    assert rope.shape[1]==2 and torch.equal(rope[0,:,0],torch.tensor([11.,11.]))
    older=m.memory_rotary_emb(x.device,current_global_start=16)
    assert torch.equal(older[0,:,0],torch.tensor([3.,3.]))
    class Attn:
        heads=2; to_k=torch.nn.Linear(4,4); to_v=torch.nn.Linear(4,4); norm_k=torch.nn.Identity()
    key=torch.randn(1,3,2,2); value=torch.randn(1,3,2,2)
    final_k,final_v,meta=m.append_native_attention(Attn(),key,value,None,lambda tensor,rotary:tensor,current_chunk=1,current_global_start=8)
    assert final_k.shape[1]==final_v.shape[1]==5 and meta['memory_tokens']==2

def test_memory_rope_recent_past_latent_is_position_18():
    memory=LongTermKVMemory(budget=8,pool=1)
    class Rope:
        def forward_with_positions(self,t,y,x,device): return torch.stack((t,y,x,t,y,x),1)
    memory.rope=Rope(); memory.capture(torch.randn(1,8,4),torch.randn(1,8,7),0,grid_shape=(8,1,1))
    rope=memory.memory_rotary_emb(torch.device('cpu'),8)
    assert torch.equal(rope[0,:,0],torch.tensor([11.,12.,13.,14.,15.,16.,17.,18.]))

def test_memory_append_extends_attention_mask_to_final_key_length():
    class Rope:
        def forward_with_positions(self,t,y,x,device): return torch.stack((t,y,x,t,y,x),1)
    memory=LongTermKVMemory(budget=4,pool=1); memory.rope=Rope(); memory.capture(torch.randn(1,1,8),torch.randn(1,1,7),0,grid_shape=(1,1,1))
    class A:
        heads=2; is_amplify_history=False; to_q=torch.nn.Linear(8,8); to_k=torch.nn.Linear(8,8); to_v=torch.nn.Linear(8,8); norm_q=torch.nn.Identity(); norm_k=torch.nn.Identity(); to_out=torch.nn.ModuleList([torch.nn.Identity(),torch.nn.Identity()])
    attn=A(); conditioner=SightlineConditioner(8)
    class Provider:
        context={'chunk_index':1}
        def __call__(self,h,**kw): return torch.zeros(h.shape[0],h.shape[1],7),torch.zeros(h.shape[0],h.shape[1],7)
    provider=Provider()
    def dispatch(q,k,v,**kwargs): assert kwargs['attn_mask'].shape[-1]==k.shape[1]; return v[:,:q.shape[1]]
    proc=SightlineHeliosAttnProcessor(conditioner,provider,memory=memory,qkv_projection=lambda a,h,e:(a.to_q(h),a.to_k(h),a.to_v(h)),rotary_apply=lambda x,r:x,attention_dispatch=dispatch)
    proc(attn,torch.randn(1,2,8),attention_mask=torch.zeros(1,1,1,2),rotary_emb=torch.zeros(1,2,4),original_context_length=2,current_chunk=1)

def test_checkpoint_restores_alpha_timestamp_and_lora(tmp_path):
    from long_video.training.sightline import SightlineTrainable, install_lora
    from long_video.sightline.memory import LayerKVMemoryBank
    from long_video.training.sightline_checkpoint import save_runtime_checkpoint, restore_runtime_checkpoint
    class Attn(torch.nn.Module):
        def __init__(self): super().__init__(); self.to_q=torch.nn.Linear(4,4); self.to_k=torch.nn.Linear(4,4); self.to_v=torch.nn.Linear(4,4); self.to_out=torch.nn.ModuleList([torch.nn.Linear(4,4),torch.nn.Identity()]); self.to_qkv=None; self.fused_projections=False
        def unfuse_projections(self): self.to_qkv=None; self.fused_projections=False
    class Block(torch.nn.Module):
        def __init__(self): super().__init__(); self.attn1=Attn()
    class Transformer(torch.nn.Module):
        def __init__(self): super().__init__(); self.transformer_blocks=torch.nn.ModuleList([Block()])
    config={'version':1}; memory_config={'layers':[0],'pool':2,'budget':8}; layers=(0,)
    trainable=SightlineTrainable(4,heads=1); memory=LayerKVMemoryBank((0,),8,2,hidden_dim=4); transformer=Transformer(); install_lora(transformer,[0])
    for alpha in trainable.conditioner.alpha_parameters(): alpha.data.fill_(.7)
    memory.timestamp.weight.data.fill_(.3)
    next(p for n,p in transformer.named_parameters() if 'lora_up' in n).data.fill_(.2)
    optimizer=torch.optim.AdamW(list(trainable.parameters())+list(memory.parameters())); scheduler=torch.optim.lr_scheduler.StepLR(optimizer,1); optimizer.step(); scheduler.step()
    torch.manual_seed(123); np.random.seed(456); path=tmp_path/'checkpoint.pt'; save_runtime_checkpoint(path,trainable,memory,transformer,optimizer,scheduler,12,config=config,helios_fingerprint='h',layers=layers,memory_config=memory_config); expected_random=torch.rand(1); expected_numpy=np.random.rand()
    target=SightlineTrainable(4,heads=1); target_memory=LayerKVMemoryBank((0,),8,2,hidden_dim=4); target_transformer=Transformer(); install_lora(target_transformer,[0])
    target_optimizer=torch.optim.AdamW(list(target.parameters())+list(target_memory.parameters())); target_scheduler=torch.optim.lr_scheduler.StepLR(target_optimizer,1)
    payload=torch.load(path)
    assert {'trainable','memory','lora','optimizer','scheduler','step','rng_torch','rng_python','rng_numpy','rng_cuda','rng_states'}.issubset(payload)
    step=restore_runtime_checkpoint(payload,target,target_memory,target_transformer,config=config,helios_fingerprint='h',layers=layers,memory_config=memory_config,optimizer=target_optimizer,scheduler=target_scheduler,restore_rng=True)
    assert step==12 and all(torch.allclose(alpha,torch.tensor(.7)) for alpha in target.conditioner.alpha_parameters()) and torch.allclose(target_memory.timestamp.weight,torch.full_like(target_memory.timestamp.weight,.3))
    assert torch.allclose(next(p for n,p in target_transformer.named_parameters() if 'lora_up' in n),torch.full_like(next(p for n,p in target_transformer.named_parameters() if 'lora_up' in n),.2))
    assert torch.equal(torch.rand(1),expected_random) and np.random.rand()==expected_numpy and target_scheduler.last_epoch==scheduler.last_epoch

def test_temporal_group_camera_shapes():
    c=torch.eye(4).repeat(2,33,1,1); k=torch.eye(3).repeat(2,33,1,1); cg,kg=temporal_group_cameras(c,k); assert cg.shape==(2,9,4,4) and kg.shape==(2,9,3,3)

def test_lora_fused_and_unfused_change_real_projection():
    from long_video.training.sightline import install_lora, LoRALinear
    class Attn(torch.nn.Module):
        def __init__(self, fused):
            super().__init__()
            self.to_q=torch.nn.Linear(4,4); self.to_k=torch.nn.Linear(4,4); self.to_v=torch.nn.Linear(4,4); self.fused_projections=fused
            self.to_qkv=torch.nn.Linear(4,12) if fused else None
            self.to_out=torch.nn.ModuleList([torch.nn.Linear(4,4),torch.nn.Identity()])
        def unfuse_projections(self): self.to_qkv=None; self.fused_projections=False
    class Block(torch.nn.Module):
        def __init__(self,fused): super().__init__(); self.attn1=Attn(fused)
    class T(torch.nn.Module):
        def __init__(self,fused): super().__init__(); self.transformer_blocks=torch.nn.ModuleList([Block(fused)])
    for fused in (False,True):
        model=T(fused); assert install_lora(model,[0])==(0,)
        target=model.transformer_blocks[0].attn1.to_q
        assert isinstance(target,LoRALinear)
        with torch.no_grad(): target.lora_up.weight.fill_(0.1)
        assert not torch.equal(target(torch.ones(1,4)),target.base(torch.ones(1,4)))

def test_current_and_memory_share_same_lora_kv_modules():
    from long_video.training.sightline import LoRALinear
    class Attn:
        heads=2; norm_k=torch.nn.Identity()
        to_k=LoRALinear(torch.nn.Linear(4,4)); to_v=LoRALinear(torch.nn.Linear(4,4))
    attn=Attn(); seen=[]
    attn.to_k.register_forward_hook(lambda module,args,out: seen.append(('k',id(module))))
    attn.to_v.register_forward_hook(lambda module,args,out: seen.append(('v',id(module))))
    attn.to_k(torch.randn(1,2,4)); attn.to_v(torch.randn(1,2,4))
    memory=LongTermKVMemory(4,1); memory.tokens=[]
    from long_video.sightline.memory import MemoryToken
    memory.tokens=[MemoryToken(torch.randn(1,1,4),torch.randn(1,1,7),0,0,0,0,0)]
    class Rope:
        def forward_with_positions(self,t,y,x,device): return torch.stack((t,y,x,t,y,x),1)
    memory.rope=Rope(); memory.append_native_attention(attn,torch.randn(1,1,2,2),torch.randn(1,1,2,2),None,lambda x,r:x,current_chunk=1,current_global_start=8)
    assert seen==[('k',id(attn.to_k)),('v',id(attn.to_v)),('k',id(attn.to_k)),('v',id(attn.to_v))]

def test_processor_qk_only_cpu_shape_and_v_unchanged():
    class A:
        heads=2; is_amplify_history=False; to_q=torch.nn.Linear(8,8); to_k=torch.nn.Linear(8,8); to_v=torch.nn.Linear(8,8); norm_q=torch.nn.Identity(); norm_k=torch.nn.Identity(); to_out=torch.nn.ModuleList([torch.nn.Identity(),torch.nn.Identity()])
    a=A(); c=SightlineConditioner(8); provider=lambda h,**kw:(torch.zeros(h.shape[0],h.shape[1],7),torch.zeros(h.shape[0],h.shape[1],7))
    def qkv(attn,h,e): return attn.to_q(h),attn.to_k(h),attn.to_v(h)
    def rope(x,r): return x
    h=torch.randn(1,4,8); expected_v=a.to_v(h).unflatten(2,(a.heads,-1)); observed=[]
    def dispatch(q,k,v,**kw): observed.append(v.detach().clone()); return torch.nn.functional.scaled_dot_product_attention(q.transpose(1,2),k.transpose(1,2),v.transpose(1,2)).transpose(1,2)
    proc=SightlineHeliosAttnProcessor(c,provider,qkv_projection=qkv,rotary_apply=rope,attention_dispatch=dispatch); out=proc(a,h,rotary_emb=torch.zeros(1,4,1,2)); assert out.shape==h.shape; assert torch.equal(observed[0],expected_v)
    assert not hasattr(proc,'last_value_native') and not hasattr(proc,'last_value')

def test_exact_flow_matching_three_stage_shapes_and_noise_scale():
    from types import SimpleNamespace
    from long_video.training.flow_matching_exact import exact_flow_matching_items
    class Scheduler:
        config={'stages':3,'num_train_timesteps':4}; start_sigmas=[1.,.8,.4]; end_sigmas=[.8,.4,0.]
        timesteps_per_stage=[torch.arange(4.)]*3; sigmas_per_stage=[torch.linspace(1,0,4)]*3
    pipe=SimpleNamespace(scheduler=Scheduler()); target=torch.randn(1,2,3,16,20); generator=torch.Generator().manual_seed(7)
    items=exact_flow_matching_items(pipe,target,generator=generator)
    assert [item['noisy_latents'].shape[-2:] for item in items]==[(4,5),(8,10),(16,20)]
    for item in items:
        assert item['noisy_latents'].shape==item['target'].shape==item['start_point'].shape==item['end_point'].shape==item['noise'].shape
        assert item['sigmas'].shape==(1,1,1,1,1)
    assert items[0]['noise'].std()>items[-1]['noise'].std()

def test_detached_autoregressive_chunk_has_no_gt_argument():
    from types import SimpleNamespace
    from scripts.train_sightline_dl3dv import _generate_detached_chunk
    calls=[]
    class Pipe:
        transformer=SimpleNamespace(dtype=torch.float32)
        def stage2_sample(self,**kwargs): calls.append(kwargs); return kwargs['latents']
    source=torch.zeros(1,2,1,4,4); history={name:(torch.zeros(1,2,n,4,4),torch.arange(n).view(1,-1)) for name,n in [('long',16),('mid',2),('short',2)]}
    result=_generate_detached_chunk(Pipe(),source,history,torch.zeros(1,2,3),SimpleNamespace(pyramid_steps=(2,2,2)),1)
    assert result.shape==(1,2,9,4,4) and 'target' not in calls[0] and calls[0]['latents'].requires_grad is False

def test_single_pose_canonicalization():
    from long_video.sightline.rays import canonicalize_c2w
    pose=torch.eye(4).repeat(2,1,1); pose[:,0,3]=torch.tensor([2.,3.])
    assert torch.allclose(canonicalize_c2w(pose),torch.eye(4).repeat(2,1,1))

def test_shared_boundary_correspondence_identity_is_not_deduplicated():
    from scripts.build_sightline_correspondences import correspondence_identity
    base={'query_chunk':0,'key_chunk':0,'query_latent_temporal':8,'key_latent_temporal':8,'query_y':1,'query_x':2,'key_y':3,'key_x':4}
    boundary=dict(base,query_chunk=1,query_latent_temporal=0)
    assert correspondence_identity(base)!=correspondence_identity(boundary)

def test_six_overlap_chunks_assemble_49_latents():
    from long_video.sightline.pipeline import SightlinePipeline
    accumulated=None
    for chunk in range(6):
        values=torch.arange(chunk*8,chunk*8+9).view(1,1,9,1,1)
        accumulated=SightlinePipeline.append_stride32_latents(accumulated,values)
    assert accumulated.shape[2]==49
    assert accumulated.flatten().tolist()==list(range(49))
    assert 1+(accumulated.shape[2]-1)*4==193

def test_overlap_chunk_cache_loads_all_six_and_drops_boundaries(tmp_path):
    from long_video.training.sightline_data import load_latent_tensor
    for chunk in range(6):
        value=torch.arange(chunk*8,chunk*8+9).view(1,9,1,1).float()
        torch.save({'latent':value},tmp_path/f'chunk_{chunk:02d}.pt')
    loaded=load_latent_tensor(tmp_path,schema='overlap_chunks_6x9')
    assert loaded.shape==(1,1,49,1,1) and loaded.flatten().tolist()==list(range(49))

def test_continuous_latent_cache_accepts_tchw_and_rejects_rgb_frames(tmp_path):
    from long_video.training.sightline_data import load_latent_tensor
    good=tmp_path/'good.pt'; torch.save({'video_latents':torch.zeros(49,16,2,3)},good)
    assert load_latent_tensor(good).shape==(1,16,49,2,3)
    bad=tmp_path/'bad.pt'; torch.save({'latents':torch.zeros(16,193,2,3)},bad)
    with pytest.raises(ValueError): load_latent_tensor(bad)

def test_rgbd_continuous_25_latent_cache(tmp_path):
    from long_video.training.sightline_data import load_latent_tensor, validate_latent_cache
    cache=tmp_path/'continuous_25.pt'; torch.save({'latents':torch.zeros(1,16,25,60,104)},cache)
    assert validate_latent_cache(cache)[0]=='continuous_25'
    assert load_latent_tensor(cache).shape==(1,16,25,60,104)

def test_lora_and_timestamp_follow_requested_dtype_device():
    from long_video.training.sightline import LoRALinear
    from long_video.sightline.memory import LayerKVMemoryBank
    base=torch.nn.Linear(4,4,dtype=torch.float64); lora=LoRALinear(base)
    assert lora.lora_down.weight.dtype==base.weight.dtype and lora.lora_down.weight.device==base.weight.device
    memory=LayerKVMemoryBank((0,),8,2,hidden_dim=4).to(dtype=torch.float64)
    assert memory.timestamp.weight.dtype==torch.float64

def test_runner_reset_clears_camera_memory_and_processor_capture():
    from types import SimpleNamespace
    from long_video.sightline.pipeline import SightlinePipeline
    class Processor: pass
    processor=Processor(); processor.last_q=processor.last_k=processor.last_hidden_states=processor.last_key_identities=torch.ones(1); processor.last_attention_meta={'x':1}; processor.last_current_length=1
    transformer=SimpleNamespace(_sightline_processors={0:processor}); helios=SimpleNamespace(transformer=transformer)
    config=SimpleNamespace(memory_layers=(0,),memory_budget=8,memory_pool=2,pyramid_steps=(2,2,2))
    runner=SightlinePipeline(helios,config=config); runner._active_chunk=4; runner.memory.banks[0].tokens.append(object()); runner.camera_history._items[1]=object()
    runner.reset_sequence()
    assert runner._active_chunk==0 and not runner.memory.banks[0].tokens and not runner.camera_history.indices()
    assert processor.last_q is None and processor.last_key_identities is None and processor.last_attention_meta=={}

@pytest.mark.skipif(not os.environ.get('HELIOS_ROOT'),reason='pinned Helios source is not configured')
def test_native_helios_attention_equivalence_alpha_zero_cpu():
    root=os.environ['HELIOS_ROOT']; sys.path.insert(0,root)
    import helios.diffusers_version.transformer_helios_diffusers as native
    torch.manual_seed(4); attention=native.HeliosAttention(dim=8,heads=2,dim_head=4,is_cross_attention=False,is_amplify_history=False).float(); hidden=torch.randn(1,5,8)
    expected=native.HeliosAttnProcessor()(attention,hidden,original_context_length=5)
    conditioner=SightlineConditioner(8).float(); conditioner.alpha_q.data.zero_(); conditioner.alpha_k.data.zero_()
    provider=lambda states,**kwargs:(torch.zeros(states.shape[0],states.shape[1],7),torch.zeros(states.shape[0],kwargs['key_length'],7))
    processor=SightlineHeliosAttnProcessor(conditioner,provider,qkv_projection=native._get_qkv_projections,rotary_apply=native.apply_rotary_emb_transposed,attention_dispatch=native.dispatch_attention_fn,attention_backend=native.HeliosAttnProcessor._attention_backend,parallel_config=native.HeliosAttnProcessor._parallel_config)
    actual=processor(attention,hidden,original_context_length=5)
    assert torch.allclose(actual,expected,rtol=1e-6,atol=1e-7)

def test_selected_q_correspondence_never_allocates_full_query_axis():
    from long_video.training.sightline import selected_qk_logits
    query=torch.randn(1,100,2,4); key=torch.randn(1,70,2,4)
    logits=selected_qk_logits(query,key,[3,91])
    assert logits.shape==(1,2,2,70)
    reference=torch.einsum('bqhd,bkhd->bhqk',query[:,[3,91]],key)/2
    assert torch.allclose(logits,reference)

def test_three_stage_current_and_native_history_ray_counts_match():
    provider=SightlineRayProvider(source_height=32,source_width=32)
    cameras=torch.eye(4).view(1,1,4,4).expand(1,9,-1,-1); K=torch.eye(3).view(1,1,3,3).expand(1,9,-1,-1)
    history_shapes={'long':(4,1,1),'mid':(1,2,2),'short':(2,4,4)}; history_count=4+4+32
    history=torch.randn(1,history_count,7); stages=((9,1,1),(9,2,2),(9,4,4))
    provider.set_context(chunk_index=0,c2w=cameras,intrinsics=K,history_rays=history,stage_shapes=stages,token_shape=stages[0],history_token_shapes=history_shapes)
    for shape in stages:
        current=shape[0]*shape[1]*shape[2]; q,k=provider(torch.zeros(1,history_count+current,8),key_length=history_count+current,current_length=current)
        assert q.shape[1]==k.shape[1]==history_count+current

def test_key_identity_map_contains_native_current_and_memory():
    from long_video.sightline.memory import LongTermKVMemory,MemoryToken
    provider=SightlineRayProvider(source_height=8,source_width=8)
    cameras=torch.eye(4).view(1,1,4,4).expand(1,9,-1,-1); K=torch.eye(3).view(1,1,3,3).expand(1,9,-1,-1)
    shapes={'long':(1,1,1),'mid':(1,1,1),'short':(1,1,1)}; coverage={'long':((1,2,3,4),),'mid':((5,6),),'short':((7,),)}
    provider.set_context(chunk_index=1,c2w=cameras,intrinsics=K,history_token_shapes=shapes,history_global_coverages=coverage,stage_shapes=((9,1,1),),token_shape=(9,1,1))
    memory=LongTermKVMemory(); memory.tokens=[MemoryToken(torch.zeros(1,1,4),torch.zeros(1,1,7),0,0,4,2,3)]
    identities=provider.key_identities(9,memory)
    assert identities[:3]==(('native',(1,2,3,4),0,0,'long'),('native',(5,6),0,0,'mid'),('native',(7,),0,0,'short'))
    assert ('current',(8,),0,0,'current') in identities and identities[-1]==('memory',(4,),2,3,'memory')

def test_p3_corr_loss_selected_queries_forward_backward():
    from types import SimpleNamespace
    from scripts.train_sightline_dl3dv import _corr_loss
    from long_video.training.sightline import SightlineTrainable
    q=torch.randn(1,2,2,2,requires_grad=True); k=torch.randn(1,2,2,2,requires_grad=True)
    context={'stage_shapes':((1,1,1),)}; provider=SimpleNamespace(context=context)
    processor=SimpleNamespace(last_q=q,last_k=k,last_current_length=1,ray_provider=provider,last_key_identities=(('native',(0,),0,0,'short'),('current',(8,),0,0,'current')))
    rows=[{'query_chunk':1,'query_latent_temporal':0,'query_y':0,'query_x':0,'key_chunk':0,'key_latent_temporal':0,'key_y':0,'key_x':0,'weight':.7}]
    trainable=SightlineTrainable(4,heads=2); loss=_corr_loss(trainable,{3:processor},rows,1,(3,),8); loss.backward()
    assert loss.ndim==0 and q.grad is not None and k.grad is not None

def test_memory_shared_boundaries_are_unique_and_future_filtered():
    memory=LongTermKVMemory(budget=100,pool=1); hidden=torch.randn(1,9,4); rays=torch.randn(1,9,7)
    memory.capture(hidden,rays,0,grid_shape=(9,1,1)); memory.capture(hidden,rays,1,grid_shape=(9,1,1))
    global_ids=[token.chunk_index*8+token.temporal for token in memory.tokens]
    assert len(global_ids)==17 and len(set(global_ids))==17 and global_ids.count(8)==1
    active=memory.active_tokens(16); assert [token.chunk_index*8+token.temporal for token in active]==list(range(16))

def test_scale_augmentation_delta_is_reused_for_q_k_and_memory():
    torch.manual_seed(2); conditioner=SightlineConditioner(8,scale_aug_prob=1.0); rays=torch.randn(1,3,7); delta=conditioner.sample_scale_delta(rays,True)
    calls=[]
    original=conditioner.project
    def recorded(value,**kwargs): calls.append(kwargs['scale_delta']); return original(value,**kwargs)
    conditioner.project=recorded
    conditioner(rays,rays,training=True,scale_delta=delta); conditioner.project(rays,kind='k',training=True,scale_delta=delta)
    assert len(calls)==3 and all(call is delta for call in calls)

def test_steps_override_is_effective_and_symmetric():
    from long_video.sightline.pipeline import SightlinePipeline
    assert SightlinePipeline.resolve_pyramid_steps((2,2,2),None)==(2,2,2)
    assert SightlinePipeline.resolve_pyramid_steps((2,2,2),4)==(4,4,4)

def test_training_preflight_and_fixed_2500_warmup_schedule():
    from types import SimpleNamespace
    from scripts.train_sightline_dl3dv import _preflight,_lr_multiplier
    cfg=SimpleNamespace(sightline_layers=tuple(range(25)),camera_layers=(1,2,3,4,5,6),correspondence_layers=(16,20,24),memory_layers=(16,20,24),lora_layers=(),warmup_ratio=.04)
    with pytest.raises(ValueError,match='P2'): _preflight(cfg,SimpleNamespace(train=True,max_steps=401),())
    cfg.lora_layers=(0,)
    cfg.memory_layers=(); cfg.correspondence_layers=()
    with pytest.raises(ValueError,match='Memory/correspondence'): _preflight(cfg,SimpleNamespace(train=True,max_steps=1001),())
    assert _lr_multiplier(0)==pytest.approx(.01) and _lr_multiplier(99)==pytest.approx(1.) and _lr_multiplier(100)==pytest.approx(1.) and _lr_multiplier(2499)<1e-5

def test_p3_chunk_curriculum_has_fixed_global_step_boundaries():
    from long_video.training.sightline import curriculum_phase
    assert [curriculum_phase(step)['max_chunks'] for step in (1000, 1399, 1400, 1699, 1700, 1899, 1900, 2099, 2100, 2299, 2300, 2499)] == [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6]

def test_teacher_overlap_screening_is_deterministic_and_causal():
    from scripts.build_sightline_correspondences import screen_overlap
    xyz=torch.tensor([[[[0.,0.,1.],[1.,0.,1.]],[[0.,1.,1.],[1.,1.,1.]]]]).numpy()
    valid=torch.ones((1,2,2),dtype=torch.bool).numpy(); confidence=torch.ones((1,2,2)).numpy()
    first=screen_overlap(xyz,valid,confidence,0,0,screening_stride=1,screening_distance_threshold=.01)
    second=screen_overlap(xyz,valid,confidence,0,0,screening_stride=1,screening_distance_threshold=.01)
    assert first==second and first['accepted']

def test_teacher_token_vote_keeps_multi_positive_and_no_membership_double_count():
    from scripts.build_sightline_correspondences import token_vote_rows
    rows=[{'query_frame':32,'key_frame':0,'query_pixel':0,'query_y':0,'query_x':0,'key_y':0,'key_x':0,'weight':.8},
          {'query_frame':32,'key_frame':0,'query_pixel':0,'query_y':0,'query_x':0,'key_y':0,'key_x':1,'weight':.75},
          {'query_frame':32,'key_frame':0,'query_pixel':1,'query_y':0,'query_x':0,'key_y':0,'key_x':1,'weight':.8}]
    out=token_vote_rows(rows,token_height=1,token_width=2,near_top_ratio=.5,total_frames=193)
    assert len(out)==4 and {row['key_x'] for row in out}=={0,1} and all(row['matched_count']==2 for row in out)

def test_odd_grid_alignment_covers_detached_rollout_scheduler():
    from types import SimpleNamespace
    from scripts.train_sightline_rgbd import _install_odd_grid_scheduler_alignment
    class Scheduler:
        def convert_flow_pred_to_x0(self,flow_pred,xt,timestep,sigmas,timesteps):
            return xt-flow_pred
    pipe=SimpleNamespace(scheduler=Scheduler())
    _install_odd_grid_scheduler_alignment(pipe)
    flow=torch.randn(1,2,9,14,26); latents=torch.randn(1,2,9,15,26)
    output=pipe.scheduler.convert_flow_pred_to_x0(flow,latents,None,None,None)
    assert output.shape==latents.shape
