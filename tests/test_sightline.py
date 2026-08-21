import pytest
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
    m.alpha.data.zero_(); q,k=m(r,training=True); assert torch.count_nonzero(q)==0 and torch.count_nonzero(k)==0

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
    from long_video.training.sightline import chunk_grad_policy, assert_single_backward_chunk, causal_chunk_plan, run_single_graph_chunks
    train_chunk=2; policies=[chunk_grad_policy(i,train_chunk) for i in range(6)]
    assert policies[:2]==["forward_detached"]*2 and policies[2]=="backward"
    assert policies[3:]==["rollout_detached"]*3
    assert_single_backward_chunk(policies,train_chunk)
    assert causal_chunk_plan(6,2)==tuple(policies)
    weight=torch.nn.Parameter(torch.tensor(2.)); seen=[]
    outputs,_=run_single_graph_chunks(4,2,lambda chunk,grad: seen.append((chunk,grad)) or weight*(chunk+1))
    assert [out.requires_grad for out in outputs]==[False,False,True,False]

def test_memory_is_kv_only_and_eviction():
    m=LongTermKVMemory(budget=2,pool=1); x=torch.randn(1,4,4); r=torch.randn(1,4,7); m.capture(x,r,0,grid_shape=(1,2,2)); assert len(m)==2; k,v=m.get(); assert k.shape[-1]==4 and v.shape[-1]==7

def test_native_history_order_matches_camera_slots():
    from long_video.training.sightline import native_history_16_2_1
    source=torch.zeros(1,1,1,1,1); latents=[torch.full_like(source,float(i)) for i in range(1,21)]
    packed=native_history_16_2_1(latents,list(range(1,21)),source)
    assert packed['long'][1].tolist()==[list(range(2,18))]
    assert packed['mid'][1].tolist()==[[18,19]] and packed['short'][1].tolist()==[[0,20]]

def test_memory_rope_uses_saved_metadata_and_second_chunk_reads():
    class Rope:
        def forward_with_positions(self,t,y,x,device): return torch.stack((t,y,x,t,y,x),1)
    m=LongTermKVMemory(budget=8,pool=1); m.rope=Rope()
    x=torch.randn(1,2,4); r=torch.randn(1,2,7); m.capture(x,r,1,grid_shape=(1,1,2))
    rope=m.memory_rotary_emb(x.device)
    assert rope.shape[1]==2 and torch.equal(rope[0,:,0],torch.tensor([8.,8.]))
    class Attn:
        heads=2; to_k=torch.nn.Linear(4,4); to_v=torch.nn.Linear(4,4); norm_k=torch.nn.Identity()
    key=torch.randn(1,3,2,2); value=torch.randn(1,3,2,2)
    final_k,final_v,meta=m.append_native_attention(Attn(),key,value,None,lambda tensor,rotary:tensor,current_chunk=2)
    assert final_k.shape[1]==final_v.shape[1]==5 and meta['memory_tokens']==2

def test_memory_append_extends_attention_mask_to_final_key_length():
    class Rope:
        def forward_with_positions(self,t,y,x,device): return torch.stack((t,y,x,t,y,x),1)
    memory=LongTermKVMemory(budget=4,pool=1); memory.rope=Rope(); memory.capture(torch.randn(1,1,8),torch.randn(1,1,7),0,grid_shape=(1,1,1))
    class A:
        heads=2; is_amplify_history=False; to_q=torch.nn.Linear(8,8); to_k=torch.nn.Linear(8,8); to_v=torch.nn.Linear(8,8); norm_q=torch.nn.Identity(); norm_k=torch.nn.Identity(); to_out=torch.nn.ModuleList([torch.nn.Identity(),torch.nn.Identity()])
    attn=A(); conditioner=SightlineConditioner(8); provider=lambda h,**kw:(torch.zeros(h.shape[0],h.shape[1],7),torch.zeros(h.shape[0],h.shape[1],7))
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
    trainable.conditioner.alpha.data.fill_(.7); memory.timestamp.weight.data.fill_(.3)
    next(p for n,p in transformer.named_parameters() if 'lora_up' in n).data.fill_(.2)
    optimizer=torch.optim.AdamW(list(trainable.parameters())+list(memory.parameters())); scheduler=torch.optim.lr_scheduler.StepLR(optimizer,1); optimizer.step(); scheduler.step()
    torch.manual_seed(123); path=tmp_path/'checkpoint.pt'; save_runtime_checkpoint(path,trainable,memory,transformer,optimizer,scheduler,12,config=config,helios_fingerprint='h',layers=layers,memory_config=memory_config); expected_random=torch.rand(1)
    target=SightlineTrainable(4,heads=1); target_memory=LayerKVMemoryBank((0,),8,2,hidden_dim=4); target_transformer=Transformer(); install_lora(target_transformer,[0])
    target_optimizer=torch.optim.AdamW(list(target.parameters())+list(target_memory.parameters())); target_scheduler=torch.optim.lr_scheduler.StepLR(target_optimizer,1)
    step=restore_runtime_checkpoint(torch.load(path),target,target_memory,target_transformer,config=config,helios_fingerprint='h',layers=layers,memory_config=memory_config,optimizer=target_optimizer,scheduler=target_scheduler,restore_rng=True)
    assert step==12 and torch.allclose(target.conditioner.alpha,torch.tensor(.7)) and torch.allclose(target_memory.timestamp.weight,torch.full_like(target_memory.timestamp.weight,.3))
    assert torch.allclose(next(p for n,p in target_transformer.named_parameters() if 'lora_up' in n),torch.full_like(next(p for n,p in target_transformer.named_parameters() if 'lora_up' in n),.2))
    assert torch.equal(torch.rand(1),expected_random) and target_scheduler.last_epoch==scheduler.last_epoch

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
    memory.rope=Rope(); memory.append_native_attention(attn,torch.randn(1,1,2,2),torch.randn(1,1,2,2),None,lambda x,r:x)
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
