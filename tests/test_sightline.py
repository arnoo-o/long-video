import pytest
import numpy as np
import os, sys
from pathlib import Path
torch=pytest.importorskip('torch')
from long_video.sightline.rays import plucker_rays, temporal_group_cameras
from long_video.sightline.conditioning import SightlineConditioner
from long_video.sightline.history import HistoryManager, CameraHistoryState
from long_video.sightline.memory import LongTermKVMemory
from long_video.sightline.rays import temporal_group_cameras
from long_video.sightline.helios_integration import SightlineHeliosAttnProcessor, SightlineRayProvider

def _camera_trajectory(x=0.,yaw=0.,frames=9):
    poses=torch.eye(4).view(1,1,4,4).repeat(1,frames,1,1); poses[:,:,0,3]=x
    c,s=np.cos(yaw),np.sin(yaw); poses[:,:,0,0]=c; poses[:,:,0,2]=s; poses[:,:,2,0]=-s; poses[:,:,2,2]=c
    return poses

def test_plucker_ray_geometry():
    K=torch.tensor([[[100.,0,2],[0,100,2],[0,0,1]]]); c=torch.eye(4).unsqueeze(0)
    r=plucker_rays(c,K,4,4,source_height=4,source_width=4); assert r.shape==(1,1,4,4,7)
    assert torch.allclose(r[0,0,2,2,:3],torch.tensor([0.,0.,1.]),atol=1e-5)

def test_scale_augmentation_gate_only_and_zero_alpha():
    torch.manual_seed(1); m=SightlineConditioner(16); r=torch.randn(2,3,7); q,k=m(r,training=False); assert q.shape==k.shape==(2,3,16)
    m.alpha_q.data.zero_(); m.alpha_k.data.zero_(); q,k=m(r,training=True); assert torch.count_nonzero(q)==0 and torch.count_nonzero(k)==0

def test_conditioner_numeric_capture_is_opt_in_and_value_preserving():
    torch.manual_seed(7); module=SightlineConditioner(16).eval(); rays=torch.randn(2,3,7)
    module.q_proj.weight.data.normal_(std=.01)
    expected=module.project(rays,kind='q',training=False)
    assert module.last_pre_norm_rms['q'] is None
    module.capture_numeric_diagnostics=True
    actual=module.project(rays,kind='q',training=False)
    assert torch.equal(expected,actual)
    assert module.last_pre_norm_rms['q'] > 0

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
    from long_video.training.sightline import chunk_grad_policy, assert_single_backward_chunk, causal_chunk_plan, run_single_graph_chunks, run_causal_prefix_chunks, prefix_chunk_should_capture_memory, correspondence_capture_for_stage
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
    assert [chunk for chunk in range(2) if prefix_chunk_should_capture_memory(chunk,1)]==[0]
    assert [chunk for chunk in range(5) if prefix_chunk_should_capture_memory(chunk,4)]==[0,1,2,3]
    assert [correspondence_capture_for_stage(stage,3,True) for stage in range(3)]==[False,False,True]

def test_camera_first_curriculum_uses_fixed_unit_origin():
    from long_video.training.sightline import curriculum_phase
    assert curriculum_phase(0)=={'name':'P1','max_chunks':1,'lora':False,'correspondence':False,'memory':False}
    assert curriculum_phase(299)['max_chunks']==1
    assert not curriculum_phase(300)['lora'] and curriculum_phase(300)['max_chunks']==2
    assert curriculum_phase(400)['lora'] and curriculum_phase(400)['max_chunks']==2
    assert curriculum_phase(999)['max_chunks']==2
    assert curriculum_phase(1000)['memory'] and curriculum_phase(1000)['correspondence'] and curriculum_phase(1000)['max_chunks']==2
    assert curriculum_phase(2499)['max_chunks']==6
    source=Path(__file__).parents[1].joinpath('scripts/train_sightline_rgbd.py').read_text()
    assert 'window_start=0' in source and 'select_chunk_window' not in source

def test_memory_is_chunk_atomic_and_never_token_truncated():
    m=LongTermKVMemory(budget=4,pool=1); x=torch.randn(1,4,4); r=torch.randn(1,4,7); m.capture(x,r,0,grid_shape=(2,1,2),camera_poses=_camera_trajectory())
    assert len(m)==2 and len(m.archive[0].tokens)==2
    k,v=m.get(8,query_camera_poses=_camera_trajectory(1.)); assert k.shape[-1]==4 and v.shape[-1]==7
    m.capture(x,r,1,grid_shape=(2,1,2),camera_poses=_camera_trajectory(1.))
    m.budget=3
    with pytest.raises(RuntimeError,match='complete active Memory chunks'):
        m.active_tokens(16,query_camera_poses=_camera_trajectory(2.))
    assert [len(m.archive[index].tokens) for index in (0,1)]==[2,2]

def test_memory_chunk_block_storage_matches_tokenwise_values():
    hidden=torch.arange(1*9*4,dtype=torch.float32).reshape(1,9,4)
    rays=torch.arange(1*9*7,dtype=torch.float32).reshape(1,9,7)
    memory=LongTermKVMemory(budget=32,pool=1); memory.capture(hidden,rays,0,grid_shape=(9,1,1),camera_poses=_camera_trajectory())
    chunk=memory.archive[0]
    assert chunk.hidden.is_contiguous() and chunk.rays.is_contiguous()
    expected_rays=rays[:,1:].clone(); expected_rays[...,:3]/=expected_rays[...,:3].norm(dim=-1,keepdim=True).clamp_min(1e-6); expected_rays[...,3:6]/=expected_rays[...,3:6].norm(dim=-1,keepdim=True).clamp_min(1e-6)
    assert torch.equal(chunk.hidden,hidden[:,1:].cpu()) and torch.equal(chunk.rays,expected_rays.cpu())
    memory.prepare_active_memory(query_chunk=4,query_camera_poses=_camera_trajectory(4.),native_history_chunk_ids=(),device='cpu',dtype=torch.float32)
    active_hidden,active_rays=memory.get(32,query_camera_poses=_camera_trajectory(4.),device='cpu',dtype=torch.float32)
    assert torch.equal(active_hidden,hidden[:,1:]) and torch.equal(active_rays,expected_rays)

def test_selected_layers_have_independent_qk_geometry_and_alphas():
    from long_video.training.sightline import SightlineTrainable
    trainable=SightlineTrainable(8,layers=(16,20,24),heads=2)
    conditioners=[trainable.conditioner.for_layer(layer) for layer in (16,20,24)]
    for name in ('q_proj','k_proj','gate','rms_norm_q','rms_norm_k'):
        assert len({id(getattr(layer,name).weight) for layer in conditioners})==3
    assert all(float(layer.alpha_q.detach())==1.0 and float(layer.alpha_k.detach())==1.0 for layer in conditioners)
    assert len([name for name,_ in trainable.named_parameters() if name.endswith(('alpha_q','alpha_k'))])==6

def test_streaming_correspondence_matches_dense_loss_and_gradients():
    from long_video.training.sightline import SightlineTrainable
    torch.manual_seed(91)
    model=SightlineTrainable(8,layers=(0,),heads=2)
    q=torch.randn(1,17,2,4,requires_grad=True); k=torch.randn(1,23,2,4,requires_grad=True)
    indices=(1,7,12); positives=((0,(1,4)),(1,(2,9)),(2,(0,22))); weights=torch.tensor([1.,2.,3.])
    selected=q[:,indices]; logits=torch.einsum('bqhd,bkhd->bhqk',selected,k)*0.5
    z=torch.logsumexp(logits.log_softmax(-1),1)-torch.log(torch.tensor(2.))
    dense=-(torch.stack([torch.logsumexp(z[:,i,list(keys)],-1).mean() for i,keys in enumerate((p for _,p in positives))])*weights).sum()/weights.sum()
    streamed=model.correspondence_streaming(q,k,query_indices=indices,multi_positive=positives,weights=weights,key_block=5)
    dq_dense,dk_dense=torch.autograd.grad(dense,(q,k),retain_graph=True)
    dq_stream,dk_stream=torch.autograd.grad(streamed,(q,k))
    assert torch.allclose(dense,streamed,atol=2e-5,rtol=2e-5)
    assert torch.allclose(dq_dense,dq_stream,atol=2e-5,rtol=2e-5)
    assert torch.allclose(dk_dense,dk_stream,atol=2e-5,rtol=2e-5)

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
    x=torch.randn(1,4,4); r=torch.randn(1,4,7); m.capture(x,r,0,grid_shape=(2,1,2),camera_poses=_camera_trajectory())
    tokens=m.active_tokens(8,query_camera_poses=_camera_trajectory(1.)); rope=m.memory_rotary_emb(tokens,x.device,current_global_start=8)
    assert rope.shape[1]==2 and torch.equal(rope[0,:,0],torch.tensor([12.,12.]))
    older=m.memory_rotary_emb(tokens,x.device,current_global_start=16)
    assert torch.equal(older[0,:,0],torch.tensor([4.,4.]))
    class Attn:
        heads=2; to_k=torch.nn.Linear(4,4); to_v=torch.nn.Linear(4,4); norm_k=torch.nn.Identity()
    key=torch.randn(1,3,2,2); value=torch.randn(1,3,2,2)
    final_k,final_v,meta=m.append_native_attention(Attn(),key,value,None,lambda tensor,rotary:tensor,current_chunk=1,current_global_start=8,query_camera_poses=_camera_trajectory(1.))
    assert final_k.shape[1]==final_v.shape[1]==5 and meta['memory_tokens']==2

def test_memory_rope_recent_past_latent_is_position_18():
    memory=LongTermKVMemory(budget=8,pool=1)
    class Rope:
        def forward_with_positions(self,t,y,x,device): return torch.stack((t,y,x,t,y,x),1)
    memory.rope=Rope(); memory.capture(torch.randn(1,9,4),torch.randn(1,9,7),0,grid_shape=(9,1,1),camera_poses=_camera_trajectory())
    tokens=memory.active_tokens(8,query_camera_poses=_camera_trajectory(1.)); rope=memory.memory_rotary_emb(tokens,torch.device('cpu'),8)
    assert torch.equal(rope[0,:,0],torch.tensor([12.,13.,14.,15.,16.,17.,18.,18.]))

def test_memory_type_embedding_changes_only_long_term_k():
    from long_video.sightline.memory import LayerKVMemoryBank
    class Rope:
        def forward_with_positions(self,t,y,x,device): return torch.zeros(1,6,t.shape[1],device=device)
    class Attn:
        heads=2; to_k=torch.nn.Identity(); to_v=torch.nn.Identity(); norm_k=torch.nn.Identity()
    bank=LayerKVMemoryBank((16,),budget=8,pool=1,hidden_dim=4); bank.bind_rope(Rope()); memory=bank.for_layer(16)
    memory.capture(torch.arange(8,dtype=torch.float32).reshape(1,2,4),torch.randn(1,2,7),0,grid_shape=(2,1,1),camera_poses=_camera_trajectory())
    memory.prepare_active_memory(query_chunk=1,query_camera_poses=_camera_trajectory(1.),device='cpu',dtype=torch.float32)
    key=torch.zeros(1,1,2,2); value=torch.zeros_like(key)
    zero_k,zero_v,_=memory.append_native_attention(Attn(),key,value,None,lambda tensor,rotary:tensor,current_chunk=1,current_global_start=8)
    bank.memory_type_embedding.data.copy_(torch.tensor([1.,2.,3.,4.]))
    typed_k,typed_v,_=memory.append_native_attention(Attn(),key,value,None,lambda tensor,rotary:tensor,current_chunk=1,current_global_start=8)
    assert torch.equal(typed_v,zero_v) and torch.equal(typed_k[:,:1],zero_k[:,:1])
    assert torch.allclose(typed_k[:,1:]-zero_k[:,1:],torch.tensor([[[[1.,2.],[3.,4.]]]]))

def test_memory_append_extends_attention_mask_to_final_key_length():
    class Rope:
        def forward_with_positions(self,t,y,x,device): return torch.stack((t,y,x,t,y,x),1)
    memory=LongTermKVMemory(budget=4,pool=1); memory.rope=Rope(); memory.capture(torch.randn(1,2,8),torch.randn(1,2,7),0,grid_shape=(2,1,1),camera_poses=_camera_trajectory())
    class A:
        heads=2; is_amplify_history=False; to_q=torch.nn.Linear(8,8); to_k=torch.nn.Linear(8,8); to_v=torch.nn.Linear(8,8); norm_q=torch.nn.Identity(); norm_k=torch.nn.Identity(); to_out=torch.nn.ModuleList([torch.nn.Identity(),torch.nn.Identity()])
    attn=A(); conditioner=SightlineConditioner(8)
    class Provider:
        context={'chunk_index':1,'c2w':_camera_trajectory(1.),'history_global_coverages':{}}
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
    memory.memory_type_embedding.data.fill_(.4)
    next(p for n,p in transformer.named_parameters() if 'lora_up' in n).data.fill_(.2)
    optimizer=torch.optim.AdamW(list(trainable.parameters())+list(memory.parameters())); scheduler=torch.optim.lr_scheduler.StepLR(optimizer,1); optimizer.step(); scheduler.step()
    torch.manual_seed(123); np.random.seed(456); path=tmp_path/'checkpoint.pt'; save_runtime_checkpoint(path,trainable,memory,transformer,optimizer,scheduler,12,config=config,helios_fingerprint='h',layers=layers,memory_config=memory_config); expected_random=torch.rand(1); expected_numpy=np.random.rand()
    target=SightlineTrainable(4,heads=1); target_memory=LayerKVMemoryBank((0,),8,2,hidden_dim=4); target_transformer=Transformer(); install_lora(target_transformer,[0])
    target_optimizer=torch.optim.AdamW(list(target.parameters())+list(target_memory.parameters())); target_scheduler=torch.optim.lr_scheduler.StepLR(target_optimizer,1)
    payload=torch.load(path)
    assert {'trainable','memory','lora','optimizer','scheduler','step','rng_torch','rng_python','rng_numpy','rng_cuda','rng_states','rng_world_size'}.issubset(payload)
    assert payload['rng_world_size']==1 and len(payload['rng_states'])==1 and payload['rng_states'][0].dtype==torch.uint8
    step=restore_runtime_checkpoint(payload,target,target_memory,target_transformer,config=config,helios_fingerprint='h',layers=layers,memory_config=memory_config,optimizer=target_optimizer,scheduler=target_scheduler,restore_rng=True)
    assert step==12 and all(torch.allclose(alpha,torch.tensor(.7)) for alpha in target.conditioner.alpha_parameters()) and torch.allclose(target_memory.timestamp.weight,torch.full_like(target_memory.timestamp.weight,.3)) and torch.allclose(target_memory.memory_type_embedding,torch.full_like(target_memory.memory_type_embedding,.4))
    assert torch.allclose(next(p for n,p in target_transformer.named_parameters() if 'lora_up' in n),torch.full_like(next(p for n,p in target_transformer.named_parameters() if 'lora_up' in n),.2))
    assert torch.equal(torch.rand(1),expected_random) and np.random.rand()==expected_numpy and target_scheduler.last_epoch==scheduler.last_epoch
    inference_target=SightlineTrainable(4,heads=1); inference_memory=LayerKVMemoryBank((0,),8,2,hidden_dim=4); inference_transformer=Transformer(); install_lora(inference_transformer,[0])
    restore_runtime_checkpoint(payload,inference_target,inference_memory,inference_transformer,config=config,helios_fingerprint='h',layers=layers,memory_config=memory_config,restore_rng=False)
    assert torch.equal(inference_memory.timestamp.weight,memory.timestamp.weight) and torch.equal(inference_memory.memory_type_embedding,memory.memory_type_embedding)
    legacy_payload={key:value for key,value in payload.items()}; legacy_payload['memory']=dict(payload['memory']); legacy_payload['memory'].pop('memory_type_embedding')
    legacy_memory=LayerKVMemoryBank((0,),8,2,hidden_dim=4); legacy_target=SightlineTrainable(4,heads=1); legacy_transformer=Transformer(); install_lora(legacy_transformer,[0])
    legacy_optimizer=torch.optim.AdamW(list(legacy_target.parameters())+list(legacy_memory.parameters()))
    restore_runtime_checkpoint(legacy_payload,legacy_target,legacy_memory,legacy_transformer,config=config,helios_fingerprint='h',layers=layers,memory_config=memory_config,optimizer=legacy_optimizer,restore_rng=False)
    assert torch.count_nonzero(legacy_memory.memory_type_embedding)==0
    with pytest.raises(RuntimeError,match='world size'):
        restore_runtime_checkpoint(payload,inference_target,inference_memory,inference_transformer,config=config,helios_fingerprint='h',layers=layers,memory_config=memory_config,restore_rng=True,rank=0,world_size=2)
    migrated_config={'version':1,'ddp_world_size':4}; payload['config']={'version':1,'ddp_world_size':3}
    from long_video.training.sightline_checkpoint import config_fingerprint
    payload['config_fingerprint']=config_fingerprint(payload['config'])
    with pytest.raises(RuntimeError,match='config mismatch'):
        restore_runtime_checkpoint(payload,inference_target,inference_memory,inference_transformer,config=migrated_config,helios_fingerprint='h',layers=layers,memory_config=memory_config,restore_rng=False)
    assert restore_runtime_checkpoint(payload,inference_target,inference_memory,inference_transformer,config=migrated_config,helios_fingerprint='h',layers=layers,memory_config=memory_config,restore_rng=True,rank=1,world_size=4,allow_world_size_migration=True)==12
    inference_source=Path(__file__).parents[1].joinpath('scripts/infer_sightline.py').read_text()
    assert 'timestamp.weight.data.zero_' not in inference_source

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
    memory=LongTermKVMemory(4,1)
    memory.capture(torch.randn(1,2,4),torch.randn(1,2,7),0,grid_shape=(2,1,1),camera_poses=_camera_trajectory())
    class Rope:
        def forward_with_positions(self,t,y,x,device): return torch.stack((t,y,x,t,y,x),1)
    memory.rope=Rope(); memory.append_native_attention(attn,torch.randn(1,1,2,2),torch.randn(1,1,2,2),None,lambda x,r:x,current_chunk=1,current_global_start=8,query_camera_poses=_camera_trajectory(1.))
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
    runner=SightlinePipeline(helios,config=config); runner._active_chunk=4; runner.memory.banks[0].archive[0]=object(); runner.camera_history._items[1]=object()
    runner.reset_sequence()
    assert runner._active_chunk==0 and not runner.memory.banks[0].archive and not runner.camera_history.indices()
    assert processor.last_q is None and processor.last_key_identities is None and processor.last_attention_meta=={}

@pytest.mark.skipif(not os.environ.get('HELIOS_ROOT'),reason='pinned Helios source is not configured')
def test_native_helios_attention_equivalence_alpha_zero_cpu():
    root=os.environ['HELIOS_ROOT']; sys.path.insert(0,root)
    import helios.diffusers_version.transformer_helios_diffusers as native
    torch.manual_seed(4); attention=native.HeliosAttention(dim=8,heads=2,dim_head=4,is_cross_attention=False,is_amplify_history=False).float(); hidden=torch.randn(1,5,8)
    expected=native.HeliosAttnProcessor()(attention,hidden,original_context_length=5)
    conditioner=SightlineConditioner(8).float(); conditioner.alpha_q.data.zero_(); conditioner.alpha_k.data.zero_()
    provider=lambda states,**kwargs:(torch.zeros(states.shape[0],states.shape[1],7),torch.zeros(states.shape[0],kwargs['key_length'],7))
    memory=LongTermKVMemory(); memory.enabled=False
    processor=SightlineHeliosAttnProcessor(conditioner,provider,memory=memory,qkv_projection=native._get_qkv_projections,rotary_apply=native.apply_rotary_emb_transposed,attention_dispatch=native.dispatch_attention_fn,attention_backend=native.HeliosAttnProcessor._attention_backend,parallel_config=native.HeliosAttnProcessor._parallel_config)
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

def test_history_rays_project_each_camera_then_match_helios_temporal_pooling():
    from long_video.sightline.rays import plucker_rays
    provider=SightlineRayProvider(source_height=32,source_width=32)
    current=_camera_trajectory(frames=9); K=torch.eye(3).view(1,1,3,3).repeat(1,9,1,1)
    def trajectory(count):
        poses=_camera_trajectory(frames=count)
        for index in range(count):
            angle=.05*index; c,s=np.cos(angle),np.sin(angle); poses[0,index,0,0]=c; poses[0,index,0,2]=s; poses[0,index,2,0]=-s; poses[0,index,2,2]=c
        return poses,torch.eye(3).view(1,1,3,3).repeat(1,count,1,1)
    groups={'long':trajectory(16),'mid':trajectory(2),'short':trajectory(2)}
    shapes={'long':(4,1,1),'mid':(1,2,2),'short':(2,4,4)}
    provider.set_context(chunk_index=2,c2w=current,intrinsics=K,history_groups=groups,history_token_shapes=shapes,stage_shapes=((9,4,4),))
    class Projector(torch.nn.Module):
        def __init__(self):super().__init__();self.dummy=torch.nn.Parameter(torch.zeros(()))
        def project(self,rays,**kwargs):return torch.cat((rays[...,:3],rays[...,:1]),-1)
    projector=Projector(); actual=provider.project_history(projector,kind='q',scale_delta=None)
    expected=[]
    for name,factor in (('long',4),('mid',2),('short',1)):
        cameras,intrinsics=groups[name]; out_t,h,w=shapes[name]
        embedded=projector.project(plucker_rays(cameras,intrinsics,h,w,source_height=32,source_width=32))
        expected.append(embedded.reshape(1,out_t,factor,h,w,4).mean(2).reshape(1,-1,4))
    assert torch.allclose(actual,torch.cat(expected,1))
    long_last=projector.project(plucker_rays(groups['long'][0][:,3::4],groups['long'][1][:,3::4],1,1,source_height=32,source_width=32)).reshape(1,4,4)
    assert not torch.allclose(actual[:,:4],long_last)

def test_memory_selector_is_anchor_plus_three_geometry_chunks_and_may_overlap_native_history():
    from long_video.sightline.memory import select_memory_chunks
    memory=LongTermKVMemory(budget=100,pool=1)
    hidden=torch.randn(1,2,4); rays=torch.randn(1,2,7)
    for chunk in range(9): memory.capture(hidden,rays,chunk,grid_shape=(2,1,1),camera_poses=_camera_trajectory(float(chunk)))
    selected=select_memory_chunks(memory.archive,query_chunk=8,query_camera_poses=_camera_trajectory(8.),native_history_chunk_ids={6,7},tau_pos=1.,tau_angle=1.)
    assert [chunk.chunk_id for chunk in selected]==[0,7,6,5]
    assert len({chunk.chunk_id for chunk in selected})==4 and all(chunk.chunk_id<8 for chunk in selected)
    # The scene Memory path is independent and may overlap native history.
    selected=select_memory_chunks(memory.archive,query_chunk=8,query_camera_poses=_camera_trajectory(8.),native_history_chunk_ids={0,6,7},tau_pos=1.,tau_angle=1.)
    assert [chunk.chunk_id for chunk in selected]==[0,7,6,5]
    assert {0,6,7}.issubset({chunk.chunk_id for chunk in selected})

def test_memory_anchor_archive_is_permanent_and_active_inside_native_history():
    from long_video.sightline.memory import select_memory_chunks
    memory=LongTermKVMemory(budget=100,pool=1); hidden=torch.randn(1,2,4); rays=torch.randn(1,2,7)
    for chunk in range(5): memory.capture(hidden,rays,chunk,grid_shape=(2,1,1),camera_poses=_camera_trajectory(float(chunk)))
    assert not memory.evict_archive_chunk(0) and 0 in memory.archive
    assert memory.evict_archive_chunk(4) and 4 not in memory.archive
    with pytest.raises(RuntimeError,match='permanent Memory anchor'):
        memory.capture(hidden,rays,0,grid_shape=(2,1,1),camera_poses=_camera_trajectory())
    while_native=select_memory_chunks(memory.archive,query_chunk=3,query_camera_poses=_camera_trajectory(3.),native_history_chunk_ids={0,1,2})
    assert [chunk.chunk_id for chunk in while_native]==[0,2,1]
    after_exit=select_memory_chunks(memory.archive,query_chunk=4,query_camera_poses=_camera_trajectory(4.),native_history_chunk_ids={1,2,3})
    assert [chunk.chunk_id for chunk in after_exit]==[0,3,2,1]

def test_native_history_chunk_ownership_does_not_treat_source_as_anchor():
    from long_video.sightline.history import covered_history_chunk_ids
    assert covered_history_chunk_ids({'short':((0,),()),'mid':(), 'long':()},4)==set()
    assert covered_history_chunk_ids({'short':((0,),(1,8)),'mid':((9,16),),'long':()},4)=={0,1}

def test_memory_active_bank_prepared_once_and_cleared_without_archive_loss():
    from long_video.sightline.memory import LongTermKVMemory
    memory=LongTermKVMemory(budget=100,pool=1); hidden=torch.randn(1,2,4); rays=torch.randn(1,2,7)
    for chunk in range(3): memory.capture(hidden,rays,chunk,grid_shape=(2,1,1),camera_poses=_camera_trajectory(float(chunk)))
    calls=[]; original=memory.select_chunks
    memory.select_chunks=lambda *args,**kwargs:(calls.append(1) or original(*args,**kwargs))
    memory.prepare_active_memory(query_chunk=2,query_camera_poses=_camera_trajectory(2.),device='cpu',dtype=torch.float32)
    memory.prepare_active_memory(query_chunk=2,query_camera_poses=_camera_trajectory(2.),device='cpu',dtype=torch.float32)
    assert len(calls)==1 and memory.last_selected_chunk_ids==(0,1)
    assert memory.get(16,query_camera_poses=_camera_trajectory(2.),device='cpu')[0].shape[1]==2
    memory.clear_active_memory()
    assert not memory.last_active_tokens and memory.archive.keys()=={0,1,2}

def test_layer_memory_banks_select_once_and_share_chunk_ids():
    from long_video.sightline.memory import LayerKVMemoryBank
    layers=(4,6,8,16,20,24,32,34,36)
    memory=LayerKVMemoryBank(layers,budget=100,pool=1); hidden=torch.randn(1,2,4); rays=torch.randn(1,2,7)
    for bank in memory.banks.values():
        for chunk in range(3): bank.capture(hidden+chunk,rays,chunk,grid_shape=(2,1,1),camera_poses=_camera_trajectory(float(chunk)))
    calls=[]
    for bank in memory.banks.values():
        original=bank.select_chunks
        bank.select_chunks=lambda *args,_original=original,**kwargs:(calls.append(1) or _original(*args,**kwargs))
    memory.prepare_active_memory(query_chunk=2,query_camera_poses=_camera_trajectory(2.),native_history_chunk_ids={0,1},device='cpu',dtype=torch.float32)
    assert calls==[1] and all(bank.last_selected_chunk_ids==(0,1) for bank in memory.banks.values())
    identity_objects=[bank.active_identity_metadata() for bank in memory.banks.values()]
    assert all(value is identity_objects[0] for value in identity_objects)

def test_memory_active_empty_bank_does_not_reselect():
    from long_video.sightline.memory import LongTermKVMemory
    memory=LongTermKVMemory(); calls=[]; original=memory.select_chunks
    memory.select_chunks=lambda *args,**kwargs:(calls.append(1) or original(*args,**kwargs))
    memory.prepare_active_memory(query_chunk=0,query_camera_poses=_camera_trajectory(),device='cpu',dtype=torch.float32)
    assert memory.get(0,query_camera_poses=_camera_trajectory(),device='cpu')==(None,None)
    assert len(calls)==1

def test_clean_memory_capture_uses_scheduler_endpoint_and_ignores_stale_hidden():
    from types import SimpleNamespace
    from long_video.sightline.pipeline import SightlinePipeline
    from long_video.sightline.memory import LongTermKVMemory
    class Provider:
        context={'stage_shapes':((2,1,1),)}
        def current_rays(self,shape): return torch.ones(1,2,7)
    memory=LongTermKVMemory(budget=8,pool=1)
    processor=SimpleNamespace(memory=memory,last_hidden_states=torch.full((1,2,4),99.),last_current_length=2)
    transformer=SimpleNamespace(_sightline_processors={0:processor})
    helios=SimpleNamespace(transformer=transformer,scheduler=SimpleNamespace(sigmas=torch.tensor([1.,.5,0.]),timesteps=torch.tensor([999,499]),_sigma_to_t=lambda sigma:sigma*1000))
    config=SimpleNamespace(memory_layers=(0,),memory_budget=8,memory_pool=1,memory_write_sigma=0.0,pyramid_steps=(2,2,2))
    runner=SightlinePipeline(helios,config=config,ray_provider=Provider())
    cameras=_camera_trajectory(); runner._pending_camera_chunk=(list(cameras.unbind(1)),[],[])
    seen=[]
    def capture(clean,timestep):
        seen.append((clean.clone(),timestep.clone())); processor.last_hidden_states=torch.ones(1,2,4)
    runner._capture_clean_memory(0,torch.zeros(1,1,2,1,1),capture)
    assert seen[0][1].shape==(1,) and int(seen[0][1])==0 and torch.equal(memory.archive[0].hidden[:,0:1],torch.ones(1,1,4))

def test_disabled_memory_skips_clean_feature_capture():
    from types import SimpleNamespace
    from long_video.sightline.pipeline import SightlinePipeline
    transformer=SimpleNamespace(_sightline_processors={})
    config=SimpleNamespace(memory_layers=(0,),memory_budget=8,memory_pool=1,pyramid_steps=(2,2,2))
    runner=SightlinePipeline(SimpleNamespace(transformer=transformer),config=config)
    runner.memory.set_enabled(False); called=[]
    runner._finalize_chunk(0,clean_latent=torch.zeros(1),capture_fn=lambda *_:called.append(1))
    assert called==[]

def test_train_chunk_finalization_skips_memory_but_required_prefix_captures():
    from types import SimpleNamespace
    from long_video.sightline.pipeline import SightlinePipeline
    transformer=SimpleNamespace(_sightline_processors={})
    config=SimpleNamespace(memory_layers=(0,),memory_budget=8,memory_pool=1,pyramid_steps=(2,2,2))
    runner=SightlinePipeline(SimpleNamespace(transformer=transformer),config=config,ray_provider=SimpleNamespace())
    runner.memory.banks[0].enabled=True; calls=[]
    runner._capture_clean_memory=lambda *args,**kwargs:(calls.append(args[0]) or {'memory_clean_forward_seconds':1.,'memory_archive_write_seconds':2.})
    skipped=runner._finalize_chunk(1,clean_latent=torch.zeros(1),capture_fn=lambda *_:None,capture_memory=False)
    captured=runner._finalize_chunk(0,clean_latent=torch.zeros(1),capture_fn=lambda *_:None,capture_memory=True)
    assert calls==[0] and skipped=={'memory_clean_forward_seconds':0.0,'memory_archive_write_seconds':0.0}
    assert captured['memory_clean_forward_seconds']==1.

def test_inference_can_disable_only_memory():
    from types import SimpleNamespace
    from scripts.infer_sightline import configure_inference_memory
    calls=[]; runner=SimpleNamespace(memory=SimpleNamespace(set_enabled=lambda value:calls.append(value)))
    assert configure_inference_memory(runner,True) is False and calls==[False]

def test_inference_global_sightline_residual_scale():
    from types import SimpleNamespace
    from scripts.infer_sightline import configure_sightline_residual_scale
    processors={index:SimpleNamespace() for index in range(40)}
    transformer=SimpleNamespace(_sightline_processors=processors)
    assert configure_sightline_residual_scale(transformer,.2)==.2
    assert all(processor.residual_scale==.2 for processor in processors.values())
    with pytest.raises(ValueError): configure_sightline_residual_scale(transformer,float('nan'))

def test_inference_dynamic_sigma_scale_uses_active_scheduler_grid():
    from scripts.infer_sightline import scheduler_sigma_for_timestep,sightline_sigma_scale
    scheduler=type('Scheduler',(),{'timesteps':torch.tensor([900.,650.,300.]),'sigmas':torch.tensor([.9,.65,.3,0.])})()
    assert scheduler_sigma_for_timestep(scheduler,torch.tensor([649]))==pytest.approx(.65)
    assert sightline_sigma_scale(.8)==1.0
    assert sightline_sigma_scale(.35)==pytest.approx(.5)

def test_processor_residual_scale_matches_q_plus_s_delta():
    class A:
        heads=2; is_amplify_history=False; to_q=torch.nn.Linear(8,8); to_k=torch.nn.Linear(8,8); to_v=torch.nn.Linear(8,8); norm_q=torch.nn.Identity(); norm_k=torch.nn.Identity(); to_out=torch.nn.ModuleList([torch.nn.Identity(),torch.nn.Identity()])
    a=A(); c=SightlineConditioner(8).eval()
    torch.nn.init.constant_(c.q_proj.weight,.1); torch.nn.init.constant_(c.k_proj.weight,.2)
    rays=torch.ones(1,4,7); provider=lambda h,**kw:(rays,rays); h=torch.randn(1,4,8); seen=[]
    def qkv(attn,states,_): return attn.to_q(states),attn.to_k(states),attn.to_v(states)
    def dispatch(q,k,v,**kw): seen.append((q.clone(),k.clone())); return v
    proc=SightlineHeliosAttnProcessor(c,provider,qkv_projection=qkv,rotary_apply=lambda x,r:x,attention_dispatch=dispatch); proc.residual_scale=.2
    proc(a,h)
    native_q=a.to_q(h).unflatten(2,(a.heads,-1)); native_k=a.to_k(h).unflatten(2,(a.heads,-1))
    dq=c.project(rays,kind='q',training=False).unflatten(-1,(a.heads,-1)); dk=c.project(rays,kind='k',training=False).unflatten(-1,(a.heads,-1))
    assert torch.allclose(seen[0][0],native_q+.2*dq) and torch.allclose(seen[0][1],native_k+.2*dk)

def test_key_identity_map_contains_native_current_and_memory():
    from long_video.sightline.memory import LongTermKVMemory
    provider=SightlineRayProvider(source_height=8,source_width=8)
    cameras=torch.eye(4).view(1,1,4,4).expand(1,9,-1,-1); K=torch.eye(3).view(1,1,3,3).expand(1,9,-1,-1)
    shapes={'long':(1,1,1),'mid':(1,1,1),'short':(1,1,1)}; coverage={'long':((1,2,3,4),),'mid':((5,6),),'short':((7,),)}
    provider.set_context(chunk_index=1,c2w=cameras,intrinsics=K,history_token_shapes=shapes,history_global_coverages=coverage,stage_shapes=((9,1,1),),token_shape=(9,1,1))
    memory=LongTermKVMemory(budget=200,pool=1)
    memory.capture(torch.zeros(1,9*3*4,4),torch.zeros(1,9*3*4,7),0,grid_shape=(9,3,4),camera_poses=_camera_trajectory())
    memory.prepare_active_memory(query_chunk=1,query_camera_poses=_camera_trajectory(1.),device='cpu',dtype=torch.float32)
    identities=provider.key_identities(9,memory)
    assert provider.key_identities(9,memory) is identities
    assert identities[:3]==(('native',(1,2,3,4),0,0,'long'),('native',(5,6),0,0,'mid'),('native',(7,),0,0,'short'))
    assert ('current',(8,),0,0,'current') in identities and identities[-1]==('memory',(8,),2,3,'memory')

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

def test_correspondence_hash_mapping_matches_identity_semantics_and_reuses_layers(monkeypatch):
    from types import SimpleNamespace
    import scripts.train_sightline_rgbd as training
    from long_video.training.sightline import SightlineTrainable
    identities=(('native',(0,),0,0,'short'),('current',(8,),0,0,'current'),('memory',(1,),0,0,'memory'))
    rows=[{'query_chunk':1,'query_latent_temporal':0,'query_y':0,'query_x':0,'key_chunk':0,'key_latent_temporal':0,'key_y':0,'key_x':0,'weight':.7}]
    processors={}
    layers=(4,6,8,16,20,24,32,34,36)
    for layer in layers:
        q=torch.randn(1,1,2,2,requires_grad=True); k=torch.randn(1,3,2,2,requires_grad=True)
        processors[layer]=SimpleNamespace(last_q=q,last_k=k,last_current_length=1,ray_provider=SimpleNamespace(context={'stage_shapes':((1,1,1),)}),last_key_identities=identities,last_attention_bias=None)
    selected,positives,weights,flags=training._mapped_correspondences(processors[4],rows,1)
    assert selected==[0] and positives==[[0]] and weights==[.7] and flags==[{'has_native_positive':True,'has_memory_positive':False}]
    calls=[]; original=training._mapped_correspondences
    def counted(*args,**kwargs): calls.append(1); return original(*args,**kwargs)
    monkeypatch.setattr(training,'_mapped_correspondences',counted)
    loss=training._corr_loss(SightlineTrainable(4,layers=layers,heads=2),processors,rows,1,layers,8)
    assert len(calls)==1 and loss.requires_grad
    processors[36].last_key_identities=identities[:-1]
    with pytest.raises(RuntimeError,match='identical key identity maps'):
        training._corr_loss(SightlineTrainable(4,layers=layers,heads=2),processors,rows,1,layers,8)

def test_native_memory_overlap_is_multi_positive_and_memory_rows_survive_sampling():
    from types import SimpleNamespace
    import scripts.train_sightline_rgbd as training
    identities=(('native',(1,),0,0,'short'),('current',(8,),0,0,'current'),('memory',(1,),0,0,'memory'))
    processor=SimpleNamespace(last_q=torch.randn(1,1,2,2),last_k=torch.randn(1,3,2,2),last_current_length=1,ray_provider=SimpleNamespace(context={'stage_shapes':((1,1,1),)}),last_key_identities=identities)
    rows=[{'query_chunk':1,'query_latent_temporal':0,'query_y':0,'query_x':0,'key_chunk':0,'key_latent_temporal':1,'key_y':0,'key_x':0,'weight':.9}]
    selected,positives,weights,flags=training._mapped_correspondences(processor,rows,1)
    assert selected==[0] and positives==[[0,2]] and weights==[.9]
    assert flags==[{'has_native_positive':True,'has_memory_positive':True}]
    sampled=training._sample_correspondence_mapping(list(range(5)),[[i] for i in range(5)],[1.]*5,[{'has_native_positive':True,'has_memory_positive':i in (3,4)} for i in range(5)],2,7)
    assert len(sampled[0])==2 and all(flag['has_memory_positive'] for flag in sampled[3])

def test_memory_shared_boundaries_are_unique_and_future_filtered():
    memory=LongTermKVMemory(budget=100,pool=1); hidden=torch.randn(1,9,4); rays=torch.randn(1,9,7)
    memory.capture(hidden,rays,0,grid_shape=(9,1,1),camera_poses=_camera_trajectory()); memory.capture(hidden,rays,1,grid_shape=(9,1,1),camera_poses=_camera_trajectory(1.))
    global_ids=[token.chunk_index*8+token.temporal for token in memory.tokens]
    assert len(global_ids)==16 and len(set(global_ids))==16 and 0 not in global_ids and global_ids.count(8)==1
    active=memory.active_tokens(16,query_camera_poses=_camera_trajectory(2.)); assert [token.chunk_index*8+token.temporal for token in active]==list(range(1,17))

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

def test_formal_config_accepts_explicit_reduced_ddp_world_size(tmp_path):
    from long_video.config import load_sightline_config
    source=Path(__file__).parents[1].joinpath('configs/sightline.yaml').read_text()
    formal=load_sightline_config(Path(__file__).parents[1]/'configs/sightline.yaml')
    assert formal.memory_layers==formal.correspondence_layers==(4,6,8,16,20,24,32,34,36)
    runtime=tmp_path/'sightline-3gpu.yaml'; runtime.write_text(source.replace('ddp_world_size: 4','ddp_world_size: 3'))
    assert load_sightline_config(runtime).ddp_world_size==3
    invalid=tmp_path/'sightline-5gpu.yaml'; invalid.write_text(source.replace('ddp_world_size: 4','ddp_world_size: 5'))
    with pytest.raises(ValueError,match='DDP=1..4'): load_sightline_config(invalid)

def test_bucketed_manual_ddp_preserves_gradient_average_and_reduces_collectives(monkeypatch):
    import scripts.train_sightline_rgbd as training
    first=torch.nn.Parameter(torch.ones(4)); second=torch.nn.Parameter(torch.ones(3)); unused=torch.nn.Parameter(torch.ones(2))
    first.grad=torch.arange(4,dtype=torch.float32); second.grad=torch.arange(3,dtype=torch.float32)+4
    expected=(first.grad.clone(),second.grad.clone())
    calls=[]
    def identical_second_rank(value): calls.append(value.numel()); value.mul_(2)
    monkeypatch.setattr(training.dist,'all_reduce',identical_second_rank)
    training._average_gradients([first,second,unused],2,bucket_bytes=1024)
    assert torch.equal(first.grad,expected[0]) and torch.equal(second.grad,expected[1]) and unused.grad is None
    assert calls==[3,7]

def test_training_preflight_and_fixed_2500_warmup_schedule():
    from types import SimpleNamespace
    from scripts.train_sightline_dl3dv import _preflight,_lr_multiplier,FORMAL_MEMORY_LAYERS
    cfg=SimpleNamespace(sightline_layers=tuple(range(40)),correspondence_layers=FORMAL_MEMORY_LAYERS,memory_layers=FORMAL_MEMORY_LAYERS,lora_layers=(),warmup_ratio=.04)
    with pytest.raises(ValueError,match='P2'): _preflight(cfg,SimpleNamespace(train=True,max_steps=401),())
    cfg.lora_layers=(0,)
    cfg.memory_layers=(); cfg.correspondence_layers=()
    with pytest.raises(ValueError,match='Memory/correspondence'): _preflight(cfg,SimpleNamespace(train=True,max_steps=1001),())
    assert _lr_multiplier(0)==pytest.approx(.01) and _lr_multiplier(99)==pytest.approx(1.) and _lr_multiplier(100)==pytest.approx(1.) and _lr_multiplier(2499)<1e-5

def test_p3_chunk_curriculum_has_fixed_global_step_boundaries():
    from long_video.training.sightline import curriculum_phase,select_train_chunk
    assert [curriculum_phase(step)['max_chunks'] for step in (1000,1499,1500,1799,1800,2099,2100,2299,2300,2499)] == [2,2,3,3,4,4,5,5,6,6]
    generator=torch.Generator().manual_seed(9); selected=[select_train_chunk(6,generator,minimum=1) for _ in range(200)]
    assert 0 not in selected and set(selected)==set(range(1,6))

def test_checkpoint_cadence_switches_after_p2():
    from scripts.train_sightline_rgbd import checkpoint_interval
    assert checkpoint_interval(0)==100 and checkpoint_interval(999)==100
    assert checkpoint_interval(1000)==60 and checkpoint_interval(2499)==60

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

def test_padded_geometry_has_exact_three_stage_grids_and_output_crop():
    from long_video.sightline.geometry import assert_latent_geometry,crop_video,padded_size
    assert padded_size(480,832)==(512,832)
    latents=torch.randn(1,16,9,64,104)
    assert_latent_geometry(latents,height=480,width=832,patch_size=(1,2,2))
    with pytest.raises(ValueError): assert_latent_geometry(torch.randn(1,16,9,60,104),height=480,width=832,patch_size=(1,2,2))
    video=np.zeros((1,33,512,832,3),np.float32)
    assert crop_video(video,480,832).shape==(1,33,480,832,3)

def test_shared_clean_boundary_training_and_sampling_use_same_three_stage_flow():
    from types import SimpleNamespace
    from long_video.sightline.boundary import stage2_sample_with_boundary,constrain_flow_items,training_boundary_stages,_stage_transition_coefficients
    class Transformer(torch.nn.Module):
        def __init__(self):super().__init__();self.seen=[]
        def forward(self,hidden_states,timestep):self.seen.append(hidden_states.clone());return hidden_states
    class Pipe:
        def __init__(self):
            self.transformer=Transformer(); self.scheduler=SimpleNamespace(start_sigmas={0:1.,1:1-0.5/(2**.5*.5+.5),2:1-0.5/(2**.5*.5+.5)},end_sigmas={0:.6,1:.25,2:0.},ori_start_sigmas={0:1.,1:.5,2:.5},config=SimpleNamespace(gamma=1.),timesteps=None,sigmas=None)
            self.stage_noises=[torch.full((1,2,9,2,2),2.),torch.full((1,2,9,4,4),4.),torch.full((1,2,9,8,8),6.)]
            self.block_noise_calls=[]
        def sample_block_noise(self,*_args,**_kwargs):
            value=self.stage_noises[len(self.block_noise_calls)+1].clone(); self.block_noise_calls.append(value); return value
        def stage2_sample(self,latents,callback_on_step_end,**kwargs):
            latents=self.stage_noises[0].clone()
            for stage in range(3):
                if stage:
                    # Match the pinned inference transition: nearest upsample,
                    # followed by alpha*latent + beta*block_noise.
                    previous=torch.nn.functional.interpolate(latents.flatten(0,2).unsqueeze(1),size=self.stage_noises[stage].shape[-2:],mode='nearest').squeeze(1).reshape_as(self.stage_noises[stage])
                    alpha,beta=_stage_transition_coefficients(self.scheduler,stage)
                    latents=alpha*previous+beta*self.sample_block_noise()
                self.scheduler.timesteps=torch.tensor([2.,1.]);self.scheduler.sigmas=torch.tensor([1.,0.,0.])
                for index,timestep in enumerate(self.scheduler.timesteps):
                    self.transformer(hidden_states=latents,timestep=timestep)
                    latents=latents+.25
                    latents=callback_on_step_end(self,index,timestep,{'latents':latents})['latents']
            return latents
    pipe=Pipe(); clean=torch.arange(128.,dtype=torch.float32).reshape(1,2,1,8,8); output=stage2_sample_with_boundary(pipe,clean_boundary=clean,latents=torch.zeros(1,2,9,8,8),pyramid_num_stages=3)
    assert torch.equal(output[:,:,:1],clean)
    clean_pyramid=[clean]
    for noise in reversed(pipe.stage_noises[:-1]):
        value=clean_pyramid[-1]; resized=torch.nn.functional.interpolate(value.flatten(0,2).unsqueeze(1),size=noise.shape[-2:],mode='bilinear',align_corners=False).squeeze(1)
        clean_pyramid.append(resized.reshape(1,2,1,*noise.shape[-2:]))
    clean_pyramid=list(reversed(clean_pyramid))
    items=[]
    for stage,noise in enumerate(pipe.stage_noises):
        clean_stage=clean_pyramid[stage].expand(-1,-1,9,-1,-1)
        if stage==0: native_start=noise
        else:
            previous=clean_pyramid[stage-1].expand(-1,-1,9,-1,-1); flat=previous.flatten(0,2).unsqueeze(1)
            previous=torch.nn.functional.interpolate(flat,size=noise.shape[-2:],mode='bilinear',align_corners=False).squeeze(1).reshape_as(noise)
            native_start=pipe.scheduler.start_sigmas[stage]*noise+(1-pipe.scheduler.start_sigmas[stage])*previous
        native_end=clean_stage if stage==2 else pipe.scheduler.end_sigmas[stage]*noise+(1-pipe.scheduler.end_sigmas[stage])*clean_stage
        sigma=torch.full((1,1,1,1,1),.35); noisy=sigma*native_start+(1-sigma)*native_end
        items.append({'noisy_latents':noisy,'target':native_start-native_end,'noise':noise.clone(),'sigmas':sigma,'stage_start_sigma':pipe.scheduler.start_sigmas[stage],'stage_end_sigma':pipe.scheduler.end_sigmas[stage]})
    stages=training_boundary_stages(items,clean); constrained=constrain_flow_items(items,clean)
    for stage in range(3):
        noise=pipe.stage_noises[stage][:,:,:1]; clean_stage=clean_pyramid[stage]
        expected_start=noise if stage==0 else pipe.scheduler.start_sigmas[stage]*noise+(1-pipe.scheduler.start_sigmas[stage])*torch.nn.functional.interpolate(clean_pyramid[stage-1].flatten(0,2).unsqueeze(1),size=noise.shape[-2:],mode='bilinear',align_corners=False).squeeze(1).reshape_as(clean_stage)
        expected_end=clean_stage if stage==2 else pipe.scheduler.end_sigmas[stage]*noise+(1-pipe.scheduler.end_sigmas[stage])*clean_stage
        assert torch.allclose(stages[stage].start,expected_start)
        assert torch.allclose(stages[stage].end,expected_end)
        assert torch.allclose(constrained[stage]['target'][:,:,:1],expected_start-expected_end)
        # With identical clean/noise inputs, temporal0 is numerically identical
        # to the untouched native temporal1 flow in every stage.
        assert torch.allclose(constrained[stage]['noisy_latents'][:,:,:1],items[stage]['noisy_latents'][:,:,1:2])
        assert torch.allclose(constrained[stage]['target'][:,:,:1],items[stage]['target'][:,:,1:2])
    # Inference endpoints use recursively constructed effective noise, not the
    # raw sample_block_noise tensors (4 and 6).
    effective=[torch.full_like(pipe.stage_noises[0][:,:,:1],2.)]
    expected_starts=[effective[0]]; expected_ends=[]
    for stage in range(3):
        clean_stage=clean_pyramid[stage]
        if stage:
            previous_end=expected_ends[-1]; alpha,beta=_stage_transition_coefficients(pipe.scheduler,stage)
            native_start=alpha*torch.nn.functional.interpolate(previous_end.flatten(0,2).unsqueeze(1),size=pipe.stage_noises[stage].shape[-2:],mode='nearest').squeeze(1).reshape_as(clean_stage)+beta*pipe.stage_noises[stage][:,:,:1]
            # Training's flow basis uses bilinear previous-clean upsample;
            # the native endpoint transition above intentionally remains
            # nearest.
            prior_clean=torch.nn.functional.interpolate(clean_pyramid[stage-1].flatten(0,2).unsqueeze(1),size=pipe.stage_noises[stage].shape[-2:],mode='bilinear',align_corners=False).squeeze(1).reshape_as(clean_stage)
            effective.append((native_start-(1-pipe.scheduler.start_sigmas[stage])*prior_clean)/pipe.scheduler.start_sigmas[stage]); expected_starts.append(native_start)
        endpoint=clean_stage if stage==2 else pipe.scheduler.end_sigmas[stage]*effective[stage]+(1-pipe.scheduler.end_sigmas[stage])*clean_stage
        expected_ends.append(endpoint)
    for stage,(expected_start,expected_end) in enumerate(zip(expected_starts,expected_ends)):
        assert torch.allclose(pipe.transformer.seen[stage*2][:,:,:1],expected_start)
        assert torch.allclose(pipe.transformer.seen[stage*2+1][:,:,:1],expected_end)
    assert len(pipe.block_noise_calls)==2

def test_boundary_scheduler_resolves_helios_integer_transformer_timestep():
    from types import SimpleNamespace
    from long_video.sightline.boundary import _scheduler_coefficient
    scheduler=SimpleNamespace(timesteps=torch.tensor([998.999,412.75]),sigmas=torch.tensor([.9,.4,0.]))
    assert torch.equal(_scheduler_coefficient(scheduler,torch.tensor([998],dtype=torch.int64),after_step=False),scheduler.sigmas[0])
    assert torch.equal(_scheduler_coefficient(scheduler,scheduler.timesteps[0],after_step=True),scheduler.sigmas[1])

def test_stride32_assembly_keeps_one_shared_boundary_per_chunk():
    from long_video.sightline.pipeline import SightlinePipeline
    accumulated=None
    for chunk in range(6):
        value=torch.full((1,1,9,1,1),float(chunk))
        if accumulated is not None: value[:,:,:1]=accumulated[:,:,-1:]
        accumulated=SightlinePipeline.append_stride32_latents(accumulated,value)
    assert accumulated.shape[2]==1+8*6

def test_inference_boundary_off_is_diagnostic_and_never_affects_chunk0():
    from long_video.sightline.pipeline import boundary_enabled_for_chunk
    assert boundary_enabled_for_chunk(0,1)
    assert not boundary_enabled_for_chunk(1,1)
    assert not boundary_enabled_for_chunk(2,1)
    assert boundary_enabled_for_chunk(1,None)
    with pytest.raises(ValueError): boundary_enabled_for_chunk(1,0)

def test_alpha_zero_baseline_disables_qk_memory_and_lora():
    from long_video.training.sightline import SightlineTrainable,LoRALinear,configure_alpha_zero_baseline
    from long_video.sightline.memory import LayerKVMemoryBank
    trainable=SightlineTrainable(4,layers=(0,)); memory=LayerKVMemoryBank((0,),8,2,hidden_dim=4)
    transformer=torch.nn.Sequential(LoRALinear(torch.nn.Linear(4,4)))
    configure_alpha_zero_baseline(trainable,memory,transformer)
    assert all(float(alpha.detach())==0 for alpha in trainable.conditioner.alpha_parameters())
    assert not memory.banks[0].enabled and not transformer[0].enabled
