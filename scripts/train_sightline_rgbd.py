"""Causal Sightline training from the canonical RGB-D manifest.

The historical filename is retained as a legacy compatibility entry point.
Formal training data is loaded exclusively through RGBDMemoryRecord.
"""
from __future__ import annotations
import argparse, hashlib, json, os, random, sys, time
# Must be set before importing/initializing CUDA; callers may override it.
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
os.environ.setdefault('PYTORCH_ALLOC_CONF', 'expandable_segments:True')
from dataclasses import asdict
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
from PIL import Image
from long_video.config import load_sightline_config
from long_video.training.flow_matching_exact import exact_flow_matching_items
from long_video.training.sightline import CorrespondencePlan, SightlineTrainable, install_lora, curriculum_phase, select_train_chunk, run_single_graph_chunks, run_causal_prefix_chunks, selected_qk_logits, set_initialization_seed, set_rank_runtime_seed, broadcast_and_assert_trainables, configure_alpha_zero_baseline, set_lora_enabled, prefix_chunk_should_capture_memory, correspondence_capture_for_stage
from long_video.training.rgbd_memory_data import load_rgbd_memory_manifest
from long_video.training.sightline_data import load_latent_tensor, validate_latent_cache, require_overlap_validation, resolve_continuous_latent_cache, validate_rgbd_record_latent
from long_video.training.sightline_checkpoint import save_runtime_checkpoint, restore_runtime_checkpoint, runtime_provenance, gather_rank_rng_states
from long_video.sightline.helios_integration import SightlineRayProvider, install_sightline_attention
from long_video.sightline.rays import canonicalize_c2w
from long_video.sightline.history import NativeHistoryState,native_helios_indices
from long_video.sightline.pipeline import SightlinePipeline, prepare_source_condition
from long_video.sightline.geometry import assert_latent_geometry, padded_size
from long_video.sightline.boundary import constrain_flow_items, stage2_sample_with_boundary

TOTAL_TRAINING_STEPS=2500
WARMUP_STEPS=100
FORMAL_MEMORY_LAYERS=(4,8,16,20,24,32,36)
LEGACY_999_MEMORY_LAYERS=(4,6,8,16,20,24,32,34,36)

HELIOS_RUNTIME_PATCH_VERSION='sightline-token-blocked-v1'
_PATCH_IMPORT="""from long_video.sightline.bounded_ops import (
    token_blocked_gated_residual,
    token_blocked_layer_norm,
    token_blocked_layer_norm_modulate,
)"""
_IMPORT_ANCHOR='import torch\nimport torch.nn as nn'
_NORM1_ORIGINAL='        norm_hidden_states = (self.norm1(hidden_states.float()) * (1 + scale_msa) + shift_msa).type_as(hidden_states)'
_NORM1_EXPERIMENTAL_V1="""        norm_hidden_states = self.norm1(hidden_states.float())
        norm_hidden_states.mul_(1 + scale_msa).add_(shift_msa)
        norm_hidden_states = norm_hidden_states.type_as(hidden_states)"""
_NORM1_EXPERIMENTAL_V2="""        norm_hidden_states = torch.empty_like(hidden_states)
        norm_hidden_states.copy_(self.norm1(hidden_states.float()).mul_(1 + scale_msa).add_(shift_msa))"""
_NORM1_PATCHED='        norm_hidden_states = token_blocked_layer_norm_modulate(hidden_states, self.norm1, scale_msa, shift_msa)'
_NORM2_ORIGINAL='        norm_hidden_states = self.norm2(hidden_states.float()).type_as(hidden_states)'
_NORM2_PATCHED='        norm_hidden_states = token_blocked_layer_norm(hidden_states, self.norm2)'
_NORM3_ORIGINAL="""        norm_hidden_states = (self.norm3(hidden_states.float()) * (1 + c_scale_msa) + c_shift_msa).type_as(
            hidden_states
        )"""
_NORM3_EXPERIMENTAL_V1="""        norm_hidden_states = self.norm3(hidden_states.float())
        norm_hidden_states.mul_(1 + c_scale_msa).add_(c_shift_msa)
        norm_hidden_states = norm_hidden_states.type_as(hidden_states)"""
_NORM3_EXPERIMENTAL_V2="""        norm_hidden_states = torch.empty_like(hidden_states)
        norm_hidden_states.copy_(self.norm3(hidden_states.float()).mul_(1 + c_scale_msa).add_(c_shift_msa))"""
_NORM3_PATCHED='        norm_hidden_states = token_blocked_layer_norm_modulate(hidden_states, self.norm3, c_scale_msa, c_shift_msa)'
_RESIDUAL1_ORIGINAL='        hidden_states = (hidden_states.float() + attn_output * gate_msa).type_as(hidden_states)'
_RESIDUAL1_PATCHED='        hidden_states = token_blocked_gated_residual(hidden_states, attn_output, gate_msa)'
_RESIDUAL3_ORIGINAL='        hidden_states = (hidden_states.float() + ff_output.float() * c_gate_msa).type_as(hidden_states)'
_RESIDUAL3_PATCHED='        hidden_states = token_blocked_gated_residual(hidden_states, ff_output, c_gate_msa)'

def _install_memory_efficient_helios_norm(source_file:Path):
    """Install and identify the audited token-blocked Helios runtime patch."""
    text=source_file.read_text()
    canonical=text.replace(_PATCH_IMPORT+'\n','')
    for patched,original in (
        (_NORM1_PATCHED,_NORM1_ORIGINAL),(_NORM1_EXPERIMENTAL_V1,_NORM1_ORIGINAL),(_NORM1_EXPERIMENTAL_V2,_NORM1_ORIGINAL),
        (_NORM2_PATCHED,_NORM2_ORIGINAL),(_NORM3_PATCHED,_NORM3_ORIGINAL),(_NORM3_EXPERIMENTAL_V1,_NORM3_ORIGINAL),
        (_NORM3_EXPERIMENTAL_V2,_NORM3_ORIGINAL),(_RESIDUAL1_PATCHED,_RESIDUAL1_ORIGINAL),(_RESIDUAL3_PATCHED,_RESIDUAL3_ORIGINAL),
    ): canonical=canonical.replace(patched,original)
    required=(_NORM1_ORIGINAL,_NORM2_ORIGINAL,_NORM3_ORIGINAL,_RESIDUAL1_ORIGINAL,_RESIDUAL3_ORIGINAL)
    if any(value not in canonical for value in required): raise RuntimeError('pinned Helios source no longer matches the audited runtime patch')
    patched=canonical.replace(_IMPORT_ANCHOR,_IMPORT_ANCHOR+'\n'+_PATCH_IMPORT)
    for original,replacement in (
        (_NORM1_ORIGINAL,_NORM1_PATCHED),(_NORM2_ORIGINAL,_NORM2_PATCHED),(_NORM3_ORIGINAL,_NORM3_PATCHED),
        (_RESIDUAL1_ORIGINAL,_RESIDUAL1_PATCHED),(_RESIDUAL3_ORIGINAL,_RESIDUAL3_PATCHED),
    ): patched=patched.replace(original,replacement)
    if patched!=text: source_file.write_text(patched)
    return {
        'version':HELIOS_RUNTIME_PATCH_VERSION,
        'original_source_sha256':hashlib.sha256(canonical.encode()).hexdigest(),
        'runtime_source_sha256':hashlib.sha256(patched.encode()).hexdigest(),
    }

def checkpoint_interval(global_step: int) -> int:
    """Formal cadence: 100-step checkpoints through step 1000, then 60-step."""
    return 100 if int(global_step) < 1000 else 60

def _distributed_context():
    world_size=int(__import__('os').environ.get('WORLD_SIZE','1'))
    if world_size==1: return 0,1,torch.device('cuda')
    if not dist.is_available(): raise RuntimeError('torch.distributed is required for DDP')
    local_rank=int(__import__('os').environ['LOCAL_RANK']); torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl',init_method='env://')
    return dist.get_rank(),world_size,torch.device('cuda',local_rank)

def _average_gradients(parameters,world_size,bucket_bytes=64<<20):
    """Average dense gradients with one presence collective and large buckets.

    Unused parameters still contribute explicit zeros, matching the previous
    per-parameter implementation.  Bucketing only reduces NCCL launch and host
    synchronization overhead; it does not change the averaged gradient.
    """
    if world_size==1: return
    parameters=list(parameters)
    if not parameters: return
    presence=torch.tensor([parameter.grad is not None for parameter in parameters],device=parameters[0].device,dtype=torch.int32)
    dist.all_reduce(presence)
    globally_present=presence.cpu().tolist()
    groups={}
    for parameter,count in zip(parameters,globally_present):
        if not count: continue
        grad=parameter.grad
        if grad is not None and grad.is_sparse: raise RuntimeError('bucketed DDP requires dense gradients')
        groups.setdefault((parameter.device,parameter.dtype),[]).append((parameter,grad if grad is not None else torch.zeros_like(parameter)))
    for entries in groups.values():
        bucket=[]; size=0
        for entry in entries:
            nbytes=entry[1].numel()*entry[1].element_size()
            if bucket and size+nbytes>bucket_bytes:
                _all_reduce_gradient_bucket(bucket,world_size); bucket=[]; size=0
            bucket.append(entry); size+=nbytes
        if bucket: _all_reduce_gradient_bucket(bucket,world_size)

def _all_reduce_gradient_bucket(entries,world_size):
    flat=torch.cat([grad.reshape(-1) for _,grad in entries])
    dist.all_reduce(flat); flat.div_(world_size)
    offset=0
    for parameter,grad in entries:
        count=grad.numel(); parameter.grad=flat[offset:offset+count].view_as(parameter); offset+=count

def _ddp_record_index(step,rank,world_size,count,seed=20260823):
    if world_size==1: return random.randrange(count)
    if world_size>count: raise ValueError('DDP world size exceeds record count')
    return random.Random(int(seed)+int(step)).sample(range(count),count)[rank]

def _ddp_train_chunk(max_chunks,minimum,rank,world_size,device,forced=None):
    """Choose one curriculum path per global step so every rank reaches collectives together."""
    if forced is not None:
        return int(forced)
    if world_size==1:
        return select_train_chunk(max_chunks,minimum=minimum)
    value=torch.tensor([select_train_chunk(max_chunks,minimum=minimum) if rank==0 else minimum],device=device,dtype=torch.int64)
    dist.broadcast(value,src=0)
    return int(value.item())

def p3_chunk0_scheduled(step,p1_steps=500,p2_steps=900):
    """Deterministic 20% direct-source slots from step 500 onward."""
    return int(step)>=500 and (int(step)-500)%5==0

def _preflight(cfg,args,probe_layers):
    p1_steps=getattr(cfg,'p1_steps',400); p2_steps=getattr(cfg,'p2_steps',600); p3_steps=getattr(cfg,'p3_steps',1500)
    total_steps=p1_steps+p2_steps+p3_steps
    sightline=set(cfg.sightline_layers)
    if not cfg.sightline_layers:
        raise ValueError('formal training requires non-empty sightline_layers')
    memory_layers=tuple(cfg.memory_layers); correspondence_layers=tuple(cfg.correspondence_layers)
    if memory_layers not in (FORMAL_MEMORY_LAYERS,LEGACY_999_MEMORY_LAYERS) or correspondence_layers!=memory_layers:
        raise ValueError(f'Memory/correspondence layers must be one matching supported set: {FORMAL_MEMORY_LAYERS} or {LEGACY_999_MEMORY_LAYERS}')
    if args.train and (cfg.memory_layers or cfg.correspondence_layers):
        # These modules remain reserved but are intentionally disabled in the
        # camera-only retraining curriculum.
        pass
    if args.train and not set(probe_layers).issubset(sightline): raise ValueError('formal training probe layers must be a subset of sightline_layers')
    if not set(cfg.lora_layers).issubset(sightline): raise ValueError('lora_layers must be a subset of sightline_layers')
    if not set(cfg.memory_layers).issubset(sightline) or not set(cfg.correspondence_layers).issubset(sightline): raise ValueError('memory/correspondence layers must be a subset of sightline_layers')
    if total_steps!=TOTAL_TRAINING_STEPS or int(total_steps*cfg.warmup_ratio)!=WARMUP_STEPS: raise ValueError('formal schedule must preserve the configured 100/2500-step warmup')
    if args.train and args.max_steps>p1_steps and not cfg.lora_layers: raise ValueError('training reaches P2 but lora_layers is empty')
    if args.train and args.max_steps>p1_steps+p2_steps and (not cfg.memory_layers or not cfg.correspondence_layers): raise ValueError('training reaches P3 but Memory/correspondence layers are empty')
    if not 1<=args.max_steps<=total_steps: raise ValueError(f'--max-steps must be in 1..{total_steps}')

def _lr_multiplier(step,total_steps=TOTAL_TRAINING_STEPS):
    if step<WARMUP_STEPS: return float(step+1)/WARMUP_STEPS
    progress=min(1.0,max(0.0,(step-WARMUP_STEPS)/(total_steps-WARMUP_STEPS)))
    return 0.5*(1.0+__import__('math').cos(__import__('math').pi*progress))

def _sigma_band(step, phase):
    if phase=='P1a': return (.8,1.),'high_0.8_1.0'
    return (0.,1.),'uniform_0.0_1.0'

def _set_gradient_checkpointing(transformer,enabled):
    method=getattr(transformer,'enable_gradient_checkpointing' if enabled else 'disable_gradient_checkpointing',None)
    if method is None:
        if enabled: raise RuntimeError('gradient_checkpointing=true but pinned Helios exposes no enable method')
        return
    method()

def _assert_optimizer_scope(optimizer,trainable,memory,transformer,text_encoder,vae):
    actual={id(parameter) for group in optimizer.param_groups for parameter in group['params']}
    expected={id(parameter) for parameter in trainable.parameters()}|{id(parameter) for parameter in memory.parameters()}|{id(parameter) for name,parameter in transformer.named_parameters() if 'lora_' in name}
    forbidden={id(parameter) for module in (text_encoder,vae) for parameter in module.parameters()}|{id(parameter) for name,parameter in transformer.named_parameters() if 'lora_' not in name}
    if actual!=expected or actual&forbidden: raise RuntimeError('optimizer contains frozen text/VAE/native Helios parameters or misses a Sightline trainable')

def _offload_unused_vae_decode_path(vae):
    """Keep the training-only VAE encoder on CUDA and park decode-only weights on CPU."""
    moved=0
    for name in ('decoder','post_quant_conv'):
        module=getattr(vae,name,None)
        if module is None: continue
        moved += sum(parameter.numel()*parameter.element_size() for parameter in module.parameters())
        module.to('cpu')
    return moved

def _prompt(pipe,text,device):
    with torch.no_grad(): result=pipe._get_t5_prompt_embeds(text,device=device,dtype=torch.bfloat16,max_sequence_length=512)
    if not isinstance(result,(tuple,list)) or len(result)!=2: raise RuntimeError('pinned Helios prompt API must return (embeds, mask)')
    embeds,mask=result
    if mask.ndim!=2 or mask.shape[:2]!=embeds.shape[:2]: raise RuntimeError('pinned Helios prompt mask shape mismatch')
    return embeds.detach(),mask.detach()

def _transformer_forward(pipe,noisy,timestep,prompt_embeds,history,current_start):
    """Call native Helios only. Geometry sigma is owned by the caller."""
    indices=native_helios_indices(noisy.device,noisy.shape[0])['current']
    output=pipe.transformer(hidden_states=noisy.to(pipe.transformer.dtype),timestep=timestep,encoder_hidden_states=prompt_embeds,
        indices_hidden_states=indices,latents_history_long=history['long'][0],indices_latents_history_long=history['long'][1],
        latents_history_mid=history['mid'][0],indices_latents_history_mid=history['mid'][1],
        latents_history_short=history['short'][0],indices_latents_history_short=history['short'][1],attention_kwargs={'current_chunk':current_start//8})
    prediction=output[0] if isinstance(output,(tuple,list)) else getattr(output,'sample',output)
    if prediction.shape!=noisy.shape: raise RuntimeError(f'prediction shape {prediction.shape} != {noisy.shape}')
    return prediction

def _model_prediction(pipe,noisy,item,prompt_embeds,history,current_start):
    """Active FM only: exact item sigma is scoped to this forward."""
    if not isinstance(item,dict) or 'sigmas' not in item or 'timesteps' not in item:
        raise ValueError('active Flow Matching prediction requires item["sigmas"] and item["timesteps"]')
    runner=getattr(pipe,'_sightline_pipeline',None)
    if runner is None: raise RuntimeError('active FM requires the Sightline pipeline sigma router')
    with runner.geometry_sigma_override(item['sigmas']):
        return _transformer_forward(pipe,noisy,item['timesteps'],prompt_embeds,history,current_start)

def _generate_detached_chunk(pipe,source,history,prompt_embeds,cfg,chunk,clean_boundary=None):
    """Native Helios autoregressive inference from noise; no target argument exists."""
    noise=torch.randn((source.shape[0],source.shape[1],9,source.shape[-2],source.shape[-1]),device=source.device,dtype=source.dtype)
    indices=native_helios_indices(source.device,source.shape[0])['current']
    class Progress:
        def update(self): pass
    pipe._guidance_scale=1.0; pipe._attention_kwargs={'current_chunk':chunk}; pipe._current_timestep=None; pipe._interrupt=False
    return stage2_sample_with_boundary(pipe,clean_boundary=clean_boundary,latents=noise,pyramid_num_stages=3,pyramid_num_inference_steps_list=list(cfg.pyramid_steps),
        prompt_embeds=prompt_embeds,negative_prompt_embeds=None,guidance_scale=1.0,indices_hidden_states=indices,
        latents_history_long=history['long'][0],indices_latents_history_long=history['long'][1],
        latents_history_mid=history['mid'][0],indices_latents_history_mid=history['mid'][1],
        latents_history_short=history['short'][0],indices_latents_history_short=history['short'][1],
        attention_kwargs={'current_chunk':chunk},device=source.device,transformer_dtype=pipe.transformer.dtype,progress_bar=Progress())

def _load_correspondence(record,query_chunk=None):
    rows=record.correspondences_for_chunk(query_chunk) if query_chunk is not None else list(record.correspondence_rows())
    if hasattr(rows,'column'):
        qf,kf=rows.column('query_frame'),rows.column('key_frame'); qc,kc=rows.column('query_chunk'),rows.column('key_chunk'); weights=rows.column('weight')
        if len(rows) and (np.any(kf>=qf) or np.any(kc>=qc) or np.any(qc!=int(query_chunk)) or np.any(kc<0) or not np.isfinite(weights).all() or np.any(weights<0)):
            raise RuntimeError('invalid causal RGB-D correspondence identity or weight')
    else:
        for row in rows:
            qf,kf=int(row['query_frame']),int(row['key_frame'])
            if kf>=qf or not (0<=int(row['key_chunk'])<int(row['query_chunk'])<record.chunk_count): raise RuntimeError('invalid causal RGB-D correspondence identity')
            if not np.isfinite(float(row['weight'])) or float(row['weight'])<0: raise RuntimeError('invalid RGB-D correspondence weight')
    return rows

def _correspondence_columns(rows):
    names={'query_chunk':'query_chunk','query_latent_temporal':'query_t','query_y':'query_y','query_x':'query_x','key_chunk':'key_chunk','key_latent_temporal':'key_t','key_y':'key_y','key_x':'key_x','weight':'weight'}
    if hasattr(rows,'column'): return {name:rows.column(column) for name,column in names.items()}
    return {name:np.asarray([row[name] for row in rows]) for name in names}

def _identity_lookup(identities):
    lookup={}
    for index,identity in enumerate(identities):
        kind,global_ids,y,x,level=identity
        for global_id in global_ids:
            lookup.setdefault((kind,int(global_id),int(y),int(x),level),[]).append(index)
    return lookup

def _mapped_correspondences(processor,rows,chunk,identity_index=None):
    q,k=processor.last_q,processor.last_k
    if q is None or k is None: raise RuntimeError('correspondence processor did not capture Q/K')
    current=processor.last_current_length; identities=processor.last_key_identities
    if identities is None or len(identities)!=k.shape[1]: raise RuntimeError('explicit attention key identity map is missing or misaligned')
    current_shape=next(shape for shape in processor.ray_provider.context['stage_shapes'] if shape[0]*shape[1]*shape[2]==current)
    return _map_correspondence_identities(rows,chunk,current_shape,q.shape[1],identities,identity_index)

def _map_correspondence_identities(rows,chunk,current_shape,query_length,identities,identity_index=None):
    """GT-to-attention mapping using layout metadata only, never Q/K values."""
    current=int(current_shape[0]*current_shape[1]*current_shape[2]); _,height,width=current_shape; q_start=int(query_length)-current
    identity_index=_identity_lookup(identities) if identity_index is None else identity_index
    columns=_correspondence_columns(rows); grouped={}
    for row_index in range(len(columns['query_chunk'])):
        if int(columns['query_chunk'][row_index])!=chunk: continue
        qt=int(columns['query_latent_temporal'][row_index]); qy=int(columns['query_y'][row_index]); qx=int(columns['query_x'][row_index])
        if not (0<=qt<current_shape[0] and 0<=qy<height and 0<=qx<width): continue
        qi=q_start+qt*height*width+qy*width+qx
        global_key=int(columns['key_chunk'][row_index])*8+int(columns['key_latent_temporal'][row_index])
        query_global=chunk*8+qt
        if global_key>query_global: continue
        ky,kx=int(columns['key_y'][row_index]),int(columns['key_x'][row_index])
        factors={'long':4,'mid':2,'short':1}
        native=[]
        for level,factor in factors.items(): native.extend(identity_index.get(('native',global_key,ky//factor,kx//factor,level),()))
        native.sort(key=lambda index:{'short':0,'mid':1,'long':2}[identities[index][4]])
        memory=list(identity_index.get(('memory',global_key,ky//2,kx//2,'memory'),()))
        current_keys=list(identity_index.get(('current',global_key,ky,kx,'current'),()))
        source_keys=list(identity_index.get(('source',global_key,ky,kx,'source'),()))
        # Preserve every legal representation of the teacher identity on the
        # actual key axis.  Source/native/current/memory are diagnostics only.
        candidates=sorted(set(native + current_keys + source_keys + memory))
        if not candidates: continue
        if 0<=qi<query_length:
            bucket=grouped.setdefault(qi,{})
            for ki in candidates:
                if 0<=ki<len(identities):
                    source='memory' if ki in memory else ('native' if ki in native else ('source' if ki in source_keys else 'current'))
                    bucket[ki]=(max(bucket.get(ki,(0.0,''))[0],float(columns['weight'][row_index])),source)
    if not grouped: raise RuntimeError('correspondence identities do not map to real attention axes')
    selected=sorted(grouped); positives=[sorted(grouped[query]) for query in selected]; weights=[max(value[0] for value in grouped[query].values()) for query in selected]
    flags=[{'has_native_positive':any(value[1] in ('native','source','current') for value in grouped[query].values()),'has_memory_positive':any(value[1]=='memory' for value in grouped[query].values())} for query in selected]
    return selected,positives,weights,flags

def _sample_correspondence_mapping(selected,positives,weights,flags,max_rows,sampling_seed):
    if len(selected)<=max_rows:return selected,positives,weights,flags
    memory_indices=[i for i,flag in enumerate(flags) if flag['has_memory_positive']]
    generator=torch.Generator(device='cpu').manual_seed(int(sampling_seed) & ((1<<63)-1))
    order=torch.randperm(len(selected),generator=generator).tolist()
    memory_set=set(memory_indices); memory_order=[i for i in order if i in memory_set]
    chosen_memory=memory_order[:max_rows]; chosen_memory_set=set(chosen_memory)
    choice=(chosen_memory+[i for i in order if i not in chosen_memory_set])[:max_rows]
    return ([selected[i] for i in choice],[positives[i] for i in choice],[weights[i] for i in choice],[flags[i] for i in choice])

def _build_correspondence_plan(processor,rows,chunk,current_length,max_rows,sampling_seed):
    """Map GT once before final-stage forward so every layer captures selected Q only."""
    identities=processor.ray_provider.key_identities(current_length,processor.memory)
    memory_count=len(processor.memory.active_identity_metadata()) if processor.memory is not None and processor.memory.enabled else 0
    query_length=len(identities)-memory_count
    current_shape=next(shape for shape in processor.ray_provider.context['stage_shapes'] if shape[0]*shape[1]*shape[2]==current_length)
    selected,positives,weights,flags=_map_correspondence_identities(rows,chunk,current_shape,query_length,identities,_identity_lookup(identities))
    selected,positives,weights,flags=_sample_correspondence_mapping(selected,positives,weights,flags,max_rows,sampling_seed)
    max_positive=max(map(len,positives),default=0); device=processor.ray_provider.context['c2w'].device
    positive_indices=torch.full((len(selected),max_positive),-1,device=device,dtype=torch.long)
    positive_mask=torch.zeros_like(positive_indices,dtype=torch.bool)
    for row,keys in enumerate(positives):
        positive_indices[row,:len(keys)]=torch.as_tensor(keys,device=device); positive_mask[row,:len(keys)]=True
    return CorrespondencePlan(torch.as_tensor(selected,device=device,dtype=torch.long),positive_indices,positive_mask,
                              torch.as_tensor(weights,device=device,dtype=torch.float32),identities,tuple(flags))

def _corr_loss(trainable,processors,rows,chunk,layers,max_rows,*,sampling_seed=0,timings=None,plan=None,vram_callback=None):
    if not layers: raise RuntimeError('correspondence is enabled but correspondence_layers is empty')
    missing=[layer for layer in layers if layer not in processors]
    if missing: raise RuntimeError(f'correspondence layers have no Sightline processor: {missing}')
    first=processors[layers[0]]; identities=first.last_key_identities
    first_q_shape=first.last_q.shape; first_k_shape=first.last_k.shape
    for layer in layers[1:]:
        processor=processors[layer]
        same_identities=(processor.last_key_identities is identities or processor.last_key_identities==identities)
        if not same_identities or processor.last_current_length!=first.last_current_length or processor.last_q.shape[1]!=first_q_shape[1] or processor.last_k.shape[1]!=first_k_shape[1]:
            raise RuntimeError('correspondence layers must have identical key identity maps for shared mapping')
    mapping_started=time.perf_counter()
    if plan is None:
        try: selected,positives,weights,flags=_mapped_correspondences(first,rows,chunk,_identity_lookup(identities))
        except RuntimeError as exc:
            if 'do not map' not in str(exc): raise
            selected=positives=weights=flags=[]
        selected,positives,weights,flags=_sample_correspondence_mapping(selected,positives,weights,flags,max_rows,sampling_seed)
    else:
        if identities != plan.identities: raise RuntimeError('CorrespondencePlan key identities changed during final-stage forward')
        selected=list(range(plan.query_indices.numel())); positives=weights=None; flags=plan.flags
    if timings is not None: timings['correspondence_mapping_seconds']+=time.perf_counter()-mapping_started
    losses=[]; loss_started=time.perf_counter()
    for layer in layers:
        processor=processors[layer]
        if not selected:
            processor.last_q=None; processor.last_k=None; processor.last_attention_bias=None
            continue
        captured_q=processor.last_q; captured_k=processor.last_k; captured_bias=getattr(processor,'last_attention_bias',None)
        if plan is None:
            numerator=captured_q.new_zeros(()); denominator=captured_q.new_zeros(())
            for start in range(0,len(selected),64):
                stop=min(start+64,len(selected)); query_indices=selected[start:stop]
                logits=selected_qk_logits(captured_q,captured_k,query_indices)
                block_positive=[(i,keys) for i,keys in enumerate(positives[start:stop])]
                weight=torch.as_tensor(weights[start:stop],device=logits.device)
                bias=captured_bias
                if bias is not None and bias.ndim in (3,4): bias=bias[:,:,query_indices,:] if bias.ndim==4 else bias[:,query_indices,:].unsqueeze(1)
                block=trainable.correspondence(logits,None,weight,multi_positive=block_positive,additive_bias=bias)
                numerator=numerator+block*weight.sum(); denominator=denominator+weight.sum()
            layer_loss=numerator/denominator.clamp_min(1e-8)
        else:
            additive_bias=captured_bias
            if additive_bias is not None and additive_bias.ndim in (3,4):
                additive_bias=additive_bias[:,:,plan.query_indices,:] if additive_bias.ndim==4 else additive_bias[:,plan.query_indices,:].unsqueeze(1)
            layer_loss=trainable.correspondence_streaming(captured_q,captured_k,plan,additive_bias=additive_bias)
        if not layer_loss.requires_grad or not captured_k.requires_grad: raise RuntimeError('correspondence Q/K lost autograd')
        if vram_callback is not None:
            captured_q.register_hook(lambda grad,callback=vram_callback: (callback('correspondence_backward'),grad)[1])
        losses.append(layer_loss)
        # The autograd node is now the sole owner of Q/K and compact plan state.
        # Do not pin nine full K tensors through processor diagnostics/finalize.
        processor.last_q=None; processor.last_k=None; processor.last_attention_bias=None
    if timings is not None: timings['correspondence_loss_seconds']+=time.perf_counter()-loss_started
    if not losses: return torch.zeros((),device=first.ray_provider.context['c2w'].device)
    return torch.stack(losses).mean()

def _reset_sequence(runner):
    runner.reset_sequence()

def _install_oom_profiler(output,device,rank,state):
    """Flush useful CUDA state even when a step dies before normal metrics."""
    original=sys.excepthook
    def hook(exc_type,exc_value,traceback):
        if isinstance(exc_value,torch.cuda.OutOfMemoryError):
            payload=dict(state)
            payload.update(rank=int(rank),timestamp=time.time(),error=str(exc_value),
                memory_allocated=int(torch.cuda.memory_allocated(device)),
                memory_reserved=int(torch.cuda.memory_reserved(device)),
                max_memory_allocated=int(torch.cuda.max_memory_allocated(device)))
            path=Path(output)/'oom_profile.jsonl'; path.parent.mkdir(parents=True,exist_ok=True)
            with path.open('a',buffering=1) as handle:
                handle.write(json.dumps(payload)+'\n'); handle.flush()
        original(exc_type,exc_value,traceback)
    sys.excepthook=hook

def main():
    p=argparse.ArgumentParser(); p.add_argument('--config',default='configs/sightline.yaml'); p.add_argument('--model',required=True); p.add_argument('--model-revision'); p.add_argument('--helios-root',required=True); p.add_argument('--manifest',required=True); p.add_argument('--p3-manifest')
    p.add_argument('--expected-records',type=int); p.add_argument('--max-steps',type=int); p.add_argument('--resume'); p.add_argument('--allow-memory-layer-migration',action='store_true'); p.add_argument('--allow-world-size-migration',action='store_true'); p.add_argument('--output-dir',required=True); p.add_argument('--save-every',type=int); p.add_argument('--latent-cache-root')
    p.add_argument('--prompt',default='A stable realistic view of the same scene.'); p.add_argument('--probe-only',action='store_true'); p.add_argument('--probe-checkpoint'); p.add_argument('--probe-layers',default=''); p.add_argument('--probe-capture'); p.add_argument('--probe-step',type=int,default=1000); p.add_argument('--alpha-zero-baseline',action='store_true'); p.add_argument('--record-index',type=int); p.add_argument('--train-chunk',type=int); p.add_argument('--checkpoint-smoke-step',type=int); p.add_argument('--smoke-max-chunks',type=int); p.add_argument('--smoke-max-chunks-sequence'); p.add_argument('--profile-timing',action='store_true'); p.add_argument('--train',action='store_true'); args=p.parse_args()
    if not (args.train or args.probe_only) or args.train==args.probe_only: raise ValueError('select exactly one of --train or --probe-only')
    cfg=load_sightline_config(args.config); total_steps=cfg.p1_steps+cfg.p2_steps+cfg.p3_steps
    args.max_steps=args.max_steps or total_steps
    save_every=args.save_every or cfg.checkpoint_every
    if args.train and args.save_every is not None and args.save_every not in (60,100): raise ValueError('formal checkpoint cadence only permits 100 before step 1000 or 60 afterward')
    rank,world_size,device=_distributed_context()
    if world_size>1 and not args.train: raise ValueError('DDP is supported only for training')
    # An explicit CLI opt-in is required when resuming with a different DDP
    # world size. restore_runtime_checkpoint then rejects shared RNG state and
    # deterministically reseeds each new rank after state restoration.
    world_size_migration=bool(args.allow_world_size_migration)
    smoke_curriculum_override=(args.profile_timing and os.environ.get('SIGHTLINE_SMOKE_ALLOW_CURRICULUM_OVERRIDE')=='1')
    smoke_chunk_sequence=tuple(int(value) for value in args.smoke_max_chunks_sequence.split(',')) if args.smoke_max_chunks_sequence else ()
    if args.smoke_max_chunks is not None and smoke_chunk_sequence:
        raise ValueError('select only one smoke curriculum override mode')
    if (args.smoke_max_chunks is not None or smoke_chunk_sequence) and (not smoke_curriculum_override or
            (args.smoke_max_chunks is not None and args.smoke_max_chunks not in (4,5,6)) or
            any(value not in (4,5,6) for value in smoke_chunk_sequence)):
        raise ValueError('--smoke-max-chunks=4/5/6 requires profile timing and the explicit smoke-only environment switch')
    if smoke_chunk_sequence and args.train_chunk!=-1:
        raise ValueError('smoke chunk sequences require --train-chunk=-1 to select each last chunk')
    if args.train and world_size!=cfg.ddp_world_size and not world_size_migration: raise ValueError(f'formal training requires exactly {cfg.ddp_world_size} DDP ranks; pass --allow-world-size-migration for an explicit deterministic resume, got {world_size}')
    probe_layers=tuple(int(x) for x in args.probe_layers.split(',') if x); _preflight(cfg,args,probe_layers); records=load_rgbd_memory_manifest(args.manifest,expected_count=args.expected_records)
    p3_records=load_rgbd_memory_manifest(args.p3_manifest) if args.p3_manifest else records
    if cfg.chunk_count!=3 or cfg.chunk_length!=33 or cfg.chunk_stride!=32 or (cfg.source_height,cfg.source_width)!=(480,832): raise ValueError('formal RGB-D training requires 3 chunks, 97 frames, and 480x832 geometry')
    sys.path.insert(0,args.helios_root)
    source_file=Path(args.helios_root)/'helios/diffusers_version/transformer_helios_diffusers.py'
    runtime_patch=_install_memory_efficient_helios_norm(source_file) if rank==0 else None
    if world_size>1: dist.barrier()
    if rank!=0: runtime_patch=_install_memory_efficient_helios_norm(source_file)
    fingerprint=runtime_patch['original_source_sha256']
    from helios.diffusers_version.pipeline_helios_diffusers import HeliosPipeline
    import helios.diffusers_version.transformer_helios_diffusers as helios_source
    pipe=HeliosPipeline.from_pretrained(args.model,torch_dtype=torch.bfloat16,revision=args.model_revision).to(device); heads=int(pipe.transformer.config.num_attention_heads); inner=int(pipe.transformer.config.attention_head_dim*heads)
    pipe.text_encoder.eval().requires_grad_(False); pipe.vae.eval().requires_grad_(False)
    set_initialization_seed()
    trainable=SightlineTrainable(inner,layers=cfg.sightline_layers,heads=heads,
        lambda_corr=cfg.lambda_corr,lambda_corr_final=cfg.lambda_corr_final,
        lambda_corr_decay_start=cfg.lambda_corr_decay_start).to(device,dtype=torch.float32)
    for parameter in pipe.transformer.parameters(): parameter.requires_grad_(False)
    install_lora(pipe.transformer,cfg.lora_layers,rank=cfg.lora_rank) if cfg.lora_layers else None
    padded_h,padded_w=padded_size(cfg.source_height,cfg.source_width)
    provider=SightlineRayProvider(source_height=padded_h,source_width=padded_w); runner=SightlinePipeline(pipe,config=cfg,conditioner=trainable.conditioner,ray_provider=provider); pipe._sightline_pipeline=runner
    runner.memory.to(device=device,dtype=torch.bfloat16)
    installed_layers=tuple(sorted(set(cfg.sightline_layers).union(cfg.memory_layers).union(cfg.correspondence_layers).union(probe_layers))) if args.probe_only else tuple(sorted(set(cfg.sightline_layers).union(cfg.memory_layers).union(cfg.correspondence_layers)))
    install_sightline_attention(pipe.transformer,trainable.conditioner,provider,layers=installed_layers,helios_module=helios_source,memory=runner.memory,memory_layers=cfg.memory_layers)
    initialization_hash=broadcast_and_assert_trainables(trainable,runner.memory,pipe.transformer,world_size)
    lora_params=[p for n,p in pipe.transformer.named_parameters() if 'lora_' in n]
    memory_params=list(runner.memory.parameters())
    optimizer=torch.optim.AdamW([{'params':list(trainable.parameters()),'lr':cfg.learning_rate},{'params':lora_params,'lr':cfg.lora_learning_rate},{'params':memory_params,'lr':cfg.learning_rate}],weight_decay=.01)
    _assert_optimizer_scope(optimizer,trainable,runner.memory,pipe.transformer,pipe.text_encoder,pipe.vae)
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda step:_lr_multiplier(step,total_steps))
    prompt_embeds,_=_prompt(pipe,args.prompt,device)
    # T5 is only needed for the fixed prompt above.  Keep its CPU module for
    # checkpoint/optimizer scope validation, but release its CUDA weights
    # before the first training forward.
    pipe.text_encoder.to('cpu'); pipe.text_encoder.eval()
    # Training only calls VAE.encode() for source conditioning. Decoder
    # weights otherwise consume ~140 MiB per rank throughout P3 backward.
    # Move them once; there is no per-step CPU/GPU transfer.
    vae_decode_bytes=_offload_unused_vae_decode_path(pipe.vae)
    if rank==0: print(f'offloaded unused VAE decode path: {vae_decode_bytes/2**20:.2f} MiB',flush=True)
    if device.type == 'cuda': torch.cuda.empty_cache()
    config=asdict(cfg); memory_config={'layers':list(cfg.memory_layers),'pool':cfg.memory_pool,'budget':cfg.memory_budget,'tau_pos':cfg.memory_tau_pos,'tau_angle':cfg.memory_tau_angle}; provenance=runtime_provenance(pipe,args.model,args.helios_root,model_revision=args.model_revision,transformer_source_sha256=fingerprint,runtime_patch=runtime_patch)
    trainable.eval() if args.probe_only else trainable.train()
    start_step=args.probe_step if args.probe_only else 0
    world_size_migrated=False
    if args.resume:
        payload=torch.load(args.resume,map_location='cpu'); world_size_migrated=int(payload.get('rng_world_size',-1))!=world_size
        completed_step=restore_runtime_checkpoint(payload,trainable,runner.memory,pipe.transformer,config=config,helios_fingerprint=fingerprint,layers=cfg.sightline_layers,memory_config=memory_config,optimizer=optimizer,scheduler=scheduler,restore_rng=True,provenance=provenance,rank=rank,world_size=world_size,allow_memory_layer_migration=args.allow_memory_layer_migration,allow_world_size_migration=args.allow_world_size_migration); start_step=completed_step+1
        if world_size_migrated:
            seed=set_rank_runtime_seed(rank,start_step)
            if rank==0: print(f'checkpoint world size migration: deterministic per-rank reseed at step {start_step}, rank0 seed {seed}',flush=True)
    elif args.probe_checkpoint:
        if not args.probe_only: raise ValueError('--probe-checkpoint is only valid with --probe-only')
        payload=torch.load(args.probe_checkpoint,map_location='cpu'); restored_step=restore_runtime_checkpoint(payload,trainable,runner.memory,pipe.transformer,config=config,helios_fingerprint=fingerprint,layers=cfg.sightline_layers,memory_config=memory_config,restore_rng=False,provenance=provenance); start_step=restored_step
    if args.alpha_zero_baseline:
        configure_alpha_zero_baseline(trainable,runner.memory,pipe.transformer)
    if args.resume:
        initialization_hash=broadcast_and_assert_trainables(trainable,runner.memory,pipe.transformer,world_size)
    else:
        set_rank_runtime_seed(rank,start_step)
    output=Path(args.output_dir); output.mkdir(parents=True,exist_ok=True); metrics=output/'metrics.jsonl'
    oom_state={'stage':'initialization','step':int(start_step),'train_chunk':None,'memory_token_count':0,'k_length':0,'selected_q_count':0}
    _install_oom_profiler(output,device,rank,oom_state)
    stop=args.max_steps if args.train else min(args.max_steps,start_step+1)
    for step in range(start_step,stop):
        oom_state.update(stage='step_setup',step=int(step),train_chunk=None,memory_token_count=0,k_length=0,selected_q_count=0)
        if args.profile_timing: torch.cuda.reset_peak_memory_stats(device)
        phase=curriculum_phase(step,p1_steps=cfg.p1_steps,p2_steps=cfg.p2_steps,p3_steps=cfg.p3_steps)
        capture_geometry_diagnostics=((step+1) % 10 == 0)
        for processor in pipe.transformer._sightline_processors.values():
            processor.capture_numeric_diagnostics=capture_geometry_diagnostics
            processor.conditioner.capture_numeric_diagnostics=capture_geometry_diagnostics
        smoke_max_chunks=args.smoke_max_chunks
        if smoke_chunk_sequence:
            smoke_index=step-start_step
            if smoke_index>=len(smoke_chunk_sequence): raise ValueError('smoke chunk sequence is shorter than the requested run')
            smoke_max_chunks=smoke_chunk_sequence[smoke_index]
        if smoke_max_chunks is not None:
            if phase['name']!='P3': raise ValueError('smoke curriculum override is valid only in P3')
            phase={**phase,'max_chunks':int(smoke_max_chunks)}
        checkpointing=bool(cfg.gradient_checkpointing); _set_gradient_checkpointing(pipe.transformer,checkpointing)
        if args.alpha_zero_baseline: phase={**phase,'memory':False,'lora':False,'correspondence':False}
        phase_records=p3_records if phase['name']=='P3' else records
        eligible_records=[record for record in phase_records if record.chunk_count >= phase['max_chunks'] and (record.memory_eligible or not (phase['memory'] or phase['correspondence'] or bool(args.probe_capture)))]
        if not eligible_records: raise RuntimeError(f"{phase['name']} requires at least one memory-eligible RGB-D record")
        if args.record_index is not None:
            record=phase_records[args.record_index]
            if record not in eligible_records: raise ValueError(f"record {record.record_id} is camera-only and cannot be used in {phase['name']}")
        else:
            index=_ddp_record_index(step,rank,world_size,len(eligible_records)); record=eligible_records[index]
        latent_root=args.latent_cache_root or cfg.latent_cache_path or None
        latent_path=resolve_continuous_latent_cache(record,cache_root=latent_root); latent_schema,_=validate_latent_cache(latent_path)
        validate_rgbd_record_latent(record,latent_path)
        if phase['name'] in ('P1a','P1b','P2'):
            if record.frame_count!=97 or record.chunk_count!=3 or latent_schema!='continuous_25': raise ValueError(f'{record.record_id}: P1/P2 requires a unit-owned 97-frame continuous_25 cache')
        if args.train and latent_schema=='overlap_chunks_6x9': require_overlap_validation(latent_path,expected_provenance=str(provenance['model_identity']))
        all_latents=load_latent_tensor(latent_path)
        assert_latent_geometry(all_latents,height=cfg.source_height,width=cfg.source_width,patch_size=pipe.transformer.config.patch_size)
        required_latents=1+8*record.chunk_count
        if all_latents.shape[2] < required_latents: raise ValueError(f'{record.record_id}: latent cache is shorter than record geometry')
        window_start=0
        latent_start=window_start*8; frame_start=window_start*32
        rgb_paths=record.rgb_paths()
        if len(rgb_paths)<=frame_start: raise RuntimeError('trajectory RGB frames do not cover sampled chunk window')
        oom_state['stage']='source_condition_vae'
        source,fake,_,_=prepare_source_condition(pipe,Image.open(rgb_paths[frame_start]).convert('RGB'),height=cfg.source_height,width=cfg.source_width,device=device)
        c2w_np,K_np=record.load_cameras(); c2w_np=np.array(c2w_np[frame_start:frame_start+1+phase['max_chunks']*32],copy=True); K_np=np.array(K_np[frame_start:frame_start+1+phase['max_chunks']*32],copy=True)
        near_depth=float(record.near_depth)
        c2w=torch.from_numpy(c2w_np).to(device,dtype=torch.float32).unsqueeze(0); c2w=canonicalize_c2w(c2w,near_depth)
        K=torch.from_numpy(K_np).to(device,dtype=torch.float32).unsqueeze(0)
        _reset_sequence(runner); runner._trajectory_c2w=c2w; runner._trajectory_K=K; runner._source_camera=c2w[:,0]; runner._source_intrinsics=K[:,0]; runner.memory.set_enabled(phase['memory'])
        for name,parameter in pipe.transformer.named_parameters():
            if 'lora_' in name: parameter.requires_grad_(phase['lora'] and not args.alpha_zero_baseline)
        set_lora_enabled(pipe.transformer,phase['lora'] and not args.alpha_zero_baseline)
        active_corr_layers=probe_layers or tuple(cfg.correspondence_layers); diagnostic_correspondence=bool(args.probe_capture)
        if phase['correspondence'] or diagnostic_correspondence:
            if not record.memory_eligible: raise RuntimeError(f"{record.record_id} has no calibrated RGB-D correspondence supervision")
            if not active_corr_layers or any(layer not in pipe.transformer._sightline_processors for layer in active_corr_layers): raise RuntimeError('active correspondence/probe layers are not installed Sightline layers')
            for layer in active_corr_layers:
                pipe.transformer._sightline_processors[layer].capture_diagnostics=False
                pipe.transformer._sightline_processors[layer].capture_query_indices=None
        # Keep direct source supervision throughout P3.  The cadence is
        # aligned to each requested global-step interval: every 2 P3 steps
        # through step 1299, every 3 steps through 1699, then every 4 steps.
        scheduled_chunk0=(step>=500
                          and p3_chunk0_scheduled(step,cfg.p1_steps,cfg.p2_steps)
                          and not smoke_chunk_sequence
                          and (args.train_chunk is None or args.train_chunk<0))
        minimum_train_chunk=0 if (phase['max_chunks']==1 or scheduled_chunk0) else 1
        forced_train_chunk=(phase['max_chunks']-1 if smoke_chunk_sequence
                            else (0 if scheduled_chunk0 else args.train_chunk))
        train_chunk=_ddp_train_chunk(phase['max_chunks'],minimum_train_chunk,rank,world_size,device,forced_train_chunk)
        oom_state['train_chunk']=int(train_chunk)
        if not minimum_train_chunk<=train_chunk<phase['max_chunks']: raise ValueError('train_chunk outside curriculum or lacks required real past history')
        # Prefix rollout never reads GT latents. Keep only the selected
        # backward chunk on CUDA instead of retaining the whole curriculum
        # window for the duration of backward.
        target_latents=all_latents[:,:,latent_start+train_chunk*8:latent_start+train_chunk*8+9].to(device,dtype=torch.bfloat16)
        perf={'prefix_generation_seconds':0.0,'memory_clean_forward_seconds':0.0,'memory_archive_write_seconds':0.0,'correspondence_load_seconds':0.0,'correspondence_mapping_seconds':0.0,'correspondence_loss_seconds':0.0,'backward_seconds':0.0}
        def timing_sync():
            if args.profile_timing: torch.cuda.synchronize(device)
        def record_vram(name):
            if args.profile_timing:
                perf[f'{name}_memory_allocated']=int(torch.cuda.memory_allocated(device))
                perf[f'{name}_max_memory_allocated']=int(torch.cuda.max_memory_allocated(device))
                perf[f'{name}_memory_reserved']=int(torch.cuda.memory_reserved(device))
        if phase['correspondence'] or diagnostic_correspondence:
            load_started=time.perf_counter(); corr_rows=_load_correspondence(record,train_chunk); perf['correspondence_load_seconds']=time.perf_counter()-load_started
        else: corr_rows=None
        sigma_range,sigma_band=_sigma_band(step,phase['name'])
        history_state=NativeHistoryState(source,fake); generated_prefix=[]; losses={}; probe_payload={}; optimizer.zero_grad(set_to_none=True)
        started=time.perf_counter()
        def forward_chunk(chunk,keep_graph):
            chunk_started=time.perf_counter()
            history=history_state.groups(); coverage=history_state.coverage()
            if not keep_graph:
                oom_state['stage']='prefix_rollout'
                template=torch.empty((source.shape[0],source.shape[1],9,*source.shape[-2:]),device=source.device,dtype=source.dtype); runner._prepare_chunk(chunk,template,{},history_global_coverages=coverage,history_validity=history_state.validity())
                clean_boundary=None if chunk==0 else generated_prefix[-1][:,:,-1:]
                generated=_generate_detached_chunk(pipe,source,history,prompt_embeds,cfg,chunk,clean_boundary).detach()
                record_vram('prefix_rollout')
            else:
                target=target_latents  # The sole non-source GT read in this step.
                clean_boundary=None if chunk==0 else generated_prefix[-1][:,:,-1:].detach()
                if clean_boundary is not None: target[:,:,:1]=clean_boundary.to(target)
                items=exact_flow_matching_items(pipe,target,stage_steps=cfg.pyramid_steps,device=target.device,sigma_range=sigma_range)
                if clean_boundary is not None: items=constrain_flow_items(items,clean_boundary)
                runner._prepare_chunk(chunk,target,{},history_global_coverages=coverage,history_validity=history_state.validity())
                oom_state['stage']='active_memory'
                if active_corr_layers:
                    active=pipe.transformer._sightline_processors[active_corr_layers[0]].memory
                    oom_state['memory_token_count']=len(active.active_identity_metadata()) if active is not None and active.enabled else 0
                record_vram('active_memory')
                correspondence_plan=None
                correspondence_seed=int(hashlib.sha256(f'{step}:{record.trajectory_id}'.encode()).hexdigest()[:16],16)
                if corr_rows is not None and len(corr_rows) and not args.alpha_zero_baseline:
                    final_shape=runner.ray_provider.context['stage_shapes'][-1]
                    current_length=int(final_shape[0]*final_shape[1]*final_shape[2])
                    correspondence_plan=_build_correspondence_plan(pipe.transformer._sightline_processors[active_corr_layers[0]],corr_rows,chunk,current_length,cfg.correspondence_rows_per_batch,correspondence_seed)
                    oom_state['k_length']=len(correspondence_plan.identities)
                    oom_state['selected_q_count']=int(correspondence_plan.query_indices.numel())
                stage_losses=[]; final_prediction=None; fm_sigma_trace=[]
                for stage_index,item in enumerate(items):
                    capture_correspondence=correspondence_capture_for_stage(stage_index,len(items),phase['correspondence'] or diagnostic_correspondence)
                    for layer in active_corr_layers:
                        processor=pipe.transformer._sightline_processors[layer]
                        processor.capture_diagnostics=capture_correspondence
                        processor.capture_query_indices=correspondence_plan.query_indices if capture_correspondence and correspondence_plan is not None and args.train else None
                    oom_state['stage']='final_stage_forward' if capture_correspondence else f'flow_stage_{stage_index}_forward'
                    prediction=_model_prediction(pipe,item['noisy_latents'],item,prompt_embeds,history,chunk*8); final_prediction=prediction
                    if capture_geometry_diagnostics:
                        first_processor=pipe.transformer._sightline_processors[int(cfg.sightline_layers[0])]
                        diagnostic=first_processor.last_numeric_diagnostics or {}
                        fm_sigma_trace.append({'stage':stage_index,'item_sigma':float(item['sigmas'].detach().float().mean()),'processor_sigma':diagnostic.get('geometry_sigma'),'g_sigma':diagnostic.get('geometry_sigma_gain')})
                    if capture_correspondence: record_vram('final_stage_forward')
                    stage_loss=(prediction.float()-item['target'].float()).square().mean(); stage_losses.append(stage_loss)
                    if args.train and stage_index+1<len(items):
                        timing_sync(); backward_started=time.perf_counter(); (stage_loss/len(items)).backward(); timing_sync(); perf['backward_seconds']+=time.perf_counter()-backward_started
                        # The early-stage backward is complete. Retain only its
                        # scalar metric and sigma; its flow tensors otherwise
                        # remain referenced by ``items`` during the much larger
                        # final-stage/correspondence backward.
                        stage_losses[-1]=stage_loss.detach()
                        for disposable in ('noisy_latents','target','start_point','end_point','noise','timesteps'):
                            item.pop(disposable,None)
                fm=torch.stack([loss.detach() if args.train else loss for loss in stage_losses]).mean()
                corr_weight=trainable.lambda_corr(step/total_steps)
                oom_state['stage']='correspondence_forward'
                corr_metric=_corr_loss(trainable,pipe.transformer._sightline_processors,corr_rows,chunk,active_corr_layers,cfg.correspondence_rows_per_batch,sampling_seed=correspondence_seed,timings=perf,plan=correspondence_plan if args.train else None,vram_callback=record_vram if args.profile_timing else None) if (corr_rows is not None and len(corr_rows) and not args.alpha_zero_baseline) else fm.new_zeros(())
                record_vram('correspondence_loss')
                corr=corr_metric if (phase['correspondence'] and train_chunk>0) else fm.new_zeros(())
                final_flow=stage_losses[-1]/len(items)
                total=final_flow+corr_weight*corr if args.train else fm+corr_weight*corr
                losses.update(fm=fm,corr=corr,total=total,stage=stage_losses,sigmas=[float(item['sigmas'].mean()) for item in items])
                if args.train:
                    oom_state['stage']='correspondence_and_fm_backward'
                    timing_sync(); backward_started=time.perf_counter()
                    # Geometry alpha controls injection strength and is deliberately
                    # FM-only: correspondence trains geometric features but cannot
                    # lower its loss by merely amplifying alpha.
                    if corr.requires_grad:
                        final_flow.backward(retain_graph=True)
                        flow_alpha_grads={id(alpha):None if alpha.grad is None else alpha.grad.detach().clone() for alpha in trainable.conditioner.alpha_parameters()}
                        (corr_weight*corr).backward()
                        for alpha in trainable.conditioner.alpha_parameters(): alpha.grad=flow_alpha_grads[id(alpha)]
                    else:
                        final_flow.backward()
                    timing_sync(); perf['backward_seconds']+=time.perf_counter()-backward_started
                    record_vram('backward')
                    record_vram('fm_backward')
                final=items[-1]; generated=(final['noisy_latents']-final['sigmas']*final_prediction).detach()
                if args.probe_capture:
                    layer_captures=[]
                    for layer in active_corr_layers:
                        processor=pipe.transformer._sightline_processors[layer]
                        try: selected,positives,_,_=_mapped_correspondences(processor,corr_rows,chunk)
                        except RuntimeError as exc:
                            if 'do not map' in str(exc): continue
                            raise
                        selected=selected[:cfg.correspondence_rows_per_batch]; positives=positives[:len(selected)]
                        selected_q=processor.last_q[:,selected]; base_k=processor.last_k
                        head_logits=torch.einsum('bqhd,bkhd->bhqk',selected_q,base_k)*(selected_q.shape[-1]**-.5)
                        layer_captures.append({'layer':layer,'attention_logits':head_logits.detach().cpu(),'positive_key_indices':positives,'memory_count':processor.last_attention_meta.get('memory_tokens',0)})
                    if not layer_captures: raise RuntimeError('probe candidates have no mapped correspondence rows')
                    layer=layer_captures[0]['layer']; processor=pipe.transformer._sightline_processors[layer]
                    base_context=dict(provider.context); normal_step_time=time.perf_counter()-started; ablation_started=time.perf_counter()
                    memory_enabled_by_layer={layer:bank.enabled for layer,bank in runner.memory.banks.items()}
                    with torch.no_grad():
                        provider.context=dict(base_context); provider.context['c2w']=base_context['c2w'].flip(1)
                        wrong=_model_prediction(pipe,final['noisy_latents'],final,prompt_embeds,history,chunk*8)
                        runner.memory.set_enabled(False); provider.context=base_context
                        zero=_model_prediction(pipe,final['noisy_latents'],final,prompt_embeds,history,chunk*8)
                        for bank_layer,enabled in memory_enabled_by_layer.items(): runner.memory.banks[bank_layer].enabled=enabled
                        originals={layer:{chunk_id:chunk.hidden for chunk_id,chunk in bank.archive.items()} for layer,bank in runner.memory.banks.items()}
                        for bank in runner.memory.banks.values():
                            chunks=list(bank.archive.values())
                            if chunks:
                                shuffled=torch.cat([chunk.hidden for chunk in chunks],1).flip(1)
                                offset=0
                                for memory_chunk in chunks:
                                    count=memory_chunk.token_count; memory_chunk.hidden=shuffled[:,offset:offset+count].contiguous(); offset+=count
                        shuffled_prediction=_model_prediction(pipe,final['noisy_latents'],final,prompt_embeds,history,chunk*8)
                        for bank_layer,hiddens in originals.items():
                            for chunk_id,hidden in hiddens.items(): runner.memory.banks[bank_layer].archive[chunk_id].hidden=hidden
                        provider.context=base_context
                    alpha_q,alpha_k=trainable.conditioner.alpha_values()
                    final_stage_loss=float((final_prediction.float()-final['target'].float()).square().mean())
                    first=layer_captures[0]
                    probe_payload.update(source='real_helios_forward',baseline=bool(args.alpha_zero_baseline),layer=first['layer'],sigma=float(final['sigmas'].mean()),attention_logits=first['attention_logits'],positive_key_indices=first['positive_key_indices'],memory_count=first['memory_count'],layer_captures=layer_captures,fm_loss=float(fm.detach()),baseline_final_stage_loss=final_stage_loss,wrong_ray_loss=float((wrong.float()-final['target'].float()).square().mean()),memory_zero_loss=float((zero.float()-final['target'].float()).square().mean()),memory_shuffle_loss=float((shuffled_prediction.float()-final['target'].float()).square().mean()),corr_loss=float(corr_metric.detach()),alpha_q=alpha_q,alpha_k=alpha_k,vram_gb=float(torch.cuda.max_memory_allocated()/2**30),step_time_sec=normal_step_time,ablation_time_sec=time.perf_counter()-ablation_started)
            for layer in active_corr_layers:
                pipe.transformer._sightline_processors[layer].capture_diagnostics=False
                pipe.transformer._sightline_processors[layer].capture_query_indices=None
            if chunk==0: generated[:,:,0:1]=source
            else: generated[:,:,0:1]=clean_boundary.to(generated)
            capture_history=history
            def clean_capture(clean_input,timestep, _history=capture_history, _chunk=chunk):
                return _transformer_forward(pipe,clean_input,timestep,prompt_embeds,_history,_chunk*8)
            generated_prefix.append(generated.detach())
            history_state.append_chunk(generated,chunk)
            capture_memory=phase['name']!='P3' or prefix_chunk_should_capture_memory(chunk,train_chunk)
            oom_state['stage']='clean_memory_capture'
            timing_sync(); memory_timings=runner._finalize_chunk(chunk,clean_latent=generated.detach(),capture_fn=clean_capture,capture_memory=capture_memory); timing_sync()
            if capture_memory: record_vram('clean_memory_capture')
            for name,value in memory_timings.items(): perf[name]+=value
            for processor in pipe.transformer._sightline_processors.values():
                processor.last_q=processor.last_k=processor.last_hidden_states=processor.last_key_identities=None
                processor.last_attention_bias=None
            if not keep_graph: perf['prefix_generation_seconds']+=time.perf_counter()-chunk_started
            return generated
        if phase['name']=='P3':
            _,policies=run_causal_prefix_chunks(phase['max_chunks'],train_chunk,forward_chunk)
        elif args.probe_only and args.alpha_zero_baseline:
            with torch.no_grad(): _,policies=run_single_graph_chunks(phase['max_chunks'],train_chunk,forward_chunk)
        else:
            _,policies=run_single_graph_chunks(phase['max_chunks'],train_chunk,forward_chunk)
        if args.probe_capture:
            probe_payload['alpha_grad']={name:0.0 if alpha.grad is None else float(alpha.grad.detach().abs()) for name,alpha in ((f'{index}.q',layer.alpha_q) for index,layer in trainable.conditioner.layers.items())}
        if args.train:
            active_phase=curriculum_phase(step,p1_steps=cfg.p1_steps,p2_steps=cfg.p2_steps,p3_steps=cfg.p3_steps)
            alpha_grads=[alpha.grad for alpha in trainable.conditioner.alpha_parameters()]
            if active_phase['name'] in ('P1a','P1b') and (any(grad is None for grad in alpha_grads) or not all(torch.isfinite(grad).all() for grad in alpha_grads)): raise RuntimeError('P1 alpha gradient missing or non-finite')
            if active_phase['name']=='P2':
                lora_grads=[p.grad for p in lora_params if p.requires_grad and p.grad is not None]
                if not lora_grads or not all(torch.isfinite(g).all() for g in lora_grads): raise RuntimeError('P2 LoRA gradient missing or non-finite')
            if active_phase['name']=='P3' and losses['corr'].requires_grad:
                if any(alpha.abs().detach()>1e-6 for alpha in trainable.conditioner.alpha_parameters()):
                    geometry_params=list(trainable.conditioner.geometry_parameters())
                    corr_grads=[p.grad for p in geometry_params if p.grad is not None]
                    if not corr_grads or not all(torch.isfinite(g).all() for g in corr_grads): raise RuntimeError('P3 geometry gradient missing or non-finite')
            oom_state['stage']='ddp_gradient_average'; record_vram('pre_ddp')
            optimized=[p for group in optimizer.param_groups for p in group['params']]; _average_gradients(optimized,world_size); grad_norm=torch.nn.utils.clip_grad_norm_([p for p in optimized if p.grad is not None],cfg.grad_clip); optimizer.step(); scheduler.step()
        else: grad_norm=torch.tensor(0.)
        alpha_q,alpha_k=trainable.conditioner.alpha_values()
        step_seconds=time.perf_counter()-started
        diagnostics={str(layer):processor.last_numeric_diagnostics for layer,processor in pipe.transformer._sightline_processors.items() if processor.last_numeric_diagnostics is not None} if (step+1)%10==0 else {}
        row={'step':step,'record':record.trajectory_id,'phase':phase['name'],'max_chunks':phase['max_chunks'],'window_start_chunk':window_start,'train_chunk':train_chunk,'executed_chunks':len(policies),'policies':policies,'flow_loss':float(losses['fm'].detach()),'corr_loss':float(losses['corr'].detach()),'stage_losses':[float(x.detach()) for x in losses['stage']],'stage_sigmas':losses['sigmas'],'sampled_sigma':losses['sigmas'],'fm_sigma_trace':fm_sigma_trace if capture_geometry_diagnostics else [],'sigma_band':sigma_band,'alpha_q':alpha_q,'alpha_k':alpha_k,'geometry_diagnostics':diagnostics,'initialization_hash':initialization_hash,'grad_norm':float(grad_norm),'lr':scheduler.get_last_lr()[0],'gradient_checkpointing':checkpointing,'helios_runtime_patch':runtime_patch,'seconds':step_seconds,'step_total_seconds':step_seconds,**perf,'timing_synchronized':bool(args.profile_timing),'uses_future_gt':False}
        if rank==0 and (args.profile_timing or (step+1)%10==0 or step==start_step or step+1==stop):
            with metrics.open('a') as handle: handle.write(json.dumps(row)+'\n')
        if args.probe_capture:
            Path(args.probe_capture).parent.mkdir(parents=True,exist_ok=True); torch.save(probe_payload,args.probe_capture)
        cadence=save_every if args.save_every is not None else checkpoint_interval(step)
        checkpoint_due=(step+1)%cadence==0 or step+1==args.max_steps or (args.checkpoint_smoke_step is not None and step+1==args.checkpoint_smoke_step)
        if args.train and checkpoint_due:
            rng_states=gather_rank_rng_states(world_size,device)
            if rank==0: save_runtime_checkpoint(output/f'checkpoint-{step:06d}.pt',trainable,runner.memory,pipe.transformer,optimizer,scheduler,step,config=config,helios_fingerprint=fingerprint,layers=cfg.sightline_layers,memory_config=memory_config,provenance=provenance,rng_states=rng_states,world_size=world_size,runtime_patch=runtime_patch)
            if world_size>1: dist.barrier()
    if world_size>1: dist.destroy_process_group()

if __name__=='__main__': main()
