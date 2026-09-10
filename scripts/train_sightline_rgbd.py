"""Causal Sightline training from the canonical RGB-D manifest.

The historical filename is retained as a legacy compatibility entry point.
Formal training data is loaded exclusively through RGBDMemoryRecord.
"""
from __future__ import annotations
import argparse, copy, hashlib, json, os, random, sys, time
from contextlib import nullcontext
# Must be set before importing/initializing CUDA; callers may override it.
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
os.environ.setdefault('PYTORCH_ALLOC_CONF', 'expandable_segments:True')
from dataclasses import asdict, replace
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from PIL import Image
from long_video.config import load_sightline_config
from long_video.training.flow_matching_exact import exact_flow_matching_items
from long_video.training.sightline import CorrespondencePlan, SightlineTrainable, install_lora, curriculum_phase, gt_prefix_probability, select_train_chunk, run_single_graph_chunks, run_causal_prefix_chunks, selected_qk_logits, set_initialization_seed, set_rank_runtime_seed, broadcast_and_assert_trainables, configure_geometry_zero_baseline, set_lora_enabled, prefix_chunk_should_capture_memory, correspondence_capture_for_stage
from long_video.training.rgbd_plan import RGBDSoftTargetPlan, build_rgbd_soft_target_plan
from long_video.training.rgbd_probability import rgbd_probability_score, rgbd_probability_row_scores
from long_video.training.rgbd_memory_data import load_rgbd_memory_manifest
from long_video.training.sightline_data import load_latent_tensor, validate_latent_cache, require_overlap_validation, resolve_continuous_latent_cache, validate_rgbd_record_latent
from long_video.training.sightline_checkpoint import save_runtime_checkpoint, restore_runtime_checkpoint, runtime_provenance, gather_rank_rng_states
from long_video.sightline.helios_integration import SightlineRayProvider, install_sightline_attention
from long_video.sightline.rays import canonicalize_c2w, latent_camera_indices
from long_video.sightline.history import NativeHistoryState,native_helios_indices
from long_video.sightline.pipeline import SightlinePipeline, prepare_source_condition
from long_video.sightline.geometry import assert_latent_geometry, geometry_sigma_schedule, padded_size
from long_video.sightline.boundary import constrain_flow_items, stage2_sample_with_boundary

TOTAL_TRAINING_STEPS=2500
WARMUP_STEPS=100
FORMAL_MEMORY_LAYERS=(4,8,16,20,24,32,36)
LEGACY_999_MEMORY_LAYERS=(4,6,8,16,20,24,32,34,36)

HELIOS_RUNTIME_PATCH_VERSION='sightline-token-blocked-v2-prefix'
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
_PREFIX_SETUP_ORIGINAL='''        # 4. Transformer blocks
        logits_hidden = []
        hidden_states = hidden_states.contiguous()
        encoder_hidden_states = encoder_hidden_states.contiguous()
        rotary_emb = rotary_emb.contiguous()'''
_PREFIX_SETUP_PATCHED='''        prefix_stop_layer = None
        if attention_kwargs and attention_kwargs.get("sightline_prefix_stop_layer") is not None:
            prefix_stop_layer = int(attention_kwargs["sightline_prefix_stop_layer"])
            if not 0 <= prefix_stop_layer < len(self.blocks):
                raise ValueError(f"sightline prefix stop layer must be in [0, {len(self.blocks) - 1}]")

        # 4. Transformer blocks
        logits_hidden = []
        hidden_states = hidden_states.contiguous()'''
_PREFIX_EXEC_ORIGINAL='''        if torch.is_grad_enabled() and self.gradient_checkpointing:
            for iidx, block in enumerate(self.blocks):
                hidden_states = self._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    encoder_hidden_states,
                    timestep_proj,
                    rotary_emb,
                    navit_hidden_attention_mask,
                    navit_encoder_attention_mask,
                    original_context_length,
                    original_context_length_list,
                    is_first_denoising_step,
                    attention_kwargs,
                )
                if gan_mode and self.is_use_gan and self.is_use_gan_hooks and iidx in self.gan_hooks:
                    logits_hidden.append(hidden_states[:, -original_context_length:, :])
        else:
            for iidx, block in enumerate(self.blocks):
                hidden_states = block(
                    hidden_states,
                    encoder_hidden_states,
                    timestep_proj,
                    rotary_emb,
                    navit_hidden_attention_mask,
                    navit_encoder_attention_mask,
                    original_context_length,
                    original_context_length_list,
                    is_first_denoising_step,
                    attention_kwargs,
                )
                if gan_mode and self.is_use_gan and self.is_use_gan_hooks and iidx in self.gan_hooks:
                    logits_hidden.append(hidden_states[:, -original_context_length:, :])
'''
_PREFIX_EXEC_PATCHED='''        if torch.is_grad_enabled() and self.gradient_checkpointing:
            for block_index, block in enumerate(self.blocks):
                hidden_states = self._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    encoder_hidden_states,
                    timestep_proj,
                    rotary_emb,
                    navit_hidden_attention_mask,
                    navit_encoder_attention_mask,
                    original_context_length,
                    original_context_length_list,
                    is_first_denoising_step,
                    attention_kwargs,
                )
                if gan_mode and self.is_use_gan and self.is_use_gan_hooks and block_index in self.gan_hooks:
                    logits_hidden.append(hidden_states[:, -original_context_length:, :])
                if prefix_stop_layer is not None and block_index >= prefix_stop_layer:
                    return (None,) if not return_dict else Transformer2DModelOutput(sample=None)
        else:
            for block_index, block in enumerate(self.blocks):
                hidden_states = block(
                    hidden_states,
                    encoder_hidden_states,
                    timestep_proj,
                    rotary_emb,
                    navit_hidden_attention_mask,
                    navit_encoder_attention_mask,
                    original_context_length,
                    original_context_length_list,
                    is_first_denoising_step,
                    attention_kwargs,
                )
                if gan_mode and self.is_use_gan and self.is_use_gan_hooks and block_index in self.gan_hooks:
                    logits_hidden.append(hidden_states[:, -original_context_length:, :])
                if prefix_stop_layer is not None and block_index >= prefix_stop_layer:
                    return (None,) if not return_dict else Transformer2DModelOutput(sample=None)
'''

# The pinned training entry point currently imports the older diffusers-style
# Helios module.  Keep its audited block-loop shape as a second source variant
# so a prior v1 in-place runtime patch can be upgraded without changing the
# actual model implementation.
_PREFIX_SETUP_ORIGINAL_LEGACY='''        # 6. Transformer blocks
        hidden_states = hidden_states.contiguous()
        encoder_hidden_states = encoder_hidden_states.contiguous()
        rotary_emb = rotary_emb.contiguous()'''
_PREFIX_SETUP_PATCHED_LEGACY='''        prefix_stop_layer = None
        if attention_kwargs and attention_kwargs.get("sightline_prefix_stop_layer") is not None:
            prefix_stop_layer = int(attention_kwargs["sightline_prefix_stop_layer"])
            if not 0 <= prefix_stop_layer < len(self.blocks):
                raise ValueError(f"sightline prefix stop layer must be in [0, {len(self.blocks) - 1}]")

        # 6. Transformer blocks
        hidden_states = hidden_states.contiguous()
        encoder_hidden_states = encoder_hidden_states.contiguous()
        rotary_emb = rotary_emb.contiguous()'''
_PREFIX_EXEC_ORIGINAL_LEGACY='''        if torch.is_grad_enabled() and self.gradient_checkpointing:
            for block in self.blocks:
                hidden_states = self._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    encoder_hidden_states,
                    timestep_proj,
                    rotary_emb,
                    original_context_length,
                )
        else:
            for block in self.blocks:
                hidden_states = block(
                    hidden_states,
                    encoder_hidden_states,
                    timestep_proj,
                    rotary_emb,
                    original_context_length,
                )
'''
_PREFIX_EXEC_PATCHED_LEGACY='''        if torch.is_grad_enabled() and self.gradient_checkpointing:
            for block_index, block in enumerate(self.blocks):
                hidden_states = self._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    encoder_hidden_states,
                    timestep_proj,
                    rotary_emb,
                    original_context_length,
                )
                if prefix_stop_layer is not None and block_index >= prefix_stop_layer:
                    return (None,) if not return_dict else Transformer2DModelOutput(sample=None)
        else:
            for block_index, block in enumerate(self.blocks):
                hidden_states = block(
                    hidden_states,
                    encoder_hidden_states,
                    timestep_proj,
                    rotary_emb,
                    original_context_length,
                )
                if prefix_stop_layer is not None and block_index >= prefix_stop_layer:
                    return (None,) if not return_dict else Transformer2DModelOutput(sample=None)
'''
_PREFIX_VARIANTS=(
    (_PREFIX_SETUP_ORIGINAL,_PREFIX_SETUP_PATCHED,_PREFIX_EXEC_ORIGINAL,_PREFIX_EXEC_PATCHED),
    (_PREFIX_SETUP_ORIGINAL_LEGACY,_PREFIX_SETUP_PATCHED_LEGACY,_PREFIX_EXEC_ORIGINAL_LEGACY,_PREFIX_EXEC_PATCHED_LEGACY),
)

def _install_memory_efficient_helios_norm(source_file:Path):
    """Install and identify the audited token-blocked Helios runtime patch."""
    text=source_file.read_text()
    canonical=text.replace(_PATCH_IMPORT+'\n','')
    for patched,original in (
        (_NORM1_PATCHED,_NORM1_ORIGINAL),(_NORM1_EXPERIMENTAL_V1,_NORM1_ORIGINAL),(_NORM1_EXPERIMENTAL_V2,_NORM1_ORIGINAL),
        (_NORM2_PATCHED,_NORM2_ORIGINAL),(_NORM3_PATCHED,_NORM3_ORIGINAL),(_NORM3_EXPERIMENTAL_V1,_NORM3_ORIGINAL),
        (_NORM3_EXPERIMENTAL_V2,_NORM3_ORIGINAL),(_RESIDUAL1_PATCHED,_RESIDUAL1_ORIGINAL),(_RESIDUAL3_PATCHED,_RESIDUAL3_ORIGINAL),
    ): canonical=canonical.replace(patched,original)
    for setup_original,setup_patched,exec_original,exec_patched in _PREFIX_VARIANTS:
        canonical=canonical.replace(setup_patched,setup_original).replace(exec_patched,exec_original)
    prefix_variant=next((variant for variant in _PREFIX_VARIANTS if variant[0] in canonical and variant[2] in canonical),None)
    required=(_NORM1_ORIGINAL,_NORM2_ORIGINAL,_NORM3_ORIGINAL,_RESIDUAL1_ORIGINAL,_RESIDUAL3_ORIGINAL)
    if prefix_variant is None or any(value not in canonical for value in required): raise RuntimeError('pinned Helios source no longer matches the audited runtime patch')
    patched=canonical.replace(_IMPORT_ANCHOR,_IMPORT_ANCHOR+'\n'+_PATCH_IMPORT)
    for original,replacement in (
        (_NORM1_ORIGINAL,_NORM1_PATCHED),(_NORM2_ORIGINAL,_NORM2_PATCHED),(_NORM3_ORIGINAL,_NORM3_PATCHED),
        (_RESIDUAL1_ORIGINAL,_RESIDUAL1_PATCHED),(_RESIDUAL3_ORIGINAL,_RESIDUAL3_PATCHED),
        (prefix_variant[0],prefix_variant[1]),(prefix_variant[2],prefix_variant[3]),
    ): patched=patched.replace(original,replacement)
    if patched!=text: source_file.write_text(patched)
    return {
        'version':HELIOS_RUNTIME_PATCH_VERSION,
        'original_source_sha256':hashlib.sha256(canonical.encode()).hexdigest(),
        'runtime_source_sha256':hashlib.sha256(patched.encode()).hexdigest(),
    }

def checkpoint_interval(global_step: int) -> int:
    """Return the default formal checkpoint cadence."""
    return 100

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

def p3_chunk0_scheduled(step,p1_steps=500,p2_steps=500):
    """Rank-stable Bernoulli(0.25) source supervision for P3."""
    return random.Random(20260907+int(step)).random() < .25

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
    if tuple(cfg.sightline_layers) != (0,2,4,6,8,10,12):
        raise ValueError('formal training requires sightline_layers=[0,2,4,6,8,10,12]')
    if cfg.lora_layers or cfg.lora_scope!='disabled':
        raise ValueError('formal Sightline-v9 training requires LoRA disabled and lora_layers=[]')
    if total_steps!=TOTAL_TRAINING_STEPS or int(total_steps*cfg.warmup_ratio)!=WARMUP_STEPS: raise ValueError('formal schedule must preserve the configured 100/2500-step warmup')
    if args.train and args.max_steps>1000 and (not cfg.memory_layers or not cfg.correspondence_layers): raise ValueError('training reaches Memory/correspondence stages but layers are empty')
    if not 1<=args.max_steps<=total_steps: raise ValueError(f'--max-steps must be in 1..{total_steps}')

def _lr_multiplier(step,total_steps=TOTAL_TRAINING_STEPS):
    if step<WARMUP_STEPS: return float(step+1)/WARMUP_STEPS
    progress=min(1.0,max(0.0,(step-WARMUP_STEPS)/(total_steps-WARMUP_STEPS)))
    return 0.5*(1.0+__import__('math').cos(__import__('math').pi*progress))

def _sigma_band(step, phase):
    return (0.,1.),'uniform_0.0_1.0'

def _set_gradient_checkpointing(transformer,enabled):
    method=getattr(transformer,'enable_gradient_checkpointing' if enabled else 'disable_gradient_checkpointing',None)
    if method is None:
        if enabled: raise RuntimeError('gradient_checkpointing=true but pinned Helios exposes no enable method')
        return
    method()

def _assert_optimizer_scope(optimizer,trainable,memory,transformer,text_encoder,vae,helios_trainable):
    actual={id(parameter) for group in optimizer.param_groups for parameter in group['params']}
    expected={id(parameter) for parameter in trainable.parameters()}|{id(parameter) for parameter in memory.parameters()}|{id(parameter) for parameter in helios_trainable}
    forbidden={id(parameter) for module in (text_encoder,vae) for parameter in module.parameters()}|{id(parameter) for name,parameter in transformer.named_parameters() if not parameter.requires_grad}
    if actual!=expected or actual&forbidden: raise RuntimeError('optimizer contains frozen text/VAE/native Helios parameters or misses a declared trainable parameter')

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

def _transformer_forward(pipe,noisy,timestep,prompt_embeds,history,current_start,*,prefix_stop_layer=None):
    """Call native Helios only. Geometry sigma is owned by the caller."""
    indices=native_helios_indices(noisy.device,noisy.shape[0])['current']
    attention_kwargs={'current_chunk':current_start//8}
    if prefix_stop_layer is not None:
        attention_kwargs['sightline_prefix_stop_layer']=int(prefix_stop_layer)
    output=pipe.transformer(hidden_states=noisy.to(pipe.transformer.dtype),timestep=timestep,encoder_hidden_states=prompt_embeds,
        indices_hidden_states=indices,latents_history_long=history['long'][0],indices_latents_history_long=history['long'][1],
        latents_history_mid=history['mid'][0],indices_latents_history_mid=history['mid'][1],
        latents_history_short=history['short'][0],indices_latents_history_short=history['short'][1],attention_kwargs=attention_kwargs,
        return_dict=False if prefix_stop_layer is not None else True)
    if prefix_stop_layer is not None:
        return None
    prediction=output[0] if isinstance(output,(tuple,list)) else getattr(output,'sample',output)
    if prediction.shape!=noisy.shape: raise RuntimeError(f'prediction shape {prediction.shape} != {noisy.shape}')
    return prediction

def _model_prediction(pipe,noisy,item,prompt_embeds,history,current_start,*,routing_scope_active=False,prefix_stop_layer=None):
    """Active FM wrapper; Geometry is timestep-independent."""
    if not isinstance(item,dict) or 'sigmas' not in item or 'timesteps' not in item:
        raise ValueError('active Flow Matching prediction requires item["sigmas"] and item["timesteps"]')
    runner=getattr(pipe,'_sightline_pipeline',None)
    if runner is None or runner.ray_provider.context is None:
        raise RuntimeError('active FM Geometry routing requires a bound Sightline pipeline context')
    # Keep the resolved stage contract alive through loss construction and
    # checkpoint recomputation of the stage.  ``sigmas`` is the Helios-local
    # coordinate; Geometry receives the shared absolute coordinate.
    sigma_local=item.get('sigma_local',item['sigmas']).detach().float().mean()
    sigma_start=float(item.get('sigma_start',item.get('stage_start_sigma',1.0)))
    sigma_end=float(item.get('sigma_end',item.get('stage_end_sigma',0.0)))
    sigma_abs,geometry_sigma_scale=geometry_sigma_schedule(sigma_local,sigma_start,sigma_end)
    runner.ray_provider.context.update({'stage_index':int(item.get('stage_index',item.get('stage_id',0))),
        'sigma_local':sigma_local,'sigma_start':sigma_start,'sigma_end':sigma_end,
        'sigma_abs':sigma_abs.detach(),'geometry_sigma_scale':geometry_sigma_scale.detach(),
        'sigma':sigma_abs.detach(),'sigma_override':True})
    return _transformer_forward(pipe,noisy,item['timesteps'],prompt_embeds,history,current_start,prefix_stop_layer=prefix_stop_layer)

def _rgbd_prefix_forward(pipe,noisy,item,prompt_embeds,history,current_start,stop_layer):
    """Run the shared Helios path only through the last RGB-D capture block."""
    if stop_layer is None or int(stop_layer)<0:
        raise ValueError('RGB-D prefix forward requires a non-negative stop layer')
    with torch.no_grad():
        return _model_prediction(pipe,noisy,item,prompt_embeds,history,current_start,prefix_stop_layer=int(stop_layer))

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

def _load_correspondence(record,query_chunk=None,*,kind='all'):
    if kind not in ('all','intra','cross'): raise ValueError(f'unknown correspondence kind {kind}')
    rows=record.correspondences_for_chunk(query_chunk) if query_chunk is not None else list(record.correspondence_rows())
    if hasattr(rows,'column'):
        qf,kf=rows.column('query_frame'),rows.column('key_frame'); qc,kc=rows.column('query_chunk'),rows.column('key_chunk'); weights=rows.column('weight')
        qt,kt=rows.column('query_t'),rows.column('key_t')
        same=kc==qc
        query_scope=(query_chunk is None or np.all(qc==int(query_chunk)))
        invalid_intra=np.any(same & (kt==qt))
        invalid_cross=np.any((~same) & (kc>=qc))
        invalid_identity=(invalid_intra if kind=='intra' else invalid_cross if kind=='cross' else (invalid_intra or invalid_cross))
        if len(rows) and (invalid_identity or (kind!='intra' and np.any(kc>qc)) or not query_scope or np.any(kc<0) or not np.isfinite(weights).all() or np.any(weights<0)):
            raise RuntimeError('invalid RGB-D correspondence identity or weight')
        if kind=='intra':
            rows=type(rows)(rows.arrays,rows.indices[(kc==qc)&(kt!=qt)])
        elif kind=='cross':
            rows=type(rows)(rows.arrays,rows.indices[kc<qc])
    else:
        filtered=[]
        for row in rows:
            qf,kf=int(row['query_frame']),int(row['key_frame'])
            qc,kc=int(row['query_chunk']),int(row['key_chunk']); qt,kt=int(row['query_latent_temporal']),int(row['key_latent_temporal'])
            invalid=(kc>qc or not (0<=kc<record.chunk_count and 0<=qc<record.chunk_count) or
                     (kind=='intra' and kc==qc and kt==qt) or
                     (kind=='cross' and kc>=qc) or
                     (kind=='all' and ((kc!=qc and kf>=qf) or (kc==qc and kt==qt))))
            if invalid: raise RuntimeError('invalid RGB-D correspondence identity')
            if not np.isfinite(float(row['weight'])) or float(row['weight'])<0: raise RuntimeError('invalid RGB-D correspondence weight')
            if kind=='intra' and kc==qc and kt!=qt: filtered.append(row)
            elif kind=='cross' and kc<qc: filtered.append(row)
            elif kind=='all': filtered.append(row)
        rows=filtered
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

def _mapped_correspondences(processor,rows,chunk,identity_index=None,*,allowed_key_kinds=None,same_chunk_only=False):
    q,k=processor.last_q,processor.last_k
    if q is None or k is None: raise RuntimeError('correspondence processor did not capture Q/K')
    current=processor.last_current_length; identities=processor.last_key_identities
    if identities is None or len(identities)!=k.shape[1]: raise RuntimeError('explicit attention key identity map is missing or misaligned')
    current_shape=next(shape for shape in processor.ray_provider.context['stage_shapes'] if shape[0]*shape[1]*shape[2]==current)
    return _map_correspondence_identities(rows,chunk,current_shape,q.shape[1],identities,identity_index,allowed_key_kinds=allowed_key_kinds,same_chunk_only=same_chunk_only)

def _map_correspondence_identities(rows,chunk,current_shape,query_length,identities,identity_index=None,*,source_shape=None,allowed_key_kinds=None,same_chunk_only=False):
    """GT-to-attention mapping using layout metadata only, never Q/K values."""
    current=int(current_shape[0]*current_shape[1]*current_shape[2]); _,height,width=current_shape; source_shape=current_shape if source_shape is None else tuple(int(value) for value in source_shape); _,source_height,source_width=source_shape; q_start=int(query_length)-current
    identity_index=_identity_lookup(identities) if identity_index is None else identity_index
    columns=_correspondence_columns(rows); grouped={}
    for row_index in range(len(columns['query_chunk'])):
        if int(columns['query_chunk'][row_index])!=chunk: continue
        qt=int(columns['query_latent_temporal'][row_index]); qy_final=int(columns['query_y'][row_index]); qx_final=int(columns['query_x'][row_index])
        if not (0<=qt<current_shape[0] and 0<=qy_final<source_height and 0<=qx_final<source_width): continue
        qy=min(height-1,(qy_final*height)//source_height); qx=min(width-1,(qx_final*width)//source_width)
        qi=q_start+qt*height*width+qy*width+qx
        key_chunk=int(columns['key_chunk'][row_index]); key_t=int(columns['key_latent_temporal'][row_index])
        if same_chunk_only and (key_chunk!=int(chunk) or key_t>=qt): continue
        global_key=key_chunk*8+key_t
        query_global=chunk*8+qt
        if global_key>query_global: continue
        ky_final,kx_final=int(columns['key_y'][row_index]),int(columns['key_x'][row_index])
        if not (0<=ky_final<source_height and 0<=kx_final<source_width): continue
        ky=min(height-1,(ky_final*height)//source_height); kx=min(width-1,(kx_final*width)//source_width)
        factors={'long':4,'mid':2,'short':1}
        native=[]
        for level,factor in factors.items(): native.extend(identity_index.get(('native',global_key,ky//factor,kx//factor,level),()))
        native.sort(key=lambda index:{'short':0,'mid':1,'long':2}[identities[index][4]])
        memory=list(identity_index.get(('memory',global_key,ky//2,kx//2,'memory'),()))
        current_keys=list(identity_index.get(('current',global_key,ky,kx,'current'),()))
        source_keys=list(identity_index.get(('source',global_key,ky,kx,'source'),()))
        # Preserve every legal representation of the teacher identity on the
        # actual key axis.  Source/native/current/memory are diagnostics only.
        if allowed_key_kinds is None:
            candidates=sorted(set(native + current_keys + source_keys + memory))
        else:
            candidates=[]
            if 'native' in allowed_key_kinds: candidates.extend(native)
            if 'current' in allowed_key_kinds: candidates.extend(current_keys)
            if 'source' in allowed_key_kinds: candidates.extend(source_keys)
            if 'memory' in allowed_key_kinds: candidates.extend(memory)
            candidates=sorted(set(candidates))
        if not candidates: continue
        if 0<=qi<query_length:
            bucket=grouped.setdefault(qi,{})
            for ki in candidates:
                if 0<=ki<len(identities):
                    source='memory' if ki in memory else ('native' if ki in native else ('source' if ki in source_keys else 'current'))
                    # Downsampling can map several cache rows to one token
                    # pair.  Keep one confidence per pair; the max is a
                    # conservative aggregation and prevents duplicate counts.
                    bucket[ki]=(max(bucket.get(ki,(0.0,''))[0],float(columns['weight'][row_index])),source)
    if not grouped: raise RuntimeError('correspondence identities do not map to real attention axes')
    selected=sorted(grouped)
    positives=[sorted(grouped[query]) for query in selected]
    positive_weights=[[float(grouped[query][key][0]) for key in keys] for query,keys in zip(selected,positives)]
    weights=[max(value[0] for value in grouped[query].values()) for query in selected]
    flags=[{'has_native_positive':any(value[1] in ('native','source','current') for value in grouped[query].values()),'has_memory_positive':any(value[1]=='memory' for value in grouped[query].values())} for query in selected]
    return selected,positives,positive_weights,weights,flags

def _sample_correspondence_mapping(selected,positives,positive_weights,weights,flags,max_rows,sampling_seed):
    if len(selected)<=max_rows:return selected,positives,positive_weights,weights,flags
    memory_indices=[i for i,flag in enumerate(flags) if flag['has_memory_positive']]
    generator=torch.Generator(device='cpu').manual_seed(int(sampling_seed) & ((1<<63)-1))
    order=torch.randperm(len(selected),generator=generator).tolist()
    memory_set=set(memory_indices); memory_order=[i for i in order if i in memory_set]
    chosen_memory=memory_order[:max_rows]; chosen_memory_set=set(chosen_memory)
    choice=(chosen_memory+[i for i in order if i not in chosen_memory_set])[:max_rows]
    return ([selected[i] for i in choice],[positives[i] for i in choice],[positive_weights[i] for i in choice],[weights[i] for i in choice],[flags[i] for i in choice])

RGBD_AUX_PADDED_IMAGE_SHAPE=(512,832)

def _skew_translation(translation):
    tx,ty,tz=translation.unbind(-1)
    zero=torch.zeros_like(tx)
    return torch.stack((torch.stack((zero,-tz,ty),-1),torch.stack((tz,zero,-tx),-1),torch.stack((-ty,tx,zero),-1)),-2)

def _hard_negative_indices(selected,positives,identities,current_shape,query_length,*,c2w,intrinsics,max_negatives=4,padded_image_shape=RGBD_AUX_PADDED_IMAGE_SHAPE):
    """Choose deterministic same-time negatives outside the RGB-D epipolar band."""
    _,height,width=map(int,current_shape); padded_height,padded_width=map(float,padded_image_shape)
    if c2w.ndim==4: c2w=c2w[0]
    if intrinsics.ndim==4: intrinsics=intrinsics[0]
    if c2w.ndim!=3 or intrinsics.ndim!=3 or c2w.shape[-2:]!=(4,4) or intrinsics.shape[-2:]!=(3,3):
        raise ValueError('RGB-D hard-negative cameras must be [F,4,4] and [F,3,3] or batched')
    if int(c2w.shape[0])<33 or int(intrinsics.shape[0])<33:
        raise ValueError('RGB-D hard-negative cameras require the current 33-frame chunk')
    # Index candidates by their exact key frame.  The previous implementation
    # scanned every current token for every positive, which made the
    # off-epipolar filter quadratic in the number of stage tokens.  A negative
    # is allowed only at the positive's key time, so this grouping is exactly
    # equivalent while reducing each query to one frame-sized candidate set.
    candidates_by_global={}
    for index,identity in enumerate(identities):
        if identity[0]!='current': continue
        global_id=int(identity[1][0])
        candidates_by_global.setdefault(global_id,[]).append((index,int(identity[2]),int(identity[3])))
    camera_indices=latent_camera_indices().tolist()
    token_pixel_size=((padded_height/height)*(padded_width/width))**0.5
    # The camera tensors are tiny and have no gradient.  Move this metadata
    # once for the geometric plan construction; doing tiny GPU inverses and a
    # device-to-host sync for every positive made the planner dominate a step.
    c2w_np=c2w.detach().float().cpu().numpy()
    intrinsics_np=intrinsics.detach().float().cpu().numpy()
    c2w_inverse=np.linalg.inv(c2w_np)
    intrinsics_inverse=np.linalg.inv(intrinsics_np)
    epipolar_cache={}
    negatives=[]; masks=[]; matched_key_t=True; pair_count=0
    for query,positive in zip(selected,positives):
        if not 0<=int(query)<len(identities): raise RuntimeError('RGB-D query index is outside the identity map')
        query_identity=identities[int(query)]
        query_global=int(query_identity[1][0]); qy,qx=int(query_identity[2]),int(query_identity[3])
        query_chunk=query_global//8
        positive_set=set(int(value) for value in positive)
        row_negatives=[]; row_masks=[]
        for positive_index in positive:
            positive_index=int(positive_index)
            if not 0<=positive_index<len(identities): raise RuntimeError('RGB-D positive key is outside the identity map')
            positive_identity=identities[positive_index]
            if positive_identity[0]!='current': raise RuntimeError('RGB-D hard negatives require current-token positives')
            positive_global=int(positive_identity[1][0]); positive_y=int(positive_identity[2]); positive_x=int(positive_identity[3])
            key_chunk=positive_global//8
            key_t=positive_global-key_chunk*8
            query_t=query_global-query_chunk*8
            # The plan already enforces causal same-chunk positives.  Keep the
            # check here so a malformed hand-built plan cannot silently select
            # another temporal identity as a negative.
            if key_chunk!=query_chunk or not (0<=key_t<9 and 0<=query_t<9) or key_t>=query_t:
                row_negatives.append([]); row_masks.append([]); continue
            camera_pair=(query_t,key_t)
            if camera_pair not in epipolar_cache:
                query_camera=c2w_np[camera_indices[query_t]]
                key_camera_inverse=c2w_inverse[camera_indices[key_t]]
                query_K_inverse=intrinsics_inverse[camera_indices[query_t]]
                key_K_inverse=intrinsics_inverse[camera_indices[key_t]]
                T_kq=key_camera_inverse@query_camera
                R=T_kq[:3,:3]; translation=T_kq[:3,3]
                tx,ty,tz=translation
                skew=np.array(((0.0,-tz,ty),(tz,0.0,-tx),(-ty,tx,0.0)),dtype=np.float32)
                epipolar_cache[camera_pair]=(key_K_inverse.T@skew@R@query_K_inverse,translation)
            fundamental,translation=epipolar_cache[camera_pair]
            baseline=float(np.linalg.norm(translation))
            if not np.isfinite(baseline) or baseline<=1e-6:
                row_negatives.append([]); row_masks.append([]); continue
            query_xy=np.asarray([(qx+0.5)*padded_width/width,(qy+0.5)*padded_height/height,1.0],dtype=np.float32)
            line=fundamental@query_xy
            line_norm=float(np.linalg.norm(line[:2]))
            if not np.isfinite(line_norm) or line_norm<=1e-8:
                row_negatives.append([]); row_masks.append([]); continue
            candidates=candidates_by_global.get(positive_global,())
            band_candidates=[]
            if candidates:
                candidate_indices=[value[0] for value in candidates]
                candidate_yx=np.asarray([(value[1],value[2]) for value in candidates],dtype=np.float32)
                candidate_xy=np.column_stack(((candidate_yx[:,1]+0.5)*padded_width/width,(candidate_yx[:,0]+0.5)*padded_height/height,np.ones((len(candidates),),dtype=np.float32)))
                epi_tokens=(np.abs(candidate_xy@line)/line_norm)/token_pixel_size
                legal=np.isfinite(epi_tokens)&(epi_tokens>=2.0)
                legal_indices=np.flatnonzero(legal).tolist()
                epi_values=epi_tokens.tolist()
                for candidate_position in legal_indices:
                    candidate_index,candidate_y,candidate_x=candidates[candidate_position]
                    if candidate_index in positive_set:
                        continue
                    epi_value=float(epi_values[candidate_position])
                    spatial=((candidate_y-positive_y)**2+(candidate_x-positive_x)**2)**0.5
                    band=0 if epi_value<4.0 else 1
                    band_candidates.append((band,spatial,candidate_index))
            if not band_candidates:
                row_negatives.append([]); row_masks.append([]); continue
            ordered_bands=[]
            for band in (0,1):
                ordered_bands.append(sorted((value for value in band_candidates if value[0]==band),key=lambda value:(value[1],value[2])))
            chosen=ordered_bands[0][:2]+ordered_bands[1][:2]
            # If one epipolar band is sparse, fill from the other legal band;
            # the d_epi>=2 filter remains in force and no time fallback exists.
            if len(chosen)<int(max_negatives):
                chosen_indices={value[2] for value in chosen}
                supplement=sorted((value for value in band_candidates if value[2] not in chosen_indices),key=lambda value:(value[1],value[2]))
                chosen.extend(supplement[:int(max_negatives)-len(chosen)])
            chosen=[value[2] for value in chosen[:int(max_negatives)]]
            if not chosen:
                row_negatives.append([]); row_masks.append([]); continue
            if any(identities[index][1][0]!=positive_global for index in chosen): matched_key_t=False
            row_negatives.append(chosen); row_masks.append([True]*len(chosen)); pair_count+=1
        negatives.append(row_negatives); masks.append(row_masks)
    return negatives,masks,matched_key_t,pair_count

def _build_correspondence_plan(processor,rows,chunk,current_length,max_rows,sampling_seed,*,source_shape=None,allowed_key_kinds=None,same_chunk_only=False,with_hard_negatives=False,max_negatives=4):
    """Map GT once for the selected pyramid stage so every layer captures selected Q only."""
    identities=processor.ray_provider.key_identities(current_length,processor.memory)
    memory_count=len(processor.memory.active_identity_metadata()) if processor.memory is not None and processor.memory.enabled else 0
    query_length=len(identities)-memory_count
    current_shape=next(shape for shape in processor.ray_provider.context['stage_shapes'] if shape[0]*shape[1]*shape[2]==current_length)
    source_shape=current_shape if source_shape is None else tuple(int(value) for value in source_shape)
    columns=_correspondence_columns(rows); pre_count=0
    for row_index in range(len(columns['query_chunk'])):
        qchunk=int(columns['query_chunk'][row_index]); kchunk=int(columns['key_chunk'][row_index]); qt=int(columns['query_latent_temporal'][row_index]); kt=int(columns['key_latent_temporal'][row_index]); qy=int(columns['query_y'][row_index]); qx=int(columns['query_x'][row_index]); ky=int(columns['key_y'][row_index]); kx=int(columns['key_x'][row_index])
        if qchunk!=int(chunk) or not (0<=qt<current_shape[0] and 0<=qy<source_shape[1] and 0<=qx<source_shape[2] and 0<=ky<source_shape[1] and 0<=kx<source_shape[2]): continue
        if same_chunk_only and (kchunk!=int(chunk) or kt>=qt): continue
        if kchunk>qchunk or kchunk<0 or kt<0: continue
        pre_count+=1
    selected,positives,positive_weights,weights,flags=_map_correspondence_identities(rows,chunk,current_shape,query_length,identities,_identity_lookup(identities),source_shape=source_shape,allowed_key_kinds=allowed_key_kinds,same_chunk_only=same_chunk_only)
    selected,positives,positive_weights,weights,flags=_sample_correspondence_mapping(selected,positives,positive_weights,weights,flags,max_rows,sampling_seed)
    device=processor.ray_provider.context['c2w'].device
    negative_indices=negative_mask=None; sparse_key_indices=key_index_map=None
    negative_key_t_match=True; negative_pair_count=0
    if with_hard_negatives:
        context=processor.ray_provider.context
        negatives,negative_masks,negative_key_t_match,negative_pair_count=_hard_negative_indices(
            selected,positives,identities,current_shape,query_length,
            c2w=context['c2w'],intrinsics=context['intrinsics'],max_negatives=max_negatives)
        # Remove positives that have no legal same-key-time negative, then
        # remove queries that have no remaining paired supervision.  This is
        # done before capture so invalid rows never retain Q/K graph state.
        filtered=[]
        for row,row_negatives in enumerate(negatives):
            keep=[index for index,value in enumerate(row_negatives) if value]
            if keep:
                filtered.append((row,keep))
        if not filtered:
            return None
        selected=[selected[row] for row,_ in filtered]
        positives=[[positives[row][index] for index in keep] for row,keep in filtered]
        positive_weights=[[positive_weights[row][index] for index in keep] for row,keep in filtered]
        weights=[weights[row] for row,_ in filtered]
        flags=[flags[row] for row,_ in filtered]
        negative_rows=[[negatives[row][index] for index in keep] for row,keep in filtered]
        # This is a count of valid (query, positive) pairs, not the number of
        # negative tokens attached to those pairs.
        negative_pair_count=sum(1 for row in negative_rows for keys in row if keys)
    mapped_count=sum(len(keys) for keys in positives)
    max_positive=max(map(len,positives),default=0)
    positive_indices=torch.full((len(selected),max_positive),-1,device=device,dtype=torch.long)
    positive_mask=torch.zeros_like(positive_indices,dtype=torch.bool)
    positive_weight_tensor=torch.zeros((len(selected),max_positive),device=device,dtype=torch.float32)
    for row,(keys,confidences) in enumerate(zip(positives,positive_weights)):
        positive_indices[row,:len(keys)]=torch.as_tensor(keys,device=device); positive_mask[row,:len(keys)]=True
        positive_weight_tensor[row,:len(keys)]=torch.as_tensor(confidences,device=device,dtype=torch.float32)
    if with_hard_negatives:
        max_negative=max((len(value) for row in negative_rows for value in row),default=0)
        negative_indices=torch.full((len(selected),max_positive,max_negative),-1,device=device,dtype=torch.long); negative_mask=torch.zeros_like(negative_indices,dtype=torch.bool)
        for row,row_keys in enumerate(negative_rows):
            for positive_index,keys in enumerate(row_keys):
                negative_indices[row,positive_index,:len(keys)]=torch.as_tensor(keys,device=device); negative_mask[row,positive_index,:len(keys)]=True
        # Capture only the union of valid positive and negative full-axis keys.
        full_key_values=[positive_indices[positive_mask],negative_indices[negative_mask]]
        sparse_key_indices=torch.unique(torch.cat(full_key_values),sorted=True)
        key_index_map=torch.full((len(identities),),-1,device=device,dtype=torch.long)
        key_index_map[sparse_key_indices]=torch.arange(sparse_key_indices.numel(),device=device,dtype=torch.long)
        def remap(values,mask):
            mapped=torch.full_like(values,-1)
            mapped[mask]=key_index_map[values[mask]]
            return mapped
        positive_indices=remap(positive_indices,positive_mask)
        negative_indices=remap(negative_indices,negative_mask)
    return CorrespondencePlan(torch.as_tensor(selected,device=device,dtype=torch.long),positive_indices,positive_mask,
                              torch.as_tensor(weights,device=device,dtype=torch.float32),identities,tuple(flags),negative_indices,negative_mask,
                              int(pre_count),int(mapped_count),tuple(int(value) for value in current_shape),bool(negative_key_t_match),int(negative_pair_count),
                              sparse_key_indices,key_index_map,positive_weight_tensor)

def _captured_queries(processor,plan,captured,*,native=False,capture_indices=None):
    """Select a plan's queries from the union captured for sparse losses."""
    if captured is None: raise RuntimeError('correspondence processor did not capture Q')
    indices=getattr(processor,'last_capture_query_indices',None) if capture_indices is None else capture_indices
    if indices is None:
        if captured.shape[1]==plan.query_indices.numel(): return captured
        indices=plan.query_indices
    indices=torch.as_tensor(indices,device=captured.device,dtype=torch.long)
    wanted=plan.query_indices.to(device=captured.device,dtype=torch.long)
    if indices.numel()==0 or wanted.numel()==0: return captured[:, :0]
    positions=torch.searchsorted(indices,wanted)
    if torch.any(positions>=indices.numel()) or not torch.equal(indices.index_select(0,positions.clamp_max(indices.numel()-1)),wanted):
        raise RuntimeError('CorrespondencePlan queries were not included in the captured sparse Q union')
    return captured.index_select(1,positions)

def _captured_keys(processor,plan,captured,*,capture_indices=None):
    """Select a plan's keys from the exact sparse/full capture union."""
    if captured is None: raise RuntimeError('correspondence processor did not capture K')
    wanted=plan.sparse_key_indices
    if wanted is None:
        if plan.positive_mask.any():
            wanted=plan.positive_indices[plan.positive_mask]
        elif plan.negative_mask is not None and plan.negative_mask.any():
            wanted=plan.negative_indices[plan.negative_mask]
        else:
            return captured[:, :0]
        wanted=torch.unique(wanted.to(device=captured.device,dtype=torch.long),sorted=True)
    else:
        wanted=wanted.to(device=captured.device,dtype=torch.long)
    indices=getattr(processor,'last_capture_key_indices',None) if capture_indices is None else capture_indices
    if indices is None:
        if captured.shape[1]==wanted.numel(): return captured
        indices=torch.arange(captured.shape[1],device=captured.device,dtype=torch.long)
    else:
        indices=torch.as_tensor(indices,device=captured.device,dtype=torch.long)
    if indices.numel()==0 or wanted.numel()==0: return captured[:, :0]
    positions=torch.searchsorted(indices,wanted)
    if torch.any(positions>=indices.numel()) or not torch.equal(indices.index_select(0,positions.clamp_max(indices.numel()-1)),wanted):
        raise RuntimeError('CorrespondencePlan keys were not included in the captured sparse K union')
    return captured.index_select(1,positions)

def _captured_rgbd_legal_mask(processor,plan,capture_indices,capture_key_indices):
    legal=getattr(processor,'last_rgbd_legal_mask',None)
    if legal is None: return plan.legal_mask
    qaxis=torch.as_tensor(capture_indices,device=legal.device,dtype=torch.long)
    kaxis=torch.as_tensor(capture_key_indices,device=legal.device,dtype=torch.long)
    qwanted=plan.query_indices.to(legal.device); kwanted=plan.sparse_key_indices.to(legal.device)
    qpos=torch.searchsorted(qaxis,qwanted); kpos=torch.searchsorted(kaxis,kwanted)
    if qpos.numel() and (qpos.max()>=qaxis.numel() or not torch.equal(qaxis.index_select(0,qpos),qwanted)): raise RuntimeError('RGB-D legal mask is missing a selected query')
    if kpos.numel() and (kpos.max()>=kaxis.numel() or not torch.equal(kaxis.index_select(0,kpos),kwanted)): raise RuntimeError('RGB-D legal mask is missing a selected key')
    return plan.legal_mask & legal.index_select(1,qpos).index_select(2,kpos)

def _rgbd_loss(trainable,processors,layers,plan,*,margin,temperature,local_scale=1.0,timings=None,captures=None):
    if plan is None or plan.query_indices.numel()==0: return torch.zeros((),device=next(iter(processors.values())).ray_provider.context['c2w'].device)
    missing=[layer for layer in layers if layer not in processors]
    if missing: raise RuntimeError(f'RGB-D correspondence layers have no Sightline processor: {missing}')
    losses=[]; started=time.perf_counter()
    for layer in layers:
        processor=processors[layer]
        saved=None if captures is None else captures.get(layer)
        if saved is None:
            augmented_q_value=processor.last_rgbd_q
            native_q_value=processor.last_rgbd_native_q
            augmented_k_value=processor.last_rgbd_k
            native_k_value=processor.last_rgbd_native_k
            dq_value=processor.last_rgbd_dq
            dk_value=processor.last_rgbd_dk
            capture_indices=processor.last_rgbd_query_indices
            capture_key_indices=processor.last_rgbd_key_indices
        else:
            if len(saved)==5:
                augmented_q_value,native_q_value,augmented_k_value,native_k_value,capture_indices=saved; capture_key_indices=None; dq_value=augmented_q_value-native_q_value; dk_value=augmented_k_value-native_k_value
            elif len(saved)==6 and isinstance(plan,RGBDSoftTargetPlan):
                native_q_value,native_k_value,dq_value,dk_value,capture_indices,capture_key_indices=saved
            else:
                augmented_q_value,native_q_value,augmented_k_value,native_k_value,capture_indices,capture_key_indices=saved; dq_value=augmented_q_value-native_q_value; dk_value=augmented_k_value-native_k_value
        native_q=_captured_queries(processor,plan,native_q_value,native=True,capture_indices=capture_indices)
        native_k=_captured_keys(processor,plan,native_k_value,capture_indices=capture_key_indices)
        dq=_captured_queries(processor,plan,dq_value,capture_indices=capture_indices)
        dk=_captured_keys(processor,plan,dk_value,capture_indices=capture_key_indices)
        if native_q is None or native_k is None or dq is None or dk is None:
            raise RuntimeError('RGB-D processor did not capture native and Sightline Q/K')
        if (not dq.requires_grad or not dk.requires_grad) and processor.conditioner is not None:
            # Prefix capture is deliberately no-grad.  Recompute only this
            # layer's projector from the detached sparse rays/native RMS so no
            # Helios or earlier-layer graph survives the prefix forward.
            ray_q=_captured_queries(processor,plan,processor.last_rgbd_rays_q,capture_indices=capture_indices)
            ray_k=_captured_keys(processor,plan,processor.last_rgbd_rays_k,capture_indices=capture_key_indices)
            dq=processor.conditioner.project(ray_q,native_q.flatten(2,3),kind='q',native_rms=processor.conditioner.native_rms(native_q.flatten(2,3)),detach_rho=True).unflatten(-1,(native_q.shape[2],native_q.shape[3]))
            dk=processor.conditioner.project(ray_k,native_k.flatten(2,3),kind='k',native_rms=processor.conditioner.native_rms(native_k.flatten(2,3)),detach_rho=True).unflatten(-1,(native_k.shape[2],native_k.shape[3]))
        if isinstance(plan,RGBDSoftTargetPlan):
            zero_q=torch.zeros_like(dq); zero_k=torch.zeros_like(dk)
            legal_mask=_captured_rgbd_legal_mask(processor,plan,capture_indices,capture_key_indices)
            s_native=rgbd_probability_row_scores(native_q.detach(),native_k.detach(),zero_q,zero_k,plan.target_indices,plan.target_weights,plan.target_mask,legal_mask,local_scale=1.0).detach()
            scales=sorted({0.1,max(0.1,min(1.0,float(local_scale))),1.0})
            scores=[rgbd_probability_row_scores(native_q.detach(),native_k.detach(),dq,dk,plan.target_indices,plan.target_weights,plan.target_mask,legal_mask,local_scale=value) for value in scales]
            wrong_scores=[]
            if plan.wrong_query_rays is not None and plan.wrong_key_rays is not None:
                wrong_dq=processor.conditioner.project(plan.wrong_query_rays,native_q.flatten(2,3),kind='q',native_rms=processor.conditioner.native_rms(native_q.flatten(2,3)),detach_rho=True).unflatten(-1,(native_q.shape[2],native_q.shape[3]))
                wrong_dk=processor.conditioner.project(plan.wrong_key_rays,native_k.flatten(2,3),kind='k',native_rms=processor.conditioner.native_rms(native_k.flatten(2,3)),detach_rho=True).unflatten(-1,(native_k.shape[2],native_k.shape[3]))
                wrong_scores=[rgbd_probability_row_scores(native_q.detach(),native_k.detach(),wrong_dq,wrong_dk,plan.target_indices,plan.target_weights,plan.target_mask,legal_mask,local_scale=value) for value in scales]
            target_values=plan.target_weights.masked_fill(~plan.target_mask,0.0)
            entropy=-(target_values.clamp_min(1e-12)*target_values.clamp_min(1e-12).log()).sum(-1)
            legal_count=legal_mask.sum(-1).float().clamp_min(1.0)
            smax=(target_values.clamp_min(1e-12)*target_values.clamp_min(1e-12).log()).sum(-1)
            headroom=(smax-s_native).clamp_min(0.0).detach()
            margins=torch.stack([torch.minimum(torch.full_like(headroom,0.1*value),0.5*headroom) for value in scales])
            row_native=torch.stack([F.relu(margins[i]-(scores[i]-s_native)) for i in range(len(scores))]).mean(0)
            if wrong_scores:
                pixel_per_token={8:64.0,16:32.0,32:16.0}.get(int(plan.stage_shape[1]),512.0/max(int(plan.stage_shape[1]),1))
                separation_ok=torch.ones((plan.query_indices.numel(),),device=s_native.device,dtype=torch.bool) if plan.separation_px is None else plan.separation_px.ge(0.5*pixel_per_token)
                wrong_gap=torch.stack([scores[i]-wrong_scores[i] for i in range(len(scores))])
                row_wrong=torch.stack([F.relu(0.1*scales[i]-wrong_gap[i]) for i in range(len(scores))]).mean(0)
                row_wrong=row_wrong*separation_ok.float()
            else:
                row_wrong=s_native.new_zeros(s_native.shape)
            row_loss=row_native+row_wrong
            # The native improvement is already the mean over local auxiliary
            # scales.  Add the absolute term at s=1 after the legal-mask
            # normalization, then reduce rows by the prescribed motion buckets.
            e_target=torch.minimum(torch.full_like(headroom,math.log(2.0)),0.9*headroom).detach()
            absolute=F.relu(e_target-(scores[-1]+legal_count.log()))
            row_loss=row_loss+0.25*absolute
            bucket_losses=[]
            for bucket in range(4):
                selected=plan.bucket.eq(bucket)
                if bool(selected.any()):
                    weights=plan.query_weights[selected]
                    bucket_losses.append((weights*row_loss[selected]).sum()/weights.sum().clamp_min(1e-12))
            if bucket_losses:
                losses.append(torch.stack(bucket_losses).mean())
            continue
        augmented_q=_captured_queries(processor,plan,augmented_q_value,capture_indices=capture_indices)
        augmented_k=_captured_keys(processor,plan,augmented_k_value,capture_indices=capture_key_indices)
        if augmented_q is None or augmented_k is None:
            raise RuntimeError('RGB-D processor did not capture augmented Q/K')
        if native_k.shape[1] < augmented_k.shape[1] and plan.negative_indices is not None:
            # Same-chunk positives/negatives must stay on the native Helios
            # history+current axis; Memory-only keys are never legal here.
            positive_max=plan.positive_indices.masked_fill(~plan.positive_mask,0).max()
            negative_max=plan.negative_indices.masked_fill(~plan.negative_mask,0).max()
            if max(int(positive_max.item()),int(negative_max.item()))>=native_k.shape[1]: raise RuntimeError('same-chunk RGB-D key escaped the native Helios key axis')
        raise TypeError('formal RGB-D supervision requires RGBDSoftTargetPlan')
    if timings is not None: timings['rgbd_loss_seconds']=timings.get('rgbd_loss_seconds',0.0)+time.perf_counter()-started
    if not losses: return torch.zeros((),device=plan.query_indices.device)
    return torch.stack(losses).mean()

def _release_rgbd_capture(processors,layers,*,preserve_cross_capture=False):
    """Drop RGB-D-only Q/K references while preserving an overlapping cross plan."""
    for layer in layers:
        processor=processors[layer]
        processor.last_rgbd_q=processor.last_rgbd_k=None
        processor.last_rgbd_native_q=processor.last_rgbd_native_k=None
        processor.last_rgbd_dq=processor.last_rgbd_dk=None
        processor.last_rgbd_rays_q=processor.last_rgbd_rays_k=None
        processor.last_rgbd_legal_mask=None
        processor.last_rgbd_query_indices=processor.last_rgbd_key_indices=None
        processor.capture_rgbd=False
        processor.last_native_q=processor.last_native_k=None
        processor.last_augmented_q=processor.last_augmented_k=None
        if not preserve_cross_capture:
            processor.last_q=processor.last_k=None
            processor.last_capture_query_indices=None
            processor.last_capture_key_indices=None
            processor.last_key_identities=None
            processor.last_attention_bias=None
            processor.capture_full_key=False

def _backward_rgbd_stage(term,trainable,*,retain_graph):
    """Backprop one RGB-D stage without letting it update the rho schedule."""
    if not term.requires_grad: return
    saved={id(beta):(None if beta.grad is None else beta.grad.detach().clone())
           for beta in trainable.conditioner.rho_parameters()}
    term.backward(retain_graph=bool(retain_graph))
    for beta in trainable.conditioner.rho_parameters():
        beta.grad=saved[id(beta)]

def _total_metric(fm,rgbd_term,cross_term):
    """Report the complete objective without creating a second backward path."""
    return fm + rgbd_term.detach() + cross_term.detach()

def _calibration_state(parameters):
    """Snapshot RNG streams and trainable bytes for side-effect-free calibration."""
    return {
        'python': random.getstate(),
        'numpy': np.random.get_state(),
        'torch_cpu': torch.random.get_rng_state(),
        'torch_cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        'parameter_hash': _parameter_hash(parameters),
    }

def _restore_calibration_state(state, parameters):
    random.setstate(state['python']); np.random.set_state(state['numpy'])
    torch.random.set_rng_state(state['torch_cpu'])
    if state['torch_cuda'] is not None: torch.cuda.set_rng_state_all(state['torch_cuda'])
    current=_parameter_hash(parameters)
    if current != state['parameter_hash']:
        raise RuntimeError('RGB-D lambda calibration changed model parameters')

def _parameter_hash(parameters):
    digest=hashlib.sha256()
    for parameter in parameters:
        digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()

def calibrate_rgbd_lambda(fm_total, rgbd_total, projector_parameters):
    """Return the formal gradient-ratio recommendation without optimizer.step.

    The caller fixes each batch's record/chunk/noise/timestep/stage choices
    before producing the two totals.  This helper only performs the two
    requested autograd.grad calls and is side-effect free for parameters.
    """
    parameters=tuple(projector_parameters)
    fm_grads=torch.autograd.grad(fm_total,parameters,retain_graph=True,allow_unused=True)
    rgbd_grads=torch.autograd.grad(rgbd_total,parameters,retain_graph=True,allow_unused=True)
    ratios=[]; cosines=[]; details=[]
    for index in range(0,len(parameters),2):
        gq_fm=fm_grads[index]; gk_fm=fm_grads[index+1] if index+1<len(parameters) else None
        gq_rgbd=rgbd_grads[index]; gk_rgbd=rgbd_grads[index+1] if index+1<len(parameters) else None
        if any(value is None for value in (gq_fm,gk_fm,gq_rgbd,gk_rgbd)): continue
        fm_norm=torch.sqrt(gq_fm.float().square().sum()+gk_fm.float().square().sum())/torch.sqrt(torch.as_tensor(gq_fm.numel()+gk_fm.numel(),device=gq_fm.device,dtype=torch.float32))
        rgbd_norm=torch.sqrt(gq_rgbd.float().square().sum()+gk_rgbd.float().square().sum())/torch.sqrt(torch.as_tensor(gq_rgbd.numel()+gk_rgbd.numel(),device=gq_fm.device,dtype=torch.float32))
        ratios.append(float((fm_norm/(rgbd_norm+1e-12)).detach()))
        cosine=F.cosine_similarity(torch.cat((gq_fm.reshape(-1).float(),gk_fm.reshape(-1).float())),torch.cat((gq_rgbd.reshape(-1).float(),gk_rgbd.reshape(-1).float())),dim=0)
        cosines.append(float(cosine.detach()))
        details.append({'layer_pair_index':index//2,'g_fm':float(fm_norm.detach()),'g_rgbd':float(rgbd_norm.detach()),'ratio':ratios[-1],'gradient_cosine':cosines[-1]})
    if not ratios: raise RuntimeError('RGB-D lambda calibration produced no valid projector gradients')
    recommendation=0.2*float(np.median(np.asarray(ratios,dtype=np.float64)))
    return {'recommended_lambda_rgbd':recommendation,'gradient_cosine_mean':float(np.mean(cosines)),'ratios':ratios,'layers':details}

def run_rgbd_lambda_calibration(batch_builder, projector_parameters, *, batches=16):
    """Calibrate over fixed batches without optimizer steps or RNG drift.

    ``batch_builder(index)`` must fix record/chunk/noise/timestep/stage choices
    before returning ``(L_FM_total, L_RGBD_total_lambda_1)``.  Keeping this
    callback boundary explicit prevents calibration from silently sampling a
    different batch for the two objectives.
    """
    if int(batches)<1: raise ValueError('calibration batches must be positive')
    parameters=tuple(projector_parameters); state=_calibration_state(parameters)
    ratios=[]; cosines=[]; batch_details=[]
    try:
        for index in range(int(batches)):
            fm_total,rgbd_total=batch_builder(index)
            result=calibrate_rgbd_lambda(fm_total,rgbd_total,parameters)
            ratios.extend(result['ratios']); cosines.append(result['gradient_cosine_mean'])
            batch_details.append(result)
    finally:
        _restore_calibration_state(state,parameters)
    if not ratios: raise RuntimeError('RGB-D lambda calibration produced no valid batch/layer ratios')
    return {'recommended_lambda_rgbd':0.2*float(np.median(np.asarray(ratios,dtype=np.float64))),
            'gradient_cosine_mean':float(np.mean(cosines)), 'ratios':ratios,
            'batches':batch_details, 'parameter_hash':state['parameter_hash']}

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
        try: selected,positives,positive_weights,weights,flags=_mapped_correspondences(first,rows,chunk,_identity_lookup(identities))
        except RuntimeError as exc:
            if 'do not map' not in str(exc): raise
            selected=positives=positive_weights=weights=flags=[]
        selected,positives,positive_weights,weights,flags=_sample_correspondence_mapping(selected,positives,positive_weights,weights,flags,max_rows,sampling_seed)
    else:
        if identities != plan.identities: raise RuntimeError('CorrespondencePlan key identities changed during final-stage forward')
        selected=list(range(plan.query_indices.numel())); positives=weights=None; flags=plan.flags
    if timings is not None: timings['correspondence_mapping_seconds']+=time.perf_counter()-mapping_started
    losses=[]; loss_started=time.perf_counter()
    for layer in layers:
        processor=processors[layer]
        if not selected:
            if plan is not None:
                processor.last_q=processor.last_k=processor.last_native_q=processor.last_native_k=processor.last_augmented_q=processor.last_augmented_k=processor.last_capture_query_indices=None
                processor.last_rgbd_q=processor.last_rgbd_k=processor.last_rgbd_native_q=processor.last_rgbd_native_k=processor.last_rgbd_dq=processor.last_rgbd_dk=processor.last_rgbd_rays_q=processor.last_rgbd_rays_k=processor.last_rgbd_legal_mask=processor.last_rgbd_query_indices=processor.last_rgbd_key_indices=None
                processor.last_attention_bias=None
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
            captured_q=_captured_queries(processor,plan,captured_q)
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
        if plan is not None:
            processor.last_q=processor.last_k=processor.last_native_q=processor.last_native_k=processor.last_augmented_q=processor.last_augmented_k=processor.last_capture_query_indices=None
            processor.last_rgbd_q=processor.last_rgbd_k=processor.last_rgbd_native_q=processor.last_rgbd_native_k=processor.last_rgbd_dq=processor.last_rgbd_dk=processor.last_rgbd_rays_q=processor.last_rgbd_rays_k=processor.last_rgbd_legal_mask=processor.last_rgbd_query_indices=processor.last_rgbd_key_indices=None
            processor.last_attention_bias=None
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
    p.add_argument('--expected-records',type=int); p.add_argument('--max-steps',type=int); p.add_argument('--resume'); p.add_argument('--allow-memory-layer-migration',action='store_true'); p.add_argument('--allow-world-size-migration',action='store_true'); p.add_argument('--skip-manifest-validation',action='store_true',help='Skip repeated per-record validation only when these exact manifests were already validated successfully.'); p.add_argument('--output-dir',required=True); p.add_argument('--save-every',type=int); p.add_argument('--latent-cache-root')
    p.add_argument('--prompt',default='A stable realistic view of the same scene.'); p.add_argument('--probe-only',action='store_true'); p.add_argument('--probe-checkpoint'); p.add_argument('--probe-layers',default=''); p.add_argument('--probe-capture'); p.add_argument('--probe-step',type=int,default=1000); p.add_argument('--alpha-zero-baseline',action='store_true'); p.add_argument('--record-index',type=int); p.add_argument('--train-chunk',type=int); p.add_argument('--checkpoint-smoke-step',type=int); p.add_argument('--smoke-max-chunks',type=int); p.add_argument('--smoke-max-chunks-sequence'); p.add_argument('--profile-timing',action='store_true'); p.add_argument('--calibrate-rgbd-lambda',action='store_true'); p.add_argument('--calibration-batches',type=int,default=16); p.add_argument('--lambda-rgbd',type=float); p.add_argument('--train',action='store_true'); args=p.parse_args()
    if not (args.train or args.probe_only) or args.train==args.probe_only: raise ValueError('select exactly one of --train or --probe-only')
    cfg=load_sightline_config(args.config)
    if args.train and args.lambda_rgbd is None:
        raise SystemExit('formal training requires explicit --lambda-rgbd')
    if args.lambda_rgbd is not None:
        if args.lambda_rgbd <= 0: raise ValueError('--lambda-rgbd must be positive')
        cfg=replace(cfg,lambda_rgbd=float(args.lambda_rgbd))
    if args.calibrate_rgbd_lambda and args.calibration_batches<1:
        raise ValueError('--calibration-batches must be positive')
    total_steps=cfg.p1_steps+cfg.p2_steps+cfg.p3_steps
    args.max_steps=args.max_steps or total_steps
    save_every=args.save_every or cfg.checkpoint_every
    if args.train and args.save_every is not None and args.save_every not in (50,60,100): raise ValueError('formal checkpoint cadence only permits 50, 60, or 100')
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
    probe_layers=tuple(int(x) for x in args.probe_layers.split(',') if x); _preflight(cfg,args,probe_layers); records=load_rgbd_memory_manifest(args.manifest,expected_count=args.expected_records,validate=not args.skip_manifest_validation)
    records=tuple(record for record in records if record.raw.get('dataset')!='arkitscenes')
    if not records: raise RuntimeError('P1/P2 manifest has no non-arkitscenes records')
    p3_records=load_rgbd_memory_manifest(args.p3_manifest,validate=not args.skip_manifest_validation) if args.p3_manifest else records
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
        lambda_corr_decay_start=cfg.lambda_corr_decay_start,rho_init=cfg.rho_init,
        scale_aug_prob=cfg.scale_augmentation_probability,
        scale_aug_range=cfg.scale_augmentation_range).to(device,dtype=torch.float32)
    for parameter in pipe.transformer.parameters(): parameter.requires_grad_(False)
    # The formal Sightline run keeps the entire Helios backbone frozen.  This
    # includes block 0..11 modulation/norm parameters as well as attention,
    # FFN, and blocks 12+.  Keep an explicit empty scope so optimizer,
    # checkpoint provenance, and resume preflight cannot silently re-enable a
    # subset of the backbone.
    helios_trainable_names=[]; helios_trainable=[]
    if rank==0: print('Helios trainable parameters: none (entire backbone frozen)', flush=True)
    install_lora(pipe.transformer,cfg.lora_layers,rank=cfg.lora_rank) if cfg.lora_layers else None
    padded_h,padded_w=padded_size(cfg.source_height,cfg.source_width)
    provider=SightlineRayProvider(source_height=padded_h,source_width=padded_w); runner=SightlinePipeline(pipe,config=cfg,conditioner=trainable.conditioner,ray_provider=provider); pipe._sightline_pipeline=runner
    runner.memory.to(device=device,dtype=torch.bfloat16)
    installed_layers=tuple(sorted(set(cfg.sightline_layers).union(cfg.memory_layers).union(cfg.correspondence_layers).union(probe_layers))) if args.probe_only else tuple(sorted(set(cfg.sightline_layers).union(cfg.memory_layers).union(cfg.correspondence_layers)))
    install_sightline_attention(pipe.transformer,trainable.conditioner,provider,layers=installed_layers,sightline_layers=cfg.sightline_layers,memory_layers=cfg.memory_layers,correspondence_layers=cfg.correspondence_layers,helios_module=helios_source,memory=runner.memory)
    initialization_hash=broadcast_and_assert_trainables(trainable,runner.memory,pipe.transformer,world_size)
    lora_params=[p for n,p in pipe.transformer.named_parameters() if 'lora_' in n]
    memory_params=list(runner.memory.parameters())
    # Keep Geometry optimization semantics explicit: zero-decay projector
    # matrices and the separately scheduled beta amplitudes are independent
    # parameter groups.
    projector_params=[]; beta_params=[]
    for layer in trainable.conditioner.layers.values():
        projector_params.extend(layer.q_proj.parameters()); projector_params.extend(layer.k_proj.parameters())
        beta_params.extend((layer.beta_q,layer.beta_k))
    optimizer_groups=[
        {'name':'projector','params':projector_params,'lr':cfg.learning_rate,'weight_decay':0.0},
        {'name':'beta','params':beta_params,'lr':cfg.geometry_beta_learning_rate,'weight_decay':0.0},
        {'name':'memory','params':memory_params,'lr':cfg.memory_learning_rate,'weight_decay':0.01},
    ]
    if lora_params: optimizer_groups.insert(-1,{'name':'lora','params':lora_params,'lr':cfg.lora_learning_rate,'weight_decay':0.01})
    optimizer=torch.optim.AdamW(optimizer_groups,weight_decay=0.0)
    _assert_optimizer_scope(optimizer,trainable,runner.memory,pipe.transformer,pipe.text_encoder,pipe.vae,helios_trainable)
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
    config=asdict(cfg); memory_config={'layers':list(cfg.memory_layers),'pool':cfg.memory_pool,'budget':cfg.memory_budget,'tau_pos':cfg.memory_tau_pos,'tau_angle':cfg.memory_tau_angle}; provenance=runtime_provenance(pipe,args.model,args.helios_root,model_revision=args.model_revision,transformer_source_sha256=fingerprint,runtime_patch=runtime_patch,lora_scope=cfg.lora_scope,helios_trainable_scope=helios_trainable_names)
    trainable.eval() if args.probe_only else trainable.train()
    start_step=args.probe_step if args.probe_only else 0
    world_size_migrated=False
    if args.resume:
        payload=torch.load(args.resume,map_location='cpu'); world_size_migrated=int(payload.get('rng_world_size',-1))!=world_size
        completed_step=restore_runtime_checkpoint(payload,trainable,runner.memory,pipe.transformer,config=config,helios_fingerprint=fingerprint,layers=cfg.sightline_layers,memory_config=memory_config,optimizer=optimizer,scheduler=scheduler,restore_rng=True,provenance=provenance,rank=rank,world_size=world_size,allow_memory_layer_migration=args.allow_memory_layer_migration,allow_world_size_migration=args.allow_world_size_migration,helios_trainable_names=helios_trainable_names); start_step=completed_step+1
        if world_size_migrated:
            seed=set_rank_runtime_seed(rank,start_step)
            if rank==0: print(f'checkpoint world size migration: deterministic per-rank reseed at step {start_step}, rank0 seed {seed}',flush=True)
    elif args.probe_checkpoint:
        if not args.probe_only: raise ValueError('--probe-checkpoint is only valid with --probe-only')
        payload=torch.load(args.probe_checkpoint,map_location='cpu'); restored_step=restore_runtime_checkpoint(payload,trainable,runner.memory,pipe.transformer,config=config,helios_fingerprint=fingerprint,layers=cfg.sightline_layers,memory_config=memory_config,restore_rng=False,provenance=provenance,helios_trainable_names=helios_trainable_names); start_step=restored_step
    if args.alpha_zero_baseline:
        configure_geometry_zero_baseline(trainable,runner.memory,pipe.transformer)
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
        if args.profile_timing or (step+1) % cfg.diagnostics_frequency == 0: torch.cuda.reset_peak_memory_stats(device)
        phase=curriculum_phase(step,p1_steps=cfg.p1_steps,p2_steps=cfg.p2_steps,p3_steps=cfg.p3_steps)
        capture_geometry_diagnostics=((step+1) % cfg.diagnostics_frequency == 0)
        for processor in pipe.transformer._sightline_processors.values():
            processor.capture_numeric_diagnostics=capture_geometry_diagnostics
            if processor.conditioner is not None:
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
        train_rgbd=bool(args.train and phase.get('rgbd',False) and not args.alpha_zero_baseline)
        phase_records=p3_records if phase['name']=='P3' else records
        requires_rgbd_data=bool(train_rgbd or phase['memory'] or phase['correspondence'] or bool(args.probe_capture))
        eligible_records=[record for record in phase_records if record.chunk_count >= phase['max_chunks'] and (record.memory_eligible if requires_rgbd_data else True)]
        if not eligible_records: raise RuntimeError(f"{phase['name']} requires at least one memory-eligible RGB-D record")
        if args.record_index is not None:
            record=phase_records[args.record_index]
            if record not in eligible_records: raise ValueError(f"record {record.record_id} is camera-only and cannot be used in {phase['name']}")
        else:
            index=_ddp_record_index(step,rank,world_size,len(eligible_records)); record=eligible_records[index]
        latent_root=args.latent_cache_root or cfg.latent_cache_path or None
        latent_path=resolve_continuous_latent_cache(record,cache_root=latent_root); latent_schema,_=validate_latent_cache(latent_path)
        validate_rgbd_record_latent(record,latent_path)
        if phase['name'] in ('P1','P2'):
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
        active_rgbd_layers=tuple(cfg.sightline_layers) if args.train else ()
        active_corr_layers=probe_layers or tuple(cfg.correspondence_layers); diagnostic_correspondence=bool(args.probe_capture)
        # Every multi-chunk phase samples the newest frontier with p=.45 and
        # distributes the remaining p=.55 uniformly over earlier chunks.
        forced_train_chunk=(phase['max_chunks']-1 if smoke_chunk_sequence else args.train_chunk)
        train_chunk=_ddp_train_chunk(phase['max_chunks'],0,rank,world_size,device,forced_train_chunk)
        oom_state['train_chunk']=int(train_chunk)
        if not 0<=train_chunk<phase['max_chunks']: raise ValueError('train_chunk outside curriculum')
        gt_prefix_p=float(phase.get('gt_prefix_probability',0.0))
        use_gt_prefix=bool(train_chunk>0 and gt_prefix_p>0.0 and random.Random(20260908+int(step)).random()<gt_prefix_p)
        # Same-chunk RGB-D supervision is valid for every selected train chunk,
        # including the direct-source chunk0 path.  Cross-chunk correspondence
        # remains restricted to Memory-backed P3 frontiers.
        train_correspondence=bool(phase['correspondence'] and train_chunk>0 and not args.alpha_zero_baseline)
        runner.memory.set_enabled(phase['memory'] and train_chunk>0)
        capture_layers=()
        if train_rgbd or train_correspondence or diagnostic_correspondence:
            if not record.memory_eligible: raise RuntimeError(f"{record.record_id} has no calibrated RGB-D correspondence supervision")
            capture_layers=tuple(sorted(set(active_rgbd_layers).union(active_corr_layers if train_correspondence or diagnostic_correspondence else ())))
            if not capture_layers or any(layer not in pipe.transformer._sightline_processors for layer in capture_layers): raise RuntimeError('active correspondence/probe layers are not installed Sightline layers')
            for layer in capture_layers:
                pipe.transformer._sightline_processors[layer].capture_diagnostics=False
                pipe.transformer._sightline_processors[layer].capture_query_indices=None
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
        if train_rgbd or train_correspondence or diagnostic_correspondence:
            load_started=time.perf_counter(); rgbd_rows=_load_correspondence(record,train_chunk,kind='intra') if train_rgbd else None; cross_rows=_load_correspondence(record,train_chunk,kind='cross') if train_correspondence or diagnostic_correspondence else None; perf['correspondence_load_seconds']=time.perf_counter()-load_started
            if train_rgbd and not len(rgbd_rows): raise RuntimeError(f'{record.record_id}: P1/P2/P3 requires same-chunk RGB-D correspondences for chunk {train_chunk}')
        else: rgbd_rows=cross_rows=None
        sigma_range,sigma_band=_sigma_band(step,phase['name'])
        geometry_memory_diagnostics={}
        def record_geometry_memory(name):
            if capture_geometry_diagnostics and device.type=='cuda':
                geometry_memory_diagnostics[name]={
                    'memory_allocated':int(torch.cuda.memory_allocated(device)),
                    'max_memory_allocated':int(torch.cuda.max_memory_allocated(device)),
                }
        history_state=NativeHistoryState(source,fake); generated_prefix=[]; losses={}; probe_payload={}; fm_sigma_trace=[]; backward_geometry_diagnostics={}; optimizer.zero_grad(set_to_none=True)
        started=time.perf_counter()
        def forward_chunk(chunk,keep_graph):
            chunk_started=time.perf_counter()
            history=history_state.groups(); coverage=history_state.coverage()
            if not keep_graph:
                # Only the selected backward chunk owns Geometry diagnostics;
                # detached rollout must not overwrite that snapshot.
                for processor in pipe.transformer._sightline_processors.values():
                    processor.capture_numeric_diagnostics=False
                    processor.capture_diagnostics=False
                    if processor.conditioner is not None: processor.conditioner.capture_numeric_diagnostics=False
                oom_state['stage']='prefix_rollout'
                template=torch.empty((source.shape[0],source.shape[1],9,*source.shape[-2:]),device=source.device,dtype=source.dtype); runner._prepare_chunk(chunk,template,{},history_global_coverages=coverage,history_validity=history_state.validity())
                clean_boundary=source if chunk==0 else generated_prefix[-1][:,:,-1:]
                if use_gt_prefix and chunk < train_chunk:
                    gt_start=latent_start+chunk*8
                    generated=all_latents[:,:,gt_start:gt_start+9].to(device,dtype=torch.bfloat16).detach()
                    generated[:,:,:1]=clean_boundary.to(generated)
                else:
                    generated=_generate_detached_chunk(pipe,source,history,prompt_embeds,cfg,chunk,clean_boundary).detach()
                record_vram('prefix_rollout')
            else:
                target=target_latents  # The sole non-source GT read in this step.
                clean_boundary=source if chunk==0 else generated_prefix[-1][:,:,-1:].detach()
                target[:,:,:1]=clean_boundary.to(target)
                items=exact_flow_matching_items(pipe,target,stage_steps=cfg.pyramid_steps,device=target.device,sigma_range=sigma_range)
                items=constrain_flow_items(items,clean_boundary)
                runner._prepare_chunk(chunk,target,{},history_global_coverages=coverage,history_validity=history_state.validity())
                for processor in pipe.transformer._sightline_processors.values():
                    processor.capture_numeric_diagnostics=capture_geometry_diagnostics
                    processor.last_numeric_diagnostics=None
                    if processor.conditioner is not None: processor.conditioner.capture_numeric_diagnostics=capture_geometry_diagnostics
                oom_state['stage']='active_memory'
                if active_corr_layers:
                    active=pipe.transformer._sightline_processors[active_corr_layers[0]].memory
                    oom_state['memory_token_count']=len(active.active_identity_metadata()) if active is not None and active.enabled else 0
                record_vram('active_memory')
                rgbd_plans={}; cross_plan=None
                correspondence_seed=int(hashlib.sha256(f'{step}:{record.trajectory_id}'.encode()).hexdigest()[:16],16)
                final_shape=tuple(int(value) for value in runner.ray_provider.context['stage_shapes'][-1])
                if rgbd_rows is not None and len(rgbd_rows) and not args.alpha_zero_baseline:
                    if len(items)!=3: raise RuntimeError('RGB-D stage1/stage2 supervision requires exactly three pyramid stages')
                    for rgbd_stage_index in (1,2):
                        rgbd_stage_shape=tuple(int(value) for value in runner.ray_provider.context['stage_shapes'][rgbd_stage_index])
                        current_length=int(rgbd_stage_shape[0]*rgbd_stage_shape[1]*rgbd_stage_shape[2])
                        rgbd_processor=pipe.transformer._sightline_processors[active_rgbd_layers[0]]
                        rgbd_identities=rgbd_processor.ray_provider.key_identities(current_length,rgbd_processor.memory)
                        stage_plan=build_rgbd_soft_target_plan(rgbd_rows,rgbd_identities,rgbd_stage_shape,chunk=chunk,max_rows=cfg.max_intra_corr_rows,sampling_seed=correspondence_seed+rgbd_stage_index,device=source.device,source_height=512,source_width=832,c2w=runner.ray_provider.context['c2w'],intrinsics=runner.ray_provider.context['intrinsics'])
                        if stage_plan is not None and stage_plan.mapping_output_count>0:
                            rgbd_plans[rgbd_stage_index]=stage_plan
                if cross_rows is not None and len(cross_rows) and not args.alpha_zero_baseline:
                    current_length=int(final_shape[0]*final_shape[1]*final_shape[2])
                    cross_plan=_build_correspondence_plan(pipe.transformer._sightline_processors[active_corr_layers[0]],cross_rows,chunk,current_length,cfg.correspondence_rows_per_batch,correspondence_seed)
                plans=tuple(list(rgbd_plans.values())+([cross_plan] if cross_plan is not None else []))
                if plans:
                    oom_state['k_length']=max(len(plan.identities) for plan in plans)
                    oom_state['selected_q_count']=int(torch.unique(torch.cat([plan.query_indices for plan in plans])).numel())
                # RGB-D auxiliary supervision is intentionally independent of
                # the FM Geometry routing strength.  The valid stage losses
                # share the weight equally; geometry_sigma_scale remains a
                # diagnostic only and never attenuates this auxiliary loss.
                rgbd_scales={stage_index:items[stage_index]['geometry_sigma_scale'].detach().float().mean() for stage_index in (1,2)}
                valid_rgbd_stages=tuple(stage_index for stage_index in (1,2) if stage_index in rgbd_plans and rgbd_plans[stage_index].mapping_output_count>0)
                rgbd_weights={stage_index:(torch.as_tensor(1.0/len(valid_rgbd_stages),device=source.device,dtype=torch.float32).detach() if stage_index in valid_rgbd_stages else torch.zeros((),device=source.device,dtype=torch.float32)) for stage_index in (1,2)}
                backward_geometry_diagnostics.clear(); active_stage_trace=[]; rgbd_stage_losses={}; rgbd_capture_seen=set()
                stage_losses=[]; final_prediction=None; fm_sigma_trace.clear()
                for stage_index,item in enumerate(items):
                    stage_rgbd_plan=rgbd_plans.get(stage_index)
                    capture_rgbd=bool(stage_rgbd_plan is not None)
                    capture_cross=bool(stage_index+1==len(items) and (cross_plan is not None or diagnostic_correspondence))
                    # The first pass of an RGB-D stage is RGB-D-only.  A final
                    # stage with cross-chunk supervision gets a separate
                    # full-K capture on the second pass below.
                    stage_capture_layers=tuple(active_rgbd_layers if capture_rgbd else (active_corr_layers if capture_cross else ()))
                    for layer in capture_layers:
                        processor=pipe.transformer._sightline_processors[layer]
                        processor.capture_diagnostics=False; processor.capture_rgbd=False; processor.capture_query_indices=None; processor.capture_key_indices=None; processor.capture_full_key=False
                        processor.reuse_scale_delta=False; processor.scale_delta_override=None; processor.last_scale_delta=None
                    for layer in stage_capture_layers:
                        processor=pipe.transformer._sightline_processors[layer]
                        # RGB-D-only passes use only the explicit auxiliary
                        # fields; keeping real last_q/last_k here would pin a
                        # second copy of the sparse capture unnecessarily.
                        processor.capture_diagnostics=bool(capture_cross and not capture_rgbd)
                        processor.capture_numeric_diagnostics=bool(capture_geometry_diagnostics and not capture_rgbd)
                        if processor.conditioner is not None:
                            processor.conditioner.capture_numeric_diagnostics=bool(capture_geometry_diagnostics and not capture_rgbd)
                        processor.capture_rgbd=bool(capture_rgbd and layer in active_rgbd_layers)
                        stage_plans=tuple(plan for plan in (stage_rgbd_plan if capture_rgbd else None,cross_plan if capture_cross and not capture_rgbd else None) if plan is not None)
                        plan_queries=torch.unique(torch.cat([plan.query_indices for plan in stage_plans])) if stage_plans else None
                        processor.capture_query_indices=plan_queries if plan_queries is not None and (args.train or args.probe_capture) else None
                        rgbd_key_plans=(stage_rgbd_plan,) if capture_rgbd else ()
                        plan_keys=tuple(plan.sparse_key_indices for plan in rgbd_key_plans if plan.sparse_key_indices is not None)
                        processor.capture_key_indices=torch.unique(torch.cat(plan_keys)) if plan_keys and (args.train or args.probe_capture) else None
                        processor.capture_full_key=bool(capture_cross and not capture_rgbd and layer in active_corr_layers)
                    capture_correspondence=bool(stage_capture_layers)
                    oom_state['stage']='final_stage_forward' if capture_correspondence else f'flow_stage_{stage_index}_forward'
                    is_final_stage=stage_index+1==len(items)
                    stage_scope=nullcontext()
                    with stage_scope:
                        if capture_geometry_diagnostics: record_geometry_memory(f'pyramid_stage_{stage_index}_geometry_forward_before')
                        if capture_rgbd:
                            # The RGB-D pass only needs the shared Helios path
                            # through the last active Sightline block.  It
                            # captures Q/K and returns before norm_out/proj_out.
                            prediction=_rgbd_prefix_forward(pipe,item['noisy_latents'],item,prompt_embeds,history,chunk*8,int(cfg.rgbd_prefix_stop_layer))
                            final_prediction=None
                        else:
                            prediction=_model_prediction(pipe,item['noisy_latents'],item,prompt_embeds,history,chunk*8,routing_scope_active=True); final_prediction=prediction
                        if capture_geometry_diagnostics: record_geometry_memory(f'pyramid_stage_{stage_index}_geometry_forward_after')
                        stage_scale_deltas={layer:(None if pipe.transformer._sightline_processors[layer].last_scale_delta is None else pipe.transformer._sightline_processors[layer].last_scale_delta.detach()) for layer in capture_layers}
                        if args.train and capture_rgbd and prediction is not None:
                            # The first RGB-D backward needs only the captured
                            # Q/K branches.  Drop the unrelated stage output
                            # graph before checkpoint recomputation.  Both
                            # RGB-D and FM/cross therefore use separate graphs.
                            final_prediction=None
                            del prediction
                        if capture_rgbd:
                            stage_rgbd_captures={layer:(pipe.transformer._sightline_processors[layer].last_rgbd_native_q,pipe.transformer._sightline_processors[layer].last_rgbd_native_k,pipe.transformer._sightline_processors[layer].last_rgbd_dq,pipe.transformer._sightline_processors[layer].last_rgbd_dk,pipe.transformer._sightline_processors[layer].last_rgbd_query_indices,pipe.transformer._sightline_processors[layer].last_rgbd_key_indices) for layer in active_rgbd_layers}
                            rgbd_capture_seen.add(stage_index)
                            # Backpropagate one captured stage at a time.  The
                            # stage loss is still the mean over the configured
                            # layers.  Each entry is only a custom-autograd
                            # scalar; pairwise graphs are recomputed in its
                            # streaming backward implementation.
                            layer_rgbd_metrics=[]
                            for rgbd_layer in active_rgbd_layers:
                                layer_rgbd_metric=_rgbd_loss(trainable,pipe.transformer._sightline_processors,(rgbd_layer,),stage_rgbd_plan,margin=cfg.m_geo,temperature=cfg.tau_geo,local_scale=float(item['geometry_sigma_scale'].detach().float().mean()),timings=perf,captures={rgbd_layer:stage_rgbd_captures[rgbd_layer]})
                                layer_rgbd_metrics.append(layer_rgbd_metric)
                                del layer_rgbd_metric
                            stage_rgbd_metric=torch.stack(layer_rgbd_metrics).mean() if layer_rgbd_metrics else torch.zeros((),device=source.device)
                            if args.train and stage_rgbd_metric.requires_grad:
                                _backward_rgbd_stage(float(cfg.lambda_rgbd)*rgbd_weights[stage_index]*stage_rgbd_metric,trainable,retain_graph=False)
                            rgbd_stage_losses[stage_index]=stage_rgbd_metric.detach()
                            for rgbd_layer in active_rgbd_layers:
                                _release_rgbd_capture(pipe.transformer._sightline_processors,(rgbd_layer,),preserve_cross_capture=False)
                                del stage_rgbd_captures[rgbd_layer]
                            del stage_rgbd_metric
                            del layer_rgbd_metrics
                            del stage_rgbd_captures
                        if args.train and capture_rgbd:
                            # Reconfigure the second pass after every RGB-D
                            # stage.  Only final-stage cross layers retain full
                            # K; RGB-D-only layers never do.
                            final_prediction=None
                            for layer in capture_layers:
                                processor=pipe.transformer._sightline_processors[layer]
                                processor.capture_diagnostics=False; processor.capture_rgbd=False; processor.capture_query_indices=None; processor.capture_key_indices=None; processor.capture_full_key=False
                                if layer in stage_scale_deltas:
                                    processor.reuse_scale_delta=True; processor.scale_delta_override=stage_scale_deltas[layer]
                            second_capture_layers=tuple(active_corr_layers) if capture_cross else ()
                            for layer in second_capture_layers:
                                processor=pipe.transformer._sightline_processors[layer]
                                processor.capture_diagnostics=True
                                processor.capture_numeric_diagnostics=bool(capture_geometry_diagnostics)
                                if processor.conditioner is not None:
                                    processor.conditioner.capture_numeric_diagnostics=bool(capture_geometry_diagnostics)
                                processor.capture_rgbd=False
                                processor.capture_query_indices=cross_plan.query_indices if cross_plan is not None and (args.train or args.probe_capture) else None
                                processor.capture_key_indices=None
                                processor.capture_full_key=bool(layer in active_corr_layers)
                            capture_correspondence=bool(second_capture_layers)
                            oom_state['stage']=f'flow_stage_{stage_index}_reforward' if not capture_correspondence else 'final_stage_reforward'
                            prediction=_model_prediction(pipe,item['noisy_latents'],item,prompt_embeds,history,chunk*8,routing_scope_active=True); final_prediction=prediction
                            if capture_geometry_diagnostics: record_geometry_memory(f'pyramid_stage_{stage_index}_geometry_reforward_after')
                        if capture_geometry_diagnostics:
                            first_processor=pipe.transformer._sightline_processors[int(cfg.sightline_layers[0])]
                            diagnostic=first_processor.last_numeric_diagnostics or {}
                            stage_fields={'stage_index':int(item['stage_index']),'sigma_local':float(item['sigma_local'].detach().float().mean()),'sigma_start':float(item['sigma_start']),'sigma_end':float(item['sigma_end']),'sigma_abs':float(item['sigma_abs'].detach().float().mean()),'geometry_sigma_scale':float(item['geometry_sigma_scale'].detach().float().mean())}
                            fm_sigma_trace.append({'stage':stage_index,
                                'item_timestep':float(item['timesteps'].detach().float().mean()),
                                'processor_timestep':diagnostic.get('timestep'),
                                'item_sigma':float(item['sigmas'].detach().float().mean()),**stage_fields})
                            active_stage_trace.append(stage_fields)
                            if not capture_rgbd:
                                # Snapshot the diagnostics from the forward
                                # that owns this FM backward before the next
                                # prefix/rollout can touch the processors.
                                backward_geometry_diagnostics.update(copy.deepcopy({str(layer):pipe.transformer._sightline_processors[layer].last_numeric_diagnostics for layer in cfg.sightline_layers if pipe.transformer._sightline_processors[layer].last_numeric_diagnostics is not None}))
                        if capture_correspondence: record_vram('final_stage_forward')
                        stage_loss=(prediction.float()-item['target'].float()).square().mean(); stage_losses.append(stage_loss)
                        if args.train and not is_final_stage:
                            timing_sync(); backward_started=time.perf_counter(); (stage_loss/len(items)).backward(retain_graph=False); timing_sync(); perf['backward_seconds']+=time.perf_counter()-backward_started
                            # The early-stage backward is complete. Retain only its
                            # scalar metric and sigma; its flow tensors otherwise
                            # remain referenced by ``items`` during the much larger
                            # final-stage/correspondence backward.
                            stage_losses[-1]=stage_loss.detach()
                            for disposable in ('noisy_latents','target','start_point','end_point','noise','timesteps'):
                                item.pop(disposable,None)
                fm=torch.stack([loss.detach() if args.train else loss for loss in stage_losses]).mean()
                cross_weight=trainable.lambda_corr(step/total_steps)
                oom_state['stage']='correspondence_forward'
                if train_rgbd and set(rgbd_plans)!=rgbd_capture_seen:
                    raise RuntimeError('RGB-D plans did not capture exactly their valid sparse Q/K stages')
                corr_metric=_corr_loss(trainable,pipe.transformer._sightline_processors,cross_rows,chunk,active_corr_layers,cfg.correspondence_rows_per_batch,sampling_seed=correspondence_seed,timings=perf,plan=cross_plan if args.train else None,vram_callback=record_vram if args.profile_timing else None) if (train_correspondence or diagnostic_correspondence) and cross_rows is not None and len(cross_rows) else fm.new_zeros(())
                record_vram('correspondence_loss')
                rgbd_stage_values={stage_index:rgbd_stage_losses.get(stage_index,fm.new_zeros(())).detach() for stage_index in (1,2)}
                rgbd=(rgbd_weights[1]*rgbd_stage_values[1]+rgbd_weights[2]*rgbd_stage_values[2]) if train_rgbd else fm.new_zeros(())
                corr=corr_metric if train_correspondence else fm.new_zeros(())
                final_flow=stage_losses[-1]/len(items)
                rgbd_term=float(cfg.lambda_rgbd)*rgbd
                cross_term=cross_weight*corr
                total=_total_metric(fm,rgbd_term,cross_term)
                rgbd_plan1=rgbd_plans.get(1); rgbd_plan2=rgbd_plans.get(2)
                losses.update(fm=fm,rgbd=rgbd,corr=corr,total=total,stage=stage_losses,
                              sigmas=[float(item['sigmas'].mean()) for item in items],
                              sigma_local=[float(item['sigma_local'].detach().float().mean()) for item in items],
                              sigma_abs=[float(item['sigma_abs'].detach().float().mean()) for item in items],
                              geometry_sigma_scale=[float(item['geometry_sigma_scale'].detach().float().mean()) for item in items],
                              sigma_start=[float(item['sigma_start']) for item in items],sigma_end=[float(item['sigma_end']) for item in items],
                              rgbd_stage1_loss=float(rgbd_stage_values[1]),rgbd_stage2_loss=float(rgbd_stage_values[2]),
                              rgbd_stage1_scale=float(rgbd_scales[1].detach()),rgbd_stage2_scale=float(rgbd_scales[2].detach()),
                              rgbd_stage1_local_aux_scales=sorted({0.1,max(0.1,min(1.0,float(rgbd_scales[1].detach()))),1.0}),rgbd_stage2_local_aux_scales=sorted({0.1,max(0.1,min(1.0,float(rgbd_scales[2].detach()))),1.0}),
                              rgbd_stage1_weight=float(rgbd_weights[1].detach()),rgbd_stage2_weight=float(rgbd_weights[2].detach()),
                              rgbd_stage1_mapping_input_count=0 if rgbd_plan1 is None else int(rgbd_plan1.mapping_input_count),rgbd_stage1_mapping_output_count=0 if rgbd_plan1 is None else int(rgbd_plan1.mapping_output_count),
                              rgbd_stage2_mapping_input_count=0 if rgbd_plan2 is None else int(rgbd_plan2.mapping_input_count),rgbd_stage2_mapping_output_count=0 if rgbd_plan2 is None else int(rgbd_plan2.mapping_output_count),
                              rgbd_stage1_negative_key_t_match=True if rgbd_plan1 is None else bool(rgbd_plan1.negative_key_t_match),rgbd_stage2_negative_key_t_match=True if rgbd_plan2 is None else bool(rgbd_plan2.negative_key_t_match),
                              rgbd_stage1_negative_pair_count=0 if rgbd_plan1 is None else int(rgbd_plan1.negative_pair_count),rgbd_stage2_negative_pair_count=0 if rgbd_plan2 is None else int(rgbd_plan2.negative_pair_count),
                              rgbd_term=rgbd_term.detach(),cross_term=cross_term,cross_weight=float(cross_weight))
                if args.train:
                    oom_state['stage']='correspondence_and_fm_backward'
                    timing_sync(); backward_started=time.perf_counter()
                    # Geometry rho controls the bounded residual and is deliberately
                    # FM-only: correspondence trains geometric features but cannot
                    # lower its loss by merely amplifying rho.
                    if corr.requires_grad:
                        final_flow.backward(retain_graph=True)
                        flow_rho_grads={id(beta):None if beta.grad is None else beta.grad.detach().clone() for beta in trainable.conditioner.rho_parameters()}
                        if cross_term.requires_grad: cross_term.backward()
                        for beta in trainable.conditioner.rho_parameters(): beta.grad=flow_rho_grads[id(beta)]
                    else:
                        final_flow.backward()
                    timing_sync(); perf['backward_seconds']+=time.perf_counter()-backward_started
                    record_vram('backward')
                    record_vram('fm_backward')
                    record_geometry_memory('backward_after')
                final=items[-1]; generated=(final['noisy_latents']-final['sigmas']*final_prediction).detach()
                if args.probe_capture:
                    layer_captures=[]
                    for layer in active_corr_layers:
                        processor=pipe.transformer._sightline_processors[layer]
                        try: selected,positives,_,_,_=_mapped_correspondences(processor,cross_rows,chunk)
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
                    rho_q,rho_k=trainable.conditioner.rho_values()
                    final_stage_loss=float((final_prediction.float()-final['target'].float()).square().mean())
                    first=layer_captures[0]
                    correct_ray_loss=final_stage_loss
                    wrong_ray_loss=float((wrong.float()-final['target'].float()).square().mean())
                    camera_sensitivity=(wrong_ray_loss-correct_ray_loss)/max(correct_ray_loss,1e-8)
                    probe_payload.update(source='real_helios_forward',baseline=bool(args.alpha_zero_baseline),layer=first['layer'],sigma=float(final['sigmas'].mean()),attention_logits=first['attention_logits'],positive_key_indices=first['positive_key_indices'],memory_count=first['memory_count'],layer_captures=layer_captures,fm_loss=float(fm.detach()),baseline_final_stage_loss=final_stage_loss,correct_ray_loss=correct_ray_loss,wrong_ray_loss=wrong_ray_loss,camera_sensitivity=camera_sensitivity,memory_zero_loss=float((zero.float()-final['target'].float()).square().mean()),memory_shuffle_loss=float((shuffled_prediction.float()-final['target'].float()).square().mean()),corr_loss=float(corr_metric.detach()),rho_q=rho_q,rho_k=rho_k,vram_gb=float(torch.cuda.max_memory_allocated()/2**30),step_time_sec=normal_step_time,ablation_time_sec=time.perf_counter()-ablation_started)
            for layer in active_corr_layers:
                pipe.transformer._sightline_processors[layer].capture_diagnostics=False
                pipe.transformer._sightline_processors[layer].capture_query_indices=None
            # Numerical closure only; the actual source/previous-chunk boundary
            # was enforced by the three-stage flow and constrain_flow_items.
            generated[:,:,0:1]=clean_boundary.to(generated)
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
                processor.last_q=processor.last_k=processor.last_native_q=processor.last_native_k=processor.last_augmented_q=processor.last_augmented_k=processor.last_capture_query_indices=None
                processor.last_rgbd_q=processor.last_rgbd_k=processor.last_rgbd_native_q=processor.last_rgbd_native_k=processor.last_rgbd_dq=processor.last_rgbd_dk=processor.last_rgbd_rays_q=processor.last_rgbd_rays_k=processor.last_rgbd_legal_mask=processor.last_rgbd_query_indices=processor.last_rgbd_key_indices=None
                processor.capture_rgbd=False
                processor.last_hidden_states=processor.last_key_identities=None
                processor.last_attention_bias=None
            if not keep_graph: perf['prefix_generation_seconds']+=time.perf_counter()-chunk_started
            return generated
        try:
            if phase['name']=='P3':
                _,policies=run_causal_prefix_chunks(phase['max_chunks'],train_chunk,forward_chunk)
            elif args.probe_only and args.alpha_zero_baseline:
                with torch.no_grad(): _,policies=run_single_graph_chunks(phase['max_chunks'],train_chunk,forward_chunk)
            else:
                _,policies=run_single_graph_chunks(phase['max_chunks'],train_chunk,forward_chunk)
        finally:
            pass
        if args.probe_capture:
            probe_payload['rho_grad']={name:0.0 if beta.grad is None else float(beta.grad.detach().abs()) for name,beta in ((f'{index}.q',layer.beta_q) for index,layer in trainable.conditioner.layers.items())}
        if args.train:
            active_phase=curriculum_phase(step,p1_steps=cfg.p1_steps,p2_steps=cfg.p2_steps,p3_steps=cfg.p3_steps)
            rho_grads=[beta.grad for beta in trainable.conditioner.rho_parameters()]
            if active_phase['name']=='P1' and (any(grad is None for grad in rho_grads) or not all(torch.isfinite(grad).all() for grad in rho_grads)): raise RuntimeError('P1 rho gradient missing or non-finite')
            if active_phase['name']=='P2':
                lora_grads=[p.grad for p in lora_params if p.requires_grad and p.grad is not None]
                if lora_params and (not lora_grads or not all(torch.isfinite(g).all() for g in lora_grads)): raise RuntimeError('P2 LoRA gradient missing or non-finite')
            if active_phase['name']=='P3' and losses['corr'].requires_grad:
                if any(beta.abs().detach()>1e-6 for beta in trainable.conditioner.rho_parameters()):
                    geometry_params=list(trainable.conditioner.geometry_parameters())
                    corr_grads=[p.grad for p in geometry_params if p.grad is not None]
                    if not corr_grads or not all(torch.isfinite(g).all() for g in corr_grads): raise RuntimeError('P3 geometry gradient missing or non-finite')
            oom_state['stage']='ddp_gradient_average'; record_vram('pre_ddp')
            optimized=[p for group in optimizer.param_groups for p in group['params']]; _average_gradients(optimized,world_size); grad_norm=torch.nn.utils.clip_grad_norm_([p for p in optimized if p.grad is not None],cfg.grad_clip); optimizer.step(); scheduler.step()
        else: grad_norm=torch.tensor(0.)
        rho_q,rho_k=trainable.conditioner.rho_values()
        lr_groups={f"lr_{group.get('name', index)}":float(group['lr']) for index,group in enumerate(optimizer.param_groups)}
        step_seconds=time.perf_counter()-started
        diagnostics=copy.deepcopy(backward_geometry_diagnostics) if capture_geometry_diagnostics else {}
        if capture_geometry_diagnostics:
            def _parameter_rms(parameter): return float(parameter.detach().float().square().mean().sqrt().cpu())
            def _grad_rms(parameter): return None if parameter.grad is None else float(parameter.grad.detach().float().square().mean().sqrt().cpu())
            for layer_text,diagnostic in diagnostics.items():
                conditioner=trainable.conditioner.for_layer(int(layer_text))
                diagnostic.update({
                    'q_projector_weight_rms':_parameter_rms(conditioner.q_proj.weight),'k_projector_weight_rms':_parameter_rms(conditioner.k_proj.weight),
                    'q_projector_grad_rms':_grad_rms(conditioner.q_proj.weight),'k_projector_grad_rms':_grad_rms(conditioner.k_proj.weight),
                    'q_projector_raw_rms':conditioner.last_pre_norm_rms['q'],'k_projector_raw_rms':conditioner.last_pre_norm_rms['k'],
                    'q_projector_soft_gain':conditioner.last_soft_gain['q'],'k_projector_soft_gain':conditioner.last_soft_gain['k'],
                    'rho_q_grad_rms':_grad_rms(conditioner.beta_q),'rho_k_grad_rms':_grad_rms(conditioner.beta_k),
                    'rho_q':float(conditioner.rho_values()[0].detach().cpu()),'rho_k':float(conditioner.rho_values()[1].detach().cpu()),
                })
        def _summary(field):
            values=[float(value[field]) for value in diagnostics.values() if value.get(field) is not None]
            if not values: return {}
            ordered=sorted(values); return {'mean':sum(values)/len(values),'p50':ordered[len(ordered)//2],'p95':ordered[min(len(ordered)-1,round(.95*(len(ordered)-1)))],'max':max(values),'min':min(values)}
        geometry_aggregate={} if not capture_geometry_diagnostics else {
            'q_residual_ratio':_summary('delta_q_over_q_native'),'k_residual_ratio':_summary('delta_k_over_k_native'),
            'q_projector_pre_norm_rms':_summary('proj_q_raw_rms'),'k_projector_pre_norm_rms':_summary('proj_k_raw_rms'),
            'rho_q':_summary('rho_q'),'rho_k':_summary('rho_k'),
        }
        final_stage=len(losses['sigma_abs'])-1 if losses.get('sigma_abs') else -1
        geometry_context={} if not capture_geometry_diagnostics else {'step':step,'phase':phase['name'],'train_chunk':train_chunk,'pyramid_stage':final_stage,'stage_index':final_stage,'sigma_local':losses['sigma_local'][final_stage],'sigma_start':losses['sigma_start'][final_stage],'sigma_end':losses['sigma_end'][final_stage],'sigma_abs':losses['sigma_abs'][final_stage],'geometry_sigma_scale':losses['geometry_sigma_scale'][final_stage],'soft_rms_epsilon':0.05,'sightline_residual_scale':1.0}
        row={'step':step,'record':record.trajectory_id,'phase':phase['name'],'max_chunks':phase['max_chunks'],'window_start_chunk':window_start,'train_chunk':train_chunk,'executed_chunks':len(policies),'policies':policies,'gt_prefix_probability':gt_prefix_p,'gt_prefix_used':use_gt_prefix,'correct_ray_loss':probe_payload.get('correct_ray_loss'),'wrong_ray_loss':probe_payload.get('wrong_ray_loss'),'camera_sensitivity':probe_payload.get('camera_sensitivity'),'flow_loss':float(losses['fm'].detach()),'rgbd_loss':float(losses['rgbd'].detach()),'corr_loss':float(losses['corr'].detach()),'total_loss':float(losses['total'].detach()),'rgbd_stage1_loss':losses['rgbd_stage1_loss'],'rgbd_stage2_loss':losses['rgbd_stage2_loss'],'rgbd_stage1_scale':losses['rgbd_stage1_scale'],'rgbd_stage2_scale':losses['rgbd_stage2_scale'],'rgbd_stage1_local_aux_scales':losses['rgbd_stage1_local_aux_scales'],'rgbd_stage2_local_aux_scales':losses['rgbd_stage2_local_aux_scales'],'rgbd_stage1_weight':losses['rgbd_stage1_weight'],'rgbd_stage2_weight':losses['rgbd_stage2_weight'],'rgbd_stage1_mapping_input_count':losses['rgbd_stage1_mapping_input_count'],'rgbd_stage1_mapping_output_count':losses['rgbd_stage1_mapping_output_count'],'rgbd_stage2_mapping_input_count':losses['rgbd_stage2_mapping_input_count'],'rgbd_stage2_mapping_output_count':losses['rgbd_stage2_mapping_output_count'],'rgbd_stage1_negative_key_t_match':losses['rgbd_stage1_negative_key_t_match'],'rgbd_stage2_negative_key_t_match':losses['rgbd_stage2_negative_key_t_match'],'rgbd_stage1_negative_pair_count':losses['rgbd_stage1_negative_pair_count'],'rgbd_stage2_negative_pair_count':losses['rgbd_stage2_negative_pair_count'],'rgbd_effective_weight':float(cfg.lambda_rgbd),'cross_effective_weight':losses['cross_weight'],'stage_losses':[float(x.detach()) for x in losses['stage']],'stage_sigmas':losses['sigmas'],'stage_sigma_local':losses['sigma_local'],'stage_sigma_start':losses['sigma_start'],'stage_sigma_end':losses['sigma_end'],'stage_sigma_abs':losses['sigma_abs'],'stage_geometry_sigma_scale':losses['geometry_sigma_scale'],'sampled_sigma':losses['sigmas'],'fm_sigma_trace':fm_sigma_trace if capture_geometry_diagnostics else [],'sigma_band':sigma_band,'rho_q':rho_q,'rho_k':rho_k,'geometry_diagnostics':diagnostics,'geometry_diagnostic_context':geometry_context,'geometry_aggregate':geometry_aggregate,'geometry_memory_diagnostics':geometry_memory_diagnostics,'lambda_rgbd':float(cfg.lambda_rgbd),'m_geo':float(cfg.m_geo),'tau_geo':float(cfg.tau_geo),'max_intra_corr_rows':int(cfg.max_intra_corr_rows),'initialization_hash':initialization_hash,'grad_norm':float(grad_norm),'lr':scheduler.get_last_lr()[0],**lr_groups,'gradient_checkpointing':checkpointing,'helios_runtime_patch':runtime_patch,'seconds':step_seconds,'step_total_seconds':step_seconds,**perf,'timing_synchronized':bool(args.profile_timing),'uses_future_gt':False}
        if rank==0 and (args.profile_timing or capture_geometry_diagnostics or step==start_step or step+1==stop):
            with metrics.open('a') as handle: handle.write(json.dumps(row)+'\n')
        if args.probe_capture:
            Path(args.probe_capture).parent.mkdir(parents=True,exist_ok=True); torch.save(probe_payload,args.probe_capture)
        cadence=save_every if args.save_every is not None else checkpoint_interval(step)
        checkpoint_due=(step+1)%cadence==0 or step+1==args.max_steps or (args.checkpoint_smoke_step is not None and step+1==args.checkpoint_smoke_step)
        if args.train and checkpoint_due:
            rng_states=gather_rank_rng_states(world_size,device)
            if rank==0: save_runtime_checkpoint(output/f'checkpoint-{step:06d}.pt',trainable,runner.memory,pipe.transformer,optimizer,scheduler,step,config=config,helios_fingerprint=fingerprint,layers=cfg.sightline_layers,memory_config=memory_config,provenance=provenance,rng_states=rng_states,world_size=world_size,runtime_patch=runtime_patch,helios_trainable_names=helios_trainable_names)
            if world_size>1: dist.barrier()
    if world_size>1: dist.destroy_process_group()

if __name__=='__main__': main()
