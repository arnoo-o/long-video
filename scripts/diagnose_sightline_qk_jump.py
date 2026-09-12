"""Standalone real-batch Sightline Q/K zero-init and checkpoint diagnostics.

This script is deliberately separate from the formal trainer.  It never saves a
checkpoint and only enables detached statistics on the installed processors.
"""
from __future__ import annotations

import argparse, csv, hashlib, json, statistics, sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from long_video.config import load_sightline_config
from long_video.sightline.geometry import assert_latent_geometry, padded_size
from long_video.sightline.helios_integration import SightlineRayProvider, install_sightline_attention
from long_video.sightline.history import NativeHistoryState
from long_video.sightline.pipeline import SightlinePipeline, prepare_source_condition
from long_video.sightline.rays import anchored_inverse_c2w, canonicalize_c2w
from long_video.training.flow_matching_exact import exact_flow_matching_items
from long_video.training.rgbd_plan import build_rgbd_soft_target_plan
from long_video.training.rgbd_probability import rgbd_log_mass_rows
from long_video.training.rgbd_memory_data import load_rgbd_memory_manifest
from long_video.training.sightline import (
    SightlineTrainable, install_lora, set_initialization_seed, set_lora_enabled,
)
from long_video.training.sightline_checkpoint import restore_runtime_checkpoint, runtime_provenance
from long_video.training.sightline_data import (
    load_latent_tensor, resolve_continuous_latent_cache, validate_latent_cache,
    validate_rgbd_record_latent,
)
from scripts.train_sightline_rgbd import (_install_memory_efficient_helios_norm, _lr_multiplier, _model_prediction, _prompt,
    _set_gradient_checkpointing, _load_correspondence, _rgbd_loss, _rgbd_prefix_forward, _release_rgbd_capture)


PROMPT = "A realistic video of the same scene."
NUMERIC_FIELDS = (
    "proj_q_raw_rms", "proj_k_raw_rms", "delta_q_rms",
    "delta_k_rms", "delta_q_over_q_native", "delta_k_over_k_native",
)


def _setup(args):
    cfg=load_sightline_config(args.config); device=torch.device("cuda")
    sys.path.insert(0,args.helios_root)
    source_file=Path(args.helios_root)/'helios/diffusers_version/transformer_helios_diffusers.py'
    runtime_patch=_install_memory_efficient_helios_norm(source_file)
    from helios.diffusers_version.pipeline_helios_diffusers import HeliosPipeline
    import helios.diffusers_version.transformer_helios_diffusers as helios_source
    pipe=HeliosPipeline.from_pretrained(args.model,torch_dtype=torch.bfloat16).to(device)
    pipe.text_encoder.eval().requires_grad_(False); pipe.vae.eval().requires_grad_(False)
    heads=int(pipe.transformer.config.num_attention_heads); inner=int(pipe.transformer.config.attention_head_dim*heads)
    set_initialization_seed()
    trainable=SightlineTrainable(inner,layers=cfg.sightline_layers,heads=heads,rho_init=cfg.rho_init,scale_aug_prob=cfg.scale_augmentation_probability,scale_aug_range=cfg.scale_augmentation_range).to(device,dtype=torch.float32)
    for parameter in pipe.transformer.parameters(): parameter.requires_grad_(False)
    install_lora(pipe.transformer,cfg.lora_layers,rank=cfg.lora_rank)
    ph,pw=padded_size(cfg.source_height,cfg.source_width)
    provider=SightlineRayProvider(source_height=ph,source_width=pw)
    runner=SightlinePipeline(pipe,config=cfg,conditioner=trainable.conditioner,ray_provider=provider)
    # Keep the diagnostic path on the exact same active-routing contract as
    # training: _model_prediction resolves sigma/Geometry through this bound
    # pipeline rather than relying on a copied context.
    pipe._sightline_pipeline=runner
    runner.memory.to(device=device,dtype=torch.bfloat16); runner.memory.set_enabled(False)
    install_sightline_attention(pipe.transformer,trainable.conditioner,provider,
        layers=tuple(sorted(set(cfg.sightline_layers).union(cfg.memory_layers).union(cfg.correspondence_layers))),
        sightline_layers=cfg.sightline_layers,memory_layers=cfg.memory_layers,
        correspondence_layers=cfg.correspondence_layers,helios_module=helios_source,memory=runner.memory)
    pipe._sightline_runtime_patch=runtime_patch
    return cfg,pipe,trainable,runner,provider


def _real_batch(args,cfg,pipe,runner,device):
    records=load_rgbd_memory_manifest(args.manifest)
    matches=[record for record in records if args.record_id in record.record_id]
    record=matches[0] if matches else records[0]
    latent_path=resolve_continuous_latent_cache(record,cache_root=args.latent_cache_root)
    schema,_=validate_latent_cache(latent_path); validate_rgbd_record_latent(record,latent_path)
    if record.frame_count!=97 or record.chunk_count!=3 or schema!="continuous_25":
        raise ValueError("diagnostic requires one unit-owned 97-frame continuous_25 record")
    all_latents=load_latent_tensor(latent_path)
    assert_latent_geometry(all_latents,height=cfg.source_height,width=cfg.source_width,patch_size=pipe.transformer.config.patch_size)
    target=all_latents[:,:,:9].to(device,dtype=torch.bfloat16)
    source,fake,_,_=prepare_source_condition(pipe,Image.open(record.rgb_paths()[0]).convert("RGB"),
        height=cfg.source_height,width=cfg.source_width,device=device)
    c2w_np,K_np=record.load_cameras()
    c2w=torch.from_numpy(np.array(c2w_np[:97],copy=True)).to(device,dtype=torch.float32).unsqueeze(0)
    c2w=canonicalize_c2w(c2w,record.near_depth)
    K=torch.from_numpy(np.array(K_np[:97],copy=True)).to(device,dtype=torch.float32).unsqueeze(0)
    runner.reset_sequence(); runner._trajectory_c2w=c2w; runner._trajectory_K=K
    runner._source_camera=c2w[:,0]; runner._source_intrinsics=K[:,0]
    history_state=NativeHistoryState(source,fake)
    prompt_embeds,_=_prompt(pipe,PROMPT,device)
    return record,latent_path,target,prompt_embeds,history_state


def _prepare(runner,target,history_state):
    runner._prepare_chunk(0,target,{},history_global_coverages=history_state.coverage(),history_validity=history_state.validity())


def _fixed_item(pipe,target,sigma,stage_id=2):
    # Start/end/noise are constructed by the pinned Helios scheduler semantics;
    # then choose the nearest actual scheduler grid point and its real timestep.
    items=exact_flow_matching_items(pipe,target,stage_steps=(2,2,2),device=target.device)
    item=dict(items[stage_id]); grid=torch.as_tensor(pipe.scheduler.sigmas_per_stage[stage_id]).flatten()
    index=int((grid-float(sigma)).abs().argmin()); actual=grid[index].to(target.device,dtype=item["start_point"].dtype)
    shaped=actual
    while shaped.ndim<item["start_point"].ndim: shaped=shaped.unsqueeze(-1)
    item["sigmas"]=shaped.expand(target.shape[0],*([1]*(target.ndim-1)))
    item["timesteps"]=torch.as_tensor(pipe.scheduler.timesteps_per_stage[stage_id][index],device=target.device).reshape(1)
    item["noisy_latents"]=shaped*item["start_point"]+(1-shaped)*item["end_point"]
    return item,float(actual),float(item["timesteps"].item())


def _capture_enabled(pipe,enabled):
    for processor in pipe.transformer._sightline_processors.values():
        processor.capture_numeric_diagnostics=bool(enabled)
        processor.conditioner.capture_numeric_diagnostics=bool(enabled)
        if not enabled: processor.last_numeric_diagnostics=None


def _collect(pipe,trainable):
    rows=[]
    for layer in sorted(pipe.transformer._sightline_processors):
        processor=pipe.transformer._sightline_processors[layer]; values=processor.last_numeric_diagnostics
        if values is None: raise RuntimeError(f"layer {layer} produced no numeric diagnostic")
        conditioner=trainable.conditioner.for_layer(layer)
        rho_q,rho_k=conditioner.rho_values()
        rows.append({"layer":layer,"rho_q":float(rho_q.detach()),
            "rho_k":float(rho_k.detach()),**values})
    return rows


def _diagnostic_forward(pipe,trainable,runner,target,prompt,history,item,rng_state=None):
    if rng_state is not None:
        torch.set_rng_state(rng_state[0]); torch.cuda.set_rng_state(rng_state[1])
    _prepare(runner,target,history); _capture_enabled(pipe,True)
    with torch.no_grad(): _model_prediction(pipe,item["noisy_latents"],item,prompt,history.groups(),0)
    rows=_collect(pipe,trainable); _capture_enabled(pipe,False)
    return rows


def _module_norm(module,attribute):
    values=[]
    for parameter in module.parameters():
        value=getattr(parameter,attribute) if attribute!="data" else parameter.detach()
        if value is not None: values.append(value.detach().float().reshape(-1))
    return float(torch.cat(values).norm().cpu()) if values else 0.0


def _initialization_jump(args,cfg,pipe,trainable,runner,target,prompt,history):
    trainable.train(); set_lora_enabled(pipe.transformer,False); runner.memory.set_enabled(False)
    _set_gradient_checkpointing(pipe.transformer,bool(cfg.gradient_checkpointing))
    lora=[p for n,p in pipe.transformer.named_parameters() if "lora_" in n]
    optimizer=torch.optim.AdamW([{"params":list(trainable.parameters()),"lr":cfg.learning_rate},
        {"params":lora,"lr":cfg.lora_learning_rate},{"params":list(runner.memory.parameters()),"lr":cfg.learning_rate}],weight_decay=.01)
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda step:_lr_multiplier(step,2500))
    torch.manual_seed(args.noise_seed); torch.cuda.manual_seed(args.noise_seed)
    items=exact_flow_matching_items(pipe,target,stage_steps=cfg.pyramid_steps,device=target.device,sigma_range=(.9,1.0))
    fixed=items[-1]; fixed_sigma=float(fixed["sigmas"].mean()); fixed_timestep=float(fixed["timesteps"].item())
    forward_rng=(torch.get_rng_state().clone(),torch.cuda.get_rng_state().clone())
    before=_diagnostic_forward(pipe,trainable,runner,target,prompt,history,fixed,forward_rng)
    snapshots={}
    for key,layer in trainable.conditioner.layers.items():
        snapshots[int(key)]={name:{n:p.detach().clone() for n,p in module.named_parameters()}
            for name,module in (("q_proj",layer.q_proj),("k_proj",layer.k_proj))}
    optimizer.zero_grad(set_to_none=True)
    torch.set_rng_state(forward_rng[0]); torch.cuda.set_rng_state(forward_rng[1])
    for stage,item in enumerate(items):
        _prepare(runner,target,history)
        prediction=_model_prediction(pipe,item["noisy_latents"],item,prompt,history.groups(),0)
        loss=(prediction.float()-item["target"].float()).square().mean()/len(items)
        loss.backward()
        if stage+1<len(items): del prediction,loss
    grad_rows=[]
    for key,layer in trainable.conditioner.layers.items():
        layer_id=int(key)
        grad_rows.append({"layer":layer_id,"q_proj_grad_norm":_module_norm(layer.q_proj,"grad"),
            "k_proj_grad_norm":_module_norm(layer.k_proj,"grad")})
    optimized=[p for group in optimizer.param_groups for p in group["params"]]
    total_grad=float(torch.nn.utils.clip_grad_norm_([p for p in optimized if p.grad is not None],cfg.grad_clip))
    optimizer_lr_for_update=float(optimizer.param_groups[0]["lr"])
    optimizer.step(); scheduler.step()
    for row in grad_rows:
        layer=trainable.conditioner.for_layer(row["layer"])
        for name,module in (("q_proj",layer.q_proj),("k_proj",layer.k_proj)):
            diffs=[]
            for pname,parameter in module.named_parameters(): diffs.append((parameter.detach()-snapshots[row["layer"]][name][pname]).float().reshape(-1))
            row[f"{name}_update_norm"]=float(torch.cat(diffs).norm().cpu())
    after=_diagnostic_forward(pipe,trainable,runner,target,prompt,history,fixed,forward_rng)
    joined=[]
    for left,right,grad in zip(before,after,grad_rows):
        row={"layer":left["layer"]}
        row.update({f"before_{key}":left[key] for key in ("rho_q","rho_k",*NUMERIC_FIELDS)})
        row.update({f"after_{key}":right[key] for key in ("rho_q","rho_k",*NUMERIC_FIELDS)})
        row.update({key:value for key,value in grad.items() if key!="layer"}); joined.append(row)
    return joined,{"sampled_sigma":fixed_sigma,"timestep":fixed_timestep,"total_grad_norm_before_clip":total_grad,
        "optimizer_lr_for_update":optimizer_lr_for_update,"lr_after_scheduler":float(optimizer.param_groups[0]["lr"])}


def _fixed_checkpoint_probe(args,cfg,pipe,trainable,runner,provider,record,target,prompt,history):
    """One deterministic stage-0 probe: RGB-D scores plus cumulative FM prefixes."""
    trainable.eval(); set_lora_enabled(pipe.transformer,False); runner.memory.set_enabled(False)
    _set_gradient_checkpointing(pipe.transformer,False)
    torch.manual_seed(args.noise_seed); torch.cuda.manual_seed(args.noise_seed)
    item,actual,timestep=_fixed_item(pipe,target,args.sigma,stage_id=0)
    original_scales={layer:float(processor.residual_scale) for layer,processor in pipe.transformer._sightline_processors.items()}
    original_context=dict(provider.context) if provider.context is not None else None
    sightline_layers=tuple(cfg.sightline_layers)
    all_prefixes=[(0,),(0,2),(0,2,4),(0,2,4,6),(0,2,4,6,8),(0,2,4,6,8,10),(0,2,4,6,8,10,12)]
    # The active Helios chunk context carries 33 RGB-frame cameras (not the
    # full 97-frame record trajectory).  Prefix comparisons must use that
    # exact current-chunk axis so c2w and K remain shape-aligned.
    base_c2w=runner._trajectory_c2w[:, :33]
    def run_fm(prefix,c2w):
        _prepare(runner,target,history)
        provider.context['c2w']=c2w
        for layer,processor in pipe.transformer._sightline_processors.items():
            processor.residual_scale=1.0 if layer in prefix else 0.0
            processor.capture_rgbd=False; processor.capture_diagnostics=False; processor.capture_numeric_diagnostics=False
            if processor.conditioner is not None: processor.conditioner.capture_numeric_diagnostics=False
        with torch.no_grad(): prediction=_model_prediction(pipe,item['noisy_latents'],item,prompt,history.groups(),0,routing_scope_active=True)
        return float((prediction.float()-item['target'].float()).square().mean())
    prefixes=[]
    try:
        wrong_c2w=anchored_inverse_c2w(base_c2w)
        for prefix in all_prefixes:
            correct=run_fm(prefix,base_c2w); wrong=run_fm(prefix,wrong_c2w)
            prefixes.append({'prefix':list(prefix),'fm_correct':correct,'fm_wrong':wrong,'fm_wrong_minus_correct':wrong-correct})
        full_correct=prefixes[-1]['fm_correct']; full_wrong=prefixes[-1]['fm_wrong']
        off=run_fm((),base_c2w)

        # Capture all three real RGB-D stages with the same fixed record,
        # seed, trajectory and target.  Each stage keeps its real token grid.
        rows=_load_correspondence(record,0,kind='intra')
        rgbd_stages={}; plans={}
        for stage_index in (0,1,2):
            stage_shape=tuple(int(value) for value in provider.context['stage_shapes'][stage_index])
            processor=pipe.transformer._sightline_processors[sightline_layers[0]]
            identities=processor.ray_provider.key_identities(int(np.prod(stage_shape)),processor.memory)
            stage_plan=build_rgbd_soft_target_plan(rows,identities,stage_shape,chunk=0,max_rows=cfg.max_intra_corr_rows,sampling_seed=args.noise_seed+stage_index,device=target.device,source_height=512,source_width=832,c2w=provider.context['c2w'],intrinsics=provider.context['intrinsics'],near_depth=record.near_depth)
            if stage_plan is None:
                continue
            plans[stage_index]=stage_plan
            stage_item,_,_=_fixed_item(pipe,target,args.sigma,stage_id=stage_index)
            query_parts=[stage_plan.query_indices]
            if stage_plan.spacetime_query_indices is not None and stage_plan.spacetime_query_indices.numel():
                query_parts.append(stage_plan.spacetime_query_indices)
            query_indices=torch.unique(torch.cat(query_parts))
            key_indices=stage_plan.sparse_key_indices
            for layer in sightline_layers:
                proc=pipe.transformer._sightline_processors[layer]
                proc.capture_rgbd=True; proc.capture_query_indices=query_indices; proc.capture_key_indices=key_indices; proc.capture_numeric_diagnostics=False
                if proc.conditioner is not None: proc.conditioner.capture_numeric_diagnostics=False
            _prepare(runner,target,history)
            with torch.no_grad():
                _rgbd_prefix_forward(pipe,stage_item['noisy_latents'],stage_item,prompt,history.groups(),0,int(cfg.rgbd_prefix_stop_layer))
            captures={layer:(pipe.transformer._sightline_processors[layer].last_rgbd_native_q,pipe.transformer._sightline_processors[layer].last_rgbd_native_k,pipe.transformer._sightline_processors[layer].last_rgbd_dq,pipe.transformer._sightline_processors[layer].last_rgbd_dk,pipe.transformer._sightline_processors[layer].last_rgbd_query_indices,pipe.transformer._sightline_processors[layer].last_rgbd_key_indices) for layer in sightline_layers}
            layer_values={}
            for layer in sightline_layers:
                diag={}
                with torch.enable_grad():
                    loss=_rgbd_loss(trainable,pipe.transformer._sightline_processors,(layer,),stage_plan,margin=cfg.m_geo,temperature=cfg.tau_geo,local_scales=(.25,.5,1.0),captures={layer:captures[layer]},rgbd_diagnostics=diag)
                layer_values[str(layer)]={'loss':float(loss.detach()),**diag.get(str(layer),{})}
                del loss
            rgbd_stages[str(stage_index)]={'loss':float(np.mean([v['loss'] for v in layer_values.values()])), 'stage_shape':list(stage_shape), 'mapping_input_count':stage_plan.mapping_input_count, 'mapping_output_count':stage_plan.mapping_output_count, 'layers':layer_values}
            _release_rgbd_capture(pipe.transformer._sightline_processors,sightline_layers)
    finally:
        provider.context=original_context
        for layer,processor in pipe.transformer._sightline_processors.items():
            processor.residual_scale=original_scales[layer]
            processor.capture_rgbd=False; processor.capture_diagnostics=False; processor.capture_numeric_diagnostics=False
            if processor.conditioner is not None: processor.conditioner.capture_numeric_diagnostics=False
    return {'checkpoint':args.checkpoint,'completed_step':args.completed_step,'requested_sigma':args.sigma,'sampled_sigma':actual,'timestep':timestep,
        'fm_correct':full_correct,'fm_wrong':full_wrong,'fm_geometry_off':off,'relative_wrong_gap':(full_wrong-full_correct)/max(full_correct,.02),
        'rgbd_stage_losses':{stage:values['loss'] for stage,values in rgbd_stages.items()},
        'rgbd_stages':rgbd_stages,
        'rgbd_conditional_spatial':rgbd_stages.get('0',{}).get('layers',{}),
        'rgbd_plan_mapping_input_count':0 if '0' not in plans else plans[0].mapping_input_count,
        'rgbd_plan_mapping_output_count':0 if '0' not in plans else plans[0].mapping_output_count,
        'prefixes':prefixes}


def _summary(rows,fields):
    result={}
    for field in fields:
        values=[float(row[field]) for row in rows]
        result[field]={"mean":statistics.fmean(values),"median":statistics.median(values),"max":max(values)}
    return result


def _write(output,payload,rows):
    output.mkdir(parents=True,exist_ok=True)
    (output/"diagnostic.json").write_text(json.dumps(payload,indent=2,ensure_ascii=False))
    with (output/"layers.csv").open("w",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    columns=list(rows[0]); lines=["|"+"|".join(columns)+"|","|"+"|".join(["---"]*len(columns))+"|"]
    def cell(value):
        if isinstance(value,(int,float)) and not isinstance(value,bool):
            return f"{float(value):.8g}"
        if isinstance(value,(list,dict,tuple)):
            return json.dumps(value,ensure_ascii=False,separators=(',',':'))
        return str(value)
    for row in rows: lines.append("|"+"|".join(cell(row[key]) for key in columns)+"|")
    (output/"layers.md").write_text("\n".join(lines)+"\n")


def main():
    p=argparse.ArgumentParser(); p.add_argument("--mode",choices=("init-jump","checkpoint","probe"),required=True)
    p.add_argument("--config",default="configs/sightline.yaml"); p.add_argument("--model",required=True); p.add_argument("--helios-root",required=True)
    p.add_argument("--manifest",required=True); p.add_argument("--latent-cache-root"); p.add_argument("--record-id",default="bonn__rgbd_bonn_static_close_far__000010")
    p.add_argument("--checkpoint"); p.add_argument("--sigma",type=float,default=.8); p.add_argument("--noise-seed",type=int,default=260827)
    p.add_argument("--allow-objective-migration",action="store_true"); p.add_argument("--output",required=True); args=p.parse_args()
    cfg,pipe,trainable,runner,provider=_setup(args); record,latent_path,target,prompt,history=_real_batch(args,cfg,pipe,runner,torch.device("cuda"))
    metadata={"mode":args.mode,"record_id":record.record_id,"latent_cache":str(latent_path),"prompt":PROMPT,"memory_enabled":False,
        "noise_seed":args.noise_seed,"git_commit":__import__("subprocess").check_output(["git","rev-parse","HEAD"],text=True).strip()}
    if args.mode=="init-jump":
        rows,extra=_initialization_jump(args,cfg,pipe,trainable,runner,target,prompt,history); metadata.update(extra)
        fields=[f"{side}_{field}" for side in ("before","after") for field in NUMERIC_FIELDS]+["q_proj_grad_norm","k_proj_grad_norm","q_proj_update_norm","k_proj_update_norm"]
    else:
        if not args.checkpoint: raise ValueError("--checkpoint is required in checkpoint mode")
        runtime_patch=getattr(pipe,'_sightline_runtime_patch',None)
        if runtime_patch is None: raise RuntimeError('diagnostic runtime patch provenance is missing')
        fingerprint=runtime_patch['original_source_sha256']; config=asdict(cfg)
        memory_config={"layers":list(cfg.memory_layers),"pool":cfg.memory_pool,"budget":cfg.memory_budget,"tau_pos":cfg.memory_tau_pos,"tau_angle":cfg.memory_tau_angle}
        provenance=runtime_provenance(pipe,args.model,args.helios_root,transformer_source_sha256=fingerprint,
            runtime_patch=runtime_patch,lora_scope=cfg.lora_scope,helios_trainable_scope=())
        payload=torch.load(args.checkpoint,map_location="cuda")
        step=restore_runtime_checkpoint(payload,trainable,runner.memory,pipe.transformer,config=config,helios_fingerprint=fingerprint,
            layers=cfg.sightline_layers,memory_config=memory_config,restore_rng=False,
            provenance=provenance,allow_objective_migration=args.allow_objective_migration)
        trainable.eval(); set_lora_enabled(pipe.transformer,False); runner.memory.set_enabled(False); _set_gradient_checkpointing(pipe.transformer,False)
        metadata.update({"checkpoint":args.checkpoint,"completed_step":step,"requested_sigma":args.sigma})
        if args.mode=="probe":
            args.completed_step=step
            probe=_fixed_checkpoint_probe(args,cfg,pipe,trainable,runner,provider,record,target,prompt,history)
            metadata.update({key:value for key,value in probe.items() if key not in ('prefixes','checkpoint','completed_step')})
            rows=probe['prefixes']; fields=['fm_correct','fm_wrong','fm_wrong_minus_correct']
        else:
            torch.manual_seed(args.noise_seed); torch.cuda.manual_seed(args.noise_seed)
            item,actual,timestep=_fixed_item(pipe,target,args.sigma)
            rows=_diagnostic_forward(pipe,trainable,runner,target,prompt,history,item)
            metadata.update({"sampled_sigma":actual,"timestep":timestep})
            fields=["rho_q","rho_k",*NUMERIC_FIELDS]
    payload={"metadata":metadata,"summary":_summary(rows,fields),"layers":rows}; _write(Path(args.output),payload,rows)
    print(json.dumps({"metadata":metadata,"summary":payload["summary"]},indent=2),flush=True)


if __name__=="__main__": main()
