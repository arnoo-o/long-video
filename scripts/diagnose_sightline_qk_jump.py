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
from long_video.training.flow_matching_exact import exact_flow_matching_items
from long_video.training.rgbd_memory_data import load_rgbd_memory_manifest
from long_video.training.sightline import (
    SightlineTrainable, install_lora, set_initialization_seed, set_lora_enabled,
)
from long_video.training.sightline_checkpoint import restore_runtime_checkpoint, runtime_provenance
from long_video.training.sightline_data import (
    load_latent_tensor, resolve_continuous_latent_cache, validate_latent_cache,
    validate_rgbd_record_latent,
)
from scripts.train_sightline_rgbd import _lr_multiplier, _model_prediction, _prompt, _set_gradient_checkpointing


PROMPT = "A stable realistic view of the same scene."
NUMERIC_FIELDS = (
    "proj_q_rms_before_norm", "proj_k_rms_before_norm", "delta_q_rms",
    "delta_k_rms", "delta_q_over_q_native", "delta_k_over_k_native",
)


def _setup(args):
    cfg=load_sightline_config(args.config); device=torch.device("cuda")
    sys.path.insert(0,args.helios_root)
    from helios.diffusers_version.pipeline_helios_diffusers import HeliosPipeline
    import helios.diffusers_version.transformer_helios_diffusers as helios_source
    pipe=HeliosPipeline.from_pretrained(args.model,torch_dtype=torch.bfloat16).to(device)
    pipe.text_encoder.eval().requires_grad_(False); pipe.vae.eval().requires_grad_(False)
    heads=int(pipe.transformer.config.num_attention_heads); inner=int(pipe.transformer.config.attention_head_dim*heads)
    set_initialization_seed()
    trainable=SightlineTrainable(inner,layers=cfg.sightline_layers,heads=heads).to(device,dtype=torch.float32)
    for parameter in pipe.transformer.parameters(): parameter.requires_grad_(False)
    install_lora(pipe.transformer,cfg.lora_layers,rank=cfg.lora_rank)
    ph,pw=padded_size(cfg.source_height,cfg.source_width)
    provider=SightlineRayProvider(source_height=ph,source_width=pw)
    runner=SightlinePipeline(pipe,config=cfg,conditioner=trainable.conditioner,ray_provider=provider)
    runner.memory.to(device=device,dtype=torch.bfloat16); runner.memory.set_enabled(False)
    install_sightline_attention(pipe.transformer,trainable.conditioner,provider,layers=cfg.sightline_layers,
        helios_module=helios_source,memory=runner.memory,memory_layers=cfg.memory_layers)
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
    c2w=torch.linalg.inv(c2w[:,:1])@c2w
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
        rows.append({"layer":layer,"alpha_q":float(conditioner.alpha_q.detach()),
            "alpha_k":float(conditioner.alpha_k.detach()),**values})
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
        row.update({f"before_{key}":left[key] for key in ("alpha_q","alpha_k",*NUMERIC_FIELDS)})
        row.update({f"after_{key}":right[key] for key in ("alpha_q","alpha_k",*NUMERIC_FIELDS)})
        row.update({key:value for key,value in grad.items() if key!="layer"}); joined.append(row)
    return joined,{"sampled_sigma":fixed_sigma,"timestep":fixed_timestep,"total_grad_norm_before_clip":total_grad,
        "optimizer_lr_for_update":optimizer_lr_for_update,"lr_after_scheduler":float(optimizer.param_groups[0]["lr"])}


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
    for row in rows: lines.append("|"+"|".join(str(row[key]) if key=="layer" else f"{float(row[key]):.8g}" for key in columns)+"|")
    (output/"layers.md").write_text("\n".join(lines)+"\n")


def main():
    p=argparse.ArgumentParser(); p.add_argument("--mode",choices=("init-jump","checkpoint"),required=True)
    p.add_argument("--config",default="configs/sightline.yaml"); p.add_argument("--model",required=True); p.add_argument("--helios-root",required=True)
    p.add_argument("--manifest",required=True); p.add_argument("--latent-cache-root"); p.add_argument("--record-id",default="nrgbd__complete_kitchen__000005")
    p.add_argument("--checkpoint"); p.add_argument("--sigma",type=float,default=.8); p.add_argument("--noise-seed",type=int,default=260827)
    p.add_argument("--output",required=True); args=p.parse_args()
    cfg,pipe,trainable,runner,_=_setup(args); record,latent_path,target,prompt,history=_real_batch(args,cfg,pipe,runner,torch.device("cuda"))
    metadata={"mode":args.mode,"record_id":record.record_id,"latent_cache":str(latent_path),"prompt":PROMPT,"memory_enabled":False,
        "noise_seed":args.noise_seed,"git_commit":__import__("subprocess").check_output(["git","rev-parse","HEAD"],text=True).strip()}
    if args.mode=="init-jump":
        rows,extra=_initialization_jump(args,cfg,pipe,trainable,runner,target,prompt,history); metadata.update(extra)
        fields=[f"{side}_{field}" for side in ("before","after") for field in NUMERIC_FIELDS]+["q_proj_grad_norm","k_proj_grad_norm","q_proj_update_norm","k_proj_update_norm"]
    else:
        if not args.checkpoint: raise ValueError("--checkpoint is required in checkpoint mode")
        source=Path(args.helios_root)/"helios/diffusers_version/transformer_helios_diffusers.py"
        fingerprint=hashlib.sha256(source.read_bytes()).hexdigest(); config=asdict(cfg)
        memory_config={"layers":list(cfg.memory_layers),"pool":cfg.memory_pool,"budget":cfg.memory_budget,"tau_pos":cfg.memory_tau_pos,"tau_angle":cfg.memory_tau_angle}
        payload=torch.load(args.checkpoint,map_location="cpu")
        step=restore_runtime_checkpoint(payload,trainable,runner.memory,pipe.transformer,config=config,helios_fingerprint=fingerprint,
            layers=cfg.sightline_layers,memory_config=memory_config,restore_rng=False,
            provenance=runtime_provenance(pipe,args.model,args.helios_root))
        trainable.eval(); set_lora_enabled(pipe.transformer,True); runner.memory.set_enabled(False); _set_gradient_checkpointing(pipe.transformer,False)
        torch.manual_seed(args.noise_seed); torch.cuda.manual_seed(args.noise_seed)
        item,actual,timestep=_fixed_item(pipe,target,args.sigma)
        rows=_diagnostic_forward(pipe,trainable,runner,target,prompt,history,item)
        metadata.update({"checkpoint":args.checkpoint,"completed_step":step,"requested_sigma":args.sigma,"sampled_sigma":actual,"timestep":timestep})
        fields=["alpha_q","alpha_k",*NUMERIC_FIELDS]
    payload={"metadata":metadata,"summary":_summary(rows,fields),"layers":rows}; _write(Path(args.output),payload,rows)
    print(json.dumps({"metadata":metadata,"summary":payload["summary"]},indent=2),flush=True)


if __name__=="__main__": main()
