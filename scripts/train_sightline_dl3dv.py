"""Causal single-graph Sightline training entrypoint.

This file defines the complete 1->6 chunk control flow.  It intentionally does
not start unless --train is passed, so repository validation cannot launch a
formal run accidentally.
"""
from __future__ import annotations
import argparse, hashlib, json, sys, time
from dataclasses import asdict
from pathlib import Path
import torch
from long_video.config import load_sightline_config
from long_video.training.flow_matching_exact import exact_flow_matching_items
from long_video.training.sightline import (
    SightlineTrainable, install_lora, curriculum_phase, select_train_chunk,
    run_single_graph_chunks, native_history_16_2_1,
)
from long_video.training.sightline_data import load_sightline_manifest, load_latent_tensor
from long_video.sightline.helios_integration import SightlineRayProvider, install_sightline_attention
from long_video.sightline.pipeline import SightlinePipeline
from long_video.training.sightline_checkpoint import save_runtime_checkpoint

def _prompt(pipe, text, device):
    result=pipe._get_t5_prompt_embeds(text,device=device,dtype=torch.bfloat16)
    if not isinstance(result,(tuple,list)) or len(result)!=2:
        raise RuntimeError("pinned Helios prompt API must return (prompt_embeds, prompt_mask)")
    return result

def _model_prediction(pipe, noisy, item, prompt_embeds, prompt_mask, history, current_start):
    latent_t=noisy.shape[2]
    current_indices=torch.arange(current_start,current_start+latent_t,device=noisy.device).view(1,-1)
    output=pipe.transformer(
        hidden_states=noisy,timestep=item["timesteps"],encoder_hidden_states=prompt_embeds,
        encoder_attention_mask=prompt_mask,indices_hidden_states=current_indices,
        latents_history_long=history["long"][0],indices_latents_history_long=history["long"][1],
        latents_history_mid=history["mid"][0],indices_latents_history_mid=history["mid"][1],
        latents_history_short=history["short"][0],indices_latents_history_short=history["short"][1],
        attention_kwargs={"current_chunk":current_start//8},
    )
    prediction=output[0] if isinstance(output,(tuple,list)) else getattr(output,"sample",output)
    if prediction.shape!=noisy.shape: raise RuntimeError(f"prediction shape {prediction.shape} != {noisy.shape}")
    return prediction

def _load_correspondence(path):
    payload=json.loads(Path(path).read_text())
    if payload.get("schema_version")!="sightline-correspondence-v1": raise RuntimeError("stale correspondence cache")
    return payload["rows"]

def _corr_loss(trainable, processors, rows, chunk, device):
    candidates=[row for row in rows if chunk in row["query_chunk_memberships"]]
    if not candidates: raise RuntimeError(f"no correspondence rows for train chunk {chunk}")
    processor=next((processors[layer] for layer in sorted(processors) if processors[layer].last_q is not None),None)
    if processor is None: raise RuntimeError("selected correspondence layer did not expose real Q/K")
    q=processor.last_q; k=processor.last_k
    logits=torch.einsum("bqhd,bkhd->bhqk",q,k)*(q.shape[-1]**-0.5)
    q_count=q.shape[1]; k_count=k.shape[1]; current_len=processor.last_current_length
    memory_count=processor.last_attention_meta.get('memory_tokens',0); native_key_len=k_count-memory_count
    selected=[]; positives=[]; weights=[]
    for row in candidates:
        if int(row['query_chunk'])!=chunk: continue
        qi=q_count-current_len+int(row["query_attention_index"])
        if int(row['key_chunk'])==chunk:
            ki=native_key_len-current_len+int(row["key_attention_index"])
        else:
            wanted=(int(row['key_chunk']),int(row['key_latent_temporal']),int(row['key_y'])//2,int(row['key_x'])//2)
            memory_index=next((i for i,token in enumerate(processor.memory.tokens) if (token.chunk_index,token.temporal,token.pooled_y,token.pooled_x)==wanted),None)
            if memory_index is None: continue
            ki=native_key_len+memory_index
        if qi<q_count and ki<k_count: selected.append(qi); positives.append(ki); weights.append(float(row["weight"]))
    if not selected: raise RuntimeError("correspondence identities do not map to the real attention axes")
    sampled=logits[:,:,torch.tensor(selected,device=device)]
    positive=torch.tensor(positives,device=device).view(1,-1).expand(sampled.shape[0],-1)
    weight=torch.tensor(weights,device=device,dtype=torch.float32).view(1,-1).expand_as(positive)
    return trainable.correspondence(sampled,positive,weight)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--config",default="configs/sightline.yaml"); p.add_argument("--model",required=True)
    p.add_argument("--helios-root",required=True); p.add_argument("--manifest",required=True)
    p.add_argument("--expected-records",type=int,default=100); p.add_argument("--step",type=int,default=0)
    p.add_argument("--record-index",type=int,default=0); p.add_argument("--prompt",default="A stable realistic view of the same scene.")
    p.add_argument("--metrics",default="sightline_metrics.jsonl"); p.add_argument("--checkpoint-out"); p.add_argument("--train",action="store_true")
    args=p.parse_args()
    if not args.train: raise SystemExit("training entrypoint validated; pass --train explicitly to start")
    cfg=load_sightline_config(args.config); phase=curriculum_phase(args.step)
    records=load_sightline_manifest(args.manifest,expected_count=args.expected_records)
    record=records[args.record_index%len(records)]
    latent_key="gt_latent_cache" if "gt_latent_cache" in record.raw else "latent_cache"
    latents=load_latent_tensor(record.path(latent_key)).to("cuda",dtype=torch.bfloat16)
    if latents.shape[2]<49: raise ValueError("six chunks require at least 49 overlapping latent frames")
    c2w_np,K_np=record.load_cameras()
    c2w=torch.from_numpy(c2w_np).to("cuda",dtype=torch.float32).unsqueeze(0)
    K=torch.from_numpy(K_np).to("cuda",dtype=torch.float32).unsqueeze(0)
    sys.path.insert(0,args.helios_root)
    from helios.diffusers_version.pipeline_helios_diffusers import HeliosPipeline
    import helios.diffusers_version.transformer_helios_diffusers as helios_source
    pipe=HeliosPipeline.from_pretrained(args.model,torch_dtype=torch.bfloat16).to("cuda")
    heads=int(pipe.transformer.config.num_attention_heads); inner=int(pipe.transformer.config.attention_head_dim*heads)
    trainable=SightlineTrainable(inner,heads=heads).to("cuda",dtype=torch.bfloat16)
    for parameter in pipe.transformer.parameters(): parameter.requires_grad_(False)
    installed_lora=install_lora(pipe.transformer,cfg.lora_layers,rank=cfg.lora_rank) if cfg.lora_layers else ()
    provider=SightlineRayProvider(source_height=cfg.source_height,source_width=cfg.source_width)
    runner=SightlinePipeline(pipe,config=cfg,conditioner=trainable.conditioner,ray_provider=provider)
    install_sightline_attention(pipe.transformer,trainable.conditioner,provider,layers=cfg.sightline_layers,
        helios_module=helios_source,memory=runner.memory)
    runner.memory.set_enabled(phase["memory"])
    for name,parameter in pipe.transformer.named_parameters():
        if "lora_" in name: parameter.requires_grad_(phase["lora"])
    params=[{"params":list(trainable.parameters()),"lr":cfg.learning_rate},
            {"params":list(runner.memory.parameters()),"lr":cfg.learning_rate},
            {"params":[p for p in pipe.transformer.parameters() if p.requires_grad],"lr":cfg.lora_learning_rate}]
    optimizer=torch.optim.AdamW([group for group in params if group["params"]],weight_decay=.01)
    prompt_embeds,prompt_mask=_prompt(pipe,args.prompt,"cuda")
    corr_rows=_load_correspondence(record.path("correspondence_cache")) if phase["correspondence"] else None
    train_chunk=select_train_chunk(phase["max_chunks"]); completed=[]; completed_ids=[]
    source=latents[:,:,:1]; losses={}
    def forward_chunk(chunk,keep_graph):
        target=latents[:,:,chunk*8:chunk*8+9]
        history=native_history_16_2_1(completed,completed_ids,source)
        items=exact_flow_matching_items(pipe,target,stage_steps=cfg.pyramid_steps,device=target.device)
        item=items[-1]
        runner._trajectory_c2w=c2w; runner._trajectory_K=K; runner._source_camera=c2w[:,0]; runner._source_intrinsics=K[:,0]
        runner._prepare_chunk(chunk,item["noisy_latents"],{})
        prediction=_model_prediction(pipe,item["noisy_latents"],item,prompt_embeds,prompt_mask,history,chunk*8)
        fm=(prediction.float()-item["target"].float()).square().mean()
        corr=_corr_loss(trainable,pipe.transformer._sightline_processors,corr_rows,chunk,target.device) if keep_graph and corr_rows else fm.new_zeros(())
        if keep_graph: losses.update(fm=fm,corr=corr,total=fm+trainable.lambda_corr(args.step/2500)*corr)
        estimate=(item["noisy_latents"]-item["sigmas"].view(-1,1,1,1,1)*prediction).detach()
        for local in range(1,estimate.shape[2]): completed.append(estimate[:,:,local:local+1]); completed_ids.append(chunk*8+local)
        runner._finalize_chunk(chunk)
        return estimate
    started=time.perf_counter(); _,policies=run_single_graph_chunks(phase["max_chunks"],train_chunk,forward_chunk)
    losses["total"].backward(); optimizer.step()
    record_out={"step":args.step,"phase":phase["name"],"max_chunks":phase["max_chunks"],"train_chunk":train_chunk,
        "policies":policies,"flow_loss":float(losses["fm"].detach()),"corr_loss":float(losses["corr"].detach()),
        "total_loss":float(losses["total"].detach()),"step_time_sec":time.perf_counter()-started,"uses_future_gt":False}
    with Path(args.metrics).open("a") as handle: handle.write(json.dumps(record_out)+"\n")
    if args.checkpoint_out:
        source_file=Path(args.helios_root)/'helios/diffusers_version/transformer_helios_diffusers.py'
        fingerprint=hashlib.sha256(source_file.read_bytes()).hexdigest()
        save_runtime_checkpoint(args.checkpoint_out,trainable,runner.memory,pipe.transformer,optimizer,None,args.step,
            config=asdict(cfg),helios_fingerprint=fingerprint,layers=cfg.sightline_layers,
            memory_config={'layers':list(cfg.memory_layers),'pool':cfg.memory_pool,'budget':cfg.memory_budget})

if __name__=="__main__": main()
