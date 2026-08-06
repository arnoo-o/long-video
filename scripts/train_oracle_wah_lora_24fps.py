#!/usr/bin/env python3
"""Multi-window 24 FPS Oracle external-warp WAH/Helios LoRA training."""
from __future__ import annotations
import argparse,hashlib,json,os,random,subprocess,sys,time
from pathlib import Path

def _args():
    p=argparse.ArgumentParser(); p.add_argument("--config",default="configs/oracle_wah_training.yaml")
    p.add_argument("--mode",choices=("smoke","train"),required=True); p.add_argument("--manifest",required=True)
    p.add_argument("--resume",default=""); p.add_argument("--max-steps",type=int,default=None)
    p.add_argument("--set",action="append",default=[],dest="overrides"); return p.parse_args()

def _sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""): h.update(block)
    return h.hexdigest()

def _frames(path):
    from PIL import Image
    return [Image.open(item).convert("RGB") for item in sorted(Path(path).glob("*.png"))]

def _mask_frames(values):
    import numpy as np
    from PIL import Image
    return [Image.fromarray(np.rint(np.clip(frame,0,1)*255).astype(np.uint8),mode="L") for frame in values]

def _history(opt,pipe,exact,device,mean,std,first,prompt,warp,visibility,confidence,seq):
    exact.history_visibility_extra_mask_frames=_mask_frames(visibility)
    exact.history_confidence_extra_mask_frames=_mask_frames(confidence)
    prompt_embeds,image_latents,fake_image_latents,video_latents=opt.prepare_condition(
        pipe,first,prompt,exact,device,mean,std,history_frames=warp)
    histories=opt.make_histories(pipe,image_latents,fake_image_latents,exact,device,
                                 video_latents=video_latents,seq=seq)
    return prompt_embeds,histories

def _read_sample(path,chunk_frames):
    import numpy as np
    root=Path(path); metadata=json.loads((root/"metadata.json").read_text(encoding="utf-8"))
    target=_frames(root/"target"/"target_rgb_for_loss")[:chunk_frames]
    warp=_frames(root/"single_chunk_warp"/"warp_rgb")
    visibility=np.load(root/"single_chunk_warp"/"warp_visibility.npy")
    confidence=np.load(root/"single_chunk_warp"/"warp_confidence.npy")
    weights=np.load(root/"primary_loss_weight_latent.npy").astype(np.float32)
    if not(len(target)==len(warp)==len(visibility)==len(confidence)==chunk_frames):
        raise ValueError(f"frame contract mismatch in {root}")
    if metadata["anchor_model_indices"][:5]!=[0,8,16,24,32]:
        raise ValueError("single-chunk anchors are not aligned at 0/8/16/24/32")
    return {"root":root,"metadata":metadata,"target":target,"warp":warp,
            "visibility":visibility,"confidence":confidence,"weights":weights,
            "prompt":(root/"prompt.txt").read_text(encoding="utf-8")}

def _gpu_processes():
    result=subprocess.run(["nvidia-smi","--query-compute-apps=gpu_uuid,pid,used_gpu_memory,name","--format=csv,noheader,nounits"],text=True,capture_output=True)
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]

def _tree_to(value, device):
    """Move a nested tensor tree without changing its non-tensor metadata."""
    import torch
    if isinstance(value, torch.Tensor):
        return value.detach().to(device=device, non_blocking=False)
    if isinstance(value, dict):
        return {key:_tree_to(item,device) for key,item in value.items()}
    if isinstance(value, list):
        return [_tree_to(item,device) for item in value]
    if isinstance(value, tuple):
        return tuple(_tree_to(item,device) for item in value)
    return value

def main():
    args=_args(); repo=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(repo))
    from long_video.config import load_yaml
    config=load_yaml(args.config,args.overrides); physical=int(config.get("physical_gpu",1))
    if physical!=1: raise ValueError("training is restricted to physical GPU 1")
    os.environ["CUDA_VISIBLE_DEVICES"]="1"; os.environ.setdefault("XFORMERS_DISABLED","1")
    import numpy as np
    import torch
    from PIL import Image
    if torch.cuda.device_count()!=1: raise RuntimeError("training must see exactly one GPU")
    torch.cuda.set_device(0); torch.cuda.reset_peak_memory_stats(0)
    required=[key for key in ("wah_root","wah_model","checkpoint_root") if not config.get(key)]
    if required: raise ValueError(f"missing machine path overrides: {required}")
    manifest_path=Path(args.manifest); manifest=json.loads(manifest_path.read_text(encoding="utf-8"))
    train_paths=[item["path"] for item in manifest["sequences"] if item["split"]=="train"]
    diagnostic_paths=[item["path"] for item in manifest["sequences"] if item["split"]=="diagnostic"]
    if len(train_paths)<8 or len(diagnostic_paths)<2: raise ValueError("manifest needs 8 train and 2 diagnostic windows")
    sys.path.insert(0,str(Path(config["wah_root"])))
    from warp_as_history.training import core as opt
    from long_video.oracle_training.contracts import assert_history_frames_are_generated
    from long_video.oracle_training.wah_training import assert_only_lora_gradients,load_training_checkpoint,masked_flow_matching_loss,save_training_checkpoint
    exact=opt.parse_args([]); exact.base_model_path=str(config["wah_model"]); exact.transformer_path=str(config["wah_model"])
    exact.height,exact.width=map(int,config["perspective_resolution"]); exact.num_frames=int(config["chunk_frames"])
    exact.num_latent_frames_per_chunk=(exact.num_frames-1)//int(config["vae_temporal_scale"])+1
    exact.history_sizes=[16,2,1]; exact.history_temporal_layout="long_mid_short"
    exact.pyramid_num_inference_steps_list=list(config["training"]["pyramid_num_inference_steps_list"])
    exact.attention_backend="native"; exact.use_warp_as_history=True; exact.warp_history_downsample_mode="short"
    exact.history_positioning="last_n_same_order"; exact.history_position_count=exact.num_latent_frames_per_chunk; exact.history_position_delta=0
    exact.history_visible_token_drop=True; exact.visible_token_mode="drop"; exact.history_visible_token_threshold=.05
    exact.history_confidence_threshold=.1; exact.history_confidence_lambda=1.; exact.history_confidence_epsilon=1e-6
    exact.add_noise_to_video_latents=False; exact.add_noise_to_image_latents=False
    exact.flow_matching_mode="train_exact"; exact.flow_matching_stage_sampling="fixed"; exact.flow_matching_stage_id=0
    exact.flow_matching_train_exact_timestep_sampling="training_density"; exact.flow_matching_use_dynamic_shifting="off"; exact.weighting_scheme="none"
    exact.seed=int(config["training"].get("seed",config["seed"])); exact.lora_rank=int(config["training"]["lora_rank"])
    exact.lora_alpha=int(config["training"]["lora_alpha"]); exact.lora_dropout=float(config["training"]["lora_dropout"])
    exact.lora_target_modules=str(config["training"]["lora_target_modules"]); exact.lora_adapter_name="oracle_wah_24fps"
    exact.iters=int(args.max_steps or config["training"]["max_steps"]); exact.gradient_checkpointing=True; opt.validate_args(exact)
    random.seed(exact.seed); np.random.seed(exact.seed); opt.seed_global_rng(exact.seed)
    device=torch.device("cuda:0"); started=time.perf_counter(); gpu_before=_gpu_processes()
    pipe=opt.load_pipeline(exact,device); mean,std=opt.latent_stats(pipe,device)
    adapter_name,trainable,lora_stats=opt.setup_visible_lora(pipe.transformer,exact,"oracle_wah_24fps")
    optimizer=torch.optim.AdamW(trainable,lr=float(config["training"]["learning_rate"]),weight_decay=.01)
    warmup=max(1,int(config["training"]["warmup_steps"])); scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda step:min(1.,(step+1)/warmup))
    ckpt_root=Path(config["checkpoint_root"]); ckpt_root.mkdir(parents=True,exist_ok=True); ckpt=ckpt_root/"oracle_wah_24fps_checkpoint.pt"
    git_sha=subprocess.check_output(["git","rev-parse","HEAD"],cwd=repo,text=True).strip(); manifest_sha=_sha(manifest_path)
    rife_sha=manifest["sequences"][0]["metadata"]["rife_checkpoint_sha256"]
    checkpoint_metadata={"manifest_sha":manifest_sha,"git_sha":git_sha,"rife_checkpoint_sha":rife_sha,"mode":args.mode}
    global_step=0; resume_verified=False
    if args.resume:
        global_step,restored=load_training_checkpoint(args.resume,pipe.transformer,optimizer,scheduler,adapter_name)
        for key in ("manifest_sha","git_sha","rife_checkpoint_sha"):
            if restored.get(key)!=checkpoint_metadata[key]: raise ValueError(f"checkpoint {key} mismatch")
        resume_verified=True

    def encode(sample):
        assert_history_frames_are_generated(sample["warp"],sample["target"])
        with torch.no_grad(): target_latents=opt.encode_video_latents(pipe,sample["target"],exact,device,mean,std).detach()
        if target_latents.shape[2]!=len(sample["weights"]): raise ValueError("VAE latent T differs from temporal weights")
        prompt,histories=_history(opt,pipe,exact,device,mean,std,sample["target"][0],sample["prompt"],sample["warp"],sample["visibility"],sample["confidence"],sample["metadata"]["sequence_id"])
        return target_latents,prompt,histories

    # VAE encoding and external-warp history construction are deterministic for
    # a prepared window and dominated formal-training wall time. Keep only
    # these encoded conditioning values in CPU RAM. Flow-matching noise,
    # timestep and stage items are still freshly sampled on every microstep.
    encoded_train_cache={}
    def cached_train_sample(path):
        key=str(path)
        if key not in encoded_train_cache:
            sample=_read_sample(path,exact.num_frames)
            latents,prompt,histories=encode(sample)
            encoded_train_cache[key]=(
                _tree_to(latents,"cpu"),
                _tree_to(prompt,"cpu"),
                _tree_to(histories,"cpu"),
                sample["weights"].copy(),
            )
            del latents,prompt,histories
            torch.cuda.empty_cache()
        latents_cpu,prompt_cpu,histories_cpu,weights=encoded_train_cache[key]
        return (_tree_to(latents_cpu,device),_tree_to(prompt_cpu,device),
                _tree_to(histories_cpu,device),weights)

    fixed=_read_sample(diagnostic_paths[0],exact.num_frames); fixed_latents,fixed_prompt,fixed_histories=encode(fixed)
    opt.seed_global_rng(exact.seed); fixed_items=opt.flow_matching_train_exact_items(pipe,fixed_latents,exact,device)
    def fixed_loss():
        pipe.transformer.eval()
        with torch.no_grad(): value,_,_=masked_flow_matching_loss(pipe,fixed_prompt,fixed_latents,fixed_histories,exact,device,fixed["weights"],fixed_stage_items=fixed_items)
        return float(value.cpu())
    def warp_diagnostics():
        order=np.random.default_rng(exact.seed).permutation(len(fixed["warp"]))
        variants={
            "correct":(fixed["warp"],fixed["visibility"],fixed["confidence"]),
            "shuffled":([fixed["warp"][i] for i in order],fixed["visibility"][order],fixed["confidence"][order]),
            "empty":([Image.new("RGB",(exact.width,exact.height),(0,0,0)) for _ in fixed["warp"]],np.zeros_like(fixed["visibility"]),np.zeros_like(fixed["confidence"])),
        }; values={}
        pipe.transformer.eval()
        for name,(frames,visibility,confidence) in variants.items():
            prompt,histories=_history(opt,pipe,exact,device,mean,std,fixed["target"][0],fixed["prompt"],frames,visibility,confidence,f"diagnostic_{name}")
            with torch.no_grad(): value,_,_=masked_flow_matching_loss(pipe,prompt,fixed_latents,histories,exact,device,fixed["weights"],fixed_stage_items=fixed_items)
            values[name]=float(value.cpu()); del prompt,histories
        return values
    initial=fixed_loss(); diagnostics_before=warp_diagnostics(); logs=[]; max_steps=int(args.max_steps or config["training"]["max_steps"])
    if args.mode=="smoke": max_steps=min(max_steps,20)
    accumulation=1 if args.mode=="smoke" else int(config["training"]["gradient_accumulation_steps"])
    rng=np.random.default_rng(exact.seed); pipe.transformer.train()
    while global_step<max_steps:
        optimizer.zero_grad(set_to_none=True); micro_losses=[]
        for micro in range(accumulation):
            if args.mode=="smoke": latents,prompt,histories,weights=fixed_latents,fixed_prompt,fixed_histories,fixed["weights"]
            else:
                latents,prompt,histories,weights=cached_train_sample(
                    train_paths[int(rng.integers(len(train_paths)))])
            loss,_,_=masked_flow_matching_loss(pipe,prompt,latents,histories,exact,device,weights,fixed_stage_items=fixed_items if args.mode=="smoke" else None)
            if not torch.isfinite(loss): raise FloatingPointError("non-finite weighted flow-matching loss")
            (loss/accumulation).backward(); micro_losses.append(float(loss.detach().cpu()))
            if args.mode!="smoke": del latents,prompt,histories
        assert_only_lora_gradients(pipe.transformer,trainable)
        grad=torch.nn.utils.clip_grad_norm_(trainable,float(config["training"]["max_grad_norm"])); optimizer.step(); scheduler.step(); global_step+=1
        record={"step":global_step,"train_weighted_loss":float(np.mean(micro_losses)),"learning_rate":float(scheduler.get_last_lr()[0]),
                "gradient_norm":float(grad),"gpu_allocated_bytes":int(torch.cuda.memory_allocated(0)),"gpu_reserved_bytes":int(torch.cuda.memory_reserved(0))}
        if global_step==1 or global_step%10==0 or global_step==max_steps: record["fixed_diagnostic_loss"]=fixed_loss(); pipe.transformer.train()
        logs.append(record)
        every=int(config["training"]["checkpoint_every"])
        if global_step%every==0 or global_step==max_steps:
            save_training_checkpoint(ckpt,pipe.transformer,trainable,optimizer,scheduler,global_step,adapter_name,checkpoint_metadata)
            opt.save_visible_lora_state(pipe.transformer,ckpt_root,adapter_name,"oracle_wah_lora.pt")
    final=fixed_loss(); diagnostics_after=warp_diagnostics(); save_training_checkpoint(ckpt,pipe.transformer,trainable,optimizer,scheduler,global_step,adapter_name,checkpoint_metadata)
    opt.save_visible_lora_state(pipe.transformer,ckpt_root,adapter_name,"oracle_wah_lora.pt")
    result={"mode":args.mode,"optimizer_steps":global_step,"gradient_accumulation_steps":accumulation,"initial_fixed_diagnostic_loss":initial,
            "final_fixed_diagnostic_loss":final,"fixed_loss_decreased":final<initial,"checkpoint_resume_verified":resume_verified,"diagnostics_before":diagnostics_before,"diagnostics_after":diagnostics_after,
            "manifest_sha":manifest_sha,"git_sha":git_sha,"rife_checkpoint_sha":rife_sha,"lora":lora_stats,"logs":logs,
            "visible_device_count":torch.cuda.device_count(),"visible_device_name":torch.cuda.get_device_name(0),
            "peak_allocated_bytes":int(torch.cuda.max_memory_allocated(0)),"peak_reserved_bytes":int(torch.cuda.max_memory_reserved(0)),
            "elapsed_seconds":time.perf_counter()-started,"gpu_processes_before":gpu_before,"gpu_processes_after":_gpu_processes()}
    (ckpt_root/f"training_{args.mode}_result.json").write_text(json.dumps(result,indent=2),encoding="utf-8"); print(json.dumps(result,indent=2))

if __name__=="__main__": main()
