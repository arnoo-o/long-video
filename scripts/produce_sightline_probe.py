"""Produce probe JSONL exclusively from tensors captured by real Helios forwards.

The capture file is written by a forward hook and must contain Q/K from a
selected layer plus explicit ablation forwards.  No random fallback exists.
"""
from __future__ import annotations
import argparse, json, subprocess, sys, tempfile
from pathlib import Path
import torch

REQUIRED=("q","k","positive_key","k_memory","fm_loss","wrong_ray_loss","memory_zero_loss","memory_shuffle_loss","corr_loss","vram_gb","step_time_sec")

def _ranking(logits,positive):
    order=logits.argsort(-1,descending=True); ranks=(order==positive[...,None]).nonzero(as_tuple=False)[:,-1]
    return float((1/(ranks.float()+1)).mean()),float((ranks==0).float().mean()),float((ranks<5).float().mean())

def measured_row(capture):
    missing=[key for key in REQUIRED if key not in capture]
    if missing: raise RuntimeError(f"real Helios probe capture missing {missing}")
    q,k=capture["q"].float(),capture["k"].float(); scale=q.shape[-1]**-.5
    logits=torch.einsum("bqhd,bkhd->bqk",q,k)*scale; positive=capture["positive_key"].long().to(logits.device)
    mrr,top1,top5=_ranking(logits,positive)
    mass=logits.softmax(-1).gather(-1,positive[...,None]).mean()
    memory=capture["k_memory"].float(); memory_count=memory.shape[1]
    if memory_count<1: raise RuntimeError('memory probe requires real second-chunk memory tokens')
    corr_gain=float(torch.log(torch.tensor(logits.shape[-1],dtype=torch.float32))-torch.tensor(float(capture['corr_loss'])))
    return {"layer":int(capture["layer"]),"sigma":float(capture["sigma"]),"correspondence_mrr":mrr,"top1":top1,"top5":top5,"corr_gain":corr_gain,
        "positive_attention_mass":float(mass),"wrong_ray_delta":float(capture["wrong_ray_loss"]-capture["fm_loss"]),
        "memory_attention_mass":float(logits.softmax(-1)[...,-memory_count:].sum(-1).mean()),"memory_zero_delta":float(capture["memory_zero_loss"]-capture["fm_loss"]),
        "memory_shuffle_delta":float(capture["memory_shuffle_loss"]-capture["fm_loss"]),"fm_loss":float(capture["fm_loss"]),
        "corr_loss":float(capture["corr_loss"]),"vram_gb":float(capture["vram_gb"]),"step_time_sec":float(capture["step_time_sec"])}

def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--model",required=True); parser.add_argument("--helios-root",required=True); parser.add_argument("--manifest",required=True); parser.add_argument("--config",default="configs/sightline.yaml"); parser.add_argument("--checkpoint"); parser.add_argument("--alpha-zero-baseline",action="store_true"); parser.add_argument("--samples",type=int,default=10); parser.add_argument("--expected-records",type=int,default=100); parser.add_argument("--out",required=True); args=parser.parse_args()
    if bool(args.checkpoint)==bool(args.alpha_zero_baseline): raise ValueError('provide --checkpoint or explicitly select --alpha-zero-baseline')
    rows=[]; train_script=Path(__file__).with_name('train_sightline_dl3dv.py')
    with tempfile.TemporaryDirectory(prefix='sightline_probe_') as directory:
      for index in range(args.samples):
        capture=Path(directory)/f'{index}.pt'; metrics=Path(directory)/f'{index}.jsonl'
        command=[sys.executable,str(train_script),'--model',args.model,'--helios-root',args.helios_root,'--manifest',args.manifest,'--config',args.config,'--expected-records',str(args.expected_records),'--record-index',str(index),'--train-chunk','1','--probe-only','--probe-step','1000','--probe-capture',str(capture),'--metrics',str(metrics),'--output-dir',directory]
        if args.checkpoint: command.extend(['--resume',args.checkpoint])
        else: command.append('--alpha-zero-baseline')
        subprocess.run(command,check=True)
        payload=torch.load(capture,map_location='cpu')
        if payload.get("source")!="real_helios_forward": raise RuntimeError(f"{capture} is not a real Helios capture")
        rows.append(measured_row(payload))
    Path(args.out).write_text("".join(json.dumps(row)+"\n" for row in rows))

if __name__=="__main__": main()
