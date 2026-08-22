"""Produce probe JSONL exclusively from tensors captured by real Helios forwards.

The capture file is written by a forward hook and must contain Q/K from a
selected layer plus explicit ablation forwards.  No random fallback exists.
"""
from __future__ import annotations
import argparse, json, subprocess, sys, tempfile
from pathlib import Path
import torch

REQUIRED=("attention_logits","positive_key_indices","memory_count","fm_loss","baseline_final_stage_loss","wrong_ray_loss","memory_zero_loss","memory_shuffle_loss","corr_loss","alpha","alpha_grad","vram_gb","step_time_sec","ablation_time_sec")

def _ranking(logits,positive_lists):
    if logits.ndim==3: logits=logits[:,None]
    order=logits.argsort(-1,descending=True); probabilities=logits.softmax(-1); ranks=[]; masses=[]
    for query,keys in enumerate(positive_lists):
        keys=torch.tensor(keys,device=logits.device,dtype=torch.long)
        positions=(order[:,:,query,:,None]==keys.view(1,1,1,-1)).any(-1).float().argmax(-1)
        ranks.append(positions.reshape(-1)); masses.append(probabilities[:,:,query].index_select(-1,keys).sum(-1).reshape(-1))
    ranks=torch.cat(ranks); mass=torch.cat(masses).mean()
    return float((1/(ranks.float()+1)).mean()),float((ranks==0).float().mean()),float((ranks<5).float().mean()),float(mass)

def measured_row(capture):
    missing=[key for key in REQUIRED if key not in capture]
    if missing: raise RuntimeError(f"real Helios probe capture missing {missing}")
    head_logits=capture['attention_logits'].float(); positives=capture["positive_key_indices"]; baseline=bool(capture.get('baseline',False))
    raw_mrr,raw_top1,raw_top5,mass=_ranking(head_logits,positives); native_attention=head_logits.softmax(-1)
    corr_mrr=corr_top1=corr_top5=None
    memory_count=int(capture['memory_count'])
    if memory_count<0 or memory_count>head_logits.shape[-1]: raise RuntimeError('invalid captured memory token count')
    baseline_final_stage_loss=float(capture['baseline_final_stage_loss'])
    return {"layer":int(capture["layer"]),"sigma":float(capture["sigma"]),"ranking_source":"raw_qk","correspondence_mrr":raw_mrr,"top1":raw_top1,"top5":raw_top5,"raw_qk_mrr":raw_mrr,"raw_qk_top1":raw_top1,"raw_qk_top5":raw_top5,
         "positive_attention_mass":float(mass),"wrong_ray_delta":float(capture["wrong_ray_loss"]-baseline_final_stage_loss),
         "memory_attention_mass":0.0 if memory_count==0 else float(native_attention[...,-memory_count:].sum(-1).mean()),"memory_zero_delta":float(capture["memory_zero_loss"]-baseline_final_stage_loss),
         "memory_shuffle_delta":float(capture["memory_shuffle_loss"]-baseline_final_stage_loss),"fm_loss":float(capture["fm_loss"]),
        "corr_loss":float(capture["corr_loss"]),"alpha":float(capture['alpha']),"alpha_grad":float(capture['alpha_grad']),"vram_gb":float(capture["vram_gb"]),"step_time_sec":float(capture["step_time_sec"]),"ablation_time_sec":float(capture['ablation_time_sec'])}

def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--model",required=True); parser.add_argument('--model-revision'); parser.add_argument("--helios-root",required=True); parser.add_argument("--manifest",required=True); parser.add_argument("--config",default="configs/sightline.yaml"); parser.add_argument("--checkpoint"); parser.add_argument("--alpha-zero-baseline",action="store_true"); parser.add_argument('--candidate-layers',default=''); parser.add_argument("--samples",type=int,default=10); parser.add_argument("--expected-records",type=int,default=100); parser.add_argument('--latent-cache-root'); parser.add_argument("--out",required=True); args=parser.parse_args()
    if bool(args.checkpoint)==bool(args.alpha_zero_baseline): raise ValueError('provide --checkpoint or explicitly select --alpha-zero-baseline')
    rows=[]; train_script=Path(__file__).with_name('train_sightline_dl3dv.py')
    if args.checkpoint:
        checkpoint_meta=torch.load(args.checkpoint,map_location='cpu'); probe_step=int(checkpoint_meta['step'])
    else: probe_step=1000
    if probe_step<300: train_chunk=0
    elif probe_step<1000: train_chunk=0
    elif probe_step<1500: train_chunk=1
    elif probe_step<1800: train_chunk=2
    elif probe_step<2100: train_chunk=3
    elif probe_step<2300: train_chunk=4
    else: train_chunk=5
    with tempfile.TemporaryDirectory(prefix='sightline_probe_') as directory:
      for index in range(args.samples):
        capture=Path(directory)/f'{index}.pt'; metrics=Path(directory)/f'{index}.jsonl'
        command=[sys.executable,str(train_script),'--model',args.model,'--helios-root',args.helios_root,'--manifest',args.manifest,'--config',args.config,'--expected-records',str(args.expected_records),'--record-index',str(index),'--train-chunk',str(train_chunk),'--probe-only','--probe-step',str(probe_step),'--probe-layers',args.candidate_layers,'--probe-capture',str(capture),'--output-dir',directory]
        if args.model_revision: command.extend(['--model-revision',args.model_revision])
        if args.latent_cache_root: command.extend(['--latent-cache-root',args.latent_cache_root])
        if args.checkpoint: command.extend(['--probe-checkpoint',args.checkpoint])
        else: command.append('--alpha-zero-baseline')
        subprocess.run(command,check=True)
        payload=torch.load(capture,map_location='cpu')
        if payload.get("source")!="real_helios_forward": raise RuntimeError(f"{capture} is not a real Helios capture")
        captures=payload.get('layer_captures') or [payload]
        for layer_capture in captures:
            merged=dict(payload); merged.update(layer_capture); merged.pop('layer_captures',None)
            rows.append(measured_row(merged))
    Path(args.out).write_text("".join(json.dumps(row)+"\n" for row in rows))

if __name__=="__main__": main()
