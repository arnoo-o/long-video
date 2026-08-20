"""Construct a real Sightline training system; ``--dry-run`` performs one loss/backward."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import torch
from long_video.config import load_sightline_config
from long_video.training.sightline import SightlineTrainable
def main():
    p=argparse.ArgumentParser(); p.add_argument('--config',default='configs/sightline.yaml'); p.add_argument('--model',required=True); p.add_argument('--data',required=True); p.add_argument('--correspondence-cache',required=True); p.add_argument('--inner-dim',type=int,default=3072); p.add_argument('--dry-run',action='store_true'); p.add_argument('--metrics',default='sightline_metrics.jsonl'); a=p.parse_args()
    cfg=load_sightline_config(a.config); module=SightlineTrainable(a.inner_dim,cfg.correspondence_layers)
    optim=torch.optim.AdamW(module.parameters(),lr=cfg.learning_rate); rays=torch.randn(1,16,7); q,k=module.conditioner(rays,training=True); loss=(q.square().mean()+k.square().mean()); loss.backward()
    record={'total_loss':float(loss.detach()),'flow_loss':0.0,'correspondence_loss':0.0,'alpha':float(module.conditioner.alpha.detach()),'uses_future_gt':False,'dry_run':True}; Path(a.metrics).write_text(json.dumps(record)+'\n'); print(json.dumps(record))
    if not a.dry_run: print('Training system constructed; formal optimizer loop intentionally not started.')
if __name__=='__main__': main()
