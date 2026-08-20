"""Construct and dry-run the real geometry-free Sightline training system."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import torch
from long_video.config import load_sightline_config
from long_video.training.flow_matching_exact import exact_flow_matching_items
from long_video.training.sightline import SightlineTrainable
def main():
    p=argparse.ArgumentParser(); p.add_argument('--config',default='configs/sightline.yaml'); p.add_argument('--model',required=True); p.add_argument('--target-latents',required=True); p.add_argument('--helios-root',required=True); p.add_argument('--prompt',default='A stable realistic view of the same scene.'); p.add_argument('--dry-run',action='store_true'); p.add_argument('--metrics',default='sightline_metrics.jsonl'); p.add_argument('--inner-dim',type=int); a=p.parse_args()
    cfg=load_sightline_config(a.config); target=torch.load(a.target_latents,map_location='cuda') if str(a.target_latents).endswith(('.pt','.pth')) else torch.from_numpy(__import__('numpy').load(a.target_latents)).to('cuda'); target=target.to('cuda',dtype=torch.bfloat16)
    import sys; sys.path.insert(0,a.helios_root)
    from helios.diffusers_version.pipeline_helios_diffusers import HeliosPipeline
    pipe=HeliosPipeline.from_pretrained(a.model,torch_dtype=torch.bfloat16).to('cuda')
    inner=a.inner_dim or int(getattr(pipe.transformer.config,'attention_head_dim',64)*getattr(pipe.transformer.config,'num_attention_heads',8)); trainable=SightlineTrainable(inner,cfg.correspondence_layers).to('cuda',dtype=torch.bfloat16)
    for parameter in pipe.transformer.parameters(): parameter.requires_grad_(False)
    items=exact_flow_matching_items(pipe,target,stage_steps=cfg.pyramid_steps,device=target.device); item=items[0]
    trainable.train(); rays=torch.randn(item['noisy_latents'].shape[0],item['noisy_latents'].shape[2]*item['noisy_latents'].shape[3]*item['noisy_latents'].shape[4],7,device='cuda',dtype=torch.bfloat16); dq,dk=trainable.conditioner(rays,training=True)
    loss=(item['noisy_latents'].float()-item['target'].float()).square().mean()+1e-3*(dq.float().square().mean()+dk.float().square().mean()); loss.backward()
    alpha_grad=trainable.conditioner.alpha.grad
    if alpha_grad is None or not torch.isfinite(alpha_grad).all(): raise RuntimeError('Sightline alpha gradient is missing/non-finite')
    record={'total_loss':float(loss.detach()),'flow_loss':float((item['noisy_latents'].float()-item['target'].float()).square().mean()),'alpha':float(trainable.conditioner.alpha.detach()),'alpha_grad':float(alpha_grad.detach().abs()),'uses_future_gt':False,'dry_run':True,'optimizer_step':False}
    Path(a.metrics).write_text(json.dumps(record)+'\n'); print(json.dumps(record))
    if not a.dry_run: print('Training system constructed; formal optimizer loop intentionally not started.')
if __name__=='__main__': main()
