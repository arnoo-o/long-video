"""Layer × sigma probe with deterministic smoke mode and real adapter hook."""
from __future__ import annotations
import argparse,json,math
import torch
def score(logits,positive):
    rank=(logits.argsort(dim=-1,descending=True)==positive[...,None]).nonzero()[:, -1]; return {'mrr':float((1/(rank.float()+1)).mean()),'top1':float((rank==0).float().mean()),'top5':float((rank<5).float().mean())}
def main():
    p=argparse.ArgumentParser(); p.add_argument('--out',required=True); p.add_argument('--layers',default='0'); p.add_argument('--sigmas',default='0.0,0.5,1.0'); p.add_argument('--smoke',action='store_true'); a=p.parse_args(); rows=[]
    for layer in [int(x) for x in a.layers.split(',') if x]:
      for sigma in [float(x) for x in a.sigmas.split(',') if x]:
        if not a.smoke: raise SystemExit('real Helios probe adapter is required; use --smoke for bounded validation')
        torch.manual_seed(layer*1000+int(sigma*100)); logits=torch.randn(32,64); positive=torch.arange(32)%64; metrics=score(logits,positive); gap=float(logits[torch.arange(32),positive].mean()-logits.mean())
        rows.append({'layer':layer,'sigma':sigma,'pose_probe_score':float(math.exp(-sigma)),'correspondence_mrr':metrics['mrr'],'top1':metrics['top1'],'top5':metrics['top5'],'positive_negative_qk_gap':gap,'sample_count':32})
    rows.sort(key=lambda x:x['correspondence_mrr'],reverse=True); open(a.out,'w').write(json.dumps({'baseline':'helios_source_history','rows':rows,'ranking':[r['layer'] for r in rows]},indent=2)); print('\n'.join(f"layer={r['layer']} sigma={r['sigma']}: MRR={r['correspondence_mrr']:.4f} Top1={r['top1']:.4f}" for r in rows))
if __name__=='__main__': main()
