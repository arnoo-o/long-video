"""Readable metrics summary with actionable Sightline warnings."""
import argparse,json,math
from pathlib import Path
def main():
 p=argparse.ArgumentParser(); p.add_argument('metrics',type=Path); p.add_argument('--json',action='store_true',help='emit the complete latest metrics record'); a=p.parse_args()
 if not a.metrics.exists(): raise SystemExit('no Sightline metrics found')
 rows=[json.loads(x) for x in a.metrics.read_text().splitlines() if x.strip()]; latest=rows[-1] if rows else {}; warnings=[]
 def near_zero(name):
  value=latest.get(name); return value is not None and abs(float(value))<1e-8
 if len(rows)>=5 and all(near_zero('alpha') for _ in rows[-5:]): warnings.append('alpha remains near zero')
 for name,msg in [('alpha_grad','alpha gradient is zero'),('wrong_ray_delta','wrong-ray delta is zero'),('memory_zero_delta','memory-zero delta is zero'),('memory_shuffle_delta','memory-shuffle delta is zero')]:
  if near_zero(name): warnings.append(msg)
 if any(not math.isfinite(float(v)) for v in latest.values() if isinstance(v,(int,float))): warnings.append('NaN/Inf in latest metrics')
 if a.json:
  print(json.dumps({'latest':latest,'steps':len(rows),'warnings':warnings},ensure_ascii=False,indent=2)); return
 def bounds(name):
  values=list((latest.get(name) or {}).values())
  return '-' if not values else f'{min(values):.5g}..{max(values):.5g}'
 sigma=latest.get('sampled_sigma',latest.get('stage_sigmas',[]))
 sigma=','.join(f'{float(value):.4f}' for value in sigma) if isinstance(sigma,list) else str(sigma)
 print(' '.join((
  f"step={latest.get('step','-')}",f"phase={latest.get('phase','-')}",
  f"loss={float(latest.get('flow_loss',float('nan'))):.6f}",f"corr={float(latest.get('corr_loss',float('nan'))):.6f}",
  f"sigma_band={latest.get('sigma_band','-')}",f"sigma=[{sigma}]",
  f"lr={float(latest.get('lr',float('nan'))):.3g}",f"grad={float(latest.get('grad_norm',float('nan'))):.4g}",
  f"alpha_q={bounds('alpha_q')}",f"alpha_k={bounds('alpha_k')}",
  f"sec={float(latest.get('seconds',float('nan'))):.1f}",
  ('warnings='+','.join(warnings) if warnings else 'ok'),
 )))
if __name__=='__main__': main()
