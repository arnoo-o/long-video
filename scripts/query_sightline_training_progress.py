"""Readable metrics summary with actionable Sightline warnings."""
import argparse,json,math
from pathlib import Path
def main():
 p=argparse.ArgumentParser(); p.add_argument('metrics',type=Path); a=p.parse_args()
 if not a.metrics.exists(): raise SystemExit('no Sightline metrics found')
 rows=[json.loads(x) for x in a.metrics.read_text().splitlines() if x.strip()]; latest=rows[-1] if rows else {}; warnings=[]
 def near_zero(name):
  value=latest.get(name); return value is not None and abs(float(value))<1e-8
 if len(rows)>=5 and all(near_zero('alpha') for _ in rows[-5:]): warnings.append('alpha remains near zero')
 for name,msg in [('alpha_grad','alpha gradient is zero'),('wrong_ray_delta','wrong-ray delta is zero'),('memory_zero_delta','memory-zero delta is zero'),('memory_shuffle_delta','memory-shuffle delta is zero')]:
  if near_zero(name): warnings.append(msg)
 if any(not math.isfinite(float(v)) for v in latest.values() if isinstance(v,(int,float))): warnings.append('NaN/Inf in latest metrics')
 print(json.dumps({'latest':latest,'steps':len(rows),'warnings':warnings},ensure_ascii=False,indent=2))
if __name__=='__main__': main()
