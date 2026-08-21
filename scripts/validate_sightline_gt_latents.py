"""Compare overlap 6x9 latents with a continuous 193-frame VAE encoding."""
from __future__ import annotations
import argparse,json,hashlib
from pathlib import Path
import torch
from long_video.training.sightline_data import load_latent_tensor

def main():
    p=argparse.ArgumentParser(); p.add_argument('--overlap',required=True); p.add_argument('--continuous',required=True); p.add_argument('--out',required=True); p.add_argument('--boundary-count',type=int,default=5); p.add_argument('--atol',type=float); a=p.parse_args()
    overlap=load_latent_tensor(a.overlap,schema='overlap_chunks_6x9').float(); continuous=load_latent_tensor(a.continuous,schema='continuous_49').float()
    if overlap.shape!=continuous.shape: raise ValueError(f'latent shape mismatch: {overlap.shape} vs {continuous.shape}')
    delta=(overlap-continuous).abs(); boundary=torch.tensor([8,16,24,32,40],dtype=torch.long); boundary=boundary[boundary<delta.shape[2]]
    overall={'max_abs':float(delta.max()),'mean_abs':float(delta.mean())}; boundary_delta=delta.index_select(2,boundary)
    continuous_payload=torch.load(a.continuous,map_location='cpu'); provenance=continuous_payload.get('provenance')
    digest=hashlib.sha256(Path(a.continuous).read_bytes()).hexdigest()
    result={'passed':bool(a.atol is not None and float(delta.max())<=a.atol),'continuous_fingerprint':digest,'provenance':provenance,'shape':list(continuous.shape),'finite':bool(torch.isfinite(continuous).all() and torch.isfinite(overlap).all()),'overall':overall,'boundary':{'indices':boundary.tolist(),'max_abs':float(boundary_delta.max()),'mean_abs':float(boundary_delta.mean())},'atol':a.atol}
    Path(a.out).write_text(json.dumps(result,indent=2)); print(json.dumps(result,indent=2))
    if not result['passed']: raise SystemExit(2)
if __name__=='__main__': main()
