"""Encode continuous 193-frame RGB records into a canonical 49-latent cache."""
from __future__ import annotations
import argparse,json,hashlib,glob
from pathlib import Path
from PIL import Image
import torch

def main():
    p=argparse.ArgumentParser(); p.add_argument('--model',required=True); p.add_argument('--helios-root',required=True); p.add_argument('--rgb-dir',required=True); p.add_argument('--out',required=True); p.add_argument('--height',type=int,default=384); p.add_argument('--width',type=int,default=640); a=p.parse_args()
    import sys; sys.path.insert(0,a.helios_root)
    from helios.diffusers_version.pipeline_helios_diffusers import HeliosPipeline
    pipe=HeliosPipeline.from_pretrained(a.model,torch_dtype=torch.bfloat16).to('cuda'); paths=sorted(Path(a.rgb_dir).glob('*'))
    if len(paths)!=193: raise ValueError('continuous cache requires exactly 193 RGB frames')
    images=[Image.open(path).convert('RGB') for path in paths]
    processor=pipe.video_processor
    try:
        pixels=processor.preprocess_video(images,height=a.height,width=a.width)
        if pixels.ndim==4: pixels=pixels.unsqueeze(0)
        if pixels.ndim!=5: raise RuntimeError(f'preprocess_video returned {tuple(pixels.shape)}')
        pixels=pixels.to('cuda',dtype=pipe.vae.dtype)
    except Exception as exc: raise RuntimeError('pinned Helios video_processor cannot batch 193 RGB frames') from exc
    mean=torch.tensor(pipe.vae.config.latents_mean,device='cuda',dtype=pipe.vae.dtype).view(1,pipe.vae.config.z_dim,1,1,1)
    std=(1.0/torch.tensor(pipe.vae.config.latents_std,device='cuda',dtype=pipe.vae.dtype)).view(1,pipe.vae.config.z_dim,1,1,1)
    with torch.no_grad():
        encoded=pipe.vae.encode(pixels); distribution=getattr(encoded,'latent_dist',None)
        if distribution is None: raise RuntimeError('pinned Helios VAE did not return latent_dist')
        latent=(distribution.mode()-mean)*std
    if latent.ndim!=5: raise RuntimeError(f'continuous VAE latent must be 5D, got {latent.shape}')
    temporal=[axis for axis,size in enumerate(latent.shape) if size in (49,193)]
    if len(temporal)!=1: raise RuntimeError(f'cannot identify continuous latent temporal axis: {latent.shape}')
    axis=temporal[0]; latent=latent.movedim(axis,2)
    if latent.shape[2]!=49: raise RuntimeError('continuous VAE encode must natively produce T=49; refusing frame subsampling')
    model_files=[]
    for pattern in ('config.json','model_index.json','transformer/config.json','transformer/*.index.json','vae/config.json','vae/*.index.json'):
        model_files.extend(glob.glob(str(Path(a.model)/pattern)))
    digest=hashlib.sha256()
    for filename in sorted(set(model_files)):
        digest.update(Path(filename).read_bytes())
    model_identity={'model_ref':str(a.model),'config_fingerprint':digest.hexdigest()}
    vae_config=dict(pipe.vae.config) if hasattr(pipe.vae.config,'items') else str(pipe.vae.config)
    provenance={'model_identity':model_identity,'vae_config':vae_config,'preprocessing':{'height':a.height,'width':a.width,'normalization':'pinned Helios video_processor'},'latent_normalization':{'mean':list(pipe.vae.config.latents_mean),'std':list(pipe.vae.config.latents_std)},'encode_mode':'mode','encoding_dtype':str(pixels.dtype)}
    provenance['fingerprint']=hashlib.sha256(json.dumps(provenance,sort_keys=True,default=str).encode()).hexdigest()
    if not torch.isfinite(latent).all(): raise RuntimeError('continuous latent contains non-finite values')
    output=Path(a.out); output.parent.mkdir(parents=True,exist_ok=True); torch.save({'latents':latent.cpu(),'schema':'continuous_49','provenance':provenance},output)
    print(json.dumps({'out':str(output),'schema':'continuous_49','shape':list(latent.shape),'provenance_fingerprint':provenance['fingerprint']}))
if __name__=='__main__': main()
