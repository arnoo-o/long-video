"""Encode continuous RGB records into a canonical Helios temporal latent cache."""
from __future__ import annotations
import argparse,json,hashlib,glob
from pathlib import Path
from PIL import Image
import torch

def main():
    p=argparse.ArgumentParser(); p.add_argument('--model',required=True); p.add_argument('--helios-root',required=True); p.add_argument('--rgb-dir',required=True); p.add_argument('--out',required=True); p.add_argument('--height',type=int,default=384); p.add_argument('--width',type=int,default=640); p.add_argument('--expected-frames',type=int,default=193); p.add_argument('--expected-latents',type=int,default=49); a=p.parse_args()
    import sys; sys.path.insert(0,a.helios_root)
    from diffusers import AutoencoderKLWan
    from diffusers.video_processor import VideoProcessor
    vae=AutoencoderKLWan.from_pretrained(a.model,subfolder='vae',torch_dtype=torch.bfloat16).to('cuda').eval()
    processor=VideoProcessor(vae_scale_factor=vae.config.scale_factor_spatial); paths=sorted(Path(a.rgb_dir).glob('*'))
    if len(paths)!=a.expected_frames: raise ValueError(f'continuous cache requires exactly {a.expected_frames} RGB frames')
    images=[Image.open(path).convert('RGB') for path in paths]
    try:
        pixels=processor.preprocess_video(images,height=a.height,width=a.width)
        if pixels.ndim==4: pixels=pixels.unsqueeze(0)
        if pixels.ndim!=5: raise RuntimeError(f'preprocess_video returned {tuple(pixels.shape)}')
        pixels=pixels.to('cuda',dtype=vae.dtype)
    except Exception as exc: raise RuntimeError(f'pinned Helios video_processor cannot batch {a.expected_frames} RGB frames') from exc
    mean=torch.tensor(vae.config.latents_mean,device='cuda',dtype=vae.dtype).view(1,vae.config.z_dim,1,1,1)
    std=(1.0/torch.tensor(vae.config.latents_std,device='cuda',dtype=vae.dtype)).view(1,vae.config.z_dim,1,1,1)
    with torch.no_grad():
        encoded=vae.encode(pixels); distribution=getattr(encoded,'latent_dist',None)
        if distribution is None: raise RuntimeError('pinned Helios VAE did not return latent_dist')
        latent=(distribution.mode()-mean)*std
    if latent.ndim!=5: raise RuntimeError(f'continuous VAE latent must be 5D, got {latent.shape}')
    temporal=[axis for axis,size in enumerate(latent.shape) if size == a.expected_latents]
    if len(temporal)!=1: raise RuntimeError(f'cannot identify continuous latent temporal axis: {latent.shape}')
    axis=temporal[0]; latent=latent.movedim(axis,2)
    if latent.shape[2]!=a.expected_latents: raise RuntimeError(f'continuous VAE encode must natively produce T={a.expected_latents}; refusing frame subsampling')
    model_files=[]
    for pattern in ('config.json','model_index.json','transformer/config.json','transformer/*.index.json','vae/config.json','vae/*.index.json'):
        model_files.extend(glob.glob(str(Path(a.model)/pattern)))
    digest=hashlib.sha256()
    for filename in sorted(set(model_files)):
        digest.update(Path(filename).read_bytes())
    model_identity={'model_ref':str(a.model),'config_fingerprint':digest.hexdigest()}
    vae_config=dict(vae.config) if hasattr(vae.config,'items') else str(vae.config)
    if isinstance(vae_config,dict) and isinstance(vae_config.get('_use_default_values'),list):
        vae_config['_use_default_values']=sorted(vae_config['_use_default_values'])
    provenance={'model_identity':model_identity,'vae_config':vae_config,'preprocessing':{'height':a.height,'width':a.width,'normalization':'pinned Helios video_processor'},'latent_normalization':{'mean':list(vae.config.latents_mean),'std':list(vae.config.latents_std)},'encode_mode':'mode','encoding_dtype':str(pixels.dtype)}
    provenance['fingerprint']=hashlib.sha256(json.dumps(provenance,sort_keys=True,default=str).encode()).hexdigest()
    if not torch.isfinite(latent).all(): raise RuntimeError('continuous latent contains non-finite values')
    schema=f'continuous_{a.expected_latents}'
    output=Path(a.out); output.parent.mkdir(parents=True,exist_ok=True); torch.save({'latents':latent.cpu(),'schema':schema,'provenance':provenance},output)
    print(json.dumps({'out':str(output),'schema':schema,'shape':list(latent.shape),'provenance_fingerprint':provenance['fingerprint']}))
if __name__=='__main__': main()
