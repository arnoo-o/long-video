"""Sequential continuous latent caching for mixed 97/193-frame RGB-D records."""
from __future__ import annotations
import argparse, glob, hashlib, json
from pathlib import Path
from PIL import Image
import torch

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--model',required=True); p.add_argument('--helios-root',required=True)
    p.add_argument('--manifest',required=True,help='JSON list or records with trajectory_id and rgb_dir/rgb_path')
    p.add_argument('--selection-manifest',help='Optional JSON containing trajectory_ids to encode')
    p.add_argument('--out-root',required=True); p.add_argument('--height',type=int,default=480); p.add_argument('--width',type=int,default=832)
    p.add_argument('--limit',type=int,default=0)
    p.add_argument('--frame-count',type=int,choices=(97,193),help='Only encode records with this exact source length')
    p.add_argument('--ignore-declared-cache',action='store_true',help='Write canonical caches below --out-root instead of a legacy manifest path')
    p.add_argument('--shard-index',type=int,default=0); p.add_argument('--shard-count',type=int,default=1)
    a=p.parse_args(); import sys; sys.path.insert(0,str(Path(__file__).resolve().parents[1])); sys.path.insert(0,a.helios_root)
    from long_video.sightline.geometry import pad_image_bottom_right, padded_size
    from long_video.training.rgbd_memory_data import RGBDMemoryRecord
    from long_video.training.sightline_data import rgbd_unit_identity
    from diffusers import AutoencoderKLWan
    from diffusers.video_processor import VideoProcessor
    vae=AutoencoderKLWan.from_pretrained(a.model,subfolder='vae',torch_dtype=torch.bfloat16).to('cuda').eval()
    processor=VideoProcessor(vae_scale_factor=vae.config.scale_factor_spatial)
    raw=json.loads(Path(a.manifest).read_text()); records=raw if isinstance(raw,list) else raw.get('records',raw.get('items',[]))
    if a.selection_manifest:
        selected=set(json.loads(Path(a.selection_manifest).read_text()).get('trajectory_ids',()))
        records=[record for record in records if (record.get('record_id') or record.get('trajectory_id')) in selected]
    if a.frame_count is not None:
        records=[record for record in records if int(record.get('frame_count',97))==a.frame_count]
    if a.shard_count < 1 or not 0 <= a.shard_index < a.shard_count:
        raise ValueError('invalid shard index/count')
    records=records[a.shard_index::a.shard_count]
    model_files=[]
    for pattern in ('config.json','model_index.json','transformer/config.json','transformer/*.index.json','vae/config.json','vae/*.index.json'):
        model_files.extend(glob.glob(str(Path(a.model)/pattern)))
    digest=hashlib.sha256(); [digest.update(Path(f).read_bytes()) for f in sorted(set(model_files))]
    model_identity={'model_ref':str(a.model),'config_fingerprint':digest.hexdigest()}
    vae_config=dict(vae.config) if hasattr(vae.config,'items') else str(vae.config)
    if isinstance(vae_config,dict) and isinstance(vae_config.get('_use_default_values'),list):
        vae_config['_use_default_values']=sorted(vae_config['_use_default_values'])
    padded_h,padded_w=padded_size(a.height,a.width)
    provenance={'model_identity':model_identity,'vae_config':vae_config,'preprocessing':{'height':a.height,'width':a.width,'padded_height':padded_h,'padded_width':padded_w,'padding':'bottom_right_zero','normalization':'pinned Helios video_processor'},'latent_normalization':{'mean':list(vae.config.latents_mean),'std':list(vae.config.latents_std)},'encode_mode':'mode','encoding_dtype':str(vae.dtype)}
    provenance['fingerprint']=hashlib.sha256(json.dumps(provenance,sort_keys=True,default=str).encode()).hexdigest()
    mean=torch.tensor(vae.config.latents_mean,device='cuda',dtype=vae.dtype).view(1,vae.config.z_dim,1,1,1); std=(1.0/torch.tensor(vae.config.latents_std,device='cuda',dtype=vae.dtype)).view(1,vae.config.z_dim,1,1,1)
    for rec in records[:a.limit or None]:
        tid=rec.get('record_id') or rec.get('trajectory_id') or rec.get('id')
        frame_count=int(rec.get('frame_count',97)); expected_latents=1+(frame_count-1)//4
        if frame_count not in (97,193) or expected_latents not in (25,49):
            raise ValueError(f'{tid}: unsupported record geometry {frame_count} frames')
        schema=f'continuous_{expected_latents}'
        rgb=rec.get('rgb_dir') or rec.get('rgb_path') or rec.get('video_dir')
        if not rgb and rec.get('path'): rgb=str(Path(rec['path']).parent/'rgb_24fps')
        if rgb and not Path(rgb).is_absolute():
            rgb=str(Path(a.manifest).parent/rgb)
        if not tid or not rgb: continue
        all_paths=sorted(p for p in Path(rgb).glob('*') if p.suffix.lower() in ('.png','.jpg','.jpeg'))
        source_start=int(rec.get('source_frame_start',0)); paths=all_paths[source_start:source_start+frame_count]
        declared=None if a.ignore_declared_cache else (rec.get('gt_latent_cache') or rec.get('latent_cache'))
        if declared:
            out=Path(declared); out=out if out.is_absolute() else Path(a.manifest).resolve().parent/out
        else:
            out=Path(a.out_root)/tid.replace(':','_')/f'{schema}.pt'
        if out.name!=f'{schema}.pt': raise ValueError(f'{tid}: declared cache is not {schema}.pt: {out}')
        out.parent.mkdir(parents=True,exist_ok=True)
        record=RGBDMemoryRecord(rec,Path(a.manifest).resolve().parent)
        identity=rgbd_unit_identity(record)
        if out.exists():
            try:
                old=torch.load(out,map_location='cpu',weights_only=False)
                if old.get('schema')==schema and old.get('unit_identity')==identity and old.get('provenance',{}).get('fingerprint')==provenance['fingerprint'] and tuple(old['latents'].shape)==(1,16,expected_latents,padded_h//8,padded_w//8) and torch.isfinite(old['latents']).all(): print(json.dumps({'trajectory_id':tid,'status':'skip'})); continue
            except Exception: pass
        if len(paths)!=frame_count: print(json.dumps({'trajectory_id':tid,'status':'blocked','reason':f'expected {frame_count} RGB, got {len(paths)}'})); continue
        images=[pad_image_bottom_right(Image.open(x).convert('RGB'),a.height,a.width) for x in paths]; pixels=processor.preprocess_video(images,height=padded_h,width=padded_w)
        if pixels.ndim==4: pixels=pixels.unsqueeze(0)
        if pixels.ndim!=5: raise RuntimeError(f'{tid}: invalid preprocessed shape {tuple(pixels.shape)}')
        with torch.inference_mode():
            latent=(vae.encode(pixels.to('cuda',dtype=vae.dtype)).latent_dist.mode()-mean)*std
        axes=[i for i,s in enumerate(latent.shape) if s in (expected_latents,frame_count)]
        if len(axes)!=1: raise RuntimeError(f'{tid}: cannot identify temporal axis {tuple(latent.shape)}')
        latent=latent.movedim(axes[0],2)
        expected_shape=(1,16,expected_latents,padded_h//8,padded_w//8)
        if tuple(latent.shape)!=expected_shape or not torch.isfinite(latent).all(): raise RuntimeError(f'{tid}: invalid native continuous latent {tuple(latent.shape)}')
        tmp=out.with_suffix('.tmp'); torch.save({'latents':latent.cpu(),'schema':schema,'provenance':provenance,'unit_identity':identity},tmp); tmp.replace(out); del pixels,latent,images; torch.cuda.empty_cache(); print(json.dumps({'trajectory_id':tid,'status':'generated','schema':schema,'shape':list(expected_shape),'provenance_fingerprint':provenance['fingerprint']}),flush=True)
if __name__=='__main__': main()
