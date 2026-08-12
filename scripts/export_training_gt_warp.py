#!/usr/bin/env python3
"""Export GT and causal point-cloud warp videos for one DL3DV trajectory."""
import argparse, json, sys
from pathlib import Path
import imageio.v2 as imageio
import numpy as np
from PIL import Image
import torch

from long_video.config import load_yaml
from long_video.initialization.geometry_backend import Pi3GeometryBackend
from long_video.memory.memory_manager import MemoryManager
from long_video.online.pipeline import OnlineSpatialHistoryPipeline
from long_video.training.stage0_causal_world import load_film_checkpoint
from long_video.types import RAY_DISTANCE, ViewSet
from long_video.wah.stage0_causal_world_film import install_stage0_causal_world_film

def source_views(image, intrinsic):
    rgb = np.asarray(image, np.uint8); h, w = rgb.shape[:2]
    return ViewSet(rgb=np.repeat(rgb[None], 8, 0), depth=np.full((8,h,w), np.nan, np.float32),
        depth_confidence=np.zeros((8,h,w), np.float32), c2w=np.repeat(np.eye(4, dtype=np.float32)[None], 8, 0),
        intrinsics=np.repeat(np.asarray(intrinsic, np.float32)[None], 8, 0), source=np.zeros((8,h,w), np.int8),
        image_confidence=np.ones((8,h,w), np.float32), depth_convention=RAY_DISTANCE)

def main():
    p=argparse.ArgumentParser(); p.add_argument('--wah-root',type=Path,required=True); p.add_argument('--model',type=Path,required=True)
    p.add_argument('--manifest',type=Path,required=True); p.add_argument('--trajectory-id',required=True); p.add_argument('--pi3-repo',type=Path,required=True)
    p.add_argument('--pi3-checkpoint',type=Path,required=True); p.add_argument('--film-checkpoint',type=Path,required=True); p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',default='cuda:0'); a=p.parse_args(); root=a.manifest.parent
    records=json.loads(a.manifest.read_text())['records']; rec=next(x for x in records if x['trajectory_id']==a.trajectory_id)
    sys.path.insert(0,str(a.wah_root)); from warp_as_history import WarpAsHistoryPipeline
    pipe=WarpAsHistoryPipeline.from_pretrained(a.model,torch_dtype=torch.bfloat16).to(a.device)
    if not hasattr(pipe.transformer.config,'image_dim'): pipe.transformer.register_to_config(image_dim=None)
    pipe._configure_wah_lora(str(a.wah_root/'checkpoints/warp-as-history/visible_lora_state_step1000.safetensors'))
    install_stage0_causal_world_film(pipe.transformer).to(a.device,torch.float32); load_film_checkpoint(a.film_checkpoint,pipe.transformer)
    for m in (pipe.transformer,pipe.vae):
        for q in m.parameters(): q.requires_grad_(False)
    geo=Pi3GeometryBackend(a.pi3_checkpoint,a.pi3_repo,a.device); manager=MemoryManager.from_config(load_yaml('configs/online_memory.yaml'),geometry_backend=geo)
    poses=np.load(root/rec['target_c2w_local']).astype(np.float32); K=np.load(root/rec['intrinsics']).astype(np.float32); source=Image.open(root/rec['source']).convert('RGB')
    gen=torch.Generator(device=a.device).manual_seed(20260812); online=OnlineSpatialHistoryPipeline(wah_pipeline=pipe,memory_manager=manager,prompt=rec['prompt'],renderer_kwargs={'device':a.device},wah_state_kwargs={'height':384,'width':640,'num_frames':33,'output_type':'np','pyramid_num_inference_steps_list':[2,2,2],'generator':gen})
    online.initialize(source_views(source,K[0]),rec['prompt'],geo,{'node_id':'node_000','center_c2w':np.eye(4,dtype=np.float32),'created_frame':0,'view_frame_indices':[0]*8,'target_frame_start':1},first_image=source)
    gt=[]; warp=[]; reports=[]
    with torch.inference_mode():
      for c in range(int(rec['chunk_count'])):
        b=c*32; frames=[]
        for i in range(b,b+33):
            matches=list((root/rec['rgb_dir']).glob(f'{i:06d}.*'))
            if len(matches)!=1: raise FileNotFoundError(f'expected one frame {i}: {matches}')
            frames.append(np.asarray(Image.open(matches[0]).convert('RGB')))
        gt.extend(frames if not gt else frames[1:]); online.frame_index=b; online.chunk_index=c
        w,_,_,_=online.prepare_supervised_chunk(poses[b:b+33],K[b:b+33],384,640); warp.extend([np.asarray(x) for x in w.rgb] if not warp else [np.asarray(x) for x in w.rgb[1:]])
        _, report = online.generate_chunk_at_cameras(poses[b:b+33],K[b:b+33],384,640); reports.append(report)
    a.output.mkdir(parents=True,exist_ok=True); imageio.mimwrite(a.output/'gt.mp4',np.asarray(gt),fps=24,macro_block_size=1); imageio.mimwrite(a.output/'causal_warp.mp4',np.asarray(warp),fps=24,macro_block_size=1); (a.output/'reports.json').write_text(json.dumps(reports,default=str,indent=2))
    print(json.dumps({'trajectory_id':a.trajectory_id,'chunk_count':rec['chunk_count'],'gt_frames':len(gt),'warp_frames':len(warp),'output':str(a.output)}))
if __name__=='__main__': main()
