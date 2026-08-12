#!/usr/bin/env python3
"""Run causal point world -> renderer -> original WAH -> Stage2 RGB clamp."""
import argparse, json, sys
from pathlib import Path
import imageio.v2 as imageio
import numpy as np
from PIL import Image
import torch

def u8(value):
    x=np.asarray(value)
    return x if x.dtype==np.uint8 else np.rint(np.clip(x,0,1)*255).astype(np.uint8)

def main():
    p=argparse.ArgumentParser(); p.add_argument('--wah-root',type=Path,required=True); p.add_argument('--model',type=Path,required=True)
    p.add_argument('--session',type=Path,required=True); p.add_argument('--controls',type=Path,required=True); p.add_argument('--pi3-repo',type=Path,required=True)
    p.add_argument('--pi3-checkpoint',type=Path,required=True); p.add_argument('--output-dir',type=Path,required=True); p.add_argument('--device',default='cuda:0')
    p.add_argument('--height',type=int,default=384); p.add_argument('--width',type=int,default=640); p.add_argument('--prompt',default='Continue the scene consistently.'); a=p.parse_args()
    sys.path.insert(0,str(a.wah_root))
    from long_video.config import load_yaml
    from long_video.data.camera import resize_intrinsics
    from long_video.initialization.geometry_backend import Pi3GeometryBackend
    from long_video.memory.memory_manager import MemoryManager
    from long_video.memory.node_store import NodeStore
    from long_video.online.pipeline import OnlineSpatialHistoryPipeline
    from long_video.wah.rgb_clamp_pipeline import HELIOS_PYRAMID_NUM_INFERENCE_STEPS, RGBClampWarpAsHistoryPipeline
    node=NodeStore(a.session).load('node_000'); pipe=RGBClampWarpAsHistoryPipeline.from_pretrained(a.model,torch_dtype=torch.bfloat16).to(a.device)
    if not hasattr(pipe.transformer.config,'image_dim'): pipe.transformer.register_to_config(image_dim=None)
    pipe._configure_wah_lora(str(a.wah_root/'checkpoints/warp-as-history/visible_lora_state_step1000.safetensors'))
    for module in (pipe.transformer,pipe.vae):
        for q in module.parameters(): q.requires_grad_(False)
    geo=Pi3GeometryBackend(a.pi3_checkpoint,a.pi3_repo,a.device); manager=MemoryManager.from_config(load_yaml('configs/online_memory.yaml'),geometry_backend=geo)
    online=OnlineSpatialHistoryPipeline(wah_pipeline=pipe,active_node=node,memory_manager=manager,prompt=a.prompt,renderer_kwargs={'device':a.device},wah_state_kwargs={'height':a.height,'width':a.width,'num_frames':33,'output_type':'np','pyramid_num_inference_steps_list':list(HELIOS_PYRAMID_NUM_INFERENCE_STEPS)})
    online.autoregressive_state=pipe.init_autoregressive_state(prompt=a.prompt,image=Image.fromarray(node.view_rgb[0]),conditioning_type='warp',warp_history_downsample_mode='short',rope_alignment=True,height=a.height,width=a.width,num_frames=33,output_type='np',pyramid_num_inference_steps_list=list(HELIOS_PYRAMID_NUM_INFERENCE_STEPS))
    online.wah_adapter.configure_state(online.autoregressive_state); controls=json.loads(a.controls.read_text()); K=resize_intrinsics(node.view_intrinsics[0],node.view_rgb.shape[1:3],(a.height,a.width))
    generated=[]; warps=[]; panels=[]; reports=[]
    with torch.inference_mode():
      for chunk_controls in controls:
        video,_,warp,report=online.generate_chunk(chunk_controls,K,a.height,a.width); g=u8(video); w=u8(warp.rgb); v=np.asarray(warp.visibility)
        offset=0 if not generated else 1; generated.extend(g[offset:]); warps.extend(w[offset:])
        masked=w.copy(); masked[~v]=0; panel=np.concatenate([g,w,masked],axis=2); panels.extend(panel[offset:]); reports.append(report)
    a.output_dir.mkdir(parents=True,exist_ok=True); imageio.mimwrite(a.output_dir/'generated.mp4',np.asarray(generated),fps=24,macro_block_size=1); imageio.mimwrite(a.output_dir/'warp.mp4',np.asarray(warps),fps=24,macro_block_size=1); imageio.mimwrite(a.output_dir/'debug_generated_warp_visible.mp4',np.asarray(panels),fps=24,macro_block_size=1); (a.output_dir/'metrics.json').write_text(json.dumps(reports,indent=2,default=str))
if __name__=='__main__': main()

