#!/usr/bin/env python3
"""Run causal point world -> renderer -> official WAH + optional GeoToken."""
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
    p.add_argument('--session',type=Path,required=True); p.add_argument('--controls',type=Path,required=True); p.add_argument('--recal3r-repo',type=Path,required=True)
    p.add_argument('--recal3r-checkpoint',type=Path,required=True); p.add_argument('--output-dir',type=Path,required=True); p.add_argument('--device',default='cuda:0')
    p.add_argument('--wpf-adaptation-checkpoint',type=Path)
    p.add_argument('--geotoken-checkpoint',type=Path,
                   help='GeoToken training checkpoint; runs the checkpoint architecture with WPF disabled.')
    p.add_argument('--geotoken-strength',type=float,default=1.0,
                   choices=(0.0,0.25,0.5,1.0))
    p.add_argument('--height',type=int,default=384); p.add_argument('--width',type=int,default=640); p.add_argument('--prompt',default='Continue the scene consistently.'); a=p.parse_args()
    sys.path.insert(0,str(a.wah_root))
    from long_video.config import load_yaml
    from long_video.data.camera import resize_intrinsics
    from long_video.initialization.recal3r_geometry_backend import ReCal3RGeometryBackend
    from long_video.memory.memory_manager import MemoryManager
    from long_video.memory.node_store import NodeStore
    from long_video.online.pipeline import OnlineSpatialHistoryPipeline
    from long_video.wah.world_projected_pipeline import PYRAMID_INFERENCE_STEPS, WorldProjectedWarpAsHistoryPipeline
    stored_node=NodeStore(a.session).load('node_000')
    node=stored_node
    if a.geotoken_checkpoint is None:
        pipe=WorldProjectedWarpAsHistoryPipeline.from_pretrained(a.model,torch_dtype=torch.bfloat16).to(a.device)
    else:
        # GeoToken was trained with the unprojected official WAH pipeline.
        from warp_as_history import WarpAsHistoryPipeline
        pipe=WarpAsHistoryPipeline.from_pretrained(a.model,torch_dtype=torch.bfloat16).to(a.device)
    if not hasattr(pipe.transformer.config,'image_dim'): pipe.transformer.register_to_config(image_dim=None)
    pipe._configure_wah_lora(str(a.wah_root/'checkpoints/warp-as-history/visible_lora_state_step1000.safetensors'))
    adaptation_step=None
    if a.wpf_adaptation_checkpoint is not None:
        from long_video.training.wpf_adaptation import (
            adaptation_parameter_items, configure_trainable_wpf_adapter,
        )
        configure_trainable_wpf_adapter(pipe)
        checkpoint=torch.load(a.wpf_adaptation_checkpoint,map_location='cpu',weights_only=False)
        state=checkpoint.get('wpf_adaptation')
        if not isinstance(state,dict):
            raise RuntimeError('checkpoint does not contain wpf_adaptation state')
        current=dict(pipe.transformer.named_parameters())
        expected={name for name,_ in adaptation_parameter_items(pipe.transformer)}
        if set(state)!=expected:
            raise RuntimeError('checkpoint wpf_adaptation keys do not match the inference adapter')
        with torch.no_grad():
            for name,value in state.items():
                current[name].copy_(value.to(device=current[name].device,dtype=current[name].dtype))
        adaptation_step=int(checkpoint.get('global_step',-1))
    geotoken_step=None
    provider=None
    if a.geotoken_checkpoint is not None:
        from long_video.geometry.geotoken import install_geotoken
        from long_video.geometry.geotoken_runtime import PointWorldGeoTokenProvider, source_scene_scale_from_active_node
        conditioner=install_geotoken(pipe.transformer).to(device=a.device)
        conditioner.set_strength(a.geotoken_strength)
        checkpoint=torch.load(a.geotoken_checkpoint,map_location='cpu',weights_only=False)
        state=checkpoint.get('geotoken')
        named=dict(pipe.transformer.named_parameters())
        expected={name for name in named if 'geotoken.' in name}
        if not isinstance(state,dict) or set(state) != expected:
            raise RuntimeError('checkpoint GeoToken parameter set does not match inference transformer')
        with torch.no_grad():
            for name,value in state.items():
                named[name].copy_(value.to(device=named[name].device,dtype=named[name].dtype))
        geotoken_step=int(checkpoint.get('global_step',-1))
    for module in (pipe.transformer,pipe.vae):
        for q in module.parameters(): q.requires_grad_(False)
    geo=ReCal3RGeometryBackend(a.recal3r_checkpoint,a.recal3r_repo,a.device); manager=MemoryManager.from_config(load_yaml('configs/online_memory.yaml'),geometry_backend=geo)
    online=OnlineSpatialHistoryPipeline(wah_pipeline=pipe,active_node=node,memory_manager=manager,prompt=a.prompt,renderer_kwargs={'device':a.device, 'point_radius':0},wah_state_kwargs={'height':a.height,'width':a.width,'num_frames':33,'output_type':'np','pyramid_num_inference_steps_list':list(PYRAMID_INFERENCE_STEPS)})
    online.autoregressive_state=pipe.init_autoregressive_state(prompt=a.prompt,image=Image.fromarray(node.view_rgb[0]),conditioning_type='warp',warp_history_downsample_mode='short',rope_alignment=True,height=a.height,width=a.width,num_frames=33,output_type='np',pyramid_num_inference_steps_list=list(PYRAMID_INFERENCE_STEPS))
    online.autoregressive_state['is_amplify_first_chunk']=False
    online.wah_adapter.configure_state(online.autoregressive_state); controls=json.loads(a.controls.read_text()); K=resize_intrinsics(node.view_intrinsics[0],node.view_rgb.shape[1:3],(a.height,a.width))
    if a.geotoken_checkpoint is not None:
        source_c2w=np.asarray(node.view_c2w[0],np.float32)
        scene_scale=source_scene_scale_from_active_node(node,source_c2w,K,device=a.device,height=a.height,width=a.width)
        provider=PointWorldGeoTokenProvider(conditioner,device=a.device,source_center=source_c2w[:3,3],scene_scale=scene_scale)
        provider.attach(pipe.transformer)
        def pre_render_world_hook(active_node,cameras):
            provider.configure_active_node(active_node)
            provider.configure_chunk(cameras.c2w,cameras.intrinsics,online.autoregressive_state.get('_geotoken_history_snapshots',()))
            return {'world_version':getattr(active_node,'node_id',id(active_node)),'freeze_history':provider.freeze_current_snapshot}
        online.pre_render_world_hook=pre_render_world_hook
    generated=[]; warps=[]; panels=[]; reports=[]
    with torch.inference_mode():
      for chunk_controls in controls:
        video,_,warp,report=online.generate_chunk(chunk_controls,K,a.height,a.width); g=u8(video); w=u8(warp.rgb); v=np.asarray(warp.visibility)
        offset=0 if not generated else 1; generated.extend(g[offset:]); warps.extend(w[offset:])
        masked=w.copy(); masked[~v]=0; panel=np.concatenate([g,w,masked],axis=2); panels.extend(panel[offset:]); reports.append(report)
    a.output_dir.mkdir(parents=True,exist_ok=True); imageio.mimwrite(a.output_dir/'generated.mp4',np.asarray(generated),fps=24,macro_block_size=1); imageio.mimwrite(a.output_dir/'warp.mp4',np.asarray(warps),fps=24,macro_block_size=1); imageio.mimwrite(a.output_dir/'debug_generated_warp_visible.mp4',np.asarray(panels),fps=24,macro_block_size=1); (a.output_dir/'metrics.json').write_text(json.dumps({'pyramid_num_inference_steps_list':list(PYRAMID_INFERENCE_STEPS),'wpf_enabled':a.geotoken_checkpoint is None,'wpf_adaptation_checkpoint':str(a.wpf_adaptation_checkpoint) if a.wpf_adaptation_checkpoint else None,'wpf_adaptation_step':adaptation_step,'geotoken_checkpoint':str(a.geotoken_checkpoint) if a.geotoken_checkpoint else None,'geotoken_step':geotoken_step,'geotoken_strength':a.geotoken_strength if a.geotoken_checkpoint else None,'geotoken_injection':getattr(conditioner,'diagnostics',None) if a.geotoken_checkpoint else None,'chunks':reports},indent=2,default=str))
if __name__=='__main__': main()

