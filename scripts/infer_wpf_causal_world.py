#!/usr/bin/env python3
"""Run causal point world -> renderer -> official WAH + optional GeoToken."""
import argparse, json, sys, time
from pathlib import Path
import imageio.v2 as imageio
import numpy as np
from PIL import Image
import torch

def u8(value):
    x=np.asarray(value)
    return x if x.dtype==np.uint8 else np.rint(np.clip(x,0,1)*255).astype(np.uint8)

def progress_event(event, **values):
    print(json.dumps({'event':event, **values}, default=str), flush=True)

def main():
    p=argparse.ArgumentParser(); p.add_argument('--wah-root',type=Path,required=True); p.add_argument('--model',type=Path,required=True)
    p.add_argument('--session',type=Path,required=True); p.add_argument('--controls',type=Path,required=True); p.add_argument('--recal3r-repo',type=Path,required=True)
    p.add_argument('--recal3r-checkpoint',type=Path,required=True); p.add_argument('--output-dir',type=Path,required=True); p.add_argument('--device',default='cuda:0')
    p.add_argument('--recal-confidence-threshold', type=float, default=None,
                   help='Deprecated: ReCal uses each valid-grid raw-confidence 40th percentile.')
    p.add_argument('--pi3x-repo',type=Path,required=True); p.add_argument('--pi3x-checkpoint',type=Path,required=True)
    p.add_argument('--wpf-adaptation-checkpoint',type=Path)
    p.add_argument('--geotoken-checkpoint',type=Path,
                   help='GeoToken training checkpoint; runs the checkpoint architecture with WPF disabled.')
    p.add_argument('--geotoken-strength',type=float,default=1.0,
                   choices=(0.0,0.25,0.5,1.0))
    p.add_argument('--camera-strength',type=float,default=1.0)
    p.add_argument('--world-strength',type=float,default=1.0)
    p.add_argument('--allow-stale-geotoken-semantics', action='store_true')
    p.add_argument('--height',type=int,default=384); p.add_argument('--width',type=int,default=640); p.add_argument('--prompt',default='Continue the scene consistently.')
    p.add_argument('--online-fusion-voxel-size',type=float,default=0.05,
                   help='Phase-C/inference PointWorld voxel fusion size (default: 0.05).')
    p.add_argument('--recal-confidence-quantile',type=float,default=0.4,
                   help='Raw-confidence quantile over each valid original-grid frame (default: 0.4/P40).')
    a=p.parse_args()
    if not np.isfinite(a.online_fusion_voxel_size) or a.online_fusion_voxel_size <= 0:
        p.error('--online-fusion-voxel-size must be finite and positive')
    if not np.isfinite(a.recal_confidence_quantile) or not 0 <= a.recal_confidence_quantile <= 1:
        p.error('--recal-confidence-quantile must be in [0,1]')
    progress_event('inference_start', online_fusion_voxel_size=a.online_fusion_voxel_size,
                   recal_confidence_quantile=a.recal_confidence_quantile)
    sys.path.insert(0,str(a.wah_root))
    from long_video.wah.upstream import assert_wah_upstream
    assert_wah_upstream(a.wah_root)
    from long_video.data.camera import resize_intrinsics
    from long_video.initialization.recal3r_geometry_backend import ReCal3RGeometryBackend
    from long_video.initialization.recal3r_world_accumulator import ReCal3RWorldAccumulator
    from long_video.memory.node_store import NodeStore
    from long_video.online.pipeline import OnlineSpatialHistoryPipeline
    from long_video.wah.world_projected_pipeline import PYRAMID_INFERENCE_STEPS, WorldProjectedWarpAsHistoryPipeline
    stored_node=NodeStore(a.session).load('node_000')
    progress_event('source_session_loaded', resolution=list(stored_node.view_rgb.shape[1:3]))
    # W0 is rebuilt from this one source observation.  Never reuse a node
    # whose point cloud could have been seeded from earlier scene frames.
    from long_video.initialization.pi3x_geometry_backend import Pi3XGeometryBackend
    from long_video.initialization.pi3x_initial_world import build_pi3x_source_world
    if torch.cuda.is_available(): torch.cuda.synchronize()
    initial_world_started=time.perf_counter()
    progress_event('pi3x_w0_start')
    node=build_pi3x_source_world(stored_node.view_rgb[0], stored_node.view_c2w[0], stored_node.view_intrinsics[0],
                                 Pi3XGeometryBackend(a.pi3x_checkpoint,a.pi3x_repo,a.device))
    from long_video.initialization.pi3x_initial_world import revoxelize_pi3x_source_world
    online_fusion_voxel_size=float(a.online_fusion_voxel_size)
    node=revoxelize_pi3x_source_world(node,voxel_size=online_fusion_voxel_size)
    if torch.cuda.is_available(): torch.cuda.synchronize()
    initial_world_seconds=time.perf_counter()-initial_world_started
    progress_event('pi3x_w0_complete', seconds=initial_world_seconds,
                   point_count=int(len(node.points_xyz)), voxel_size=online_fusion_voxel_size)
    progress_event('wah_pipeline_load_start')
    if a.geotoken_checkpoint is None:
        pipe=WorldProjectedWarpAsHistoryPipeline.from_pretrained(a.model,torch_dtype=torch.bfloat16).to(a.device)
    else:
        # GeoToken was trained with the unprojected official WAH pipeline.
        from warp_as_history import WarpAsHistoryPipeline
        pipe=WarpAsHistoryPipeline.from_pretrained(a.model,torch_dtype=torch.bfloat16).to(a.device)
    progress_event('wah_pipeline_load_complete')
    if not hasattr(pipe.transformer.config,'image_dim'): pipe.transformer.register_to_config(image_dim=None)
    pipe._configure_wah_lora(str(a.wah_root/'checkpoints/warp-as-history/visible_lora_state_step1000.safetensors'))
    progress_event('wah_lora_configured')
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
        progress_event('geotoken_checkpoint_load_start', checkpoint=str(a.geotoken_checkpoint))
        from long_video.geometry.geotoken import install_geotoken
        from long_video.geometry.geotoken_runtime import PointWorldGeoTokenProvider, source_scene_scale_from_active_node
        conditioner=install_geotoken(pipe.transformer).to(device=a.device)
        conditioner.configure_strengths(
            geotoken=a.geotoken_strength, camera=a.camera_strength, world=a.world_strength,
        )
        checkpoint=torch.load(a.geotoken_checkpoint,map_location='cpu',weights_only=False)
        from long_video.training.geotoken import TRAINING_SEMANTICS_VERSION, GEOMETRY_SCHEMA_VERSION, GEOMETRY_IMPLEMENTATION_VERSION, WAH_RUNTIME_FINGERPRINT
        semantics_match = (checkpoint.get('training_semantics_version') == TRAINING_SEMANTICS_VERSION
                           and checkpoint.get('geometry_schema_version') == GEOMETRY_SCHEMA_VERSION
                           and checkpoint.get('geometry_implementation_version') == GEOMETRY_IMPLEMENTATION_VERSION)
        if not semantics_match and not a.allow_stale_geotoken_semantics:
            raise RuntimeError('GeoToken checkpoint has stale training/world-binding semantics')
        if not semantics_match:
            print('WARNING: stale GeoToken semantics allowed for diagnostic inference only')
        if checkpoint.get('wah_runtime_fingerprint') != WAH_RUNTIME_FINGERPRINT: raise RuntimeError('GeoToken checkpoint WAH runtime fingerprint mismatch')
        state=checkpoint.get('geotoken')
        named=dict(pipe.transformer.named_parameters())
        expected={name for name in named if 'geotoken.' in name}
        if not isinstance(state,dict) or set(state) != expected:
            raise RuntimeError('checkpoint GeoToken parameter set does not match inference transformer')
        with torch.no_grad():
            for name,value in state.items():
                named[name].copy_(value.to(device=named[name].device,dtype=named[name].dtype))
        geotoken_step=int(checkpoint.get('global_step',-1))
        progress_event('geotoken_checkpoint_load_complete', step=geotoken_step)
    for module in (pipe.transformer,pipe.vae):
        for q in module.parameters(): q.requires_grad_(False)
    geo=ReCal3RGeometryBackend(
        a.recal3r_checkpoint, a.recal3r_repo, a.device,
        confidence_threshold=1.5, confidence_quantile=a.recal_confidence_quantile,
    )
    trajectory_id=f"inference:{a.session.resolve()}"
    accumulator=ReCal3RWorldAccumulator(geo,node,trajectory_id=trajectory_id,voxel_size=online_fusion_voxel_size)
    progress_event('recal_accumulator_ready', point_count=int(len(node.points_xyz)))
    online=OnlineSpatialHistoryPipeline(wah_pipeline=pipe,active_node=node,memory_manager=None,world_accumulator=accumulator,prompt=a.prompt,renderer_kwargs={'device':a.device, 'point_radius':0},wah_state_kwargs={'height':a.height,'width':a.width,'num_frames':33,'output_type':'np','pyramid_num_inference_steps_list':list(PYRAMID_INFERENCE_STEPS)})
    online.autoregressive_state=pipe.init_autoregressive_state(prompt=a.prompt,image=Image.fromarray(node.view_rgb[0]),conditioning_type='warp',warp_history_downsample_mode='short',rope_alignment=True,height=a.height,width=a.width,num_frames=33,output_type='np',pyramid_num_inference_steps_list=list(PYRAMID_INFERENCE_STEPS))
    online.autoregressive_state['is_amplify_first_chunk']=False
    online.wah_adapter.configure_state(online.autoregressive_state); controls=json.loads(a.controls.read_text()); K=resize_intrinsics(node.view_intrinsics[0],node.view_rgb.shape[1:3],(a.height,a.width))
    progress_event('autoregressive_state_ready', chunks_total=len(controls))
    if a.geotoken_checkpoint is not None:
        source_c2w=np.asarray(node.view_c2w[0],np.float32)
        scene_scale=source_scene_scale_from_active_node(node,source_c2w,K,device=a.device,height=a.height,width=a.width)
        provider=PointWorldGeoTokenProvider(conditioner,device=a.device,source_center=source_c2w[:3,3],scene_scale=scene_scale,render_height=a.height,render_width=a.width)
        from long_video.geometry.geotoken import scheduler_progress_from_timestep
        provider.set_timing_resolver(lambda kwargs: scheduler_progress_from_timestep(pipe.scheduler, kwargs["timestep"]))
        provider.attach(pipe.transformer)
        def pre_render_world_hook(active_node,cameras):
            active_world=provider.configure_active_node(active_node)
            source_geometry=provider.ensure_source_geometry(source_c2w, K)
            existing_source=online.autoregressive_state.setdefault('_geotoken_source_geometry',source_geometry)
            if existing_source is not source_geometry: raise RuntimeError('GeoToken source geometry changed within one trajectory')
            provider.configure_chunk(
                cameras.c2w,cameras.intrinsics,online.autoregressive_state.get('_geotoken_history_snapshots',()),
                history_window=online.autoregressive_state.get('_wah_geometry_slot_refs',()),
                source_geometry=online.autoregressive_state['_geotoken_source_geometry'],
            )
            from long_video.online.pipeline import point_world_snapshot_identity
            return {'world_identity':point_world_snapshot_identity(active_node),'freeze_history':provider.freeze_current_snapshot}
        online.pre_render_world_hook=pre_render_world_hook
    from long_video.types import CameraBatch
    from long_video.geometry.point_renderer import render_geometry_cuda
    def scalar_debug_frame(values):
        values = np.asarray(values, np.float32)
        finite = values[np.isfinite(values)]
        image = np.zeros((*values.shape, 3), np.uint8)
        if len(finite):
            lo, hi = np.percentile(finite, (2, 98))
            hi = max(float(hi), float(lo) + 1e-6)
            normalized = np.clip((values - lo) / (hi - lo), 0, 1)
            image[...] = np.rint(normalized[..., None] * 255).astype(np.uint8)
        return image

    def xyz_debug_frame(values):
        values = np.asarray(values, np.float32)
        image = np.zeros((*values.shape[:2], 3), np.uint8)
        finite = np.isfinite(values).all(-1)
        if finite.any():
            lo, hi = np.percentile(values[finite], (2, 98), axis=0)
            normalized = np.clip((values - lo) / np.maximum(hi - lo, 1e-6), 0, 1)
            image[finite] = np.rint(normalized[finite] * 255).astype(np.uint8)
        return image

    generated=[]; warps=[]; panels=[]; geometry_frames=[]; reports=[]; chunk_inference_seconds=[]
    raw_depth_frames=[]; raw_confidence_frames=[]; native_world_frames=[]; commanded_world_frames=[]; association_mask_frames=[]
    with torch.inference_mode():
      for chunk_index, chunk_controls in enumerate(controls):
        progress_event('chunk_start', chunk=chunk_index + 1, chunks_total=len(controls),
                       point_count=int(len(online.active_node.points_xyz)))
        if torch.cuda.is_available(): torch.cuda.synchronize()
        chunk_inference_started=time.perf_counter()
        video,poses,warp,report=online.generate_chunk(chunk_controls,K,a.height,a.width)
        if torch.cuda.is_available(): torch.cuda.synchronize()
        chunk_inference_seconds.append(time.perf_counter()-chunk_inference_started)
        g=u8(video); w=u8(warp.rgb)
        # Render the post-update PointWorld, not the pre-generation WAH warp.
        # This makes the diagnostic video reflect the world used by the next
        # chunk and exposes whether ReCal observations actually changed it.
        cameras=CameraBatch(np.asarray(poses,np.float32), np.repeat(np.asarray(K,np.float32)[None],len(poses),axis=0), int(a.height), int(a.width))
        xyz_t=torch.as_tensor(online.active_node.points_xyz,dtype=torch.float32,device=a.device)
        conf_t=torch.as_tensor(online.active_node.points_confidence,dtype=torch.float32,device=a.device)
        _, depth_t, visible_t, confidence_t=render_geometry_cuda(xyz_t,conf_t,cameras,parent_point_count=getattr(online.active_node,'parent_point_count',None))
        depth=np.asarray(depth_t.detach().cpu()); visible=np.asarray(visible_t.detach().cpu()); confidence=np.asarray(confidence_t.detach().cpu())
        finite=visible & np.isfinite(depth) & (depth>0)
        values=depth[finite]
        lo,hi=(float(np.percentile(values,2)),float(np.percentile(values,98))) if len(values) else (0.0,1.0)
        hi=max(hi,lo+1e-6)
        normalized=np.clip((depth-lo)/(hi-lo),0,1)
        geom=np.zeros((len(depth),int(a.height),int(a.width),3),np.uint8)
        geom[...,0]=np.rint(normalized*255).astype(np.uint8)
        geom[...,1]=np.rint(np.clip(confidence,0,1)*255).astype(np.uint8)
        geom[...,2]=np.where(finite,255,0).astype(np.uint8)
        offset=0 if not generated else 1
        generated.extend(g[offset:]); warps.extend(w[offset:]); geometry_frames.extend(geom[offset:])
        association_chunk=np.concatenate([np.zeros((1,int(a.height),int(a.width),3),np.uint8),np.asarray(accumulator.association_debug_masks(),np.uint8)],axis=0)
        association_mask_frames.extend(association_chunk[offset:])
        frame_indices = range(int(report['frame_start']) - 1, int(report['frame_end']) + 1)
        recal_debug = accumulator.debug_geometry_for_frames(frame_indices)
        raw_depth_frames.extend(scalar_debug_frame(item['raw_recal_depth']) for item in recal_debug[offset:])
        raw_confidence_frames.extend(scalar_debug_frame(item['raw_recal_confidence']) for item in recal_debug[offset:])
        native_world_frames.extend(xyz_debug_frame(item['native_recal_world']) for item in recal_debug[offset:])
        commanded_world_frames.extend(xyz_debug_frame(item['commanded_world_before_fusion']) for item in recal_debug[offset:])
        panel=np.concatenate([g,w,geom],axis=2); panels.extend(panel[offset:]); reports.append(report)
        print(json.dumps({
            'event':'chunk_complete', 'chunk':chunk_index + 1, 'chunks_total':len(controls),
            'chunk_seconds':chunk_inference_seconds[-1],
            'point_count':int(len(online.active_node.points_xyz)),
            'frame_end':int(report['frame_end']),
        }), flush=True)
    a.output_dir.mkdir(parents=True,exist_ok=True)
    imageio.mimwrite(a.output_dir/'persistent_surface_association_mask.mp4',np.asarray(association_mask_frames),fps=24,macro_block_size=1)
    confidence_mode=f"p{100 * a.recal_confidence_quantile:g}_valid_grid_raw_confidence"
    confidence_description=(f"{100 * a.recal_confidence_quantile:g}th percentile of raw confidence over valid original-grid depth/confidence pixels, per ReCal frame")
    a.output_dir.mkdir(parents=True,exist_ok=True); imageio.mimwrite(a.output_dir/'generated.mp4',np.asarray(generated),fps=24,macro_block_size=1); imageio.mimwrite(a.output_dir/'warp.mp4',np.asarray(warps),fps=24,macro_block_size=1); imageio.mimwrite(a.output_dir/'generated_pixels_and_warp_world.mp4',np.concatenate([np.asarray(generated),np.asarray(warps)],axis=2),fps=24,macro_block_size=1); imageio.mimwrite(a.output_dir/'geometry_post_update.mp4',np.asarray(geometry_frames),fps=24,macro_block_size=1); imageio.mimwrite(a.output_dir/'debug_generated_warp_geometry.mp4',np.asarray(panels),fps=24,macro_block_size=1); imageio.mimwrite(a.output_dir/'raw_recal_depth.mp4',np.asarray(raw_depth_frames),fps=24,macro_block_size=1); imageio.mimwrite(a.output_dir/'raw_recal_confidence.mp4',np.asarray(raw_confidence_frames),fps=24,macro_block_size=1); imageio.mimwrite(a.output_dir/'native_recal_world.mp4',np.asarray(native_world_frames),fps=24,macro_block_size=1); imageio.mimwrite(a.output_dir/'commanded_world_before_fusion.mp4',np.asarray(commanded_world_frames),fps=24,macro_block_size=1); (a.output_dir/'recal_debug_semantics.json').write_text(json.dumps({'raw_recal_depth':'ReCal pts3d_in_self_view.z remapped to the original RGB grid before thresholding','raw_recal_confidence':'ReCal conf_self remapped to the original RGB grid before thresholding','confidence_threshold':confidence_description,'native_recal_world':'ReCal self-view Z remapped to the original RGB grid and backprojected with commanded intrinsics; local XYZ coordinate colors, not RGB','commanded_world_before_fusion':'the same commanded-intrinsics backprojection after fixed ReCal-to-W0 scale plus commanded c2w, before validity threshold or voxel fusion; coordinate colors, not RGB','generated_pixels_and_warp_world':'left: generated RGB pixels; right: RGB warp rendered from the pre-generation persistent PointWorld'},indent=2)); (a.output_dir/'metrics.json').write_text(json.dumps({'pyramid_num_inference_steps_list':list(PYRAMID_INFERENCE_STEPS),'wpf_enabled':a.geotoken_checkpoint is None,'wpf_adaptation_checkpoint':str(a.wpf_adaptation_checkpoint) if a.wpf_adaptation_checkpoint else None,'wpf_adaptation_step':adaptation_step,'geotoken_checkpoint':str(a.geotoken_checkpoint) if a.geotoken_checkpoint else None,'geotoken_step':geotoken_step,'camera_strength':a.camera_strength if a.geotoken_checkpoint else None,'world_strength':a.world_strength if a.geotoken_checkpoint else None,'recal_confidence_threshold_mode':confidence_mode,'recal_confidence_quantile':a.recal_confidence_quantile,'geotoken_injection':getattr(conditioner,'diagnostics',None) if a.geotoken_checkpoint else None,'chunks':reports,'initial_world_seconds':float(initial_world_seconds),'chunk_inference_seconds':chunk_inference_seconds,'chunk_inference_seconds_total':float(sum(chunk_inference_seconds)),'inference_core_seconds':float(initial_world_seconds+sum(chunk_inference_seconds)),'pixel_generation_seconds_total':float(sum(item.get('pixel_generation_seconds',0.0) for item in reports)),'pixel_generation_seconds_mean':float(np.mean([item.get('pixel_generation_seconds',0.0) for item in reports])) if reports else 0.0},indent=2,default=str))
    semantics_path=a.output_dir/'recal_debug_semantics.json'
    semantics=json.loads(semantics_path.read_text()); semantics['confidence_threshold']=confidence_description; semantics['persistent_surface_association_mask']='green=MATCH/owned duplicate, red=CONFLICT, blue=valid NOVEL before commit, cyan=chunk-local NOVEL committed, magenta=FREE_SPACE_VIOLATION, black=invalid'; semantics_path.write_text(json.dumps(semantics,indent=2))
    metrics_path=a.output_dir/'metrics.json'
    metrics_payload=json.loads(metrics_path.read_text()); metrics_payload['recal_confidence_threshold_mode']=confidence_mode; metrics_payload['recal_confidence_quantile']=a.recal_confidence_quantile; metrics_payload['surface_ownership']='immutable_xyz_chunk_local_immediate_commit_first_chunk_nonconflicting_v4'; metrics_payload['online_fusion_voxel_size']=online_fusion_voxel_size; metrics_payload['match_base_tolerance']=0.04; metrics_payload['source_free_space_base_tolerance']=0.06; metrics_path.write_text(json.dumps(metrics_payload,indent=2,default=str))
if __name__=='__main__': main()

