"""Build 24 FPS Oracle-M0 windows from sparse Holo360D anchors."""
from __future__ import annotations
import hashlib, json
from pathlib import Path
import numpy as np
from PIL import Image
from ..data.camera import rgb_to_uint8
from ..data.erp_geometry import source_relative_c2w
from ..data.holo360d import Holo360DReader
from ..geometry.point_renderer import render
from ..memory.node_store import NodeStore
from ..types import CameraBatch
from .dataset import _perspective,_resize_erp,_write_png_frames,attach_warp_provenance
from .dense24 import DenseTiming,dense_rgb_weights,interpolate_c2w,temporal_weights_to_latent
from .oracle_node import build_oracle_erp_node

def _sha256(path):
    digest=hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda:handle.read(1024*1024),b""): digest.update(chunk)
    return digest.hexdigest()

def build_dense_oracle_sequence(scene_root,output_root,*,sequence_id,split,anchor_count,rife,
        erp_resolution=(1024,2048),perspective_resolution=(384,640),fov_degrees=90.,pixel_center=.5,
        prompt="an indoor scene",voxel_size=.01,renderer_kwargs=None,rife_revision="unknown",rife_checkpoint=None):
    timing=DenseTiming(); reader=Holo360DReader(scene_root,normalize_first_pose=False)
    if len(reader.frame_ids)!=anchor_count:
        raise ValueError(f"extracted window has {len(reader.frame_ids)} frames, expected {anchor_count} anchors")
    anchors=[reader.read(i) for i in range(anchor_count)]; source=anchors[0]
    source_rgb,source_depth,source_mask=_resize_erp(source.rgb,source.depth,source.mask,erp_resolution)
    source_world=np.asarray(source.raw_c2w,np.float32)
    node=build_oracle_erp_node(source_rgb,source_depth,source_mask,source_c2w_local=np.eye(4,dtype=np.float32),
        voxel_size=voxel_size,pixel_center=pixel_center,
        model_versions={"geometry":"Holo360D_mesh_depth","builder":"oracle_erp_24fps_v1"})
    height,width=map(int,perspective_resolution)
    anchor_rgb=[]; anchor_z=[]; anchor_ray=[]; anchor_mask=[]; anchor_k=[]; anchor_world=[]; anchor_local=[]
    for frame in anchors:
        rgb,z,ray,valid,k=_perspective(frame,fov_degrees=fov_degrees,height=height,width=width)
        anchor_rgb.append(rgb); anchor_z.append(z); anchor_ray.append(ray); anchor_mask.append(valid); anchor_k.append(k)
        anchor_world.append(frame.raw_c2w); anchor_local.append(source_relative_c2w(source_world,frame.raw_c2w))
    anchor_rgb=np.stack(anchor_rgb); anchor_z=np.stack(anchor_z).astype(np.float32)
    anchor_ray=np.stack(anchor_ray).astype(np.float32); anchor_mask=np.stack(anchor_mask)
    dense_rgb=rife.interpolate(anchor_rgb,Path(output_root)/"_rife_work"/sequence_id)
    dense_local=interpolate_c2w(np.stack(anchor_local),timing.anchor_stride)
    dense_world=interpolate_c2w(np.stack(anchor_world),timing.anchor_stride)
    dense_k=np.repeat(np.asarray(anchor_k[:1],np.float32),len(dense_rgb),axis=0)
    dense_z=np.full((len(dense_rgb),height,width),np.nan,np.float32)
    dense_ray=np.full_like(dense_z,np.nan); dense_valid=np.zeros_like(dense_z,bool)
    anchor_indices=timing.anchor_indices(anchor_count)
    dense_z[anchor_indices]=anchor_z; dense_ray[anchor_indices]=anchor_ray; dense_valid[anchor_indices]=anchor_mask
    weights=dense_rgb_weights(anchor_count,timing); roles=np.full(len(dense_rgb),"interpolated_supervision_only",dtype="U40")
    roles[anchor_indices]="ground_truth_anchor"; roles[0]="source_prefix"
    out=Path(output_root)/sequence_id; out.mkdir(parents=True,exist_ok=True); NodeStore(out/"session").save(node)
    cameras=CameraBatch(dense_local[:timing.chunk_frames],dense_k[:timing.chunk_frames],height,width)
    options={"near":.05,"far":100.,"point_radius":1,"device":"cpu",**dict(renderer_kwargs or {})}
    warp=attach_warp_provenance(render(node,cameras,**options),node)
    source_dir=out/"source"; target_dir=out/"target"; warp_dir=out/"single_chunk_warp"
    source_dir.mkdir(exist_ok=True); target_dir.mkdir(exist_ok=True); warp_dir.mkdir(exist_ok=True)
    Image.fromarray(source_rgb).save(source_dir/"source_erp_rgb.png"); np.save(source_dir/"source_erp_depth_ray_distance.npy",source_depth)
    Image.fromarray(source_mask.astype(np.uint8)*255).save(source_dir/"source_erp_mask.png"); np.save(source_dir/"source_c2w_world.npy",source_world)
    Image.fromarray(anchor_rgb[0]).save(source_dir/"source_perspective.png"); _write_png_frames(target_dir/"target_rgb_for_loss",dense_rgb)
    for name,value in {"target_z_depth_for_eval":dense_z,"target_ray_distance_for_reference":dense_ray,"target_valid_mask":dense_valid,
                       "target_c2w_world":dense_world,"target_c2w_local":dense_local,"intrinsics":dense_k,
                       "supervision_weights_rgb":weights,"supervision_roles":roles}.items(): np.save(target_dir/f"{name}.npy",value)
    _write_png_frames(warp_dir/"warp_rgb",warp.rgb)
    for name,value in {"warp_z_depth":warp.depth,"warp_visibility":warp.visibility,"warp_confidence":warp.confidence,
                       "rgb_content_origin":warp.rgb_content_origin,"depth_content_origin":warp.depth_content_origin,
                       "evidence_role":warp.evidence_role}.items(): np.save(warp_dir/f"{name}.npy",value)
    latent=temporal_weights_to_latent(weights,timing.vae_temporal_scale)
    np.save(out/"primary_loss_weight_rgb.npy",weights); np.save(out/"primary_loss_weight_latent.npy",latent)
    np.save(out/"primary_loss_mask_rgb.npy",weights>0); np.save(out/"primary_loss_mask_latent.npy",latent>0)
    (out/"prompt.txt").write_text(prompt,encoding="utf-8")
    metadata={"sequence_id":sequence_id,"split":split,"scene_id":Path(scene_root).name,"data_fps_nominal":3,
      "model_fps":24,"output_fps":24,"anchor_stride":8,"anchor_count":anchor_count,"model_frame_count":len(dense_rgb),
      "anchor_model_indices":anchor_indices.tolist(),"chunk_frames":33,"chunk_stride":32,"vae_temporal_scale":4,
      "source_frame_id":anchors[0].frame_id,"anchor_frame_ids":[f.frame_id for f in anchors],
      "erp_resolution":list(map(int,erp_resolution)),"perspective_resolution":[height,width],"fov_degrees":float(fov_degrees),
      "source_depth_convention":"RAY_DISTANCE","target_evaluation_depth_convention":"Z_DEPTH","renderer_depth_convention":"Z_DEPTH",
      "coordinate_convention":"OpenCV c2w: +x right, +y down, +z forward","scale_mode":"dataset_calibrated","meters_per_world_unit":1.0,
      "future_geometry_used":False,"rife_version":"4.25","rife_variant":"full","rife_revision":rife_revision,
      "rife_checkpoint_sha256":_sha256(rife_checkpoint) if rife_checkpoint else None,
      "interpolated_supervision_confidence_source":"fixed_no_model_head","interpolated_supervision_weight":.25,
      "target_gt_allowed_roles":["loss","offline_evaluation"],"memory_video_source":"generated_rgb_for_memory_only"}
    (out/"metadata.json").write_text(json.dumps(metadata,indent=2),encoding="utf-8")
    return out,metadata
