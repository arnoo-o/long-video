#!/usr/bin/env python3
"""Build a sparse ERP, run official DiT360 editing, and restore observations."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from long_video.data.panorama_projection import (
    build_canonical_view_cameras,equirectangular_to_perspective,
    intrinsics_from_fov,rotation_yaw_pitch,
)


def erp_rays(height,width):
    yy,xx=np.indices((height,width),np.float32)
    lon=((xx+0.5)/width-0.5)*(2*np.pi)
    lat=(0.5-(yy+0.5)/height)*np.pi
    return np.stack((np.cos(lat)*np.sin(lon),-np.sin(lat),
                     np.cos(lat)*np.cos(lon)),axis=-1)


def observation_intrinsics(item,width,height):
    if "intrinsics" in item:
        return np.asarray(item["intrinsics"],np.float32)
    return intrinsics_from_fov(float(item["fov_degrees"]),width,height)


def project_observations_to_erp(observed,height=1024,width=2048):
    world=erp_rays(height,width)
    rgb_sum=np.zeros((height,width,3),np.float64)
    square_sum=np.zeros_like(rgb_sum)
    weight_sum=np.zeros((height,width),np.float64)
    contributor_count=np.zeros((height,width),np.uint16)
    for item in observed:
        image=np.asarray(Image.open(item["image_path"]).convert("RGB"),np.float32)/255
        ih,iw=image.shape[:2]; k=observation_intrinsics(item,iw,ih)
        rotation=rotation_yaw_pitch(np.deg2rad(float(item["yaw_degrees"])),
                                    np.deg2rad(float(item.get("pitch_degrees",0))))
        local=world@rotation
        z=local[...,2]
        map_x=k[0,0]*local[...,0]/np.maximum(z,1e-8)+k[0,2]
        map_y=k[1,1]*local[...,1]/np.maximum(z,1e-8)+k[1,2]
        valid=((z>1e-6)&(map_x>=0)&(map_x<=iw-1)&
               (map_y>=0)&(map_y<=ih-1))
        sampled=cv2.remap(image,map_x.astype(np.float32),map_y.astype(np.float32),
                          cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
        weight=valid.astype(np.float64)*np.clip(z,0,1)**4
        rgb_sum+=sampled*weight[...,None]
        square_sum+=sampled**2*weight[...,None]
        weight_sum+=weight
        contributor_count+=valid
    valid=weight_sum>1e-8
    rgb=rgb_sum/np.maximum(weight_sum[...,None],1e-8)
    variance=np.maximum(square_sum/np.maximum(weight_sum[...,None],1e-8)-rgb**2,0)
    conflict=(contributor_count>=2)&(np.sqrt(variance.mean(-1))>0.08)
    return rgb.astype(np.float32),valid,weight_sum.astype(np.float32),conflict


def periodic_distance_to_observation(valid):
    missing=(~valid).astype(np.uint8)
    tiled=np.concatenate([missing,missing,missing],axis=1)
    distance=cv2.distanceTransform(tiled,cv2.DIST_L2,5)
    width=valid.shape[1]
    return distance[:,width:2*width]


def confidence_map(valid,conflict,base):
    height,width=valid.shape
    distance=periodic_distance_to_observation(valid)
    confidence=base*np.exp(-distance/max(1,width*0.25))
    latitude=np.abs((np.arange(height)+0.5)/height-0.5)*2
    confidence*=((1-latitude[:,None])*0.5+0.5)
    seam=np.minimum(np.arange(width),np.arange(width)[::-1])
    confidence*=np.where(seam[None,:]<32,0.75,1.0)
    confidence[valid]=1.0
    confidence[conflict]*=0.5
    return np.clip(confidence,0,1).astype(np.float32)


def run_dit360(manifest,init_rgb,valid_mask):
    import torch
    repo=Path(manifest["dit360_repo"])
    sys.path.insert(0,str(repo))
    from pa_src.pipeline import RFPanoInversionParallelFluxPipeline
    from pa_src.attn_processor import PersonalizeAnythingAttnProcessor,set_flux_transformer_attn_processor
    from pa_src.utils import create_mask

    if not torch.cuda.is_available():
        raise RuntimeError("DiT360 real inference requires CUDA")
    device=torch.device("cuda:0")
    dtype=torch.float16
    pipe=RFPanoInversionParallelFluxPipeline.from_pretrained(
        manifest["base_model_path"],torch_dtype=dtype,low_cpu_mem_usage=True).to(device)
    pipe.load_lora_weights(manifest["lora_path"])
    height,width=init_rgb.shape[:2]
    torch.cuda.reset_peak_memory_stats(device)
    mask_path=Path(manifest["output_dir"])/"erp_valid_mask.png"
    latent_h=height//(pipe.vae_scale_factor*2)
    latent_w=width//(pipe.vae_scale_factor*2)
    mask=create_mask(mask_path,latent_w,latent_h).float()
    mask=torch.cat([mask[:,0:1],mask,mask[:,-1:]],dim=-1).view(-1,1)
    image=Image.fromarray(np.rint(init_rgb*255).clip(0,255).astype(np.uint8))
    steps=int(manifest.get("num_inference_steps",28))
    inverted,image_latents,image_ids=pipe.invert(
        source_prompt="",image=image,height=height,width=width,
        num_inversion_steps=steps,gamma=1.0)
    img_dims=latent_h*(latent_w+2)
    set_flux_transformer_attn_processor(
        pipe.transformer,
        set_attn_proc_func=lambda name,dh,nh,ap: PersonalizeAnythingAttnProcessor(
            name=name,tau=float(manifest.get("tau",0.5)),mask=mask,
            device=device,img_dims=img_dims),
    )
    prompt=str(manifest["prompt"])
    output=pipe(
        [prompt,prompt],inverted_latents=inverted,image_latents=image_latents,
        latent_image_ids=image_ids,height=height,width=width,start_timestep=0.0,
        stop_timestep=0.99,num_inference_steps=steps,eta=1.0,
        generator=torch.Generator(device=device).manual_seed(int(manifest.get("seed",0))),
        mask=mask,use_timestep=True,
    ).images[1]
    return np.asarray(output.convert("RGB"),np.uint8),{
        "gpu_name":torch.cuda.get_device_name(device),
        "peak_gpu_memory_allocated_bytes":int(torch.cuda.max_memory_allocated(device)),
        "peak_gpu_memory_reserved_bytes":int(torch.cuda.max_memory_reserved(device)),
    }


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--manifest",required=True)
    args=parser.parse_args()
    manifest=json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    output=Path(manifest["output_dir"]); output.mkdir(parents=True,exist_ok=True)
    start=time.time()
    observed,valid,weight,conflict=project_observations_to_erp(
        manifest["observed_views"],manifest["erp_height"],manifest["erp_width"])
    np.save(output/"erp_observed_rgb.npy",observed)
    np.save(output/"erp_valid_mask.npy",valid)
    np.save(output/"erp_fusion_weight.npy",weight)
    np.save(output/"erp_conflict.npy",conflict)
    Image.fromarray((valid*255).astype(np.uint8)).save(output/"erp_valid_mask.png")
    Image.fromarray(np.rint(observed*255).clip(0,255).astype(np.uint8)).save(
        output/"erp_observed_rgb.png")
    dit_gpu={}
    if manifest.get("prepare_only",False):
        generated=np.rint(observed*255).clip(0,255).astype(np.uint8)
    else:
        generated,dit_gpu=run_dit360(manifest,observed,valid)
    np.save(output/"erp_generated_rgb.npy",generated)
    Image.fromarray(generated).save(output/"preview_generated_panorama.png")
    restore_observations=bool(manifest.get("restore_observations",True))
    restored=generated.copy()
    if restore_observations:
        restored[valid]=np.rint(observed[valid]*255).clip(0,255).astype(np.uint8)
    np.save(output/"erp_rgb.npy",restored)
    Image.fromarray(restored).save(output/"preview_panorama.png")
    if restore_observations:
        erp_conf=confidence_map(valid,conflict,float(manifest["synthesized_confidence"]))
        source_valid=valid
    else:
        erp_conf=np.full(valid.shape,float(manifest["synthesized_confidence"]),np.float32)
        source_valid=np.zeros_like(valid)
    views=[]; masks=[]; confidences=[]
    yaws=manifest["target_yaws_degrees"]
    for yaw in yaws:
        radians=np.deg2rad(float(yaw)); pitch=np.deg2rad(float(manifest["target_pitch_degrees"]))
        views.append(equirectangular_to_perspective(
            restored,radians,pitch,manifest["target_fov_degrees"],
            manifest["height"],manifest["width"],"bilinear"))
        masks.append(equirectangular_to_perspective(
            source_valid.astype(np.uint8),radians,pitch,manifest["target_fov_degrees"],
            manifest["height"],manifest["width"],"nearest")>0)
        confidences.append(equirectangular_to_perspective(
            erp_conf,radians,pitch,manifest["target_fov_degrees"],
            manifest["height"],manifest["width"],"bilinear"))
    views=np.stack(views); masks=np.stack(masks); confidences=np.stack(confidences).astype(np.float32)
    source=np.where(masks,0,1).astype(np.int8)
    c2w,intrinsics=build_canonical_view_cameras(
        np.eye(4,dtype=np.float32),manifest["target_fov_degrees"],
        manifest["width"],manifest["height"],yaws,manifest["target_pitch_degrees"])
    np.save(output/"views_rgb.npy",views)
    np.save(output/"view_poses.npy",c2w)
    np.save(output/"intrinsics.npy",intrinsics)
    np.save(output/"observed_masks.npy",masks)
    np.save(output/"source_maps.npy",source)
    np.save(output/"image_confidence.npy",confidences)
    metadata={
        "backend":"official DiT360 FLUX.1-dev panorama editing",
        "prepare_only":bool(manifest.get("prepare_only",False)),
        "restore_observations":restore_observations,
        "erp_rgb_source":"observations_restored" if restore_observations else "dit360_generated_only",
        "observed_erp_ratio":float(valid.mean()),
        "conflict_erp_ratio":float(conflict.mean()),
        "elapsed_seconds":time.time()-start,
        "coordinate_convention":"OpenCV_c2w_x_right_y_down_z_forward",
        "erp_shape":[manifest["erp_height"],manifest["erp_width"]],
        "dit360_gpu":dit_gpu,
    }
    (output/"metadata.json").write_text(json.dumps(metadata,indent=2),encoding="utf-8")


if __name__=="__main__":
    main()
