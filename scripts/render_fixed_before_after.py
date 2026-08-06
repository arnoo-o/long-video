#!/usr/bin/env python3
"""Generate a fixed diagnostic chunk with base WAH and the trained LoRA."""
from __future__ import annotations
import argparse,json,os,sys
from pathlib import Path

def main():
    p=argparse.ArgumentParser(); p.add_argument("--config",default="configs/oracle_wah_training.yaml"); p.add_argument("--sequence",required=True)
    p.add_argument("--output",required=True); p.add_argument("--set",action="append",default=[],dest="overrides"); args=p.parse_args()
    repo=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(repo)); from long_video.config import load_yaml
    config=load_yaml(args.config,args.overrides); physical=int(config.get("physical_gpu",1))
    if physical!=1: raise ValueError("generation is restricted to physical GPU 1")
    os.environ["CUDA_VISIBLE_DEVICES"]="1"; os.environ.setdefault("XFORMERS_DISABLED","1")
    import imageio.v2 as imageio
    import numpy as np
    import torch
    from PIL import Image,ImageDraw
    if torch.cuda.device_count()!=1: raise RuntimeError("generation must see exactly one GPU")
    sys.path.insert(0,str(Path(config["wah_root"]))); from warp_as_history import WarpAsHistoryPipeline
    sequence=Path(args.sequence); metadata=json.loads((sequence/"metadata.json").read_text(encoding="utf-8")); chunk=int(metadata["chunk_frames"])
    source=Image.open(sequence/"source"/"source_perspective.png").convert("RGB")
    warp=[Image.open(path).convert("RGB") for path in sorted((sequence/"single_chunk_warp"/"warp_rgb").glob("*.png"))]
    visibility=np.load(sequence/"single_chunk_warp"/"warp_visibility.npy"); confidence=np.load(sequence/"single_chunk_warp"/"warp_confidence.npy")
    def generate(lora):
        torch.manual_seed(int(config["seed"])); torch.cuda.manual_seed_all(int(config["seed"])); pipe=WarpAsHistoryPipeline.from_pretrained(config["wah_model"],torch_dtype=torch.bfloat16).to("cuda:0")
        if not hasattr(pipe.transformer.config,"image_dim"): pipe.transformer.register_to_config(image_dim=None)
        state=pipe.init_autoregressive_state(prompt=config["prompt"],image=source,conditioning_type="warp",lora_path=lora,
            visible_token_drop=True,warp_history_downsample_mode="short",rope_alignment=True,height=384,width=640,num_frames=chunk,
            output_type="np",add_noise_to_image_latents=False,pyramid_num_inference_steps_list=config["training"]["pyramid_num_inference_steps_list"],is_amplify_first_chunk=False)
        with torch.no_grad(): video,_=pipe.generate_next_chunk(state,warp_video=warp,warp_visibility_mask=visibility[None,None].astype(np.float32),
            warp_confidence_mask=(confidence*visibility)[None,None].astype(np.float32),output_type="np")
        value=np.asarray(video); value=value[0] if value.ndim==5 else value
        if value.ndim==4 and value.shape[1]==3: value=np.moveaxis(value,1,-1)
        value=value if value.dtype==np.uint8 else np.rint(np.clip(value,0,1)*255).astype(np.uint8)
        del state,pipe,video; torch.cuda.empty_cache(); return value
    base=generate(None); lora_path=Path(config["checkpoint_root"])/"oracle_wah_lora.pt"; trained=generate(str(lora_path))
    if len(base)!=chunk or len(trained)!=chunk: raise ValueError("before/after generated frame count mismatch")
    writer=imageio.get_writer(args.output,fps=int(config.get("output_fps",24)),macro_block_size=1)
    try:
        for index,(left,right) in enumerate(zip(base,trained)):
            canvas=Image.new("RGB",(1280,384)); canvas.paste(Image.fromarray(left),(0,0)); canvas.paste(Image.fromarray(right),(640,0))
            draw=ImageDraw.Draw(canvas); draw.rectangle((0,0,1280,26),fill="black"); draw.text((6,6),f"base | step-trained LoRA   frame={index}",fill="white")
            writer.append_data(np.asarray(canvas))
    finally: writer.close()
    print(json.dumps({"output":args.output,"frames":chunk,"fps":int(config.get("output_fps",24))},indent=2))

if __name__=="__main__": main()
