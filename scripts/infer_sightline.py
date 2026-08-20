"""Run pinned Helios directly, without WAH or scene geometry."""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np
from PIL import Image
def main() -> None:
    p=argparse.ArgumentParser(); p.add_argument('--source',required=True); p.add_argument('--model',required=True); p.add_argument('--out',required=True); p.add_argument('--prompt',default=''); p.add_argument('--negative-prompt',default=''); p.add_argument('--chunks',type=int,default=6); p.add_argument('--helios-root',required=True); p.add_argument('--steps',type=int,default=2); a=p.parse_args()
    if not 1<=a.chunks<=6: raise ValueError('--chunks must be 1..6')
    sys.path.insert(0,a.helios_root)
    from helios.diffusers_version.pipeline_helios_diffusers import HeliosPipeline
    import torch
    pipe=HeliosPipeline.from_pretrained(a.model,torch_dtype=torch.bfloat16).to('cuda')
    image=Image.open(a.source).convert('RGB')
    result=pipe(prompt=a.prompt,negative_prompt=a.negative_prompt,image=image,height=384,width=640,num_frames=1+a.chunks*32,num_inference_steps=a.steps,history_sizes=[16,2,1],num_latent_frames_per_chunk=9,output_type='np',is_enable_stage2=False)
    video=getattr(result,'frames',result); arr=np.asarray(video); output=Path(a.out).with_suffix('.npy'); output.parent.mkdir(parents=True,exist_ok=True); np.save(output,arr); print({'frames':int(arr.shape[-4] if arr.ndim>=4 else len(arr)),'out':str(output)})
if __name__=='__main__': main()
