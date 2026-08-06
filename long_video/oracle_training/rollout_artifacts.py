"""Streaming 24 FPS rollout videos and real-anchor evaluation artifacts."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw

def _uint8(value):
    array=np.asarray(value)
    if array.dtype==np.uint8: return array
    if not np.isfinite(array).all(): raise ValueError("generated/warp RGB contains NaN or Inf")
    return np.rint(np.clip(array,0,1)*255).astype(np.uint8)

class RolloutArtifacts:
    def __init__(self,root,target_dir,*,fps=24,anchor_stride=8):
        import imageio.v2 as imageio
        self.root=Path(root); self.root.mkdir(parents=True,exist_ok=True); self.target_dir=Path(target_dir)
        self.anchor_stride=int(anchor_stride); self.clean=imageio.get_writer(self.root/"final_4chunk_24fps.mp4",fps=fps,macro_block_size=1)
        self.debug=imageio.get_writer(self.root/"final_4chunk_debug.mp4",fps=fps,macro_block_size=1)
        self.anchor_rows=[]; self.frame_count=0

    def append_chunk(self,generated,warp,visibility,confidence,*,global_start,chunk_index,node_id,event):
        generated=_uint8(generated); warped=_uint8(warp)
        if not(len(generated)==len(warped)==len(visibility)==len(confidence)): raise ValueError("artifact streams have unequal lengths")
        for local,frame in enumerate(generated):
            global_index=int(global_start+local); self.clean.append_data(frame); self.frame_count+=1
            vis=np.asarray(visibility[local],bool); conf=np.asarray(confidence[local],np.float32)
            vis_rgb=np.repeat((vis.astype(np.uint8)*255)[...,None],3,axis=2)
            conf_rgb=np.repeat(np.rint(np.clip(conf,0,1)*255).astype(np.uint8)[...,None],3,axis=2)
            anchor=global_index%self.anchor_stride==0
            gt=None
            if anchor:
                path=self.target_dir/f"{global_index:06d}.png"
                if path.exists(): gt=np.asarray(Image.open(path).convert("RGB"))
            fourth=gt if gt is not None else conf_rgb
            h,w=frame.shape[:2]; canvas=Image.new("RGB",(2*w,2*h),(0,0,0))
            for xy,value in (((0,0),frame),((w,0),warped[local]),((0,h),vis_rgb),((w,h),fourth)):
                canvas.paste(Image.fromarray(value),xy)
            draw=ImageDraw.Draw(canvas); event_name=(event or {}).get("status") or (event or {}).get("reason") or "none"
            draw.rectangle((0,0,2*w,26),fill=(0,0,0)); draw.text((6,6),f"chunk={chunk_index} frame={global_index} node={node_id} event={event_name}",fill="white")
            self.debug.append_data(np.asarray(canvas))
            if gt is not None and global_index>0:
                from skimage.metrics import peak_signal_noise_ratio,structural_similarity
                l1=float(np.abs(gt.astype(np.float32)-frame.astype(np.float32)).mean()/255.)
                metrics={"global_frame":global_index,"l1":l1,"psnr":float(peak_signal_noise_ratio(gt,frame,data_range=255)),
                         "ssim":float(structural_similarity(gt,frame,channel_axis=2,data_range=255))}
                error=np.rint(np.abs(gt.astype(np.float32)-frame.astype(np.float32))).astype(np.uint8)
                self.anchor_rows.append((gt,frame,warped[local],error,metrics))

    def close(self):
        self.clean.close(); self.debug.close()
        rows=self.anchor_rows; width=640; height=384
        canvas=Image.new("RGB",(4*width,max(1,len(rows))*height+32),"white"); draw=ImageDraw.Draw(canvas)
        draw.text((5,5),"GT | generation | active-node warp | absolute error",fill="black")
        for row_index,(gt,gen,warp,error,metrics) in enumerate(rows):
            y=32+row_index*height
            for column,value in enumerate((gt,gen,warp,error)): canvas.paste(Image.fromarray(value),(column*width,y))
            draw.text((5,y+5),f"f={metrics['global_frame']} PSNR={metrics['psnr']:.2f} SSIM={metrics['ssim']:.3f} L1={metrics['l1']:.4f}",fill="white",stroke_width=2,stroke_fill="black")
        canvas.save(self.root/"anchor_evaluation.png")
        values=[item[-1] for item in rows]
        means={key:float(np.mean([item[key] for item in values])) for key in ("psnr","ssim","l1")} if values else {}
        return {"frame_count":self.frame_count,"anchor_metrics":values,"anchor_mean":means,
                "clean_video":str(self.root/"final_4chunk_24fps.mp4"),"debug_video":str(self.root/"final_4chunk_debug.mp4"),
                "anchor_evaluation":str(self.root/"anchor_evaluation.png")}
