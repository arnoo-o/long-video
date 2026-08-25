"""Run with torchrun --standalone --nproc-per-node=4 to verify step0 identity."""
from __future__ import annotations
import json, os, sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
import torch.distributed as dist
from long_video.training.sightline import SightlineTrainable,install_lora,set_initialization_seed,broadcast_and_assert_trainables
from long_video.sightline.memory import LayerKVMemoryBank

class Attention(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.to_q=torch.nn.Linear(8,8); self.to_k=torch.nn.Linear(8,8); self.to_v=torch.nn.Linear(8,8); self.to_out=torch.nn.ModuleList([torch.nn.Linear(8,8),torch.nn.Identity()]); self.to_qkv=None; self.fused_projections=False
    def unfuse_projections(self): self.to_qkv=None; self.fused_projections=False
class Block(torch.nn.Module):
    def __init__(self): super().__init__(); self.attn1=Attention()
class Transformer(torch.nn.Module):
    def __init__(self): super().__init__(); self.transformer_blocks=torch.nn.ModuleList([Block()])

def main():
    dist.init_process_group('gloo',init_method='env://'); rank=dist.get_rank(); world=dist.get_world_size()
    if world!=4: raise RuntimeError(f'expected exactly 4 ranks, got {world}')
    set_initialization_seed(); trainable=SightlineTrainable(8,layers=(0,),heads=2); memory=LayerKVMemoryBank((0,),8,2,hidden_dim=8); transformer=Transformer(); install_lora(transformer,(0,),rank=8)
    if rank: next(trainable.parameters()).data.add_(rank)  # prove rank0 broadcast is authoritative
    digest=broadcast_and_assert_trainables(trainable,memory,transformer,world)
    if rank==0: print(json.dumps({'world_size':world,'step0_trainable_hash':digest,'passed':True}))
    dist.destroy_process_group()
if __name__=='__main__': main()
