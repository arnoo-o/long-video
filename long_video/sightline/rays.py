"""Deterministic Plücker sightlines on each actual transformer token grid."""
from __future__ import annotations
import torch

TEMPORAL_GROUPS = ((0,), (1,2,3,4), (5,6,7,8), (9,10,11,12), (13,14,15,16),
                   (17,18,19,20), (21,22,23,24), (25,26,27,28), (29,30,31,32))

def latent_camera_indices(device=None) -> torch.Tensor:
    """Canonical 33 RGB -> 9 latent camera representatives used everywhere.

    The final RGB frame of each native temporal group is used. This makes the
    last representative of chunk ``k`` and the first representative of chunk
    ``k+1`` the shared boundary frame 32, 64, ... exactly.
    """
    return torch.tensor([group[-1] for group in TEMPORAL_GROUPS], device=device, dtype=torch.long)

def chunk_frame_slice(chunk_index: int) -> slice:
    if chunk_index < 0 or chunk_index > 5: raise ValueError("chunk index must be 0..5")
    return slice(32 * chunk_index, 32 * chunk_index + 33)

def chunk_cameras(c2w: torch.Tensor, intrinsics: torch.Tensor, chunk_index: int):
    sl=chunk_frame_slice(chunk_index)
    if c2w.ndim != 4 or intrinsics.ndim != 4: raise ValueError("c2w/K must retain [B,F,...] dimensions")
    if c2w.shape[1] < sl.stop or intrinsics.shape[1] < sl.stop: raise ValueError("trajectory has fewer than requested chunk frames")
    return c2w[:,sl], intrinsics[:,sl]

def canonicalize_c2w(c2w: torch.Tensor) -> torch.Tensor:
    """Express trajectory poses in the first-frame coordinate system."""
    if c2w.ndim == 3:
        return torch.linalg.inv(c2w[:, :1]) @ c2w[:, :1]
    if c2w.ndim != 4:
        raise ValueError("c2w must be [B,4,4] or [B,T,4,4]")
    return torch.linalg.inv(c2w[:, :1]) @ c2w

def temporal_group_cameras(c2w: torch.Tensor, intrinsics: torch.Tensor) -> tuple[torch.Tensor,torch.Tensor]:
    if c2w.ndim == 3: c2w=c2w[:,None].expand(-1,33,-1,-1)
    if c2w.ndim != 4 or c2w.shape[1] != 33: raise ValueError("c2w must be [B,4,4] or [B,33,4,4]")
    if intrinsics.ndim == 3: intrinsics=intrinsics[:,None].expand(-1,33,-1,-1)
    if intrinsics.ndim != 4 or intrinsics.shape[:2] != c2w.shape[:2]: raise ValueError("K must be [B,3,3] or [B,33,3,3]")
    indices=latent_camera_indices(c2w.device)
    return c2w.index_select(1,indices),intrinsics.index_select(1,indices)

def plucker_rays(c2w: torch.Tensor, intrinsics: torch.Tensor, token_height: int, token_width: int, *, source_height: int, source_width: int, eps: float=1e-6) -> torch.Tensor:
    if c2w.ndim==3: c2w=c2w[:,None]
    if intrinsics.ndim==3: intrinsics=intrinsics[:,None].expand(-1,c2w.shape[1],-1,-1)
    if c2w.ndim!=4 or intrinsics.ndim!=4 or c2w.shape[:2]!=intrinsics.shape[:2]: raise ValueError("camera/K batch and temporal dimensions must match")
    B,T=c2w.shape[:2]; device,dtype=c2w.device,c2w.dtype
    u=(torch.arange(token_width,device=device,dtype=dtype)+.5)*(source_width/token_width)-.5; v=(torch.arange(token_height,device=device,dtype=dtype)+.5)*(source_height/token_height)-.5
    vv,uu=torch.meshgrid(v,u,indexing='ij'); pix=torch.stack((uu,vv,torch.ones_like(uu)),-1).reshape(1,1,-1,3).expand(B,T,-1,-1)
    K=intrinsics.reshape(B*T,3,3); p=pix.reshape(B*T,-1,3); dcam=torch.linalg.solve(K,p.transpose(1,2)).transpose(1,2); dcam=dcam/dcam.norm(dim=-1,keepdim=True).clamp_min(eps)
    R=c2w[:,:,:3,:3].reshape(B*T,3,3); origin=c2w[:,:,:3,3].reshape(B*T,1,3); direction=torch.bmm(dcam,R.transpose(1,2)); direction=direction/direction.norm(dim=-1,keepdim=True).clamp_min(eps)
    moment=torch.cross(origin.expand_as(direction),direction,dim=-1); norm=moment.norm(dim=-1,keepdim=True).clamp_min(eps)
    return torch.cat((direction,moment/norm,norm.log()),-1).reshape(B,T,token_height,token_width,7)

def token_rays_for_shape(c2w: torch.Tensor, intrinsics: torch.Tensor, shape: tuple[int,...], *, source_height:int, source_width:int) -> torch.Tensor:
    if len(shape)!=5: raise ValueError("shape must be [B,T,H,W,C]")
    B,T,H,W,_=shape
    if c2w.ndim==4 and c2w.shape[1]==33 and T==9: c2w,intrinsics=temporal_group_cameras(c2w,intrinsics)
    rays=plucker_rays(c2w,intrinsics,H,W,source_height=source_height,source_width=source_width)
    if rays.shape[:4]!=(B,T,H,W): raise RuntimeError(f"rays {rays.shape[:4]} do not match tokens {(B,T,H,W)}")
    return rays
