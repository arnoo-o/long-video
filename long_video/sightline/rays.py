"""Deterministic Plücker sightlines for arbitrary Helios token grids."""
import torch

def _as_batched(x, dims):
    if x.ndim == dims - 1:
        return x.unsqueeze(0)
    if x.ndim != dims:
        raise ValueError(f"expected {dims-1} or {dims} dimensions, got {x.shape}")
    return x

def plucker_rays(c2w: torch.Tensor, intrinsics: torch.Tensor, height: int, width: int,
                 *, temporal: int = 1, patch_size: tuple[int,int] = (1,1), eps: float = 1e-6):
    """Return ``[B,T,Ht,Wt,7]`` = direction, normalized moment, log moment norm.

    Ray centers are computed on the requested token grid, never interpolated from
    another stage.  ``c2w`` is camera-to-world and K is in source pixel units.
    """
    flattened = c2w.ndim == 4
    if flattened:
        B,T = c2w.shape[:2]; c2w = c2w.reshape(B*T,4,4)
        if intrinsics.ndim == 4: intrinsics = intrinsics.reshape(B*T,3,3)
    else:
        c2w = _as_batched(c2w, 3); B,T = c2w.shape[0], temporal
    K = _as_batched(intrinsics, 3)
    flat_B = c2w.shape[0]; device, dtype = c2w.device, c2w.dtype
    ph, pw = patch_size
    ys = (torch.arange(height, device=device, dtype=dtype) + .5) * ph - .5
    xs = (torch.arange(width, device=device, dtype=dtype) + .5) * pw - .5
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    pix = torch.stack((xx, yy, torch.ones_like(xx)), -1).reshape(1,-1,3).expand(flat_B,-1,-1)
    dcam = torch.linalg.solve(K, pix.transpose(1,2)).transpose(1,2)
    dcam = dcam / dcam.norm(dim=-1, keepdim=True).clamp_min(eps)
    R = c2w[:, :3, :3]; o = c2w[:, :3, 3]
    d = torch.bmm(dcam, R.transpose(1,2)); d = d / d.norm(dim=-1,keepdim=True).clamp_min(eps)
    m = torch.cross(o[:,None,:].expand_as(d), d, dim=-1)
    mn = m.norm(dim=-1, keepdim=True).clamp_min(eps)
    out = torch.cat((d, m / mn, mn.log()), -1).reshape(flat_B,1,height,width,7)
    if flattened: return out.reshape(B,T,height,width,7).contiguous()
    return out.expand(B, temporal, height, width, 7).contiguous()

def token_rays_for_shape(c2w, intrinsics, shape, patch_size=(1,1)):
    if len(shape) != 5: raise ValueError(f"expected [B,T,H,W,C], got {shape}")
    B,T,H,W,_ = shape
    rays = plucker_rays(c2w, intrinsics, H, W, temporal=T, patch_size=patch_size)
    if rays.shape[:4] != (B,T,H,W): raise RuntimeError("ray/token shape mismatch")
    return rays
