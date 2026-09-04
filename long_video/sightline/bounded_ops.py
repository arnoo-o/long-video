"""Bounded-memory Helios normalization and gated residual primitives.

The public functions preserve Helios' FP32 normalization/modulation arithmetic
while limiting temporary FP32 storage to one token tile.  Their custom
backward recomputes tile statistics analytically and never builds a full
[B, N, D] FP32 activation.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


DEFAULT_TOKEN_TILE = 512


def _slice_token(value: torch.Tensor | None, start: int, stop: int, tokens: int):
    if value is None:
        return None
    return value[:, start:stop] if value.ndim == 3 and value.shape[1] == tokens else value


def _accumulate_broadcast(gradient, tile_gradient, start, stop, tokens):
    if gradient is None:
        return
    if gradient.ndim == 3 and gradient.shape[1] == tokens:
        gradient[:, start:stop].copy_(tile_gradient.to(gradient.dtype))
    else:
        gradient.add_(tile_gradient.sum_to_size(gradient.shape).to(gradient.dtype))


class _TokenBlockedLayerNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden, weight, bias, scale, shift, eps, token_tile):
        tokens, width = hidden.shape[1:]
        output = torch.empty_like(hidden)
        has_weight = weight is not None
        has_bias = bias is not None
        has_scale = scale is not None
        has_shift = shift is not None
        empty = hidden.new_empty(0)
        for start in range(0, tokens, int(token_tile)):
            stop = min(tokens, start + int(token_tile))
            normalized = F.layer_norm(
                hidden[:, start:stop].float(), (width,),
                weight.float() if has_weight else None,
                bias.float() if has_bias else None, float(eps),
            )
            scale_tile = _slice_token(scale, start, stop, tokens)
            shift_tile = _slice_token(shift, start, stop, tokens)
            if scale_tile is not None:
                normalized.mul_(1 + scale_tile)
            if shift_tile is not None:
                normalized.add_(shift_tile)
            output[:, start:stop].copy_(normalized)
        ctx.save_for_backward(
            hidden,
            weight if has_weight else empty,
            bias if has_bias else empty,
            scale if has_scale else empty,
            shift if has_shift else empty,
        )
        ctx.flags = has_weight, has_bias, has_scale, has_shift
        ctx.eps = float(eps)
        ctx.token_tile = int(token_tile)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        hidden, saved_weight, saved_bias, saved_scale, saved_shift = ctx.saved_tensors
        has_weight, has_bias, has_scale, has_shift = ctx.flags
        weight = saved_weight if has_weight else None
        bias = saved_bias if has_bias else None
        scale = saved_scale if has_scale else None
        shift = saved_shift if has_shift else None
        tokens, width = hidden.shape[1:]
        grad_hidden = torch.empty_like(hidden) if ctx.needs_input_grad[0] else None
        grad_weight = torch.zeros_like(weight) if has_weight and ctx.needs_input_grad[1] else None
        grad_bias = torch.zeros_like(bias) if has_bias and ctx.needs_input_grad[2] else None
        grad_scale = torch.zeros_like(scale) if has_scale and ctx.needs_input_grad[3] else None
        grad_shift = torch.zeros_like(shift) if has_shift and ctx.needs_input_grad[4] else None

        for start in range(0, tokens, ctx.token_tile):
            stop = min(tokens, start + ctx.token_tile)
            x = hidden[:, start:stop].float()
            mean = x.mean(dim=-1, keepdim=True)
            centered = x - mean
            rstd = (centered.square().mean(dim=-1, keepdim=True) + ctx.eps).rsqrt()
            normalized = centered * rstd
            affine = normalized
            if weight is not None:
                affine = affine * weight.float()
            if bias is not None:
                affine = affine + bias.float()
            grad = grad_output[:, start:stop].float()
            scale_tile = _slice_token(scale, start, stop, tokens)
            grad_affine = grad * (1 + scale_tile) if scale_tile is not None else grad
            grad_normalized = grad_affine * weight.float() if weight is not None else grad_affine
            dx = (
                grad_normalized
                - grad_normalized.mean(dim=-1, keepdim=True)
                - normalized * (grad_normalized * normalized).mean(dim=-1, keepdim=True)
            ) * rstd
            if grad_hidden is not None:
                grad_hidden[:, start:stop].copy_(dx)
            if grad_weight is not None:
                grad_weight.add_((grad_affine * normalized).sum_to_size(weight.shape).to(weight.dtype))
            if grad_bias is not None:
                grad_bias.add_(grad_affine.sum_to_size(bias.shape).to(bias.dtype))
            _accumulate_broadcast(grad_scale, grad * affine, start, stop, tokens)
            _accumulate_broadcast(grad_shift, grad, start, stop, tokens)
        return grad_hidden, grad_weight, grad_bias, grad_scale, grad_shift, None, None


class _TokenBlockedGatedResidual(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden, update, gate, token_tile):
        tokens = hidden.shape[1]
        output = torch.empty_like(hidden)
        for start in range(0, tokens, int(token_tile)):
            stop = min(tokens, start + int(token_tile))
            gate_tile = _slice_token(gate, start, stop, tokens)
            value = hidden[:, start:stop].float() + update[:, start:stop].float() * gate_tile
            output[:, start:stop].copy_(value)
        ctx.save_for_backward(update, gate)
        ctx.token_tile = int(token_tile)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        update, gate = ctx.saved_tensors
        tokens = grad_output.shape[1]
        grad_hidden = grad_output if ctx.needs_input_grad[0] else None
        grad_update = torch.empty_like(update) if ctx.needs_input_grad[1] else None
        grad_gate = torch.zeros_like(gate) if ctx.needs_input_grad[2] else None
        for start in range(0, tokens, ctx.token_tile):
            stop = min(tokens, start + ctx.token_tile)
            grad = grad_output[:, start:stop].float()
            gate_tile = _slice_token(gate, start, stop, tokens)
            if grad_update is not None:
                grad_update[:, start:stop].copy_(grad * gate_tile)
            _accumulate_broadcast(
                grad_gate, grad * update[:, start:stop].float(), start, stop, tokens
            )
        return grad_hidden, grad_update, grad_gate, None


class _TokenBlockedSightlineProject(torch.autograd.Function):
    @staticmethod
    def forward(ctx, rays, proj_weight, proj_bias, gate_weight, gate_bias,
                norm_weight, alpha, scale_delta, eps, swap_direction_moment, token_tile):
        flat_rays=rays.reshape(-1,7)
        output=torch.empty((flat_rays.shape[0],proj_weight.shape[0]),device=rays.device,dtype=rays.dtype)
        has_scale_delta=scale_delta.numel()!=0
        delta=scale_delta.float() if has_scale_delta else None
        for start in range(0,flat_rays.shape[0],int(token_tile)):
            stop=min(flat_rays.shape[0],start+int(token_tile)); ray=flat_rays[start:stop].float(); scale=ray[:,6:7]
            gate=F.linear(scale if delta is None else scale+delta,gate_weight.float(),gate_bias.float()).sigmoid()
            geometric=torch.cat((ray[:,3:6],ray[:,:3],scale),-1) if swap_direction_moment else ray
            projected=F.linear(geometric,proj_weight.float(),proj_bias.float())
            rstd=(projected.square().mean(-1,keepdim=True)+float(eps)).rsqrt()
            normalized=projected*rstd*norm_weight.float()
            output[start:stop].copy_(alpha.float()*gate*normalized)
        ctx.save_for_backward(rays,proj_weight,proj_bias,gate_weight,gate_bias,norm_weight,alpha,scale_delta)
        ctx.eps=float(eps); ctx.swap=bool(swap_direction_moment); ctx.token_tile=int(token_tile); ctx.has_scale_delta=has_scale_delta
        return output.reshape(*rays.shape[:-1],proj_weight.shape[0])

    @staticmethod
    def backward(ctx, grad_output):
        rays,proj_weight,proj_bias,gate_weight,gate_bias,norm_weight,alpha,scale_delta=ctx.saved_tensors
        flat_rays=rays.reshape(-1,7); flat_grad=grad_output.reshape(-1,grad_output.shape[-1])
        grad_rays=torch.empty_like(flat_rays) if ctx.needs_input_grad[0] else None
        grad_proj_weight=torch.zeros_like(proj_weight) if ctx.needs_input_grad[1] else None
        grad_proj_bias=torch.zeros_like(proj_bias) if ctx.needs_input_grad[2] else None
        grad_gate_weight=torch.zeros_like(gate_weight) if ctx.needs_input_grad[3] else None
        grad_gate_bias=torch.zeros_like(gate_bias) if ctx.needs_input_grad[4] else None
        grad_norm_weight=torch.zeros_like(norm_weight) if ctx.needs_input_grad[5] else None
        grad_alpha=torch.zeros_like(alpha) if ctx.needs_input_grad[6] else None
        grad_scale_delta=torch.zeros_like(scale_delta) if ctx.has_scale_delta and ctx.needs_input_grad[7] else None
        delta=scale_delta.float() if ctx.has_scale_delta else None
        for start in range(0,flat_rays.shape[0],ctx.token_tile):
            stop=min(flat_rays.shape[0],start+ctx.token_tile); ray=flat_rays[start:stop].float(); scale=ray[:,6:7]
            gate_input=scale if delta is None else scale+delta
            gate=F.linear(gate_input,gate_weight.float(),gate_bias.float()).sigmoid()
            geometric=torch.cat((ray[:,3:6],ray[:,:3],scale),-1) if ctx.swap else ray
            projected=F.linear(geometric,proj_weight.float(),proj_bias.float())
            rstd=(projected.square().mean(-1,keepdim=True)+ctx.eps).rsqrt()
            normalized=projected*rstd*norm_weight.float(); grad=flat_grad[start:stop].float()
            if grad_alpha is not None: grad_alpha.add_((grad*gate*normalized).sum().to(grad_alpha.dtype))
            grad_normalized=grad*alpha.float()*gate
            grad_gate=grad*alpha.float()*normalized
            weighted_grad=grad_normalized*norm_weight.float()
            grad_projected=rstd*(weighted_grad-projected*rstd.square()*(weighted_grad*projected).mean(-1,keepdim=True))
            if grad_norm_weight is not None: grad_norm_weight.add_((grad_normalized*projected*rstd).sum(0).to(grad_norm_weight.dtype))
            if grad_proj_weight is not None: grad_proj_weight.add_((grad_projected.transpose(0,1)@geometric).to(grad_proj_weight.dtype))
            if grad_proj_bias is not None: grad_proj_bias.add_(grad_projected.sum(0).to(grad_proj_bias.dtype))
            grad_geometric=grad_projected@proj_weight.float()
            grad_gate_logits=grad_gate*gate*(1-gate)
            if grad_gate_weight is not None: grad_gate_weight.add_((grad_gate_logits.transpose(0,1)@gate_input).to(grad_gate_weight.dtype))
            if grad_gate_bias is not None: grad_gate_bias.add_(grad_gate_logits.sum(0).to(grad_gate_bias.dtype))
            grad_scale_gate=grad_gate_logits@gate_weight.float()
            if grad_scale_delta is not None: grad_scale_delta.add_(grad_scale_gate.sum_to_size(scale_delta.shape).to(grad_scale_delta.dtype))
            if grad_rays is not None:
                ray_grad=torch.empty_like(ray)
                if ctx.swap:
                    ray_grad[:,:3]=grad_geometric[:,3:6]; ray_grad[:,3:6]=grad_geometric[:,:3]
                else:
                    ray_grad[:,:6]=grad_geometric[:,:6]
                ray_grad[:,6:7]=grad_geometric[:,6:7]+grad_scale_gate
                grad_rays[start:stop].copy_(ray_grad.to(grad_rays.dtype))
        return (None if grad_rays is None else grad_rays.reshape_as(rays),grad_proj_weight,grad_proj_bias,
                grad_gate_weight,grad_gate_bias,grad_norm_weight,grad_alpha,grad_scale_delta,None,None,None)


def token_blocked_layer_norm(hidden, norm, token_tile=DEFAULT_TOKEN_TILE):
    if isinstance(norm, torch.nn.Identity):
        return hidden
    return _TokenBlockedLayerNorm.apply(
        hidden, getattr(norm, "weight", None), getattr(norm, "bias", None),
        None, None, float(norm.eps), int(token_tile),
    )


def token_blocked_layer_norm_modulate(hidden, norm, scale, shift, token_tile=DEFAULT_TOKEN_TILE):
    return _TokenBlockedLayerNorm.apply(
        hidden, getattr(norm, "weight", None), getattr(norm, "bias", None),
        scale, shift, float(norm.eps), int(token_tile),
    )


def token_blocked_gated_residual(hidden, update, gate, token_tile=DEFAULT_TOKEN_TILE):
    return _TokenBlockedGatedResidual.apply(hidden, update, gate, int(token_tile))


def token_blocked_sightline_project(rays, projection, gate, norm, alpha, *, kind,
                                    scale_delta=None, token_tile=DEFAULT_TOKEN_TILE):
    if kind not in ('q','k'):
        raise ValueError('kind must be q or k')
    delta=rays.new_empty(0) if scale_delta is None else torch.as_tensor(scale_delta,device=rays.device,dtype=rays.dtype)
    return _TokenBlockedSightlineProject.apply(
        rays,projection.weight,projection.bias,gate.weight,gate.bias,norm.weight,alpha,
        delta,float(norm.eps),kind=='k',int(token_tile),
    )
