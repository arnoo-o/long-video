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


class _TokenBlockedSightlineMLPProject(torch.autograd.Function):
    """SCoPE MLP + nonlinear gate with bounded saved activation memory."""
    @staticmethod
    def forward(ctx,rays,p1,p2,g1w,g1b,g2w,g2b,norm_weight,alpha,scale_delta,eps,swap,token_tile):
        flat=rays.reshape(-1,7); output=torch.empty((flat.shape[0],p2.shape[0]),device=rays.device,dtype=rays.dtype)
        has_delta=scale_delta.numel()!=0; delta=scale_delta.float() if has_delta else None
        for start in range(0,flat.shape[0],int(token_tile)):
            stop=min(flat.shape[0],start+int(token_tile)); ray=flat[start:stop].float(); scale=ray[:,6:7]
            geometric=torch.cat((ray[:,3:6],ray[:,:3],scale),-1) if swap else ray
            project_hidden=F.gelu(F.linear(geometric,p1.float()))
            projected=F.linear(project_hidden,p2.float())
            rstd=(projected.square().mean(-1,keepdim=True)+float(eps)).rsqrt()
            gate_hidden=F.silu(F.linear(scale if delta is None else scale+delta,g1w.float(),g1b.float()))
            gate=F.linear(gate_hidden,g2w.float(),g2b.float()).sigmoid()
            output[start:stop].copy_(alpha.float()*gate*projected*rstd*norm_weight.float())
        ctx.save_for_backward(rays,p1,p2,g1w,g1b,g2w,g2b,norm_weight,alpha,scale_delta)
        ctx.eps=float(eps); ctx.swap=bool(swap); ctx.tile=int(token_tile); ctx.has_delta=has_delta
        return output.reshape(*rays.shape[:-1],p2.shape[0])

    @staticmethod
    def backward(ctx,grad_output):
        rays,p1,p2,g1w,g1b,g2w,g2b,norm_weight,alpha,scale_delta=ctx.saved_tensors
        flat=rays.reshape(-1,7); grad_flat=grad_output.reshape(-1,grad_output.shape[-1])
        inputs=(rays,p1,p2,g1w,g1b,g2w,g2b,norm_weight,alpha,scale_delta)
        grads=[torch.empty_like(flat) if ctx.needs_input_grad[0] else None]
        grads.extend(torch.zeros_like(value) if ctx.needs_input_grad[index] else None for index,value in enumerate(inputs[1:],1))
        delta=scale_delta.float() if ctx.has_delta else None
        sqrt_2=2.0**.5; inv_sqrt_2pi=(2.0*torch.pi)**-.5
        for start in range(0,flat.shape[0],ctx.tile):
            stop=min(flat.shape[0],start+ctx.tile); ray=flat[start:stop].float(); scale=ray[:,6:7]
            geometric=torch.cat((ray[:,3:6],ray[:,:3],scale),-1) if ctx.swap else ray
            project_pre=F.linear(geometric,p1.float()); project_hidden=F.gelu(project_pre)
            projected=F.linear(project_hidden,p2.float()); rstd=(projected.square().mean(-1,keepdim=True)+ctx.eps).rsqrt()
            normalized=projected*rstd*norm_weight.float()
            gate_input=scale if delta is None else scale+delta
            gate_pre=F.linear(gate_input,g1w.float(),g1b.float()); gate_hidden=F.silu(gate_pre)
            gate=F.linear(gate_hidden,g2w.float(),g2b.float()).sigmoid(); grad=grad_flat[start:stop].float()
            if grads[8] is not None: grads[8].add_((grad*gate*normalized).sum().to(grads[8].dtype))
            grad_normalized=grad*alpha.float()*gate; grad_gate=grad*alpha.float()*normalized
            weighted=grad_normalized*norm_weight.float()
            grad_projected=rstd*(weighted-projected*rstd.square()*(weighted*projected).mean(-1,keepdim=True))
            if grads[7] is not None: grads[7].add_((grad_normalized*projected*rstd).sum(0).to(grads[7].dtype))
            if grads[2] is not None: grads[2].add_((grad_projected.transpose(0,1)@project_hidden).to(grads[2].dtype))
            grad_project_hidden=grad_projected@p2.float()
            gelu_grad=.5*(1+torch.erf(project_pre/sqrt_2))+project_pre*torch.exp(-.5*project_pre.square())*inv_sqrt_2pi
            grad_project_pre=grad_project_hidden*gelu_grad
            if grads[1] is not None: grads[1].add_((grad_project_pre.transpose(0,1)@geometric).to(grads[1].dtype))
            grad_geometric=grad_project_pre@p1.float()
            grad_gate_logits=grad_gate*gate*(1-gate)
            if grads[5] is not None: grads[5].add_((grad_gate_logits.transpose(0,1)@gate_hidden).to(grads[5].dtype))
            if grads[6] is not None: grads[6].add_(grad_gate_logits.sum(0).to(grads[6].dtype))
            grad_gate_hidden=grad_gate_logits@g2w.float(); sigmoid=torch.sigmoid(gate_pre)
            grad_gate_pre=grad_gate_hidden*sigmoid*(1+gate_pre*(1-sigmoid))
            if grads[3] is not None: grads[3].add_((grad_gate_pre.transpose(0,1)@gate_input).to(grads[3].dtype))
            if grads[4] is not None: grads[4].add_(grad_gate_pre.sum(0).to(grads[4].dtype))
            grad_scale_gate=grad_gate_pre@g1w.float()
            if grads[9] is not None: grads[9].add_(grad_scale_gate.sum_to_size(scale_delta.shape).to(grads[9].dtype))
            if grads[0] is not None:
                ray_grad=torch.zeros_like(ray)
                if ctx.swap: ray_grad[:,:3]=grad_geometric[:,3:6]; ray_grad[:,3:6]=grad_geometric[:,:3]
                else: ray_grad[:,:6]=grad_geometric[:,:6]
                ray_grad[:,6:7]=grad_geometric[:,6:7]+grad_scale_gate
                grads[0][start:stop].copy_(ray_grad.to(grads[0].dtype))
        grad_rays=None if grads[0] is None else grads[0].reshape_as(rays)
        return (grad_rays,*grads[1:],None,None,None)


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


class _TokenBlockedRelativeSightlineProject(torch.autograd.Function):
    """Native-relative residual with sample-wide detached native RMS.

    Each tile computes only ``gate * RMSNorm(project(ray))``.  The native
    Q/K RMS is accumulated per batch sample across all token/channel values,
    detached, and then applied as the residual amplitude.  No global RMS of
    the geometry vector is used: gate amplitude therefore directly controls
    the residual as specified by the Geometry contract.
    """
    @staticmethod
    def forward(ctx, rays, native, proj_weight, proj_bias, gate_weight, gate_bias,
                norm_weight, beta, scale_delta, eps, swap, token_tile):
        batch = rays.shape[0]
        tokens = rays.numel() // (batch * 7)
        width = native.shape[-1]
        flat_rays = rays.reshape(batch, tokens, 7)
        flat_native = native.reshape(batch, tokens, width)
        output = torch.empty_like(flat_native)
        has_delta = scale_delta.numel() != 0
        delta = scale_delta.float() if has_delta else None
        rho = beta.float().sigmoid()
        native_sq = torch.zeros((batch, 1), device=rays.device, dtype=torch.float32)
        count = float(tokens * width)
        # Single bounded pass: compute projector -> RMSNorm -> gate once per
        # tile and retain only the unscaled geometry direction in output.
        for start in range(0, tokens, int(token_tile)):
            stop = min(tokens, start + int(token_tile))
            ray = flat_rays[:, start:stop].float()
            xnative = flat_native[:, start:stop].float()
            scale = ray[:, :, 6:7]
            gate_input = scale if delta is None else scale + delta
            gate = F.linear(gate_input, gate_weight.float(), gate_bias.float()).sigmoid()
            geometric = torch.cat((ray[:, :, 3:6], ray[:, :, :3], scale), -1) if swap else ray
            projected = F.linear(geometric, proj_weight.float(), proj_bias.float())
            rstd = (projected.square().mean(-1, keepdim=True) + float(eps)).rsqrt()
            normalized = projected * rstd * norm_weight.float()
            u = gate * normalized
            native_sq.add_(xnative.square().sum((1, 2), keepdim=False).unsqueeze(1))
            output[:, start:stop].copy_(u.to(output.dtype))
        native_rms = (native_sq / count + float(eps)).sqrt().detach()
        # Apply the true Geometry amplitude.  Gate is intentionally not
        # normalized by a second global RMS, so its magnitude is observable.
        amplitude = rho * native_rms
        for start in range(0, tokens, int(token_tile)):
            stop = min(tokens, start + int(token_tile))
            output[:, start:stop].mul_(amplitude[:, None, :].to(output.dtype))
        ctx.save_for_backward(rays, proj_weight, proj_bias, gate_weight, gate_bias,
                              norm_weight, beta, scale_delta, native_rms)
        ctx.eps = float(eps); ctx.swap = bool(swap); ctx.tile = int(token_tile)
        ctx.has_delta = has_delta; ctx.tokens = int(tokens); ctx.width = int(width)
        return output.reshape_as(native)

    @staticmethod
    def backward(ctx, grad_output):
        rays, proj_weight, proj_bias, gate_weight, gate_bias, norm_weight, beta, scale_delta, native_rms = ctx.saved_tensors
        batch = rays.shape[0]; tokens = ctx.tokens; width = ctx.width
        flat_rays = rays.reshape(batch, tokens, 7)
        flat_grad = grad_output.reshape(batch, tokens, width).float()
        grad_rays = torch.empty_like(flat_rays) if ctx.needs_input_grad[0] else None
        grad_proj = torch.zeros_like(proj_weight) if ctx.needs_input_grad[2] else None
        grad_proj_bias = torch.zeros_like(proj_bias) if ctx.needs_input_grad[3] else None
        grad_gate_w = torch.zeros_like(gate_weight) if ctx.needs_input_grad[4] else None
        grad_gate_b = torch.zeros_like(gate_bias) if ctx.needs_input_grad[5] else None
        grad_norm = torch.zeros_like(norm_weight) if ctx.needs_input_grad[6] else None
        grad_beta = torch.zeros_like(beta) if ctx.needs_input_grad[7] else None
        grad_delta = torch.zeros_like(scale_delta) if ctx.has_delta and ctx.needs_input_grad[8] else None
        delta = scale_delta.float() if ctx.has_delta else None
        rho = beta.float().sigmoid()
        rho_deriv = rho * (1 - rho)
        for start in range(0, tokens, ctx.tile):
            stop = min(tokens, start + ctx.tile)
            ray = flat_rays[:, start:stop].float(); scale = ray[:, :, 6:7]
            gate_input = scale if delta is None else scale + delta
            geometric = torch.cat((ray[:, :, 3:6], ray[:, :, :3], scale), -1) if ctx.swap else ray
            projected = F.linear(geometric, proj_weight.float(), proj_bias.float())
            rstd = (projected.square().mean(-1, keepdim=True) + ctx.eps).rsqrt()
            normalized = projected * rstd * norm_weight.float()
            gate = F.linear(gate_input, gate_weight.float(), gate_bias.float()).sigmoid()
            u = gate * normalized; grad = flat_grad[:, start:stop]
            if grad_beta is not None:
                grad_beta.add_((grad * (native_rms[:, None, :] * u)).sum() * rho_deriv)
            grad_u = grad * (rho * native_rms[:, None, :])
            grad_normalized = grad_u * gate
            grad_gate = grad_u * normalized
            weighted = grad_normalized * norm_weight.float()
            grad_projected = rstd * (weighted - projected * rstd.square() *
                                     (weighted * projected).mean(-1, keepdim=True))
            if grad_norm is not None:
                grad_norm.add_((grad_normalized * projected * rstd).sum((0, 1)).to(grad_norm.dtype))
            if grad_proj is not None:
                grad_proj.add_(torch.einsum('bld,blf->df', grad_projected, geometric).to(grad_proj.dtype))
            if grad_proj_bias is not None:
                grad_proj_bias.add_(grad_projected.sum((0, 1)).to(grad_proj_bias.dtype))
            grad_geometric = grad_projected @ proj_weight.float()
            grad_gate_logits = grad_gate * gate * (1 - gate)
            if grad_gate_w is not None:
                grad_gate_w.add_(torch.einsum('bld,blf->df', grad_gate_logits, gate_input).to(grad_gate_w.dtype))
            if grad_gate_b is not None:
                grad_gate_b.add_(grad_gate_logits.sum((0, 1)).to(grad_gate_b.dtype))
            grad_scale_gate = grad_gate_logits @ gate_weight.float()
            if grad_delta is not None:
                grad_delta.add_(grad_scale_gate.sum_to_size(scale_delta.shape).to(grad_delta.dtype))
            if grad_rays is not None:
                ray_grad = torch.zeros_like(ray)
                if ctx.swap:
                    ray_grad[:, :, :3] = grad_geometric[:, :, 3:6]
                    ray_grad[:, :, 3:6] = grad_geometric[:, :, :3]
                else:
                    ray_grad[:, :, :6] = grad_geometric[:, :, :6]
                ray_grad[:, :, 6:7] = grad_geometric[:, :, 6:7] + grad_scale_gate
                grad_rays[:, start:stop].copy_(ray_grad.to(grad_rays.dtype))
        return (None if grad_rays is None else grad_rays.reshape_as(rays), None,
                grad_proj, grad_proj_bias, grad_gate_w, grad_gate_b, grad_norm,
                grad_beta, grad_delta, None, None, None)


def token_blocked_sightline_relative_project(rays, native, projection, gate, norm, beta, *, kind,
                                             scale_delta=None, token_tile=DEFAULT_TOKEN_TILE,
                                             eps=1e-6):
    if kind not in ('q', 'k'):
        raise ValueError('kind must be q or k')
    if rays.shape[:-1] != native.shape[:-1]:
        raise ValueError(f'rays/native shape mismatch: {tuple(rays.shape)} vs {tuple(native.shape)}')
    delta = rays.new_empty(0) if scale_delta is None else torch.as_tensor(
        scale_delta, device=rays.device, dtype=rays.dtype)
    return _TokenBlockedRelativeSightlineProject.apply(
        rays, native, projection.weight, projection.bias, gate.weight, gate.bias,
        norm.weight,
        beta, delta, float(eps), kind == 'k', int(token_tile))


class _TokenBlockedSightlineFixedRMSProject(torch.autograd.Function):
    """Linear ray projector + fixed RMSNorm, recomputed one token tile at a time."""
    @staticmethod
    def forward(ctx, rays, proj_weight, gate_weight, gate_bias, beta, scale_delta, eps, swap, token_tile):
        flat=rays.reshape(-1,7); output=torch.empty((flat.shape[0],proj_weight.shape[0]),device=rays.device,dtype=rays.dtype)
        has_delta=scale_delta.numel()!=0; delta=scale_delta.float() if has_delta else None
        alpha=beta.float().sigmoid()
        for start in range(0,flat.shape[0],int(token_tile)):
            stop=min(flat.shape[0],start+int(token_tile)); ray=flat[start:stop].float(); scale=ray[:,6:7]
            geometric=torch.cat((ray[:,3:6],ray[:,:3],scale),-1) if swap else ray
            raw=F.linear(geometric,proj_weight.float()); gate=F.linear(scale if delta is None else scale+delta,gate_weight.float(),gate_bias.float()).sigmoid()
            norm=raw*(raw.square().mean(-1,keepdim=True)+float(eps)).rsqrt()
            output[start:stop].copy_(alpha*gate*norm)
        ctx.save_for_backward(rays,proj_weight,gate_weight,gate_bias,beta,scale_delta)
        ctx.eps=float(eps); ctx.swap=bool(swap); ctx.tile=int(token_tile); ctx.has_delta=has_delta
        return output.reshape(*rays.shape[:-1],proj_weight.shape[0])
    @staticmethod
    def backward(ctx,grad_output):
        rays,proj_weight,gate_weight,gate_bias,beta,scale_delta=ctx.saved_tensors
        flat=rays.reshape(-1,7); grad_flat=grad_output.reshape(-1,grad_output.shape[-1])
        grad_rays=torch.empty_like(flat) if ctx.needs_input_grad[0] else None
        grad_proj=torch.zeros_like(proj_weight) if ctx.needs_input_grad[1] else None
        grad_gate_weight=torch.zeros_like(gate_weight) if ctx.needs_input_grad[2] else None
        grad_gate_bias=torch.zeros_like(gate_bias) if ctx.needs_input_grad[3] else None
        grad_beta=torch.zeros_like(beta) if ctx.needs_input_grad[4] else None
        grad_delta=torch.zeros_like(scale_delta) if ctx.has_delta and ctx.needs_input_grad[5] else None
        delta=scale_delta.float() if ctx.has_delta else None; alpha=beta.float().sigmoid()
        for start in range(0,flat.shape[0],ctx.tile):
            stop=min(flat.shape[0],start+ctx.tile); ray=flat[start:stop].float(); scale=ray[:,6:7]; gate_input=scale if delta is None else scale+delta
            geometric=torch.cat((ray[:,3:6],ray[:,:3],scale),-1) if ctx.swap else ray
            raw=F.linear(geometric,proj_weight.float()); rstd=(raw.square().mean(-1,keepdim=True)+ctx.eps).rsqrt(); norm=raw*rstd
            gate=F.linear(gate_input,gate_weight.float(),gate_bias.float()).sigmoid(); grad=grad_flat[start:stop].float()
            if grad_beta is not None: grad_beta.add_(((grad*gate*norm).sum()*alpha*(1-alpha)).to(grad_beta.dtype))
            grad_norm=grad*alpha*gate; grad_gate=grad*alpha*norm
            grad_raw=rstd*(grad_norm-raw*rstd.square()*(grad_norm*raw).mean(-1,keepdim=True))
            if grad_proj is not None: grad_proj.add_((grad_raw.transpose(0,1)@geometric).to(grad_proj.dtype))
            grad_geometric=grad_raw@proj_weight.float(); grad_logits=grad_gate*gate*(1-gate)
            if grad_gate_weight is not None: grad_gate_weight.add_((grad_logits.transpose(0,1)@gate_input).to(grad_gate_weight.dtype))
            if grad_gate_bias is not None: grad_gate_bias.add_(grad_logits.sum(0).to(grad_gate_bias.dtype))
            grad_scale=grad_logits@gate_weight.float()
            if grad_delta is not None: grad_delta.add_(grad_scale.sum_to_size(scale_delta.shape).to(grad_delta.dtype))
            if grad_rays is not None:
                ray_grad=torch.zeros_like(ray)
                if ctx.swap: ray_grad[:,:3]=grad_geometric[:,3:6]; ray_grad[:,3:6]=grad_geometric[:,:3]
                else: ray_grad[:,:6]=grad_geometric[:,:6]
                ray_grad[:,6:7]=grad_geometric[:,6:7]+grad_scale; grad_rays[start:stop].copy_(ray_grad.to(grad_rays.dtype))
        return (None if grad_rays is None else grad_rays.reshape_as(rays),grad_proj,grad_gate_weight,grad_gate_bias,grad_beta,grad_delta,None,None,None)


def token_blocked_sightline_fixed_rms_project(rays, projection, gate, beta, *, kind, eps, scale_delta=None, token_tile=DEFAULT_TOKEN_TILE):
    if kind not in ('q','k'): raise ValueError('kind must be q or k')
    delta=rays.new_empty(0) if scale_delta is None else torch.as_tensor(scale_delta,device=rays.device,dtype=rays.dtype)
    return _TokenBlockedSightlineFixedRMSProject.apply(rays,projection.weight,gate[0].weight,gate[0].bias,beta,delta,float(eps),kind=='k',int(token_tile))


def token_blocked_sightline_mlp_project(rays, projection, gate, norm, alpha, *, kind,
                                        scale_delta=None, token_tile=DEFAULT_TOKEN_TILE):
    if kind not in ('q','k'): raise ValueError('kind must be q or k')
    if len(projection)!=3 or len(gate)!=4: raise ValueError('unexpected Sightline-v2 projector/gate layout')
    delta=rays.new_empty(0) if scale_delta is None else torch.as_tensor(scale_delta,device=rays.device,dtype=rays.dtype)
    return _TokenBlockedSightlineMLPProject.apply(
        rays,projection[0].weight,projection[2].weight,
        gate[0].weight,gate[0].bias,gate[2].weight,gate[2].bias,
        norm.weight,alpha,delta,float(norm.eps),kind=='k',int(token_tile))
