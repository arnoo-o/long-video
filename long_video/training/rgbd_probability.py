"""Streaming RGB-D probability score.

The auxiliary target is a soft distribution over spatial keys.  This module
keeps the native Q/K detached and only returns gradients for the current
layer's Geometry deltas.  Both the forward and backward walk bounded key
blocks; no Q-by-K tensor or per-pair autograd graph is retained.
"""
from __future__ import annotations

import torch
from torch.autograd import Function


QUERY_BLOCK = 64
KEY_BLOCK = 256


def attention_mask_to_legal_mask(mask: torch.Tensor | None) -> torch.Tensor | None:
    """Convert Helios dispatch mask semantics to ``True == legal``.

    Helios follows PyTorch SDPA: boolean entries mark positions which may be
    attended, while additive masks use finite values for legal positions and
    ``-inf`` for blocked positions.  The helper deliberately preserves all
    query/head axes for the caller's exact sparse slice.
    """
    if mask is None:
        return None
    if mask.dtype == torch.bool:
        return mask
    return torch.isfinite(mask) & (mask > -torch.finfo(mask.dtype).max / 2)


def _as_legal(legal_mask, batch: int, row: int, key_count: int, device):
    if legal_mask is None or legal_mask.numel() == 0:
        return torch.ones((batch, key_count), device=device, dtype=torch.bool)
    mask = legal_mask.to(device=device, dtype=torch.bool)
    if mask.ndim == 2:
        return mask[row].view(1, key_count).expand(batch, -1)
    if mask.ndim == 3:
        return mask[:, row]
    raise ValueError('legal_mask must be [R,K] or [B,R,K]')


def _target_distribution(indices, values, mask, legal, key_count):
    """Return a normalized target, preserving duplicate support by sum."""
    target = torch.zeros((legal.shape[0], key_count), device=legal.device, dtype=torch.float32)
    if indices.numel():
        safe = indices.clamp_min(0).to(torch.long)
        valid = mask & indices.ge(0) & values.gt(0)
        for batch in range(target.shape[0]):
            if valid.any():
                target[batch].index_add_(0, safe[valid], values[valid].float())
    target.mul_(legal)
    return target / target.sum(-1, keepdim=True).clamp_min(1e-12)


def _log_normalizer(q, k, legal, scale, key_block):
    # q is [B,H,D], k is [B,K,H,D].  Only one key block is materialized.
    running_max = torch.full((q.shape[0], q.shape[1]), -torch.inf, device=q.device, dtype=torch.float32)
    running_sum = torch.zeros_like(running_max)
    for start in range(0, k.shape[1], int(key_block)):
        stop = min(k.shape[1], start + int(key_block))
        logits = torch.einsum('bhd,bkhd->bhk', q.float(), k[:, start:stop].float()).mul_(scale)
        valid = legal[:, start:stop].unsqueeze(1)
        logits = logits.masked_fill(~valid, -torch.inf)
        block_max = logits.amax(-1)
        new_max = torch.maximum(running_max, block_max)
        old_scale=torch.where(torch.isfinite(new_max),torch.exp(running_max-new_max),torch.zeros_like(new_max))
        block_exp=torch.where(valid,torch.exp(logits-new_max.unsqueeze(-1)),torch.zeros_like(logits)).sum(-1)
        running_sum.mul_(old_scale).add_(block_exp)
        running_max = new_max
    return running_max + running_sum.clamp_min(1e-30).log()


def _score_row(q, k, target, legal, scale, key_block):
    logz = _log_normalizer(q, k, legal, scale, key_block)
    support = torch.nonzero(target.gt(0).any(0), as_tuple=False).flatten()
    if support.numel() == 0:
        return q.new_zeros((), dtype=torch.float32)
    score = torch.zeros((q.shape[0],), device=q.device, dtype=torch.float32)
    for start in range(0, support.numel(), int(key_block)):
        stop = min(support.numel(), start + int(key_block))
        idx = support[start:stop]
        logits = torch.einsum('bhd,bkhd->bhk', q.float(), k.index_select(1, idx).float()).mul_(scale)
        logits = logits.masked_fill(~legal.index_select(1, idx).unsqueeze(1), -torch.inf)
        log_p_bar = torch.logsumexp(logits - logz.unsqueeze(-1), dim=1) - torch.log(torch.as_tensor(q.shape[1], device=q.device, dtype=torch.float32))
        block_target = target[:, idx]
        score += (block_target * log_p_bar.masked_fill(~block_target.gt(0), 0.0)).sum(-1)
    return score.mean()


class StreamingRGBDProbabilityScore(Function):
    """Custom-autograd streaming score.

    Inputs are ``native_q, native_k, dq, dk, target_indices, target_weights,
    target_mask, legal_mask, local_scale, row_weights``.  Q/K use the sparse
    axis supplied by the plan.  The returned score is the row-weighted mean
    and gradients are returned only for ``dq`` and ``dk``.
    """

    @staticmethod
    def forward(ctx, native_q, native_k, dq, dk, target_indices,
                target_weights, target_mask, legal_mask, local_scale,
                row_weights=None, return_rows=False):
        if native_q.ndim != 4 or native_k.ndim != 4 or dq.shape != native_q.shape or dk.shape != native_k.shape:
            raise ValueError('native/delta Q/K must be [B,R/H,K,H,D] with matching shapes')
        if target_indices.ndim != 2 or target_weights.shape != target_indices.shape or target_mask.shape != target_indices.shape:
            raise ValueError('target support tensors must be [R,P]')
        batch, rows, heads, dim = native_q.shape
        if target_indices.shape[0] != rows:
            raise ValueError('target rows must match query rows')
        scale_value = torch.as_tensor(local_scale, device=dq.device, dtype=torch.float32).reshape(())
        row_weight = torch.ones((rows,), device=dq.device, dtype=torch.float32) if row_weights is None or row_weights.numel() == 0 else row_weights.to(dq.device, torch.float32).reshape(rows)
        if not torch.isfinite(row_weight).all() or bool((row_weight < 0).any()):
            raise ValueError('row_weights must be finite and non-negative')
        denominator = row_weight.sum().clamp_min(1e-12)
        q = native_q.detach().float() + scale_value * dq.float()
        k = native_k.detach().float() + scale_value * dk.float()
        row_scores = torch.zeros((rows,), device=dq.device, dtype=torch.float32)
        valid_rows = 0
        for q0 in range(0, rows, QUERY_BLOCK):
            q1 = min(rows, q0 + QUERY_BLOCK)
            for row in range(q0, q1):
                legal = _as_legal(legal_mask, batch, row, k.shape[1], dq.device)
                target = _target_distribution(target_indices[row], target_weights[row], target_mask[row], legal, k.shape[1])
                if bool(target.sum() > 0):
                    row_scores[row] = _score_row(q[:, row], k, target, legal, dim ** -0.5, KEY_BLOCK)
                    valid_rows += 1
        score = (row_scores * row_weight).sum() / denominator
        ctx.save_for_backward(native_q.detach(), native_k.detach(), dq, dk,
                              target_indices, target_weights.float(), target_mask,
                              legal_mask if legal_mask is not None else dq.new_empty(0, dtype=torch.bool),
                              scale_value, row_weight, denominator)
        ctx.key_block = KEY_BLOCK
        ctx.query_block = QUERY_BLOCK
        ctx.valid_rows = valid_rows
        ctx.return_rows = bool(return_rows)
        return row_scores if ctx.return_rows else score

    @staticmethod
    def backward(ctx, grad_output):
        (native_q, native_k, dq, dk, target_indices, target_weights,
         target_mask, legal_mask, scale_value, row_weight, denominator) = ctx.saved_tensors
        batch, rows, heads, dim = native_q.shape
        q = native_q.float() + scale_value * dq.float()
        k = native_k.float() + scale_value * dk.float()
        grad_q = torch.zeros_like(q)
        grad_k = torch.zeros_like(k)
        log_head_count = torch.log(torch.as_tensor(heads, device=q.device, dtype=torch.float32))
        scale = dim ** -0.5
        has_legal = legal_mask.numel() != 0
        for q0 in range(0, rows, ctx.query_block):
            q1 = min(rows, q0 + ctx.query_block)
            for row in range(q0, q1):
                legal = _as_legal(legal_mask if has_legal else None, batch, row, k.shape[1], q.device)
                target = _target_distribution(target_indices[row], target_weights[row], target_mask[row], legal, k.shape[1])
                support = torch.nonzero(target.gt(0).any(0), as_tuple=False).flatten()
                if support.numel() == 0:
                    continue
                logz = _log_normalizer(q[:, row], k, legal, scale, ctx.key_block)
                # First construct the target head probabilities and A_h in
                # bounded support blocks.  The alpha denominator is across
                # heads for each target key; no support-wide logits are kept.
                A = torch.zeros((q.shape[0], q.shape[2]), device=q.device, dtype=torch.float32)
                for support_start in range(0, support.numel(), ctx.key_block):
                    support_stop = min(support.numel(), support_start + ctx.key_block)
                    support_idx = support[support_start:support_stop]
                    target_logits = torch.einsum('bhd,bkhd->bhk', q[:, row].float(), k.index_select(1, support_idx).float()).mul_(scale)
                    target_logits = target_logits.masked_fill(~legal.index_select(1, support_idx).unsqueeze(1), -torch.inf)
                    target_p = torch.exp(target_logits - logz.unsqueeze(-1))
                    alpha = target_p / target_p.sum(1, keepdim=True).clamp_min(1e-30)
                    A += (target[:, support_idx].unsqueeze(1) * alpha).sum(-1)
                if ctx.return_rows:
                    upstream = grad_output[row].float() / float(batch)
                else:
                    upstream = grad_output.float() * row_weight[row] / denominator / float(batch)
                for start in range(0, k.shape[1], ctx.key_block):
                    stop = min(k.shape[1], start + ctx.key_block)
                    idx = torch.arange(start, stop, device=q.device)
                    logits = torch.einsum('bhd,bkhd->bhk', q[:, row].float(), k[:, start:stop].float()).mul_(scale)
                    probs = torch.exp(logits - logz.unsqueeze(-1)).masked_fill(~legal[:, start:stop].unsqueeze(1), 0.0)
                    dlogits = -probs * A.unsqueeze(-1)
                    # Recompute only the support keys in this key block.  The
                    # target-head denominator was accumulated above, so this
                    # remains bounded even when a row has large soft support.
                    local_mask=(support >= start) & (support < stop)
                    if bool(local_mask.any()):
                        local_support=support[local_mask]
                        local_logits=torch.einsum('bhd,bkhd->bhk', q[:, row].float(), k.index_select(1, local_support).float()).mul_(scale)
                        local_logits=local_logits.masked_fill(~legal.index_select(1, local_support).unsqueeze(1), -torch.inf)
                        local_p=torch.exp(local_logits-logz.unsqueeze(-1))
                        local_alpha=local_p/local_p.sum(1,keepdim=True).clamp_min(1e-30)
                        local_target=target[:,local_support].unsqueeze(1)
                        local_positions=(local_support-start).to(torch.long)
                        for local_index, key_index in enumerate(local_positions.tolist()):
                            dlogits[:,:,key_index] += local_target[:,:,local_index]*local_alpha[:,:,local_index]
                    dlogits.mul_(upstream)
                    grad_q[:, row] += torch.einsum('bhk,bkhd->bhd', dlogits, k[:, start:stop]).mul_(scale)
                    grad_k[:, start:stop] += torch.einsum('bhk,bhd->bkhd', dlogits, q[:, row]).mul_(scale)
        return (None, None, (grad_q * scale_value).to(dq.dtype),
                (grad_k * scale_value).to(dk.dtype), None, None, None, None, None, None, None)


def rgbd_probability_score(native_q, native_k, dq, dk, target_indices,
                           target_weights, target_mask, legal_mask,
                           local_scale=1.0, row_weights=None):
    return StreamingRGBDProbabilityScore.apply(
        native_q, native_k, dq, dk, target_indices, target_weights,
        target_mask, legal_mask, torch.as_tensor(local_scale, device=dq.device),
        torch.empty(0, device=dq.device) if row_weights is None else row_weights,
    )


def rgbd_probability_row_scores(native_q, native_k, dq, dk, target_indices,
                                target_weights, target_mask, legal_mask,
                                local_scale=1.0):
    """Return one unweighted batch-mean score per plan row.

    The same streaming custom-autograd kernel is used; the vector output lets
    the training caller apply the formal motion-bucket/query-weight reduction
    without constructing a dense probability tensor.
    """
    return StreamingRGBDProbabilityScore.apply(
        native_q, native_k, dq, dk, target_indices, target_weights,
        target_mask, legal_mask, torch.as_tensor(local_scale, device=dq.device),
        torch.empty(0, device=dq.device), True,
    )


__all__ = ['StreamingRGBDProbabilityScore', 'rgbd_probability_score',
           'rgbd_probability_row_scores', 'attention_mask_to_legal_mask']
