"""LMStaR objective: per-positive multi-positive InfoNCE (float32 numerics)."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def multi_positive_infonce(scores, positive_mask, negative_mask, temperature):
    """For every positive p: -log e^{z_p} / (e^{z_p} + sum_n e^{z_n}) (the other positives are excluded), averaged
    over the positives of an interaction, then over the interactions; z = score / temperature."""
    if temperature <= 0:
        raise ValueError('temperature must be positive')
    if scores.ndim != 2 or scores.shape != positive_mask.shape or scores.shape != negative_mask.shape:
        raise ValueError('scores and masks must have identical [B, M] shapes')
    if bool((positive_mask & negative_mask).any()):
        raise ValueError('positive and negative masks overlap')
    if not bool(positive_mask.any(dim=1).all() and negative_mask.any(dim=1).all()):
        raise ValueError('every interaction requires >= 1 positive and >= 1 negative')
    logits = scores.float() / temperature
    neg_lse = torch.logsumexp(logits.masked_fill(~negative_mask, -torch.inf), dim=1, keepdim=True)
    per_positive = F.softplus(neg_lse - logits) * positive_mask
    return (per_positive.sum(dim=1) / positive_mask.sum(dim=1)).mean()
