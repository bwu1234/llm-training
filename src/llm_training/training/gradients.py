"""Gradient utilities shared by the next-token and GRPO training loops."""

import torch
from torch.nn.utils import clip_grads_with_norm_, get_total_norm


def clip_and_measure(model, max_norm):
    """Return the pre-clip global gradient norm, clipping in place when `max_norm` is set.

    Computing the norm once replaces a per-parameter `isfinite(...).all()` host sync; a
    NaN/inf gradient makes the norm non-finite, so the caller's check still catches it.
    Without `max_norm` the gradients are left untouched, so the book path is bit-identical.
    """
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    norm = get_total_norm(grads)
    if max_norm is not None and torch.isfinite(norm):
        clip_grads_with_norm_(model.parameters(), max_norm, norm)
    return norm
