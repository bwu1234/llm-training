"""Optimizer construction and gradient utilities for the training loop."""

import torch
from torch.nn.utils import clip_grads_with_norm_, get_total_norm


def build_optimizer(model, training):
    betas = (training.beta1, training.beta2)
    if training.weight_decay_policy == "all":
        # Book recipe: one group over model.parameters(), exactly as the frozen source does.
        return torch.optim.AdamW(
            model.parameters(),
            lr=training.learning_rate,
            weight_decay=training.weight_decay,
            betas=betas,
        )
    # named_parameters() lists a tied tensor once. Matrices and embeddings (dim >= 2)
    # decay; biases and LayerNorm scale/shift (dim < 2) do not.
    params = [p for _, p in model.named_parameters() if p.requires_grad]
    return torch.optim.AdamW(
        [
            {"params": [p for p in params if p.dim() >= 2], "weight_decay": training.weight_decay},
            {"params": [p for p in params if p.dim() < 2], "weight_decay": 0.0},
        ],
        lr=training.learning_rate,
        betas=betas,
    )


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
