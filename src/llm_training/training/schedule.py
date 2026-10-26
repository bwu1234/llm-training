"""Learning-rate schedule as a pure function of the zero-based optimizer update index.

Statelessness is what makes resume exact: the rate for update N depends only on N, never
on the value stored in a restored optimizer. Warmup and cosine decay follow nanoGPT's
`get_lr`: linear warmup `lr * (it + 1) / (warmup + 1)`, then cosine from `lr` down to
`min_lr` at `total_steps`, flat at `min_lr` afterwards.
"""

import math


def is_constant(training):
    """True when the schedule never touches the optimizer's learning rate (book recipe)."""
    return training.lr_schedule == "constant" and training.warmup_steps == 0


def learning_rate_at(training, step, total_steps):
    peak, warmup = training.learning_rate, training.warmup_steps
    if step < warmup:
        return peak * (step + 1) / (warmup + 1)
    if training.lr_schedule == "constant":
        return peak
    if step > total_steps:
        return training.min_lr
    ratio = (step - warmup) / (total_steps - warmup)
    return training.min_lr + 0.5 * (1.0 + math.cos(math.pi * ratio)) * (peak - training.min_lr)
