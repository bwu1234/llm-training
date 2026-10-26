# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source: https://github.com/rasbt/LLMs-from-scratch
# Modified: selected and renamed functions and imports, formatted for llm-training.

"""Batch-averaged training and validation loss monitoring."""

import torch

from .loss import next_token_cross_entropy


def mean_batch_loss(data_loader, model, device, num_batches=None):
    """Return the unweighted mean of batch losses over a loader subset."""
    total_loss = 0.0
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = next_token_cross_entropy(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches


def evaluate_train_validation_loss(model, train_loader, val_loader, device, eval_iter):
    """Monitor both splits, consuming loader RNG and returning to training mode."""
    model.eval()
    with torch.no_grad():
        train_loss = mean_batch_loss(train_loader, model, device, num_batches=eval_iter)
        val_loss = mean_batch_loss(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss
