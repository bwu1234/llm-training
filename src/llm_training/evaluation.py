"""Token-weighted evaluation, independent of the training data iterator."""

import math

import torch


def evaluate(model, loader, device, max_batches=None):
    was_training = model.training
    total_nll, tokens = 0.0, 0
    model.eval()
    try:
        with torch.inference_mode():
            for i, (inputs, targets) in enumerate(loader):
                if max_batches is not None and i >= max_batches:
                    break
                inputs, targets = inputs.to(device), targets.to(device)
                nll = torch.nn.functional.cross_entropy(
                    model(inputs).flatten(0, 1), targets.flatten(), reduction="sum"
                )
                total_nll += nll.item()
                # cross_entropy skips ignored targets (-100: padding, masked prompts).
                tokens += (targets != -100).sum().item()
    finally:
        model.train(was_training)
    if not tokens:
        raise ValueError("Evaluation loader contains no target tokens")
    loss = total_nll / tokens
    if not math.isfinite(loss):
        raise ValueError("Nonfinite evaluation loss")
    return {"loss": loss, "perplexity": math.exp(loss) if loss < 709 else None, "tokens": tokens}
