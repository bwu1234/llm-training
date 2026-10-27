"""AdamW as the next-token recipes configure it (`training.weight_decay_policy`)."""

import torch


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
