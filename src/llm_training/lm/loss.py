# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSES/LLMs-from-scratch.txt).
# Source: https://github.com/rasbt/LLMs-from-scratch
# Modified: selected and renamed symbols, reorganized imports/modules and formatting for llm-training.

import torch


def next_token_cross_entropy(input_batch, target_batch, model, device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
    return loss
