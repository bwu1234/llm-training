"""GPT-2-style weight initialization (nanoGPT convention).

OpenAI's released model.py draws embeddings and weights from N(0, 0.02) (position
table N(0, 0.01)) with zero biases and no residual scaling. nanoGPT uses 0.02 for
both embeddings and additionally shrinks the output projection of each residual
branch to 0.02 / sqrt(2 * n_layers), following the GPT-2 paper's description. This
module follows nanoGPT; `residual_scaled` switches the extra scaling off to match
OpenAI's code instead.
"""

import math

import torch
from torch import nn

STD = 0.02


def init_gpt2_weights(model, n_layers, residual_scaled=True):
    def init(module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=STD)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=STD)

    model.apply(init)
    if residual_scaled:
        # Each block adds two residual branches (attention, feed-forward); their final
        # projections are the only layers that write into the residual stream.
        std = STD / math.sqrt(2 * n_layers)
        with torch.no_grad():
            for block in model.trf_blocks:
                nn.init.normal_(block.att.out_proj.weight, mean=0.0, std=std)
                nn.init.normal_(block.ff.layers[2].weight, mean=0.0, std=std)
