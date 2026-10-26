# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source: https://github.com/rasbt/LLMs-from-scratch
# Modified: selected symbols, reorganized imports/modules and formatting for llm-training.
# Also modified: GELU's float32 constant is a non-persistent buffer instead of rebuilt per call.

import torch
from torch import nn


class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift


class GELU(nn.Module):
    def __init__(self):
        super().__init__()
        # Same float32 constant the original rebuilt on every call; as a buffer it is built
        # once and follows the module across devices without a per-forward host copy.
        self.register_buffer(
            "_sqrt_two_over_pi", torch.sqrt(torch.tensor(2.0 / torch.pi)), persistent=False
        )

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(self._sqrt_two_over_pi * (x + 0.044715 * torch.pow(x, 3))))


class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )

    def forward(self, x):
        return self.layers(x)
