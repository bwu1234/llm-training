# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt in this folder).
# Source: https://github.com/rasbt/reasoning-from-scratch (reasoning_from_scratch/qwen3.py)
# Modified: selected symbols, reorganized imports/modules and formatting for llm-training.

import torch

# ==============================================================================
# RoPE implementation summary
#
#
# There are two common styles to implement RoPE, which are
# mathematically equivalent;
# they mainly differ in how the rotation matrix pairs dimensions.
#
# 1) Split-halves style (this repo, Hugging Face Transformers):
#
#   For hidden dim d = 4 (example):
#
#       [ x0   x1 | x2   x3 ]
#         │    │    │    │
#         ▼    ▼    ▼    ▼
#        cos  cos  sin  sin
#
#   Rotation matrix:
#
#       [ cosθ0   0    -sinθ0   0   ]
#       [  0    cosθ1    0    -sinθ1]
#       [ sinθ0   0     cosθ0   0   ]
#       [  0    sinθ1    0     cosθ1]
#
#   Here, the embedding dims are split into two halves and then
#   each one is rotated in blocks.
#
#
# 2) Interleaved (even/odd) style (original paper, Llama repo):
#
#   For hidden dim d = 4 (example):
#
#       [ x0   x1   x2   x3 ]
#         │    │    │    │
#         ▼    ▼    ▼    ▼
#        cos  sin  cos  sin
#
#   Rotation matrix:
#
#       [ cosθ0  -sinθ0   0       0    ]
#       [ sinθ0   cosθ0   0       0    ]
#       [  0        0    cosθ1  -sinθ1 ]
#       [  0        0    sinθ1   cosθ1 ]
#
#
#   Here, embedding dims are interleaved as even/odd cosine/sine pairs.
#
# Both layouts encode the same relative positions; the only difference is how
# dimensions are paired.
# ==============================================================================


def compute_rope_params(head_dim, theta_base=10_000, context_length=4096, dtype=torch.float32):
    assert head_dim % 2 == 0, "Embedding dimension must be even"

    # Compute the inverse frequencies
    inv_freq = 1.0 / (
        theta_base
        ** (torch.arange(0, head_dim, 2, dtype=dtype)[: (head_dim // 2)].float() / head_dim)
    )

    # Generate position indices
    positions = torch.arange(context_length, dtype=dtype)

    # Compute the angles
    angles = positions.unsqueeze(1) * inv_freq.unsqueeze(
        0
    )  # Shape: (context_length, head_dim // 2)

    # Expand angles to match the head_dim
    angles = torch.cat([angles, angles], dim=1)  # Shape: (context_length, head_dim)

    # Precompute sine and cosine
    cos = torch.cos(angles)
    sin = torch.sin(angles)

    return cos, sin


def apply_rope(x, cos, sin, offset=0):
    # x: (batch_size, num_heads, seq_len, head_dim)
    batch_size, num_heads, seq_len, head_dim = x.shape
    assert head_dim % 2 == 0, "Head dimension must be even"

    # Split x into first half and second half
    x1 = x[..., : head_dim // 2]  # First half
    x2 = x[..., head_dim // 2 :]  # Second half

    # Adjust sin and cos shapes
    cos = (
        cos[offset : offset + seq_len, :].unsqueeze(0).unsqueeze(0)
    )  # Shape: (1, 1, seq_len, head_dim)
    sin = sin[offset : offset + seq_len, :].unsqueeze(0).unsqueeze(0)

    # Apply the rotary transformation
    rotated = torch.cat((-x2, x1), dim=-1)
    x_rotated = (x * cos) + (rotated * sin)

    # It's ok to use lower-precision after applying cos and sin rotation
    return x_rotated.to(dtype=x.dtype)
