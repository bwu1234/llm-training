# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt in this folder).
# Source: https://github.com/rasbt/reasoning-from-scratch (reasoning_from_scratch/qwen3.py)
# Modified: selected symbols, reorganized imports/modules and formatting for llm-training.
# Also modified: no `dtype` argument (float32 parameters); optional fused SDPA path
# (attn_impl="sdpa"); the default is the source code.

import torch
import torch.nn as nn

from .layers import RMSNorm
from .rope import apply_rope


class GroupedQueryAttention(nn.Module):
    def __init__(
        self, d_in, num_heads, num_kv_groups, head_dim=None, qk_norm=False, attn_impl="explicit"
    ):
        super().__init__()
        assert num_heads % num_kv_groups == 0, "num_heads must be divisible by num_kv_groups"

        self.attn_impl = attn_impl
        self.num_heads = num_heads
        self.num_kv_groups = num_kv_groups
        self.group_size = num_heads // num_kv_groups

        if head_dim is None:
            assert d_in % num_heads == 0, (
                "`d_in` must be divisible by `num_heads` if `head_dim` is not set"
            )
            head_dim = d_in // num_heads

        self.head_dim = head_dim
        self.d_out = num_heads * head_dim

        self.W_query = nn.Linear(d_in, self.d_out, bias=False)
        self.W_key = nn.Linear(d_in, num_kv_groups * head_dim, bias=False)
        self.W_value = nn.Linear(d_in, num_kv_groups * head_dim, bias=False)

        self.out_proj = nn.Linear(self.d_out, d_in, bias=False)

        if qk_norm:
            self.q_norm = RMSNorm(head_dim, eps=1e-6)
            self.k_norm = RMSNorm(head_dim, eps=1e-6)
        else:
            self.q_norm = self.k_norm = None

    def forward(self, x, mask, cos, sin, start_pos=0, cache=None):
        b, num_tokens, _ = x.shape

        # Apply projections
        queries = self.W_query(x)  # (b, num_tokens, num_heads * head_dim)
        keys = self.W_key(x)  # (b, num_tokens, num_kv_groups * head_dim)
        values = self.W_value(x)  # (b, num_tokens, num_kv_groups * head_dim)

        # Reshape to heads / kv-groups
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        keys_new = keys.view(b, num_tokens, self.num_kv_groups, self.head_dim).transpose(1, 2)
        values_new = values.view(b, num_tokens, self.num_kv_groups, self.head_dim).transpose(1, 2)

        # Optional normalization
        if self.q_norm:
            queries = self.q_norm(queries)
        if self.k_norm:
            keys_new = self.k_norm(keys_new)

        # Apply RoPE
        queries = apply_rope(queries, cos, sin, offset=start_pos)
        keys_new = apply_rope(keys_new, cos, sin, offset=start_pos)

        if cache is not None:
            prev_k, prev_v = cache
            keys = torch.cat([prev_k, keys_new], dim=2)
            values = torch.cat([prev_v, values_new], dim=2)
        else:
            start_pos = 0  # reset RoPE
            keys, values = keys_new, values_new
        next_cache = (keys, values)

        # Expand K and V to match number of heads
        keys = keys.repeat_interleave(self.group_size, dim=1)
        values = values.repeat_interleave(self.group_size, dim=1)

        if self.attn_impl == "sdpa":
            # Fused kernel: never materializes the (T, T) score matrix. `is_causal` aligns its
            # mask top-left, which is only right with no cached prefix (a square mask); after
            # a prefix, pass the explicit mask, inverted because SDPA marks allowed positions.
            square = mask.shape[-2] == mask.shape[-1]
            context = torch.nn.functional.scaled_dot_product_attention(
                queries,
                keys,
                values,
                attn_mask=None if square else ~mask,
                is_causal=square,
            )
            context = context.transpose(1, 2).reshape(b, num_tokens, self.d_out)
            return self.out_proj(context), next_cache

        # Attention
        attn_scores = queries @ keys.transpose(2, 3)
        attn_scores = attn_scores.masked_fill(mask, -torch.inf)
        attn_weights = torch.softmax(attn_scores / self.head_dim**0.5, dim=-1)

        context = (attn_weights @ values).transpose(1, 2).reshape(b, num_tokens, self.d_out)
        return self.out_proj(context), next_cache
