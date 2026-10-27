# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSES/reasoning-from-scratch.txt).
# Source: https://github.com/rasbt/reasoning-from-scratch (reasoning_from_scratch/qwen3.py)
# Modified: selected symbols, reorganized imports/modules and formatting for llm-training.
# Also modified: no `dtype` config key (float32 parameters); optional tied output head
# (`tie_embeddings`) and fused attention (`attn_impl`); defaults keep source behavior.

import torch
import torch.nn as nn

from .attention import GroupedQueryAttention
from .layers import FeedForward, RMSNorm
from .rope import compute_rope_params

# 0.6 billion parameters
QWEN3_0_6B = {
    "vocab_size": 151_936,  # Vocabulary size
    "context_length": 40_960,  # Length originally used during training
    "emb_dim": 1024,  # Embedding dimension
    "n_heads": 16,  # Number of attention heads
    "n_layers": 28,  # Number of layers
    "hidden_dim": 3072,  # Size of intermediate dim in FeedForward
    "head_dim": 128,  # Size of the heads in GQA
    "qk_norm": True,  # Whether to normalize queries & keys in GQA
    "n_kv_groups": 8,  # Key-Value groups for GQA
    "rope_base": 1_000_000.0,  # The base in RoPE's "theta"
    # Qwen3-0.6B ties its output head to the token embedding (`tie_word_embeddings`).
    "tie_embeddings": True,
}


class Qwen3Model(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        # Main model parameters
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])

        self.trf_blocks = nn.ModuleList(  # ModuleList since Sequential can only accept one input, and we need `x, mask, cos, sin`
            [TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )
        self.final_norm = RMSNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)
        # Keys are read with defaults so the source's config dicts still build the source model.
        if cfg.get("tie_embeddings", False):
            self.out_head.weight = self.tok_emb.weight

        # Reusable utilities
        if cfg["head_dim"] is None:
            head_dim = cfg["emb_dim"] // cfg["n_heads"]
        else:
            head_dim = cfg["head_dim"]
        cos, sin = compute_rope_params(
            head_dim=head_dim, theta_base=cfg["rope_base"], context_length=cfg["context_length"]
        )
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        self.cfg = cfg
        self.current_pos = 0  # Track current position in KV cache

    def forward(self, in_idx, cache=None):
        # Forward pass
        tok_embeds = self.tok_emb(in_idx)
        x = tok_embeds

        num_tokens = x.shape[1]
        pos_start = self.current_pos if cache is not None else 0
        pos_end = pos_start + num_tokens
        if pos_end > self.cfg["context_length"]:
            raise ValueError(
                f"Sequence length {pos_end} exceeds the model's context length "
                f"of {self.cfg['context_length']} tokens."
            )

        if cache is not None:
            self.current_pos = pos_end
            mask = torch.triu(
                torch.ones(pos_end, pos_end, device=x.device, dtype=torch.bool), diagonal=1
            )[pos_start:pos_end, :pos_end]
        else:
            mask = torch.triu(
                torch.ones(num_tokens, num_tokens, device=x.device, dtype=torch.bool), diagonal=1
            )
        # Prefill (no cache): mask starts as (num_tokens, num_tokens)
        # Cached decoding: mask starts as (num_tokens, prev_k_number_tokens + num_tokens)
        #
        # We add two leading dimensions so the mask becomes
        # (1, 1, num_tokens, num_tokens) during prefill and
        # (1, 1, num_tokens, total_key_tokens) during cached decoding.
        # These extra dimensions let PyTorch broadcast the same mask
        # across all batches and attention heads when applying it to
        # attn_scores of shape (batch, num_heads, num_tokens, total_key_tokens).
        mask = mask[None, None, :, :]  # broadcast mask

        for i, block in enumerate(self.trf_blocks):
            blk_cache = cache.get(i) if cache else None
            x, new_blk_cache = block(
                x, mask, self.cos, self.sin, start_pos=pos_start, cache=blk_cache
            )
            if cache is not None:
                cache.update(i, new_blk_cache)

        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits

    def reset_kv_cache(self):
        self.current_pos = 0


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = GroupedQueryAttention(
            d_in=cfg["emb_dim"],
            num_heads=cfg["n_heads"],
            head_dim=cfg["head_dim"],
            num_kv_groups=cfg["n_kv_groups"],
            qk_norm=cfg["qk_norm"],
            attn_impl=cfg.get("attn_impl", "explicit"),
        )
        self.ff = FeedForward(cfg)
        self.norm1 = RMSNorm(cfg["emb_dim"], eps=1e-6)
        self.norm2 = RMSNorm(cfg["emb_dim"], eps=1e-6)

    def forward(self, x, mask, cos, sin, start_pos=0, cache=None):
        # Shortcut connection for attention block
        shortcut = x
        x = self.norm1(x)
        x, next_cache = self.att(
            x, mask, cos, sin, start_pos=start_pos, cache=cache
        )  # Shape [batch_size, num_tokens, emb_size]
        x = x + shortcut  # Add the original input back

        # Shortcut connection for feed-forward block
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = x + shortcut  # Add the original input back

        return x, next_cache


class KVCache:
    def __init__(self, n_layers):
        self.cache = [None] * n_layers

    def get(self, layer_idx):
        return self.cache[layer_idx]

    def update(self, layer_idx, value):
        self.cache[layer_idx] = value

    def get_all(self):
        return self.cache

    def reset(self):
        for i in range(len(self.cache)):
            self.cache[i] = None
