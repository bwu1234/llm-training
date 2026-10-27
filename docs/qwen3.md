# Qwen3-0.6B base model

`llm_training.model.qwen3` is a modular port of `reasoning_from_scratch/qwen3.py` from
Sebastian Raschka's *Build a Reasoning Model (From Scratch)*. It loads the released
Qwen3-0.6B base weights as the starting point for fine-tuning. It sits beside the GPT-2
modules in `model/gpt2/` ([GPT-2 guide](gpt2.md)); each family keeps its architecture, weights, and tokenizer in its own
package.

**Status:** the architecture, pinned weight loading, and tokenizer are in place and checked
against Hugging Face Transformers. Two ways to fine-tune it exist: GRPO on MATH, with its own
commands ([GRPO guide](grpo.md)), and the next-token `train` command with `model.arch: qwen3`
and `model.init_from: qwen3-0.6b-base`, for continued pretraining on `uint32` token files or
supervised fine-tuning ([fine-tuning guide](finetuning.md)). `predict` and `generate` load
either family. Training Qwen3 from random weights is not built.

## Usage

```python
from llm_training.model.qwen3.pretrained import load_qwen3_base, prepare_qwen3_base

prepare_qwen3_base("data/qwen3-0.6b-base")  # the only network step; ~1.5 GB, skips verified files
model, tokenizer = load_qwen3_base("data/qwen3-0.6b-base", device="mps", attn_impl="sdpa")
```

`load_qwen3_base` accepts any directory that holds the two pinned files, including an existing
`reasoning-from-scratch/qwen3` download. It checks both SHA-256 hashes before loading. The
weights are stored in bfloat16 and widen exactly into the float32 model. On an M2 Max (MPS),
loading takes about 4 s.

## How it differs from the GPT model

Both models share one interface: `forward(token_ids) -> [batch, seq, vocab]` logits. Every
component is different:

| | GPT (`model/gpt2/`) | Qwen3 (`model/qwen3/`) |
| --- | --- | --- |
| Positions | learned table added to embeddings | RoPE on queries and keys, θ = 1,000,000 |
| Normalization | LayerNorm (scale + shift) | RMSNorm (scale only), also on Q and K per head |
| Attention | multi-head, `head_dim = emb_dim / n_heads` | grouped-query: 16 query heads share 8 K/V heads; `head_dim` 128 set independently |
| Feed-forward | GELU, width 4 × `emb_dim` | SwiGLU, `silu(fc1(x)) * fc2(x)`, width 3072 |
| Biases, dropout | configurable | none |
| Output head | optionally tied | tied to the token embedding |

The preset `QWEN3_0_6B` has 596,049,920 parameters: 28 layers, `emb_dim` 1024, and a
151,936-token vocabulary.

## Modules

| Module | Contents |
| --- | --- |
| [`model.py`](../src/llm_training/model/qwen3/model.py) | `QWEN3_0_6B`, `Qwen3Model`, `TransformerBlock`, `KVCache` |
| [`attention.py`](../src/llm_training/model/qwen3/attention.py) | `GroupedQueryAttention` (explicit, or `attn_impl: sdpa`) |
| [`layers.py`](../src/llm_training/model/qwen3/layers.py) | `RMSNorm`, SwiGLU `FeedForward` |
| [`rope.py`](../src/llm_training/model/qwen3/rope.py) | `compute_rope_params`, `apply_rope` |
| [`pretrained.py`](../src/llm_training/model/qwen3/pretrained.py) | pinned assets: `prepare_qwen3_base`, `load_qwen3_base` |
| [`tokenizer.py`](../src/llm_training/model/qwen3/tokenizer.py) | `Qwen3Tokenizer` (Hugging Face `tokenizers`, EOS `<\|endoftext\|>`); `Qwen3TextTokenizer`, the trainer's tiktoken-style interface |

`KVCache` is kept with the source forward pass. `generation.py` does not use it yet: it
recomputes the whole context for each new token.

## Weights

The source of truth is `Qwen/Qwen3-0.6B-Base` (revision `da87bfb6`). The package loads
`rasbt/qwen3-from-scratch` (revision `8a3862b`), Raschka's conversion of that checkpoint into
this module's parameter names, so loading needs no name-mapping code. The file stores the tied
output head as a separate copy. `load_qwen3_state` checks that the copy equals the embedding
before both load into the one shared tensor. `tokenizer-base.json` is byte-identical to the
official `tokenizer.json` (same git blob, `443909a6`).

Lineage check (2026-10-26):

- Downloaded the official `model.safetensors` at `da87bfb6`. Its SHA-256 `cd2a5120…` matches
  the Hub's LFS hash.
- Renamed the converted `.pth` tensors with `tests/reference/qwen3_weights.to_hf_state`.
- All 310 tensors (596,049,920 values) are bit-identical, with the same bfloat16 dtype.
- The official file has no `lm_head.weight`, which confirms the tied head. The converted
  file's separate head equals its embedding.

## Verification

- **CI** (`tests/unit/test_qwen3.py`, no downloads):
  - On a tiny random configuration, the default model equals the frozen source
    (`tests/reference/qwen3_baseline.py`) bit for bit, both for a full forward pass and
    for KV-cached decoding.
  - SDPA, cached decoding, tying, strict loading and the bf16 → fp32 widening each have
    their own checks.
  - The preset matches the official `config.json`.
- **Oracle** (`tests/oracle/test_qwen3_oracle.py`, opt-in):
  - Hugging Face's `Qwen3ForCausalLM`, built from the official config, runs the same
    weights.
  - Tolerances were declared before the first run: rtol 1e-4 and atol 1e-3, identical
    argmax at every position, and cross-entropy within 1e-4.
  - First run (2026-10-26, CPU, FP32): the largest logit difference was 1.26e-4 (explicit
    attention 1.09e-4, SDPA 1.26e-4). Argmax was identical everywhere.
  - The tokenizer matched `tokenizers` on 7 strings covering whitespace, newlines,
    non-ASCII text, code, and embedded special tokens.

```bash
uv sync --group oracle
LLM_ORACLE=1 uv run pytest tests/oracle/test_qwen3_oracle.py -q
# or reuse an existing download without writing to it:
LLM_ORACLE=1 LLM_QWEN3_DIR=/path/to/reasoning-from-scratch/qwen3 uv run pytest tests/oracle/test_qwen3_oracle.py -q
```
