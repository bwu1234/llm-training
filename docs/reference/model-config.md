# Model configuration

Every field of a recipe's `model:` section, per family. The family is chosen by `model.arch`
(`gpt2` when omitted). A field a family lacks is refused in that family's recipes; the
[last section](#fields-a-family-does-not-have) says why each one is absent.

Each family's tables come from its config dataclass:
[`gpt2/family.py`](../../src/llm_training/model/gpt2/family.py) and
[`qwen3/family.py`](../../src/llm_training/model/qwen3/family.py).
`tests/unit/test_model_config_docs.py` fails when a field, a default or the
"Fixed by weights" column here disagrees with them. The values the presets set are in the
[GPT-2](../gpt2.md#preset-sizes) and [Qwen3](../qwen3.md) guides and in `configs/`.

The "Fixed by weights" column marks the fields that determine the model's shape. When
`init_from` names a release or a project `model.pt`, those fields come from the weights: a
recipe may leave them out, and one that sets a different value is refused. The other fields are
choices for the run, so they may differ from the weights' source.

## Fields every family has

| Field | Default | Meaning |
| --- | --- | --- |
| `init_from` | `null` | Where the starting weights come from. `null` trains from scratch. Otherwise a pinned release (`gpt2-small-124m`, `gpt2-medium-355m`, `qwen3-0.6b-base`) or the path of a project `model.pt`, relative to the config file. Qwen3 has no from-scratch mode, so it always needs one. |
| `weights_dir` | `null` | The directory holding a pinned release's downloaded files, relative to the config file. Required when `init_from` is a release, refused otherwise. |

## GPT-2 (`arch: gpt2`)

The defaults describe a tiny model for tests, not GPT-2 small. A recipe sets the real sizes.

| Field | Default | Fixed by weights | Meaning |
| --- | --- | --- | --- |
| `arch` | `gpt2` | no | Selects this family. |
| `vocab_size` | `50,257` | yes | Rows of the token embedding and outputs of the head. The GPT-2 tokenizer fixes it at 50,257. |
| `context_length` | `16` | yes | The longest sequence, and the number of learned position embeddings and the size of the causal mask. |
| `emb_dim` | `32` | yes | Width of the residual stream. Must be divisible by `n_heads`. |
| `n_heads` | `4` | yes | Attention heads. Each has width `head_dim = emb_dim / n_heads`, which is not a field here: it is 64 in GPT-2 small. |
| `n_layers` | `2` | yes | Transformer blocks. |
| `qkv_bias` | `false` | yes | Whether the query, key and value projections have biases. GPT-2's released weights have them. |
| `drop_rate` | `0.1` | no | Dropout probability for the embeddings, the attention weights and both residual branches. Off in evaluation mode. A run that starts from a pinned release defaults to `0.0`, as the book's fine-tuning does. |
| `tie_embeddings` | `false` | no | `true` makes the output head reuse the token embedding matrix, as GPT-2 does, saving `vocab_size × emb_dim` parameters. Requires `init: gpt2` unless `init_from` is set. |
| `init` | `pytorch` | no | Weight initialization: `pytorch` keeps each module's default, `gpt2` draws from N(0, 0.02) as nanoGPT does. |
| `residual_scaled_init` | `true` | no | Only read when `init` is `gpt2`. Shrinks the init of each block's two residual-writing projections to 0.02 / √(2 × `n_layers`); `false` matches OpenAI's code instead. |
| `attn_impl` | `explicit` | no | Which attention code runs: `explicit` is the book's, building the full T × T score matrix; `sdpa` is PyTorch's fused kernel, which never stores it. Both compute the same attention, but with dropout on they draw different random numbers. |

## Qwen3 (`arch: qwen3`)

The defaults are the released 0.6B architecture.

| Field | Default | Fixed by weights | Meaning |
| --- | --- | --- | --- |
| `arch` | `qwen3` | no | Selects this family. |
| `vocab_size` | `151,936` | yes | Rows of the token embedding and outputs of the head. The Qwen3 tokenizer fixes it at 151,936. |
| `context_length` | `40,960` | yes | Only sizes the RoPE tables. The window a run trains on is `data.max_length`, which Qwen3 recipes must set. |
| `emb_dim` | `1,024` | yes | Width of the residual stream. |
| `n_heads` | `16` | yes | Query heads. |
| `n_layers` | `28` | yes | Transformer blocks. |
| `hidden_dim` | `3,072` | yes | Inner width of the SwiGLU feed-forward network. |
| `head_dim` | `128` | yes | Width of each attention head, set independently of `emb_dim`: the 16 query heads together are 2,048 wide, wider than the 1,024-wide residual stream. |
| `qk_norm` | `true` | yes | Whether queries and keys are RMS-normalized per head before RoPE. |
| `n_kv_groups` | `8` | yes | Key/value heads. Each is shared by `n_heads / n_kv_groups` query heads (grouped-query attention), which must divide evenly. |
| `rope_base` | `1,000,000` | yes | The base of RoPE's rotation frequencies. |
| `tie_embeddings` | `false` | no | `true` makes the output head reuse the token embedding matrix. The released model ties them, but this repo's fine-tuning recipes train a separate head, as the book does, so the default is `false`. |
| `attn_impl` | `sdpa` | no | `sdpa` is PyTorch's fused attention kernel; `explicit` is the book's code, with the scores, mask and softmax as separate steps. Both give the same result, and Qwen3 has no attention dropout. |

## Fields a family does not have

GPT-2 only, because Qwen3 has no mechanism for them:

- `drop_rate`: Qwen3 has no dropout anywhere.
- `qkv_bias`: Qwen3's linear layers have no biases.
- `init`, `residual_scaled_init`: a Qwen3 run always starts from existing weights (`init_from`
  is required), so no initialization is chosen.

Qwen3 only, because GPT-2 fixes them in its design:

- `hidden_dim`: GPT-2's feed-forward network is always 4 × `emb_dim`.
- `head_dim`: GPT-2's is always `emb_dim / n_heads`.
- `n_kv_groups`: GPT-2 uses full multi-head attention, with as many key/value heads as query
  heads.
- `qk_norm`: GPT-2 has no query/key normalization.
- `rope_base`: GPT-2 uses learned position embeddings (up to `context_length`), not RoPE.

The [architecture explorer](../architecture-explorer.md) shows the resolved fields for each
preset as chips, and so lists only the fields its family has.
