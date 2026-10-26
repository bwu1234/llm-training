# Upstream provenance

Source: [rasbt/LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch),
commit `cdbd33e6d57a71e20ea31434841095fac50cecd8`.

The [machine-readable manifest](../src/llm_training/upstream-provenance.json)
records SHA256 hashes of the original source files, selected symbols, destinations,
and per-file local diffs at extraction. It ships in the Python wheel and is copied
into each run's provenance. The inspected upstream checkout had an unrelated
modification to `requirements.txt`; all selected code files had empty local diffs.
The original repository was left unchanged.

| New module | Original module | Selected implementation |
| --- | --- | --- |
| `model/attention.py` | `pkg/llms_from_scratch/ch03.py` | `MultiHeadAttention` |
| `model/layers.py` | `pkg/llms_from_scratch/ch04.py` | `LayerNorm`, `GELU`, `FeedForward` |
| `model/gpt.py` | `pkg/llms_from_scratch/ch04.py` | `TransformerBlock`, `GPTModel` |
| `generation.py` | `pkg/llms_from_scratch/ch04.py` | `generate_text_simple` → `generate_greedy` |
| `data/dataset.py` | `pkg/llms_from_scratch/ch02.py` | `GPTDatasetV1` → `SlidingWindowTokenDataset` |
| `training/loss.py` | `pkg/llms_from_scratch/ch05.py` | `calc_loss_batch` → `next_token_cross_entropy` |
| `training/monitoring.py` | `ch05/01_main-chapter-code/gpt_train.py` | `calc_loss_loader` → `mean_batch_loss`, `evaluate_model` → `evaluate_train_validation_loss` |

All destinations are under `src/llm_training/`. Changes to these definitions are
limited to symbol selection and renaming, module/import organization, attribution notices,
and formatting, plus the opt-in extensions below. With their defaults (`init: pytorch`,
`tie_embeddings: false`, `attn_impl: explicit`) the model keeps the source's parameter
names, initialization, causal masking, dropout, loss computation, and untied output head;
the frozen-source parity tests assert zero difference on that path.

Opt-in extensions added after extraction (project code, not upstream):

- `model/init.py`: GPT-2-style initialization, selected by `model.init: gpt2`.
- `model/gpt.py`: optional tied output head (`model.tie_embeddings`).
- `model/attention.py`: optional `scaled_dot_product_attention` path (`model.attn_impl: sdpa`).
- `model/layers.py`: `GELU` builds its float32 constant once as a non-persistent buffer. The
  arithmetic is identical (a test asserts exact equality with the frozen reference) and
  `state_dict` is unchanged.

The extraction-time AST equality check between the source and package `GPTModel` and
`MultiHeadAttention` therefore no longer holds literally; the numerical equivalence tests
replace it.

`tests/reference/baseline.py` combines the same selected original definitions into
a frozen, attributed test fixture with only necessary imports. Keep it independent
of production refactors. New trainer, configuration, acquisition, evaluation,
artifact handling, CLI, and tests are project code; they are not represented as
verbatim upstream implementations.

`tests/reference/training_loop.py` freezes the source script's loss/evaluation helpers,
sample helpers, and `train_model_simple` without changing their bodies. The full
comparison uses this independent loop and the frozen model definitions; the
source script's `GPTModel` and `MultiHeadAttention` were also checked to have the
same ASTs as the package originals. Hashes/symbols are recorded in the manifest.

`configs/pretrain_gpt_162m.yaml` derives settings from
`ch05/01_main-chapter-code/gpt_train.py`. The `sampled_train_validation` recipe preserves source evaluation RNG
consumption and epoch sampling cadence. See the README and reproduction report
for the full-run comparison and scope. `the-verdict.txt` is downloaded
only by explicit preparation, with pinned revision and content hash; it is not
bundled in the package. The committed smoke text was authored for this project.

`LICENSE.txt` is an unchanged copy of the source repository's license file,
including its wording about books and related images. Source headers are retained
and modified files are marked. No book prose, notebook narrative, or images were
copied. Future upstream updates should refresh this mapping and run the parity
checks before changing the frozen reference.
