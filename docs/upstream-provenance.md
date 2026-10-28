# Upstream provenance

Two upstream repositories, each with its own license, shipped unchanged under `LICENSES/` and
installed with the wheel:

| Repository | License file | Derived code |
| --- | --- | --- |
| LLMs-from-scratch | [`LICENSES/LLMs-from-scratch.txt`](../LICENSES/LLMs-from-scratch.txt) | `model/gpt2/`, `data/dataset.py`, `data/instructions.py`, `lm/generation.py`, `lm/loss.py`, `lm/monitoring.py` |
| reasoning-from-scratch | [`LICENSES/reasoning-from-scratch.txt`](../LICENSES/reasoning-from-scratch.txt) | `model/qwen3/`, `reasoning/grading.py`, `reasoning/grpo.py`, `reasoning/distill.py`, and the recipe of `reasoning/trainer.py` |

Each derived file's header names its license file.

`docs/architecture/spec.js`, the architecture explorer's generated data, also holds copies of
the derived classes' `forward` methods, of derived functions that run in the forward pass
(`apply_rope`), and of the individual source lines each traced step runs. Each copied method
or function keeps its file's header, which the page
shows with the excerpt. It isn't installed with the wheel. The CI `pages` job publishes it on
[GitHub Pages](https://bwu1234.github.io/llm-training/) together with an unchanged copy of `LICENSES/`, and the page's
footer links to both licenses.

## GPT-2 (LLMs-from-scratch)

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
| `model/gpt2/attention.py` | `pkg/llms_from_scratch/ch03.py` | `MultiHeadAttention` |
| `model/gpt2/layers.py` | `pkg/llms_from_scratch/ch04.py` | `LayerNorm`, `GELU`, `FeedForward` |
| `model/gpt2/model.py` | `pkg/llms_from_scratch/ch04.py` | `TransformerBlock`, `GPTModel` |
| `lm/generation.py` | `pkg/llms_from_scratch/ch04.py` | `generate_text_simple` → `generate_greedy` |
| `data/dataset.py` | `pkg/llms_from_scratch/ch02.py` | `GPTDatasetV1` → `SlidingWindowTokenDataset` |
| `lm/loss.py` | `pkg/llms_from_scratch/ch05.py` | `calc_loss_batch` → `next_token_cross_entropy` |
| `lm/monitoring.py` | `ch05/01_main-chapter-code/gpt_train.py` | `calc_loss_loader` → `mean_batch_loss`, `evaluate_model` → `evaluate_train_validation_loss` |

All destinations are under `src/llm_training/`. Changes to these definitions are
limited to symbol selection and renaming, module/import organization, attribution notices,
and formatting, plus the opt-in extensions below. With their defaults (`init: pytorch`,
`tie_embeddings: false`, `attn_impl: explicit`) the model keeps the source's parameter
names, initialization, causal masking, dropout, loss computation, and untied output head;
the frozen-source parity tests assert zero difference on that path.

Opt-in extensions added after extraction (project code, not upstream):

- `model/gpt2/init.py`: GPT-2-style initialization, selected by `model.init: gpt2`.
- `model/gpt2/model.py`: optional tied output head (`model.tie_embeddings`).
- `model/gpt2/attention.py`: optional `scaled_dot_product_attention` path (`model.attn_impl: sdpa`).
- `model/gpt2/layers.py`: `GELU` builds its float32 constant once as a non-persistent buffer. The
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

Fine-tuning additions (same repository and commit, no local diff at extraction):

| New module | Source | Selected implementation |
| --- | --- | --- |
| `data/instructions.py` | `ch07/01_main-chapter-code/gpt_instruction_finetuning.py` | `format_input` |

`collate_instructions` is project code that reproduces chapter 7's `custom_collate_fn` and
its exercise 7.2 masking variant from examples that already end in end-of-text; a test checks
the tensors are equal. `tests/reference/instruction_baseline.py` freezes those definitions
(and chapter 8's below). The starting weights for `init_from: gpt2-small-124m` and
`gpt2-medium-355m` are the author's
[`rasbt/gpt2-from-scratch-pytorch`](https://huggingface.co/rasbt/gpt2-from-scratch-pytorch)
files at `402728f` (MIT); the small file is bit-identical to OpenAI's release
(`tests/oracle/test_gpt2_lineage.py`). The manifest records the source hashes and pins.

`configs/pretrain_gpt_162m.yaml` derives settings from
`ch05/01_main-chapter-code/gpt_train.py`. The `sampled_train_validation` recipe preserves source evaluation RNG
consumption and epoch sampling cadence. See [Reproduce the chapter-5 run](how-to/reproduce-chapter-5.md) and the reproduction report
for the full-run comparison and scope. `the-verdict.txt` is downloaded
only by explicit preparation, with pinned revision and content hash; it is not
bundled in the package. The committed smoke text was authored for this project.

`LICENSES/LLMs-from-scratch.txt` is an unchanged copy of the source repository's license
file, including its wording about books and related images. Source headers are retained
and modified files are marked. No book prose, notebook narrative, or images were
copied. Future upstream updates should refresh this mapping and run the parity
checks before changing the frozen reference.

## Qwen3 (reasoning-from-scratch)

Source: [rasbt/reasoning-from-scratch](https://github.com/rasbt/reasoning-from-scratch),
commit `a788466dc85cfe8617b6c0b809ed4ab9084485de`, file `reasoning_from_scratch/qwen3.py`
(SHA256 `873c3cbd…`, no local diff at extraction). The manifest records it under
`additional_sources`, and its license ships unchanged as
[`LICENSES/reasoning-from-scratch.txt`](../LICENSES/reasoning-from-scratch.txt). That file is
plain Apache 2.0, without the book exclusions in `LICENSES/LLMs-from-scratch.txt`.

| New module | Selected implementation |
| --- | --- |
| `model/qwen3/model.py` | `QWEN_CONFIG_06_B` → `QWEN3_0_6B`, `Qwen3Model`, `TransformerBlock`, `KVCache` |
| `model/qwen3/attention.py` | `GroupedQueryAttention` |
| `model/qwen3/layers.py` | `FeedForward`, `RMSNorm` |
| `model/qwen3/rope.py` | `compute_rope_params`, `apply_rope` |
| `model/qwen3/tokenizer.py` | `Qwen3Tokenizer` |

Changes:

- The `dtype` config key and arguments are gone. Parameters are float32, and the output head
  no longer casts its input.
- Two opt-in additions:
  - `tie_embeddings`, which the 0.6B preset sets to match the official
    `tie_word_embeddings: true`;
  - `attn_impl: sdpa`.
- Otherwise the selected definitions are copied unchanged. The download and Hugging Face
  name-mapping helpers were left out: the pinned weights already use these modules'
  parameter names.

`tests/reference/qwen3_baseline.py` freezes the selected definitions. Its unit test shows that
the package's default path matches them bit for bit.


## GRPO on MATH (reasoning-from-scratch, chapters 2–6)

Same repository and commit as the Qwen3 port, no local diff at extraction. The license is the
same file, `LICENSES/reasoning-from-scratch.txt`.

| New module | Source | Selected implementation |
| --- | --- | --- |
| `reasoning/grading.py` | `reasoning_from_scratch/ch03.py` | `get_last_boxed`, `extract_final_candidate`, `normalize_text`, `sympy_parser`, `equality_check`, `split_into_parts`, `grade_answer`, `render_prompt`, and their constants |
| `reasoning/grpo.py` | `reasoning_from_scratch/ch04.py` | `top_p_filter` |
| `reasoning/grpo.py` | `reasoning_from_scratch/ch06.py` | `sample_response`, `reward_rlvr`, `sequence_logprob`, `compute_grpo_loss` → `grpo_step` |
| `reasoning/grpo.py` | `reasoning_from_scratch/ch02.py` | `generate_text_basic_stream_cache` → `generate_greedy_ids` |
| `reasoning/trainer.py` | `ch06/02_rlvr_grpo_scripts_intro/rlvr_grpo_original_no_kl.py` | the training-loop recipe and `--skip-zero-advantage-updates` |

The grading functions are copied unchanged (AST-identical before formatting).
`reasoning/trainer.py` copies no code: it reimplements the script's loop, and its header
names the script as the source of the recipe. Changes elsewhere:

- `TimeLimitedGrader` (new) runs `grade_answer` in a worker process with a time limit. SymPy
  parses with `evaluate=True`, so a 7-character answer such as `9^9^9^9` starts an integer
  power that runs indefinitely inline (checked: still running after 20 s). Timeouts and
  grader exceptions score 0 and are recorded per rollout.
- Rollouts sample on the model's device, as the bonus script does, from an explicit generator
  whose state every checkpoint saves.
- `grpo_step` backpropagates each rollout as soon as its log-probability is computed instead
  of holding all rollouts' graphs for one `mean` loss. The gradient is the same (unit test
  against the frozen source, `rtol=1e-5`), with one rollout's activations in memory at a time.
  When all rewards are equal the advantages are exactly zero, so it sets zero gradients
  without the forward/backward pass; the optimizer still steps, as upstream does by default.
- Training is float32 with SDPA attention. Upstream holds the embedding and linear weights
  and their AdamW state in bfloat16 (RMSNorm scales stay float32), where many lr-1e-5 updates
  round away, so the two runs are not numerically
  comparable step for step.
- Data is pinned by commit URL and SHA-256 (`reasoning/math_data.py`); upstream reads `main`.

`tests/reference/reasoning_baseline.py` freezes the selected definitions, with the frozen
`KVCache` from `qwen3_baseline.py`.

## Reasoning distillation (reasoning-from-scratch, chapter 8)

| New module | Source | Selected implementation |
| --- | --- | --- |
| `reasoning/distill.py` | `ch08/04_train_with_distillation/distill.py` | `strip_think_tags`, `format_distilled_answer` |

The rest of chapter 8's recipe is reproduced by the next-token trainer rather than copied:
`data/instructions.py` encodes prompt and answer as `build_examples` does and drops
over-length examples as `filter_examples_by_max_len` does, and prompt-masked targets give the
same loss and gradient as `compute_example_loss` (tests against the frozen copies). Batch
order comes from the trainer's seeded shuffle, not `random.Random(seed)`.
