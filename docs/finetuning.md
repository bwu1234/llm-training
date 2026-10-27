# Continued pretraining and supervised fine-tuning

The next-token trainer (`llm-training train`) can start from released weights instead of
random ones. With raw text it does **continued pretraining**; with instruction/response pairs
it does **supervised fine-tuning (SFT)**. Both use the same loop as training from scratch,
with the same checkpoints, exact resume, and run records. Chapter 7 of *Build a Large
Language Model (From Scratch)* fine-tunes with chapter 5's training loop, and chapter 8 of
*Build a Reasoning Model (From Scratch)* uses the same next-token loss on the answer tokens.
So SFT here is a data format (`data.format: instructions`), not a separate trainer.

## Starting weights

`model.init_from` names a pinned release or a project `model.pt`:

| `init_from` | Family | File | Size | Source |
| --- | --- | --- | --- | --- |
| `gpt2-small-124m` | GPT-2 | `gpt2-small-124M.pth` | 703 MB | [`rasbt/gpt2-from-scratch-pytorch`](https://huggingface.co/rasbt/gpt2-from-scratch-pytorch) at `402728f` |
| `gpt2-medium-355m` | GPT-2 | `gpt2-medium-355M.pth` | 1.73 GB | same repository and revision |
| `qwen3-0.6b-base` | Qwen3 | `qwen3-0.6B-base.pth` | 1.5 GB | [`rasbt/qwen3-from-scratch`](https://huggingface.co/rasbt/qwen3-from-scratch) at `8a3862b` |
| a path to `model.pt` | from the file | | | any finished run of this trainer |

The GPT-2 files are the *LLMs-from-scratch* author's conversion of OpenAI's release into this
package's parameter names (chapter 5's alternative weight loading), so they load strictly with
no name mapping. `tests/oracle/test_gpt2_lineage.py` maps OpenAI's own `model.safetensors`
into the model and finds every tensor of the small file bit-identical (checked 2026-10-26;
the medium file was not compared). The Qwen3 file's lineage is in the [Qwen3 guide](qwen3.md).

```yaml
model:
  arch: qwen3                     # gpt2 (default) or qwen3
  init_from: qwen3-0.6b-base      # or ../runs/sft/<run>/model.pt
  weights_dir: ../data/qwen3-0.6b-base   # pinned releases only
  tie_embeddings: false
  attn_impl: sdpa
data:
  max_length: 1024                # training window; required for Qwen3
```

The weights decide the shape fields (`emb_dim`, `n_layers`, `vocab_size`, ...), so a recipe
may omit them. If a recipe does name one, it must agree with the weights. A recipe may still
choose dropout (GPT-2), the attention kernel, and whether the head is tied. When it doesn't,
a `model.pt` passes on its own choices (a tied model stays tied), while a pinned release uses
the books' defaults: untied, dropout off. Tying is checked at load time: the source head must
equal the embedding.

`data.max_length` sets the training window for text and the longest example kept for
instructions. It defaults to `model.context_length`. Qwen3 requires it, because its
`context_length` of 40,960 only sizes the RoPE tables.

A finished run writes `model.pt` with its family, full model config, and tokenizer identity.
That file is the hand-off between stages: point the next recipe's `init_from` at it
(continued pretraining, then SFT; or SFT on one dataset, then another). The run's
`provenance.json` records `initial_weights`: the release URL and SHA-256, or the `model.pt`
path and SHA-256.

Checkpoints written before this support was added still load in `predict`, `generate`, and as
`init_from`. Resuming one needs `--allow-code-drift`, because the training code changed; the
same applies to GRPO checkpoints.

`llm-training prepare --config <recipe>` downloads what the recipe reads: the tokenizer, a
pinned release named in `init_from`, and the dataset. Training never downloads anything.

## Recipes

Measured on an M2 Max (64 GB, MPS, float32) on 2026-10-26, by timing a few updates with
evaluation and sampling turned off. Full-run times are estimates from those rates.

| Recipe | What it is | Updates | Measured | Estimated full run | Peak MPS memory |
| --- | --- | --- | --- | --- | --- |
| [`sft_gpt2_medium_alpaca.yaml`](../configs/sft_gpt2_medium_alpaca.yaml) | Chapter 7: instruction fine-tuning of GPT-2 medium | 233 | 1.18 s per update (batch of 8); an evaluation every 5 updates adds about 5 s | 8–10 min | 15 GB |
| [`sft_qwen3_0_6b_math_distill.yaml`](../configs/sft_qwen3_0_6b_math_distill.yaml) | Chapter 8: distilling DeepSeek-R1 math reasoning into Qwen3-0.6B base | 13,400 | 4.4 s per example (~830 answer tokens) | 18–22 h | 41 GB |
| [`cpt_gpt2_small_tinystories.yaml`](../configs/cpt_gpt2_small_tinystories.yaml) | Continued pretraining, **plumbing test** | 200 | 9.05 s per 32,768-token update (3,600 tokens/s) | 30 min | 28 GB |
| [`cpt_qwen3_0_6b_tinystories.yaml`](../configs/cpt_qwen3_0_6b_tinystories.yaml) | Continued pretraining, **plumbing test** | 100 | 55 s per 32,768-token update (600 tokens/s) | 1.5 h | 30 GB |

The chapter 8 estimate scales the measured rate to the dataset's mean example length (1,173
tokens). The Qwen3 continued-pretraining recipe uses micro-batches of 2; at 4 it peaked at
47 GB and was slower (65 s per update).

Short runs of each recipe on the GPU behaved as expected:

- **GPT-2 medium SFT.** After 6 updates, validation loss was 1.091. Chapter 7 reports
  1.103 at the same point. The two differ in batch order and head tying.
- **Qwen3 SFT.** Its first training-time sample answered the test prompt with `\boxed{7}`
  and stopped.
- **Continued pretraining.** Before training, validation loss on TinyStories was 2.39 for
  GPT-2 small and 1.56 for Qwen3. The 51M model trained only on TinyStories reaches 1.38.

## Continued pretraining: read before a long run

The TinyStories recipes test the path from released weights through the token corpus to
`model.pt`. They are not meant to be run for long. A model drifts toward whatever text it
keeps training on. Hours on children's stories would mostly teach GPT-2 or Qwen3 to write
children's stories, and they would lose part of what they knew. This is called catastrophic
forgetting. Before a real run:

- **Choose the corpus.** Use text from the domain the model should improve at, and mix in
  some general text so the older skills keep getting used. Corpora are registered in
  `data/corpus.py` (`CORPORA`: pinned URLs, sizes and SHA-256). Registering local files and
  mixing corpora are not built yet ([backlog](backlog.md)).
- **Measure forgetting.** Track validation loss on the old domain as well as the new one,
  before and after. The starting losses above are the baseline for TinyStories.
- **Treat the learning rates as provisional.** Neither book has a continued-pretraining
  recipe, so the recipes use common practice rather than a source: re-warm to about 10%
  of the original peak, then cosine decay. Tune them on the chosen corpus.

Qwen3's 151,936-token vocabulary does not fit `uint16`, so its token files are `uint32`
(`data/tinystories-qwen3`, 451.7M tokens, 1.8 GB; tokenizing took 111 s). The manifest records
the dtype, and verification checks file sizes against it. GPT-2 corpora stay `uint16`.

## Instruction data

| `data.dataset` | Rows | Prompt | Split | Source |
| --- | --- | --- | --- | --- |
| `alpaca-1100` | 1,100 | Alpaca (`format_input`) | 85/10/5 train/test/validation in file order: 935/110/55 | `instruction-data.json`, LLMs-from-scratch `cdbd33e` |
| `math-distill-r1` | 12,000 | chapter 3 math prompt | 25 validation after `random.Random(123)` shuffle | `rasbt/math_distill` `f48286d`, `deepseek-r1-math-train.json`, Apache-2.0 |

Either dataset works with either model family; the prompt follows the dataset. Rules:

- **Stopping.** Every example ends with the tokenizer's end-of-text token, so the model
  learns to stop.
- **Prompt masking.** `mask_prompt: true` excludes prompt tokens from the loss (chapter 7's
  exercise 7.2, and chapter 8). Chapter 7 itself trains on the prompt too.
- **Length.** Examples longer than `max_length` tokens are dropped before splitting
  (chapter 8's filter), and rows with missing fields are skipped. `data-manifest.json`
  records both counts. At 2,048 tokens, `math-distill-r1` keeps 6,725 of 12,000 rows
  (6,700 train, 25 validation). The chapter's notebook keeps 6,695, because it wraps
  prompts in the reasoning tokenizer's chat template. This recipe follows the base-model
  defaults of the chapter's `distill.py`, which do not.
- **Batching.** Batches pad to the longest example. The tensors equal chapter 7's
  `custom_collate_fn`, and its exercise 7.2 masking variant, exactly.
- **Loss.** Ignored targets are `-100`. The loss and gradient of a masked example equal
  chapter 8's `compute_example_loss`. `tokens_seen` and validation token counts include only
  supervised tokens.
- **No accumulation.** `grad_accum_steps` must be 1. Batches hold different numbers of
  supervised tokens, so averaging micro-batch means would not give the token-weighted mean
  that accumulation promises.

`tests/unit/test_instructions.py` checks these against frozen copies of the chapters' code
(`tests/reference/instruction_baseline.py`). `tests/integration/test_finetune.py` checks
exact resume and the `model.pt` hand-off for both families on CPU.

## Using the result

```bash
# Chapter 7: responses to the 110 held-out instructions, one JSONL record each.
uv run llm-training respond --checkpoint runs/sft/<run>/model.pt \
  --config runs/sft/<run>/config.resolved.yaml --device mps

# Chapter 8: MATH-500 accuracy of a Qwen3 model.pt (or a full checkpoint), same scorer as GRPO.
uv run llm-training math-eval --model runs/sft/<run>/model.pt --limit 100

# Free-form text; the tokenizer defaults to the checkpoint's family.
uv run llm-training generate --checkpoint runs/cpt/<run>/model.pt --prompt "Once upon a time"
```

`respond` follows chapter 7. It decodes greedily from `format_input` until end-of-text,
writes the model's response next to the reference, and skips recorded examples when rerun.
Chapter 7 then has a larger model grade the responses; that grading step is not built
([backlog](backlog.md)).

## Tokenizers

GPT-2 uses tiktoken. Qwen3 uses its Hugging Face `tokenizers` file behind
`Qwen3TextTokenizer`, which offers the same few methods the trainer, corpus builder and
`predict` call. Raw text is encoded literally: a string like `<|im_start|>` inside a document
becomes ordinary tokens rather than the control token, as tiktoken's `encode_ordinary` does.
`data.tokenizer_cache` is the directory holding the family's tokenizer assets:
`data/tokenizer` for GPT-2, and the Qwen3 weights directory for Qwen3.

## What is not built

See [the backlog](backlog.md).
