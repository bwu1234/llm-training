# Backlog

Last updated 2026-10-26. **This is the one place that tracks pending work:** runs that are
built but not yet launched, work that is not built, known issues, decisions that block other
work, and the recommended next steps. Other docs describe what exists and link here for
what doesn't. The linked guides hold the details, measurements and design, and the
[gap register](gpt2-parity-gaps.md) holds the GPT-2 items' specifications.

When an item lands, delete it here and describe it in its guide. When work is found or
deferred, add it here rather than to a guide.

## 1. Pending runs (built, awaiting a go-ahead)

Every runnable stage is implemented and smoke-tested, but none has been run at full length.
They share one GPU (MPS on a 64 GB M2 Max), so run them **one at a time**. Each one alone
peaks at 15–41 GB. Times are estimates from measured rates. For any run over a few
minutes, start it detached with a watcher that exits on completion, failure or process death.
A hang that produces no log output is not detected.

| # | Stage | Command / recipe | Estimated time | Peak memory | Output to check | Details |
| --- | --- | --- | --- | --- | --- | --- |
| R1 | Chapter 7 SFT: GPT-2 medium on the Alpaca data | `train --config configs/sft_gpt2_medium_alpaca.yaml`, then `respond` on the test split | 8–10 min + responses | 15 GB | validation loss (chapter 7 reports 1.103 at step 5); the 110 responses | [finetuning](finetuning.md) |
| R2 | GRPO baseline: base Qwen3-0.6B on all of MATH-500 | `math-eval --base --output runs/grpo/base.math500-first500.jsonl` | ~20 min | — | accuracy (book: 15.2%, from bfloat16) | [grpo](grpo.md) |
| R3 | GRPO training, 100 steps | `grpo-train --config configs/grpo_qwen3_0_6b_math.yaml` | 1.5–3 h | ~15 GB | reward and answer length per step; snapshots every 10 steps | [grpo](grpo.md) |
| R4 | GRPO snapshot selection | `math-eval --snapshot …/stepNNNNN.pt --limit 100` for steps 30, 50, 70, 100 | ~2 h | — | best snapshot by score, not the latest (book: 47.4% at step 50, 44.0% at step 100) | [grpo](grpo.md) |
| R5 | Score the best GRPO snapshot on all of MATH-500 | `math-eval --snapshot <best>` | 2–2.5 h | — | final accuracy | [grpo](grpo.md) |
| R6 | Chapter 8 SFT: distil DeepSeek-R1 math reasoning into Qwen3-0.6B | `train --config configs/sft_qwen3_0_6b_math_distill.yaml`, then `math-eval --model` | 18–22 h + evaluation | 41 GB | validation loss; MATH-500 accuracy against R2 | [finetuning](finetuning.md) |
| R7 | Continued-pretraining **plumbing** runs | `cpt_gpt2_small_tinystories.yaml` (~30 min), `cpt_qwen3_0_6b_tinystories.yaml` (~1.5 h) | 30 min / 1.5 h | 28 / 30 GB | that the path works end to end; not a useful model | [finetuning](finetuning.md#continued-pretraining-read-before-a-long-run) |

Notes:

- **GRPO is one approved unit.** R2–R5 were proposed together (about 6–8 h in total). The
  shorter variant runs only R2 and R3 first.
- **R7 is plumbing only.** Hours of continued pretraining on TinyStories would erode the
  models' general skills. A real continued-pretraining run waits on decision D2 below.
- **Not ready to run: a GPT-2 124M pretraining run.** It is blocked on the corpus decision
  (D1) and a throughput pilot (step 6 below).

## 2. Not built

### Fine-tuning and evaluation

Recipes and design: [finetuning](finetuning.md).

| Item | Why it matters | Notes |
| --- | --- | --- |
| Register a local corpus, and mix corpora | Real continued pretraining needs domain text plus general text to limit forgetting | `data/corpus.py` `CORPORA` only takes pinned remote files; needs a local-file entry type, its hashes, and mixing weights recorded in the manifest |
| Track forgetting during continued pretraining | Shows what continued pretraining costs on the old domain | Validation on a second, old-domain corpus at each evaluation |
| Score `respond` output | Chapter 7 grades responses with a larger model (Ollama, Llama 3) | Uses an external model, which may cost money or quota; ask before building it |
| GRPO starting from an SFT `model.pt` | Chains chapter 8 into chapter 6 (distil, then reinforce) | `grpo-train` accepts only the pinned base directory |
| Chapter 8's `--use_think_tokens` variant | The chapter's main notebook uses the reasoning tokenizer, chat template and `<think>` tags | It keeps 6,695 examples; the base-default variant built here keeps 6,725 |
| Training Qwen3 from random weights | An architecture comparison against the 51M GPT TinyStories model | Needs a Qwen3 initialization, and probably the GPT-2 tokenizer: Qwen3's vocabulary at width 512 is ~78M embedding parameters |
| bfloat16 training | Both books train in bfloat16; it lowers memory and may speed up training | Changes results, because many lr-1e-5 updates vanish in bfloat16. Make it a recorded runtime choice, not a silent default |
| Faster generation | `respond`, `generate`, MATH-500 evaluation and GRPO rollouts decode one sequence at a time; the next-token path also has no KV cache | Batched decoding is the main speedup for R2–R6 |

### GPT-2 pretraining track

The [gap register](gpt2-parity-gaps.md) keeps each item's evidence, fix and acceptance gate; this list is the only place
that tracks which are open. Partly done:

- B7 host syncs: one sync per micro-batch remains.
- B8 metrics: tokens/s, step time and memory are not logged.
- F7: the scaler state is still null.
- A10: sampling and `generate` exist; there is no KV cache.
- D4: MPS resume matches to ~1e-7; CUDA is untested.
- D6: TinyStories and fine-tuning rates are measured; there is no GPT-2 124M pilot.

Open:

- **Throughput and scale:**
  - D1 bf16 autocast, D2 `torch.compile`;
  - B8 timing metrics in a separate log that exact resume does not compare;
  - B9 fused AdamW;
  - A10 KV cache;
  - A7 non-persistent mask, A8 `F.layer_norm`, A9 padded vocabulary.
- **Data (blocked on D1):**
  - C1 corpus, C4 document-level split and deduplication;
  - C6 stateless sampler, C7 test split, C8 token budget.
- **Evaluation:** E2 token-weighted monitoring, E3 benchmarks (blocked on D1), E4
  reference-loss calibration, E5 `evaluate` and `export` commands.
- **Operations:**
  - F3 checkpoint retention (a Qwen3 `last.pt` is ~9 GB, a GPT-2 one ~1.5 GB);
  - F4 split `train()` into a `Trainer` (it now also handles instructions and initial
    weights);
  - F5 relative paths in saved configs, F9 heartbeat file (hangs are currently invisible).
- **Hardware:** D3 DDP, D4 CUDA validation.
- **Tests that wait on those items:** a resume from a relocated checkpoint (F5), a resume with
  an active scaler (F7), and any accelerator run in CI (D4).

### Repository structure

- **No remote.** All work is on `main` (renamed from `tinystories-run` on 2026-10-26).

### Deferred by choice

- Porting Qwen3.5-0.8B.

## 3. Known issues

- **Old runs with the `book` recipe.** Run configs saved before the `book` recipe was
  renamed `sampled_train_validation` (`runs/chapter-reproduction-20261025/…`) no longer
  validate, so `train --resume` rejects them. Their `model.pt` files still load in `predict`
  and `generate`, and the runs are complete. A fix would map the old name when loading
  saved configs.
- **Exact resume on MPS** matches an uninterrupted run to float32 rounding (about 1e-7),
  not bit for bit.
- **Old checkpoints need `--allow-code-drift`.** GPT-2 and GRPO checkpoints written before
  2026-10-26 need it to resume, because the code their resume hash covers has changed.

## 4. Open decisions

| # | Decision | Blocks |
| --- | --- | --- |
| D1 | Corpus for GPT-2 124M pretraining: license, size for the budget, a public loss number for pretrained GPT-2 on the same split | C1, C2, C4–C8, E3, the pretraining run |
| D2 | Domain for continued pretraining, and the general text to mix in | A real continued-pretraining run, corpus registration |
| D3 | Go/no-go for R1–R7 | Results for every built stage |
| D4 | float32 versus bfloat16 as the default for fine-tuning runs | Comparability with the books' numbers; memory headroom |

## 5. Recommended next steps

1. **R1, chapter 7 SFT on GPT-2 medium.** It is the cheapest end-to-end result (about 10
   minutes), and `respond` shows whether instruction following works.
2. **R2–R5, the GRPO sequence.** R2 gives the MATH-500 baseline that both R3 and R6 are
   measured against.
3. **Decide D2.** Then build local-corpus registration, mixing, and forgetting tracking
   before any long continued-pretraining run.
4. **Batched generation.** Do this before R6, if MATH-500 evaluations remain the
   bottleneck.
5. **R6, chapter 8 distillation.** It runs overnight; compare its MATH-500 score with R2
   and the GRPO result.
6. **GPT-2 pretraining track.** Run the throughput pilot with bf16/compile (D1/D2 in the
   register), add checkpoint retention (F3), then decide the corpus (D1 above).
