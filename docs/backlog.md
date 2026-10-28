# Backlog

Last updated 2026-10-28. **This is the one place that tracks pending work:** runs that are
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
- **Evaluation:** E2 token-weighted monitoring, E3 benchmarks (blocked on D1, except CORE,
  which doesn't depend on the corpus; see [nanochat](#techniques-from-nanochat)), E4
  reference-loss calibration, E5 `evaluate` and `export` commands.
- **Operations:**
  - F3 checkpoint retention (a Qwen3 `last.pt` is ~9 GB, a GPT-2 one ~1.5 GB);
  - F5 relative paths in saved configs, F9 heartbeat file (hangs are currently invisible).
- **Hardware:** D3 DDP, D4 CUDA validation.
- **Tests that wait on those items:** a resume from a relocated checkpoint (F5), a resume with
  an active scaler (F7), and any accelerator run in CI (D4).

### Techniques from nanochat

Found by comparing this repo with
[karpathy/nanochat at `92d63d4`](https://github.com/karpathy/nanochat/tree/92d63d4e8bb4df75c3b71618f31ddde2378b2bcd)
on 2026-10-28. File references are to that commit. Items nanochat shares with ones already
listed above (bf16, `torch.compile`, KV cache and batched generation, DDP, fused AdamW, tok/s
and MFU, padded vocabulary, token-weighted monitoring, stateless sampler, corpus mixing) are
not repeated here.

**Evaluation**

| Item | What nanochat does | Why it matters | Notes |
| --- | --- | --- | --- |
| Bits per byte | `loss_eval.evaluate_bpb`: summed loss over the UTF-8 byte count of the targets, special and masked tokens excluded | Loss per token can't be compared across tokenizers ([caveats](training-loop.md#caveats-when-comparing)); bits per byte can, so GPT-2 and Qwen3 runs on the same text become comparable | Both tokenizers expose `decode_single_token_bytes`, so a byte-length table needs no family branching. Recommended as part of E5's `evaluate` command, in a module outside the resume hash |
| CORE benchmark | `core_eval.py`, `scripts/base_eval.evaluate_core`: the DCLM paper's 22 few-shot tasks, scored by lowest mean continuation loss (multiple choice, schema) or exact greedy match (language modeling), averaged as centered accuracy (0 = random, 1 = perfect) | A capability number for base models that doesn't depend on our corpus, so unlike the rest of E3 it isn't blocked on D1 | Score public GPT-2 124M first as the anchor (nanochat lists 0.2565 for GPT-2 1.6B). The eval bundle is downloaded from nanochat's S3 URL; size unchecked. The tasks keep their own licenses; check them before pinning the bundle |
| Fixed factual probe prompts | `base_train.py`: seven prompts ("The capital of France is", "If 5*x + 3 = 13, then x is", ...) decoded greedily for 16 tokens at each sample point | Shows when facts and simple reasoning appear during pretraining, which one story prompt doesn't | A list option beside `output.prompt`. Edits `lm/trainer.py`, so checkpoints need `--allow-code-drift` |
| ChatCORE during SFT | `chat_eval.run_categorical_eval` scores ARC and MMLU by comparing only the answer letters' logits; `run_generative_eval` grades GSM8K and HumanEval answers; both centered on the random baseline (`chat_sft.py`) | SFT is tracked only by validation loss; this measures whether instruction following transfers to tasks, without an external grader model | Letter-logit scoring is one forward pass per batch. The generative tasks need batched generation to be affordable on MPS |
| pass@k inside the GRPO loop | `chat_rl.run_gsm8k_eval`: every 60 steps, 8 samples at temperature 1.0 on 400 test problems, logging pass@1 through pass@8 | GRPO's loop runs no evaluation, so collapse shows only afterwards, and R4 spends about 2 h choosing a snapshot | Needs batched generation first. Use a held-out MATH subset, never MATH-500. Edits `reasoning/`, so GRPO checkpoints need `--allow-code-drift` |

**Pretraining**

| Item | What nanochat does | Why it matters | Notes |
| --- | --- | --- | --- |
| Warmup, constant, linear warmdown | `base_train.get_lr_multiplier`: linear warmup, flat, then a linear decay over the last `warmdown_ratio` (0.65) of updates to `final_lr_frac` (0.05) of the peak | An alternative to cosine that holds the peak rate longer | A third `lr_schedule` value in `lm/schedule.py`; still a pure function of the update index. Needs schedule-table tests and `--allow-code-drift` |
| Muon optimizer | `optim.py` `MuonAdamW`: Muon (orthogonalized momentum via Polar Express iterations, NorMuon variance reduction, cautious weight decay) for the blocks' matrices; AdamW with its own rate, betas and decay for each of embeddings, head and scalars | nanochat's main optimizer gain over AdamW | A large port behind an opt-in `optimizer` field. nanochat's `test_optim.py` needs a GPU; the exact-resume test must cover its state |
| Muon momentum and weight-decay schedules | `get_muon_momentum` (0.85 → 0.97 over 400 updates, → 0.90 during warmdown); `get_weight_decay` (cosine to 0) | Part of the Muon recipe | Only with Muon. Both must be pure functions of the update index |
| Hyperparameters derived from depth | `base_train.py` "Scaling laws": tokens = 12 × (matrix + head) parameters; batch ∝ D^0.383 from a d12 reference; rates ∝ √(B/B_ref), AdamW rates ∝ 1/√(d_model/768); weight decay scaled to keep T_epoch constant | One size setting gives a compute-matched recipe; overlaps C8 (token budget) | Tuned for nanochat's model and optimizer, so the constants don't transfer to GPT-2 with AdamW. The token-budget rule could inform C8 |
| Logit softcap | `GPT.forward`: `15 · tanh(logits / 15)` in float32 before the loss | Bounds the logits, which stabilizes training | A model option, so it lives in `model/gpt2/` and must default off for book parity |
| BOS-aligned best-fit packing | `dataloader.py`: each row starts at a document start and is filled with the largest buffered document that fits, cropping only when none does (~35% of tokens cropped at T=2048) | No token attends across a document boundary, unlike C5's current policy | A token-corpus option. Changes the batch stream, so resume and manifest tests change |

**Fine-tuning and RL**

| Item | What nanochat does | Why it matters | Notes |
| --- | --- | --- | --- |
| Packed SFT batches | `chat_sft.sft_data_generator_bos_bestfit`: conversations packed best-fit into full rows, the remainder padded and masked | Less padding than batching to the longest example | nanochat divides each micro-batch's loss by its own token count and then by `grad_accum`, which is not token-weighted. A port needs a summed loss divided by the group's supervised tokens, which would also lift the `grad_accum_steps: 1` rule |
| SFT task mixtures | `tasks/common.TaskMixture`: SmolTalk plus MMLU ×3 and GSM8K ×4 epochs, shuffled together | Several instruction datasets in one run, each weighted by repetition | Needs a list of datasets with epochs in the recipe and in the manifest |
| SFT optimizer warm-start | `chat_sft.py --load-optimizer`: loads the pretraining run's momentum buffers, then resets the learning rates | nanochat reports a fresh optimizer as "slightly worse" | Only for our own pretrain → SFT chain: released weights have no optimizer state, and `model.pt` is weights only |
| nanochat's GRPO variant | `chat_rl.py`: advantage = reward − mean (no division by the standard deviation); loss normalized by valid tokens (DAPO); 16 problems per step; learning rate ramped linearly to 0 | Our loss sums each rollout's log-probabilities, so longer answers carry more weight; one problem per step gives a very noisy gradient | Four separate opt-in fields; the current values reproduce the book's 47.4% run. More problems per step needs batched generation |
| A nanochat-style model family | `gpt.py`: value embeddings, smear, backout, per-layer residual and x0 scalars, relu² MLP, parameter-free RMSNorm, QK norm, zero-initialized projections, sliding-window attention | Its architecture gains, which `gpt2` options can't add without breaking GPT-2 | Would be a new `model/<name>/` family. Sliding windows need FA3; on SDPA nanochat warns that utilization is "terrible", so use an all-full-context pattern on MPS |

Constraints for all of them:

- **Book parity.** Every option defaults to today's behavior.
- **Resume hash.** Code under `lm/`, `data/`, `training/`, `model/<arch>/` or `reasoning/`
  makes existing checkpoints need `--allow-code-drift`. Evaluation commands can avoid that by
  living outside those paths.
- **License.** nanochat is MIT-licensed (`LICENSE`). Porting its code, rather than its ideas,
  adds the license to `LICENSES/` and the provenance records, as for Raschka's repositories.
- **Not applicable on MPS:** fp8 training (`fp8.py`, needs H100-class CUDA) and Flash
  Attention 3.

### Educational material

Graphics and guides that teach the two architectures from this repo's own code. Their
advantage over generic visualizers such as
[Transformer Explainer](https://poloclub.github.io/transformer-explainer/) is that every box
links to the module that implements it, and GPT-2 and Qwen3 sit side by side. The
architecture explorer (`docs/architecture/`, described in the README, live at
[GitHub Pages](https://bwu1234.github.io/llm-training/)) is built; these
items extend it.

| Item | Why it matters | Notes |
| --- | --- | --- |
| Live views on real weights | Attention heatmaps, logit lens, GQA head sharing, RoPE rotation | Uses the weights already in `data/`; could extend the explorer's traced pass with real inputs. Plotting or notebook dependencies go in an optional group, as `IMPLEMENTATION_PLAN.md` requires. Several days |
| Run viewer for GRPO | The next-token run viewer (`explain/runs.py`, see the README) rejects GRPO runs, so GRPO's reward, answer length and rollouts have no view | Plot `reward_mean`, `mean_gen_len`, `hit_token_limit` and `grad_norm` per step, mark `updated: false` steps, mark snapshots, and show each step's rollouts from `rollouts.jsonl`. About half a day |
| Run viewer: live refresh | The page is a snapshot of a running run | A `--watch` mode that rewrites the page as `metrics.jsonl` grows, or a page that rereads it. Browsers block `fetch` on `file://`, so rereading needs a local server. About half a day |
| Run viewer: smoothed training loss | One group per point makes `train_loss` noisy; a trend line is easier to read | A bias-corrected EMA (`ema / (1 - beta^(step+1))`, beta 0.9), as nanochat's `base_train.py` logs it. Computed in the page, not the trainer |
| Run viewer: publish an example | Local only by choice (2026-10-28). A downsampled copy of the TinyStories run on GitHub Pages would show it without a checkout | Publishing a run's metrics and samples is a decision for the user. About an hour |
| A guided learning path (`docs/learn.md`) | The guides say what each part does but not what order to read the code in | Tokenizer → embeddings → attention → block → training loop → GRPO, each linked to its book chapter and to the graphics. About half a day |

Constraints for all of them:

- **Keep the code outside the resume hash.** Put it in `src/llm_training/explain/` (where
  the explorer is) or in `scripts/`. Never put it in `model/<arch>/`, `lm/`, `data/` or
  `training/`, and don't add a describe method to `Family` or the models. Either would make
  every existing checkpoint need `--allow-code-drift`.
- **Introspect generically.** Use `named_modules()`, parameters and forward hooks. Branching
  on a family name would break the family rule.

### Deferred by choice

- Porting Qwen3.5-0.8B.

## 3. Known issues

- **Old runs with the `book` recipe.** Run configs saved before the `book` recipe was
  renamed `sampled_train_validation` (`runs/chapter-reproduction-20261025/…`) no longer
  validate, so `train --resume` rejects them. Their `model.pt` files still load in `predict`
  and `generate`, and the runs are complete. A fix would map the old name when loading
  saved configs.
- **`sample.json` mislabels greedy samples.** With `output.temperature: 0` and a `top_k`,
  `_sample` takes its sampling branch: the output is greedy, but it is recorded as
  `"method": "sampled"` with an unused seed. The fix is in `lm/trainer.py`, so checkpoints
  written after the last hashed change would also need `--allow-code-drift`. For text data,
  routing it to the greedy branch would also stop it ending early at end-of-text.
- **Exact resume on MPS** matches an uninterrupted run to float32 rounding (about 1e-7),
  not bit for bit.
- **Old checkpoints need `--allow-code-drift`.** Every checkpoint written before 2026-10-27
  (GPT-2, Qwen3 next-token and GRPO, including the finished TinyStories run) needs it to
  resume, because the code its resume hash covers has changed. The most recent change was
  renaming `training/optim.py` to `training/gradients.py`.

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
