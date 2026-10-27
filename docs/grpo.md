# GRPO: teaching Qwen3-0.6B base to reason on math

`llm_training.reasoning` ports chapter 6 of Sebastian Raschka's *Build a Reasoning Model (From
Scratch)*. It fine-tunes the Qwen3-0.6B **base** model with reinforcement learning and
verifiable rewards (RLVR), using GRPO without a KL term.

On each step the model writes several sampled answers (rollouts) to one MATH problem. An
answer whose last `\boxed{...}` grades equal to the reference earns reward 1, anything else 0.
Each rollout's advantage is its reward minus the group mean, divided by the group standard
deviation. The loss raises the log-probability of above-average rollouts and lowers it for
below-average ones. There is no supervised data and no reward model: the grader is the
teacher.

The book's results on MATH-500 (its numbers, from bfloat16 training; see below):

| Model | MATH-500 | Mean answer tokens |
| --- | --- | --- |
| Qwen3-0.6B base | 15.2% | 79 |
| Official Qwen3-0.6B reasoning model | 48.2% | 1370 |
| Base + 50 GRPO steps (8 rollouts, 512 tokens) | 47.4% | 586 |
| Base + 100 GRPO steps | 44.0% | 556 |

The last two rows show that more steps can make the model worse. Without a KL term nothing
holds the policy near the base model, and the book's chapter 7 reports one variant that falls
from 33.4% to 0.4% between steps 50 and 100. A rising training reward does not guarantee a
better test score. **Choose a snapshot by its evaluation score, not by recency.**

## Commands

```bash
# Pinned Qwen3-0.6B base files (~1.5 GB, skipped when already verified) and MATH (~11 MB).
uv run llm-training grpo-prepare --config configs/grpo_qwen3_0_6b_math.yaml

# Baseline: the untrained base model on the same evaluation.
uv run llm-training math-eval --base --output runs/grpo/base.math500-first500.jsonl

# Train. Each run gets runs/grpo/<timestamp>-<id>/.
uv run llm-training grpo-train --config configs/grpo_qwen3_0_6b_math.yaml

# Score snapshots on a fixed subset, then the best one on all 500 problems.
uv run llm-training math-eval --snapshot runs/grpo/<run>/snapshots/step00050.pt --limit 100
uv run llm-training math-eval --snapshot runs/grpo/<run>/snapshots/step00050.pt
```

If `reasoning-from-scratch/qwen3` already holds the pinned files, `cp -c` them into
`data/qwen3-0.6b-base` (an APFS clone: instant, no extra disk) and `grpo-prepare` only
verifies them.

`grpo-train --stop-after N` stops after step N. SIGINT/SIGTERM stop after the current step.
Either way, `--resume runs/grpo/<run>/checkpoints/last.pt` continues in a child run, as the GPT
trainer does. `math-eval` appends one JSONL record per problem and skips recorded problems on
a rerun, so an interrupted evaluation continues where it stopped.

## Recipe

`configs/grpo_qwen3_0_6b_math.yaml` is the run behind the book's 47.4% row
(`rlvr_grpo_original_no_kl.py --num_rollouts 8 --steps 100 --max_new_tokens 512`):

- one problem per step, in file order, from the 12,000 MATH problems that exclude MATH-500;
- 8 rollouts of up to 512 new tokens each, at temperature 0.8 and top-p 0.9;
- AdamW at lr 1e-5 with PyTorch's defaults (weight decay 0.01), gradient clipping at 1.0;
- no KL term and no reference model;
- separate (untied) output head and token embedding, as the book's `Qwen3Model` builds them.

When all rollouts earn the same reward (common early on, when every answer is wrong), the
gradient is zero but AdamW still steps: momentum carries on and weight decay applies. That is
upstream's default and the 47.4% run's behavior. `skip_zero_advantage_updates: true` skips
those steps instead, as upstream's `--skip-zero-advantage-updates` does.

Evaluation follows chapter 3: greedy decoding up to 2048 tokens, the same prompt, and the
answer taken from the last `\boxed{}` or else the last number in the text.

## Differences from the book

| | This repo | Book |
| --- | --- | --- |
| Embedding and linear weights, their AdamW state | float32 | bfloat16 (RMSNorm scales float32) |
| Attention | SDPA kernel | explicit |
| Rollout gradients | backpropagated one at a time | all rollouts' graphs held for one loss |
| Grading | worker process, 30 s limit per answer | inline |
| Data | pinned commit URLs, SHA-256 checked | `main` branch |
| Checkpoints | resumable full state + weights-only snapshots | weights only |

Precision is the difference that matters for results. In bfloat16, a weight near 0.02 has a
resolution of about 1e-4, so many lr-1e-5 AdamW updates round to nothing. In float32 every
update lands. The book's learning rate and step count may therefore act differently here. Use
this repo's own base-model evaluation as the baseline rather than the book's 15.2%, and watch
reward and answer length for collapse.

The other changes do not alter the math. A unit test checks that the per-rollout gradient
equals the frozen source's (`tests/unit/test_reasoning.py`). The grader limit exists because
an answer like `9^9^9^9` makes SymPy compute an enormous power, which runs indefinitely inline
(checked: still running after 20 s).

## Cost on an M2 Max (MPS)

Measured with the shipped recipe (float32, untied, SDPA):

| Operation | Measured |
| --- | --- |
| Top-p rollout sampling, 512 tokens, one sequence | 12.4 s (41 tokens/s); greedy decoding is about the same |
| Log-probability forward + backward on a 557-token rollout | 1.1 s |
| One full step, 8 rollouts averaging 133 tokens | 31 s; ~15 GB MPS driver memory afterwards |
| `checkpoints/last.pt` (weights + AdamW) / each snapshot | ~9 GB, overwritten / ~3 GB, kept |

Estimated from those rates (decoding dominates, so time scales with answer length):

| Operation | Estimate |
| --- | --- |
| One step with all 8 rollouts at the 512-token cap | about 1.8 min |
| 100 steps (answers lengthen as training goes on) | about 1.5–3 h |
| MATH-500, base model (answers average ~80 tokens) | about 20 min |
| MATH-500, trained model (~600 tokens) | about 2–2.5 h; a model that always runs to 2048 tokens, 7+ h |
| `--limit 100` evaluation of a trained snapshot | about 25–30 min |

Decoding is unbatched (one sequence at a time), as in the book's main chapter. Batched
rollouts and evaluation would be the main speedup and are not built.

## Run directory

| File | Contents |
| --- | --- |
| `config.resolved.yaml`, `environment.json`, `provenance.json` | recipe, versions, git state, pinned weights and data, fingerprint |
| `metrics.jsonl` | per step: rewards, loss, gradient norm, whether it updated, answer lengths, token-limit hits, grader issues, time |
| `rollouts.jsonl` | every rollout's text and reward, per step |
| `checkpoints/last.pt` | full resumable state |
| `snapshots/stepNNNNN.pt` | weights every `snapshot_every` steps and at the end |
| `status.json` | running, completed, interrupted, or failed |

The resume fingerprint covers only the code a GRPO run executes (`GRPO_CODE` in
`reasoning/trainer.py`), so edits to the GPT-2 code never block a GRPO resume, and the reverse.
