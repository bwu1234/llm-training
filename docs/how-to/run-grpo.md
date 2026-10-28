# Run GRPO on MATH and score it

`llm_training.reasoning` ports chapter 6 of *Build a Reasoning Model (From Scratch)*: GRPO
with verifiable rewards turns the Qwen3-0.6B base model into a step-by-step math solver. The
book reports MATH-500 going from 15.2% to 47.4% after 50 steps. How it works and how this
port differs from the book are in [the GRPO guide](../grpo.md).

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

Measured times are in [the GRPO guide](../grpo.md#cost-on-an-m2-max-mps). Longer GRPO
training can lower the score, so score the saved snapshots and keep the best.

If `reasoning-from-scratch/qwen3` already holds the pinned files, `cp -c` them into
`data/qwen3-0.6b-base` (an APFS clone: instant, no extra disk) and `grpo-prepare` only
verifies them.

`grpo-train --stop-after N` stops after step N. SIGINT/SIGTERM stop after the current step.
Either way, `--resume runs/grpo/<run>/checkpoints/last.pt` continues in a child run, as the GPT
trainer does. `math-eval` appends one JSONL record per problem and skips recorded problems on
a rerun, so an interrupted evaluation continues where it stopped.
