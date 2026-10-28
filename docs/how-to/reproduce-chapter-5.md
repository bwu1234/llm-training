# Reproduce the book's chapter-5 run (GPT 162M preset)

```bash
uv run llm-training prepare --config configs/pretrain_gpt_162m.yaml
uv run llm-training train --config configs/pretrain_gpt_162m.yaml
```

Preparation downloads `the-verdict.txt` from the pinned upstream commit and checks
its SHA256. Training uses the GPT model settings (12 layers, width 768,
context 256), ten epochs, batch size 2, AdamW learning rate 0.0005, and weight decay
0.1. The explicit untied output head gives **162,419,712 parameters**. On the pinned
text there are nine updates per epoch: **90 updates and 46,080 target-token
presentations** in ten epochs.

The preset matches the source script's hyperparameters and 90/10 character split,
then tokenizes each split separately. It preserves untied embeddings and the
explicit attention implementation. `training.recipe: sampled_train_validation` evaluates a shuffled training subset
and validation subset at zero-based updates 0, 5, 10, …, consuming RNG exactly as the source does,
and generates 50 greedy tokens after each epoch. `training.recipe: validation_only` monitors validation loss with isolated RNG
and omits epoch samples. Additional initial/final reports in either mode isolate RNG.
Use `max_steps` to bound a pilot. A document-level split is planned for larger
corpora; this source-compatible character split is only the reference path.

## Compare against the upstream loop

Run the complete independent comparison from the repository root after preparation:

```bash
uv run python -m scripts.compare_training_runs \
  --config configs/pretrain_gpt_162m.yaml \
  --output runs/chapter-reproduction-new \
  --stop-after 46
```

Use a new output directory. The workflow runs the frozen upstream loop in a
fresh process, runs the packaged trainer to update 46, resumes it in another
process through update 90, then compares final weights, optimizer/RNG state,
all monitored losses/token counts, and all ten epoch samples. It writes source
logs/metrics/weights, packaged run artifacts, and `comparison.json`. Declared
tolerances are `atol=1e-6`, `rtol=1e-5`. This verification script and frozen test
sources belong to the repository; the installed runtime wheel does not need them.

The recorded result is in the [chapter reproduction report](../experiments/chapter-reproduction.md).
