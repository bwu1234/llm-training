# Stop and resume a training run

```bash
uv run llm-training train --config configs/smoke.yaml --stop-after 2
uv run llm-training train --resume runs/<run-id>/checkpoints/last.pt
```

`--stop-after` is an **absolute completed-update count**, independent of the recipe's
total budget. It is a pause: the cosine schedule still spans the full budget, so a run never
resumed ends at a high learning rate. To end a run early for good, set `training.max_steps`
before starting ([schedule details](../training-loop.md#learning-rate-schedule)). The example stops after update 2, then resumes updates 3–4. SIGINT
(Ctrl-C) and SIGTERM also request a checkpoint after the current safe update
boundary. Inspect `status.json`: a planned/signal stop is `interrupted`, not
`completed`. SIGKILL/power loss can recover only from the last successful checkpoint.

Use `--resume` without `--config`: the original recipe and total budget are
restored, and completed budgets cannot be extended through resume. Changes to
data, tokenizer assets, Python/Torch/dependency versions, device, thread count, or
numerical settings are rejected. A change to executable training code (model, data, trainer;
not comments, docstrings, formatting, the CLI, or `predict`) is rejected too, unless you pass
`--allow-code-drift`; the child run's `provenance.json` records that the override was used.
Only the code a run executes counts: `lm_training_code` in `lm/trainer.py` lists it, so
edits to the Qwen3 modules never block a GPT-2 resume. Exact recovery is tested
on CPU FP32 in the recorded environment. On MPS, resume was checked on the TinyStories preset and
matches an uninterrupted run to float32 rounding (about 1e-7), not bitwise, because MPS kernels are
not bitwise deterministic. CUDA recovery is not yet validated.
`model.pt` and legacy weights files cannot resume training.

Keep the parent run when resuming: the child inherits `best.pt` when the matching
parent best checkpoint is available. A standalone `last.pt` still suffices for
training recovery, but cannot recreate an earlier best model if those weights
were not retained. Best selection uses periodic monitoring; the final full-split
report evaluates the last model rather than selecting on that report.

What a checkpoint holds is in [Run artifacts](../reference/run-artifacts.md#checkpoint-contents),
and how resume rebuilds the run is in [the training loop guide](../training-loop.md#resume).
GRPO runs stop and resume the same way ([GRPO commands](run-grpo.md)).
