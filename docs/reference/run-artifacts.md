# Run artifacts

What a next-token run (`llm-training train`) writes. GRPO runs have their own layout, listed in
[the GRPO guide](../grpo.md#run-directory).

## Run directory

The CLI prints a unique directory under `runs/`. It contains:

- `config.resolved.yaml`, `environment.json`, and `provenance.json` for settings,
  runtime versions, source lineage, and config/lock hashes.
- `data-manifest.json` for input bytes, tokenizer hashes, and split identity.
- `metrics.jsonl` for per-update training loss and periodic validation metrics (fields
  [below](#metricsjsonl)).
- `evaluation.json` for initial/final validation on the first `eval_iter` batches.
- `model.pt` for portable model weights/config, `sample.json` for a final sample (greedy
  unless `output.temperature` is above 0), and `status.json` for completion/failure
  information. `evaluation.json`, `model.pt` and `sample.json` are written only when the run
  completes its budget, not when it stops early or fails.
- `checkpoints/last.pt` for full-state recovery and `checkpoints/best.pt` for the
  best monitored validation checkpoint. `epoch-samples.jsonl` records end-of-epoch
  samples and their epoch/update identities.

Run artifacts and downloaded data are ignored by Git. Evaluation reports
token-weighted cross-entropy and perplexity, with dropout disabled. Periodic
monitoring uses a declared subset; completed runs additionally evaluate the full
validation split. The chapter's tiny corpus does not constitute a broad benchmark.
How to read each number is in [the training loop guide](../training-loop.md#evaluation-metrics).

## `metrics.jsonl`

One JSON line per update:

| Field | Meaning |
| --- | --- |
| `optimizer_step` | completed updates, counting this one (1-based) |
| `epoch` | epoch this update belongs to (1-based) |
| `train_loss` | mean of the group's micro-batch losses, measured before this update's step |
| `tokens_seen` | cumulative supervised target tokens (targets that aren't `-100`) |
| `learning_rate` | rate this update used |
| `grad_norm` | global gradient norm before clipping |
| `batch_sha256` | hash of the group's input tokens, which the resume tests compare |
| `validation` | on monitored updates under `validation_only`: `loss`, `perplexity`, `tokens` |
| `source_train_loss`, `source_val_loss` | on monitored updates under `sampled_train_validation` |
| `sample` | on sampled updates: the generated text |

## Checkpoint contents

Full checkpoints save weights, AdamW moments/step counters, Python/NumPy/Torch RNG
states, epoch-start RNG and batch cursor, metrics/samples, the best validation
metric/update, environment, resolved recipe, and data/tokenizer identity. The
scheduler field is null for the constant book schedule and records the schedule name and
last update for cosine/warmup runs (the rate itself is recomputed from the update index);
there is no AMP scaler yet, so that field is null. Files are written to a temporary path, flushed, then atomically
replaced. The previous `last.pt` survives a failed write.

`model.pt` holds the weights, the model config, the family and the tokenizer identity. It is
the hand-off to `init_from`, `predict` and `generate`, and it cannot resume training.
