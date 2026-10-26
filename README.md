# LLM training

A Python package built from the explicit GPT implementations in Sebastian Raschka's
[LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch).

The initial package slice is implemented: explicit causal attention/GPT modules,
GPT-2 tokenization, next-token datasets, a resumable FP32 training CLI, validation
metrics, greedy samples, and reloadable model weights. The source checkout is not
needed to install or run this project.

The chapter-sized reproduction and checkpoint recovery workflows are implemented.
See [the implementation plan](IMPLEMENTATION_PLAN.md) and
[reproduction evidence](docs/experiments/chapter-reproduction.md).
For the model's structure, tensor flow, and preset dimensions, see the
[model architecture guide](docs/model-architecture.md).
Prepared `uint16` token files exist for registered corpora (TinyStories); AMP/`torch.compile`
and distributed training remain future work.
The runtime is single-process FP32 with `num_workers=0`.

Two recipes share the trainer. `configs/pretrain_gpt_162m.yaml` reproduces the book's
chapter-5 loop (PyTorch init, untied head, context 256, constant LR, no clipping) and is
frozen as the source-parity reference. `configs/pretrain_gpt2_124m.yaml` is the GPT-2 small
recipe (see below). A corpus, token shards, bf16/compile, and a measured budget are still
missing, so **no GPT-2 pretraining run has been done**. The
[GPT-2 parity gap register](docs/gpt2-parity-gaps.md) tracks what is built and what is not.

## Quick start

Use Python **3.14.9**, pinned in `.python-version`. Install that interpreter with
your preferred Python manager, then:

```bash
uv sync --frozen
uv run llm-training inspect
uv run llm-training prepare --config configs/smoke.yaml
uv run llm-training train --config configs/smoke.yaml
```

If your `uv` download catalog does not yet include 3.14.9 but it is installed with
`pyenv`, create the environment explicitly before syncing:

```bash
uv venv --python "$(pyenv root)/versions/3.14.9/bin/python3.14"
uv sync --frozen
```

`prepare` acquires the two GPT-2 tokenizer assets into `data/tokenizer` and records
their hashes. The smoke corpus is a synthetic text fixture already in this
repository. Once dependencies and tokenizer assets are prepared, `train` reads
only local data. Paths inside a config resolve relative to that config file.

The CPU smoke preset has 3,242,240 parameters and stops after four optimizer
updates (128 target-token presentations). It is an execution check; its repetitive
synthetic corpus is unsuitable for measuring language capability.

## Run artifacts

The CLI prints a unique directory under `runs/`. It contains:

- `config.resolved.yaml`, `environment.json`, and `provenance.json` for settings,
  runtime versions, source lineage, and config/lock hashes.
- `data-manifest.json` for input bytes, tokenizer hashes, and split identity.
- `metrics.jsonl` for per-update training loss and periodic validation metrics.
- `evaluation.json` for initial/final validation on the first `eval_iter` batches.
- `model.pt` for portable model weights/config, `sample.json` for greedy output,
  and `status.json` for completion/failure information.
- `checkpoints/last.pt` for full-state recovery and `checkpoints/best.pt` for the
  best monitored validation checkpoint. `epoch-samples.jsonl` records end-of-epoch
  samples and their epoch/update identities.

Run artifacts and downloaded data are ignored by Git. Evaluation reports
token-weighted cross-entropy and perplexity, with dropout disabled. Periodic
monitoring uses a declared subset; completed runs additionally evaluate the full
validation split. The chapter's tiny corpus does not constitute a broad benchmark.

## Inspect next-token predictions

Open the [saved eight-step view](docs/experiments/next-token-predictions.md) for
the chapter-trained model, or inspect any saved model with:

```bash
uv run llm-training predict \
  --checkpoint runs/<run-id>/model.pt \
  --prompt "Every effort moves you" \
  --max-new-tokens 8 --top-k 5 \
  --output runs/<run-id>/next-token-predictions.json
```

Replace `<run-id>` with the training directory printed by the CLI. The command
prints Markdown tables to the terminal: each step shows its current context,
ranked token IDs/text, probabilities, the selected token, and the probability
mass of undisplayed tokens. Redirect stdout to a `.md` file to keep that view.
`--output` saves exact probabilities, token bytes/IDs, context-window truncation,
checkpoint SHA256, and runtime/model identity as JSON.

Probabilities are the full-vocabulary softmax at temperature 1. `--top-k` only
limits displayed rows; it does not renormalize probabilities or filter generation.
The highest-logit token is appended before the next step, matching the original
greedy generator. Exact ties choose the lowest token ID. Leading spaces and
newlines are escaped; a token can contain punctuation, part of a word, or part of
a UTF-8 character. Exact bytes are retained when a token alone cannot decode.

Both `model.pt` and full `checkpoints/last.pt` / `checkpoints/best.pt` can be inspected
without restoring optimizer or RNG state. Existing trained weights remain usable
for this inference view. Inference defaults to CPU and the local `data/tokenizer`
cache; use `--tokenizer-cache`, `--device`, and `--num-threads` to override them.
Preparation must already have cached the GPT-2 assets; prediction works offline.
Held-out NLL/perplexity remain in the run's `evaluation.json`.

The [saved JSON trace](docs/experiments/next-token-predictions.json) records the
same eight-step example. For its first step, the model assigns about 22.89% to
the punctuation token `?"` and 18.50% to ` know`; it appends `?"`.

## Stop and resume

```bash
uv run llm-training train --config configs/smoke.yaml --stop-after 2
uv run llm-training train --resume runs/<run-id>/checkpoints/last.pt
```

`--stop-after` is an **absolute completed-update count**, independent of the recipe's
total budget. The example stops after update 2, then resumes updates 3–4. SIGINT
(Ctrl-C) and SIGTERM also request a checkpoint after the current safe update
boundary. Inspect `status.json`: a planned/signal stop is `interrupted`, not
`completed`. SIGKILL/power loss can recover only from the last successful checkpoint.

Full checkpoints save weights, AdamW moments/step counters, Python/NumPy/Torch RNG
states, epoch-start RNG and batch cursor, metrics/samples, the best validation
metric/update, environment, resolved recipe, and data/tokenizer identity. The
scheduler field is null for the constant book schedule and records the schedule name and
last update for cosine/warmup runs (the rate itself is recomputed from the update index);
there is no AMP scaler yet, so that field is null. Files are written to a temporary path, flushed, then atomically
replaced. The previous `last.pt` survives a failed write.

Resume starts a **new child run**, records the parent checkpoint, and carries
forward the checkpoint's metric/sample history. It replays deterministic data
access to reconstruct the iterator, then restores the saved live RNG before the
next update. No previous optimizer updates are repeated. Checkpoint history is
authoritative if an abrupt failure left newer, uncheckpointed log lines behind.

Use `--resume` without `--config`: the original recipe and total budget are
restored, and completed budgets cannot be extended through resume. Changes to
data, tokenizer assets, Python/Torch/dependency versions, device, thread count, or
numerical settings are rejected. A change to executable training code (model, data, trainer;
not comments, docstrings, formatting, the CLI, or `predict`) is rejected too, unless you pass
`--allow-code-drift`; the child run's `provenance.json` records that the override was used. Exact recovery is tested
on CPU FP32 in the recorded environment. On MPS, resume was checked on the TinyStories preset and
matches an uninterrupted run to float32 rounding (about 1e-7), not bitwise, because MPS kernels are
not bitwise deterministic. CUDA recovery is not yet validated.
`model.pt` and legacy weights files cannot resume training.

Keep the parent run when resuming: the child inherits `best.pt` when the matching
parent best checkpoint is available. A standalone `last.pt` still suffices for
training recovery, but cannot recreate an earlier best model if those weights
were not retained. Best selection uses periodic monitoring; the final full-split
report evaluates the last model rather than selecting on that report.

## GPT-2 small recipe

`configs/pretrain_gpt2_124m.yaml` is GPT-2 small (**124,439,808 parameters**: tied head,
QKV bias, context 1024) with nanoGPT's optimization recipe. Each piece is a config field
whose default keeps the book behavior:

| Field | GPT-2 recipe | Book default |
| --- | --- | --- |
| `model.init`, `residual_scaled_init` | `gpt2`, `true` | `pytorch` |
| `model.tie_embeddings` | `true` | `false` |
| `model.attn_impl` | `sdpa` (fused) | `explicit` |
| `training.weight_decay_policy` | `matrices` (no decay on biases/LayerNorm) | `all` |
| `training.beta1`, `beta2` | 0.9, 0.95 | 0.9, 0.999 |
| `training.lr_schedule`, `warmup_steps`, `min_lr` | cosine, linear warmup | constant |
| `training.grad_clip` | 1.0 (global norm) | none |
| `training.grad_accum_steps` | micro-batches per update | 1 |

`batch_size` is the micro-batch; one optimizer update averages `grad_accum_steps` of them, so
tokens per update are `batch_size x context_length x grad_accum_steps`. A final partial group at
the end of an epoch still updates, averaged over the micro-batches it has. The learning rate is
a pure function of the update index, so resume does not depend on optimizer state. Metrics now
include the pre-clip `grad_norm`.

The preset's **data path, tokens per update (32,768, against nanoGPT's ~0.5M scale), warmup
length, token budget, and evaluation cadence are provisional**: no corpus is selected yet, so `prepare` reports the data file missing. The
model/recipe pieces are verified by unit tests, an exact fresh-process resume test with every
feature active, an accumulation-equals-full-batch test, and the pretrained-weight oracle in
the [architecture guide](docs/model-architecture.md#gpt-2-initialization-and-weights).

## TinyStories run (readable output in about a day)

`configs/tinystories_51m.yaml` trains a **51,213,824-parameter** GPT (8 layers, width 512,
8 heads, context 512, the GPT-2 recipe above) on
[TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) using the Mac GPU
(`device: mps`). The goal is output that reads as simple English, not GPT-2 quality.

```bash
uv run llm-training prepare --config configs/tinystories_51m.yaml   # ~1.9 GB download, once
uv run llm-training train --config configs/tinystories_51m.yaml
uv run llm-training generate --checkpoint runs/tinystories/<run-id>/checkpoints/last.pt \
  --prompt "Once upon a time" --temperature 0.8 --top-k 50 --num-samples 3
```

- **Data.** `data.format: tokens` names a registered corpus (`data/corpus.py`) instead of a
  text file. `prepare` downloads the original release (`TinyStories-train.txt` and
  `TinyStories-valid.txt`) at a pinned Hub revision, verifies each file's SHA-256, and writes
  `train.bin` / `val.bin` as `uint16` token ids (463,833,291 training tokens in 2,119,489
  stories; 4,667,152 validation tokens). Every story starts with `<|endoftext|>`. Windows are
  cut from the concatenated stream, so attention can cross a story boundary at that token.
  The corpus's own validation file is the validation split.
- **Loading.** Training memory-maps the token files: nothing is re-tokenized on start or resume,
  and memory does not grow with corpus size. Each start re-hashes the token files against the
  manifest (about a second).
- **Samples.** `output.sample_every` adds a sampled continuation to every Nth update's record in
  `metrics.jsonl`, using `output.temperature` / `output.top_k`. Sampling draws from a private
  generator seeded by (seed, update), so samples never change training. A resumed run records
  the same text on CPU (tested bitwise); on MPS the ~1e-7 weight drift could in principle flip a
  draw (the 6-update MPS check matched).
- **`generate`** samples from any `model.pt` or full checkpoint (temperature 0 is greedy) and
  stops at `<|endoftext|>`. Use `--device cpu` (the default) during training so it does not
  compete for the GPU.

Measured on an M2 Max (FP32): **4.41 s per update** (32,768 tokens, about 7,430 tokens/s) over
the first 60 updates, so the 14,155-update epoch takes about 17–18 hours. Stop and resume work
on MPS, but not bitwise: MPS kernels are not bitwise deterministic. Two identical fresh runs
differ by up to 1.2e-7 in the weights, and a resumed run differs from an uninterrupted one by
the same amount, with identical batches.

## GPT 162M pretraining preset

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

## Verification

```bash
uv run ruff check .
uv run ruff format --check .
uv run pytest tests/unit -q
uv run python -m tests.reference.compare
LLM_TEST_TOKENIZER_CACHE="$PWD/data/tokenizer" uv run pytest -q
uv build
# Opt-in: compare against the public GPT-2 124M weights (downloads ~550 MB once)
# and run Qwen3-0.6B against Hugging Face Transformers (~1.5 GB once; see docs/qwen3.md).
uv sync --group oracle && LLM_ORACLE=1 uv run pytest tests/oracle -q
```

Integration checks require prepared tokenizer assets. Without the environment
variable they are explicitly skipped. They cover offline training/reload, repeated
fresh-process recovery within/across epochs, identical optimizer/RNG/batch history,
SIGTERM, invalid resume inputs, and a small multi-epoch source-loop comparison.
Unit checks use synthetic tokens and require no asset downloads.
Prediction checks cover known full-vocabulary probabilities, greedy parity,
ties, truncation, token-byte rendering, invalid inputs, and offline CLI inference
from both model weights and full checkpoints.

The numerical comparison uses frozen upstream definitions, identical weights and
batches, nonzero dropout with matching RNG state, and one AdamW update. It compares
logits, loss, gradients, and updated state at `atol=1e-6`, `rtol=1e-5`.
See [initial evidence](docs/experiments/initial-baseline.md) for the original slice
and [chapter/resume evidence](docs/experiments/chapter-reproduction.md) for this extension.

## Qwen3-0.6B base model

`llm_training.model.qwen3` ports Qwen3 from Raschka's *reasoning-from-scratch* as separate
modules alongside the GPT ones. It loads the pinned Qwen3-0.6B base weights and tokenizer,
which an oracle test checks against Hugging Face Transformers. Training and CLI integration
are not wired up yet. See [the Qwen3 guide](docs/qwen3.md).

## Source and license

See [upstream provenance](docs/upstream-provenance.md) for the source commit,
symbol mapping, changes, and hashes. Derived code retains Sebastian Raschka's
attribution and the [upstream license as shipped](LICENSE.txt). The Qwen3 modules ship
*reasoning-from-scratch*'s license, which is plain Apache 2.0 without the book exclusions,
as [`model/qwen3/LICENSE.txt`](src/llm_training/model/qwen3/LICENSE.txt). The package does
not depend on Jupyter, TensorFlow, or a sibling source directory.
