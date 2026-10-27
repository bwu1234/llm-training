# LLM training

A Python package built from the explicit GPT implementations in Sebastian Raschka's
[LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch), plus the Qwen3 model and
GRPO reasoning fine-tune from his
[reasoning-from-scratch](https://github.com/rasbt/reasoning-from-scratch).

What it does, all through one `llm-training` CLI:

- **Pretraining** a GPT model from random weights, with the book's chapter-5 recipe or the
  GPT-2 recipe.
- **Continued pretraining and SFT** from released GPT-2 or Qwen3-0.6B weights.
- **GRPO** on MATH for Qwen3-0.6B, and MATH-500 evaluation.
- **Inspecting a model:** next-token predictions, sampled text, and responses to held-out
  instructions.

Training is resumable. It runs single-process in FP32 with `num_workers=0`, on CPU or the
Mac GPU (MPS). The source checkout is not needed to install or run this project.

Two full-length runs are complete: the chapter-5 reproduction
([evidence](docs/experiments/chapter-reproduction.md)) and the 51M-parameter TinyStories
model (below). For each model's structure, tensor flow and preset dimensions, see the
[GPT-2 guide](docs/gpt2.md) and the [Qwen3 guide](docs/qwen3.md). To explore them side by
side, open the [architecture explorer](https://bwu1234.github.io/llm-training/): every module that runs, its shapes,
parameters and source, traced from the code (see below).
**Pending runs, unbuilt work, open decisions and next steps are in
[the backlog](docs/backlog.md).** [`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md) is the
original plan, kept for history.

Two recipes share the trainer. `configs/pretrain_gpt_162m.yaml` reproduces the book's
chapter-5 loop (PyTorch init, untied head, context 256, constant LR, no clipping) and is
frozen as the source-parity reference. `configs/pretrain_gpt2_124m.yaml` is the GPT-2 small
recipe (see below). **No GPT-2 pretraining run has been done**; it waits on a corpus and a
throughput pilot (see the backlog). The [GPT-2 parity gap register](docs/gpt2-parity-gaps.md)
specifies each gap and records what was built.

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
for this inference view. Inference defaults to CPU and the checkpoint family's local
tokenizer (`data/tokenizer` for GPT-2, `data/qwen3-0.6b-base` for Qwen3); use
`--tokenizer-cache`, `--device`, and `--num-threads` to override them. Preparation must
already have cached the tokenizer assets; prediction works offline.
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
the [GPT-2 guide](docs/gpt2.md#gpt-2-initialization-and-weights).

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
which an oracle test checks against Hugging Face Transformers. See
[the Qwen3 guide](docs/qwen3.md).

## Adding a model family

The next-token trainer, `prepare`, `predict`, `generate` and `respond` never branch on a family
name. Each family describes itself once, as a `Family` (`model/family.py`) in
`model/<name>/family.py`: its recipe settings and their validation, how to build the model and
load weights into it, its tokenizer, its pinned releases, and any extra package versions resume
must match. To add one, write `model/<name>/` with that file, then add the name to `FAMILIES`
in `model/registry.py` and its releases to `PRETRAINED`. `tests/unit/test_families.py` checks
the declaration, and `tests/unit/test_fingerprint_scope.py` checks that the resume fingerprint
covers the family's code and nothing from another family. GRPO (`reasoning/`) is built on
Qwen3 directly and is not covered by this. To show the new family in the architecture
explorer, add a preset and notes for its module classes to `docs/architecture/explainer.json`
(below).

## Architecture explorer

[`docs/architecture/index.html`](docs/architecture/index.html) puts two presets side by side,
such as GPT-2 small and Qwen3-0.6B. Each row is something that runs in the forward pass, in the
order it runs, with its output shape (`B` batch, `T` tokens):

- a module, with its parameter count;
- a call to one of the package's functions, such as RoPE's `apply_rope`;
- a tensor operation that the repo's own code performs: residual additions, the head split,
  the causal mask, softmax, the SwiGLU product, each step of RoPE and of the norms.

Every row is named for what it does: "QK-norm, keys", "Residual connection", "Split queries
into heads", "Rotate each channel pair". Buffers such as RoPE's `cos` and `sin` tables are
listed under the module that stores them, with links to the steps that read them. Selecting a
row shows what it does and its source, with the line it runs highlighted, and its data flow:
where each input came from and what reads its output, marked in the tree. That shows, for
example, that Qwen3 adds nothing to its token embedding: the causal mask and the RoPE tables
go to every block as separate arguments. A tensor that a preset builds or stores but never
computes with is tagged "not read here": under `attn_impl: sdpa`, `is_causal=True` masks
inside the fused kernel, so GPT-2's `mask` buffer and Qwen3's built mask go unread. A
selected module also highlights the module at the same path in the other model. The
`attn_impl: explicit` presets show attention's scores, mask, scaling and softmax as separate
steps; with `sdpa` they are one fused call.

It is published at <https://bwu1234.github.io/llm-training/>. To view a local copy, open the file in a browser; it needs
no server. Locally its source links open the files in your checkout. On the published page
they open the files on GitHub at the commit it was built from, so line numbers match.
The `pages` job in `.github/workflows/ci.yml` deploys it after every push to `main` once the
checks pass, so a stale `spec.js` can't be published. It publishes `index.html`,
`spec.js` and `LICENSES/`, plus a one-line `site.js` that sets the GitHub link root. Nothing
else from `docs/` goes online.

The page's data, `docs/architecture/spec.js`, is generated from the models themselves:

```bash
uv run python -m llm_training.explain
```

`llm_training.explain` builds each preset listed in `docs/architecture/explainer.json` on
PyTorch's `meta` device (shapes without storage, so Qwen3 needs no 2.4 GB allocation). It runs
one forward pass and records, with input and output shapes:

- module calls, with module hooks;
- calls to the package's own functions, with Python's `sys.monitoring`;
- every tensor operation the package's code performs, with PyTorch's `TorchFunctionMode`.
  Operations inside a PyTorch module such as `nn.Linear` belong to that module's row;
- for each of those, which earlier node produced each tensor it reads.

A `TorchDispatchMode` sees every kernel that runs and fails the generation if one ran outside
both a recorded operation and a PyTorch module, so a step can't be dropped silently. A second
pass with a different batch and length names the dimensions that changed. The prose for each
module class, function, operation and buffer, and the name of each module and source line,
live in `explainer.json`, so `src/` holds no family-specific text. `tests/unit/test_explain.py` fails when:

- `spec.js` is stale, for example after an edit to a model's `forward`;
- a documented parameter count changes;
- a module class, function, operation or buffer has no note, a module or step has no name,
  or a note or name refers to something that no longer runs;
- folding the repeated blocks loses or adds a step.

The explainer is outside every resume fingerprint, so changing it never affects
`train --resume`.

## Continued pretraining and SFT from released weights

`model.init_from` starts the same trainer from OpenAI's GPT-2 small or medium, Qwen3-0.6B
base, or any finished run's `model.pt`. Raw text continues pretraining; `data.format:
instructions` is supervised fine-tuning, following chapter 7 of *Build a Large Language
Model (From Scratch)* and chapter 8 of *Build a Reasoning Model (From Scratch)*.

```bash
uv run llm-training prepare --config configs/sft_gpt2_medium_alpaca.yaml   # ~1.7 GB once
uv run llm-training train --config configs/sft_gpt2_medium_alpaca.yaml     # ~8-10 min on MPS
uv run llm-training respond --checkpoint runs/sft/<run>/model.pt \
  --config runs/sft/<run>/config.resolved.yaml --device mps
```

`configs/sft_qwen3_0_6b_math_distill.yaml` is chapter 8's reasoning distillation (an
estimated 18–22 hours). The two `cpt_*_tinystories.yaml` recipes are plumbing tests only: a
long run on children's stories would erode the models' general skills. See
[the fine-tuning guide](docs/finetuning.md) for measured costs and the book parity checks.

## Reasoning fine-tune: GRPO on MATH

`llm_training.reasoning` ports chapter 6 of *Build a Reasoning Model (From Scratch)*: GRPO
with verifiable rewards turns the Qwen3-0.6B base model into a step-by-step math solver. The
book reports MATH-500 going from 15.2% to 47.4% after 50 steps.

```bash
uv run llm-training grpo-prepare --config configs/grpo_qwen3_0_6b_math.yaml
uv run llm-training grpo-train --config configs/grpo_qwen3_0_6b_math.yaml
uv run llm-training math-eval --snapshot runs/grpo/<run>/snapshots/step00050.pt --limit 100
```

On an M2 Max the 100-step recipe takes an estimated 1.5–3 hours (a step is 31 s measured
with short answers, about 1.8 minutes when all rollouts reach 512 tokens), and a full
MATH-500 evaluation of a trained model about 2–2.5 hours. Longer GRPO training can lower the score, so score the saved
snapshots and keep the best. See [the GRPO guide](docs/grpo.md).

## Source and license

See [upstream provenance](docs/upstream-provenance.md) for the source commit,
symbol mapping, changes, and hashes. Derived code retains Sebastian Raschka's
attribution, and each upstream repository's license is shipped unchanged under
[`LICENSES/`](LICENSES):

- [`LLMs-from-scratch.txt`](LICENSES/LLMs-from-scratch.txt) covers the GPT-2 model, the
  next-token data, loss, monitoring and generation code, and the chapter 7 instruction format.
  It is Apache 2.0 with the books and their images excluded from "Source".
- [`reasoning-from-scratch.txt`](LICENSES/reasoning-from-scratch.txt) covers the Qwen3 model
  and the GRPO, grading and distillation code in `reasoning/`. It is plain Apache 2.0.

Each derived file's header names its license. Both files are installed with the wheel. The
package does not depend on Jupyter, TensorFlow, or a sibling source directory.
