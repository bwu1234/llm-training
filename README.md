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
([evidence](https://github.com/bwu1234/llm-training/blob/main/docs/experiments/chapter-reproduction.md)) and the 51M-parameter
[TinyStories model](https://github.com/bwu1234/llm-training/blob/main/docs/how-to/train-tinystories.md). The [documentation](#documentation)
below is grouped by what you need. **Pending runs, unbuilt work, open decisions and next steps
are in [the backlog](https://github.com/bwu1234/llm-training/blob/main/docs/backlog.md).**

Two recipes share the trainer. `configs/pretrain_gpt_162m.yaml` reproduces the book's
chapter-5 loop (PyTorch init, untied head, context 256, constant LR, no clipping) and is
frozen as the source-parity reference. `configs/pretrain_gpt2_124m.yaml` is the
[GPT-2 small recipe](https://github.com/bwu1234/llm-training/blob/main/docs/gpt2.md#gpt-2-small-recipe). **No GPT-2 pretraining run has been done**; it waits on a corpus and a
throughput pilot (see the backlog). The [GPT-2 parity gap register](https://github.com/bwu1234/llm-training/blob/main/docs/gpt2-parity-gaps.md)
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

## Documentation

The docs are organized along [Diátaxis](https://diataxis.fr/) into how-to guides, reference
and explanation, and the Quick start above is the starting point. What remains is in
[the backlog](https://github.com/bwu1234/llm-training/blob/main/docs/backlog.md#documentation).

**How-to guides**, for a task you already have in mind:

- [Stop and resume a training run](https://github.com/bwu1234/llm-training/blob/main/docs/how-to/stop-and-resume.md)
- [Train a model with readable output (TinyStories, about a day)](https://github.com/bwu1234/llm-training/blob/main/docs/how-to/train-tinystories.md)
- [Reproduce the book's chapter-5 run](https://github.com/bwu1234/llm-training/blob/main/docs/how-to/reproduce-chapter-5.md), and compare it with
  the upstream loop
- [Fine-tune from released weights (continued pretraining and SFT)](https://github.com/bwu1234/llm-training/blob/main/docs/how-to/fine-tune.md)
- [Run GRPO on MATH and score it](https://github.com/bwu1234/llm-training/blob/main/docs/how-to/run-grpo.md)
- [Inspect next-token predictions](https://github.com/bwu1234/llm-training/blob/main/docs/how-to/inspect-predictions.md)
- [View a run's metrics and samples](https://github.com/bwu1234/llm-training/blob/main/docs/how-to/view-a-run.md)
- [Run the checks](https://github.com/bwu1234/llm-training/blob/main/docs/how-to/run-the-checks.md)
- [Add a model family](https://github.com/bwu1234/llm-training/blob/main/docs/how-to/add-a-model-family.md)

**Reference**, to look something up:

- [Run artifacts](https://github.com/bwu1234/llm-training/blob/main/docs/reference/run-artifacts.md): the files a run writes, `metrics.jsonl`
  fields, and what a checkpoint holds

**Explanation**, for how and why things work:

- [GPT-2 model](https://github.com/bwu1234/llm-training/blob/main/docs/gpt2.md): structure, tensor flow, preset sizes, and the GPT-2 small recipe
- [Qwen3-0.6B base model](https://github.com/bwu1234/llm-training/blob/main/docs/qwen3.md): how it differs from GPT-2, weights, and verification
- [Training loop](https://github.com/bwu1234/llm-training/blob/main/docs/training-loop.md): what one update does, the schedule, resume, and how to
  read the evaluation metrics
- [Continued pretraining and SFT](https://github.com/bwu1234/llm-training/blob/main/docs/finetuning.md): starting weights, recipes and their
  measured costs, instruction data
- [GRPO](https://github.com/bwu1234/llm-training/blob/main/docs/grpo.md): the recipe, differences from the book, and measured costs
- [Architecture explorer](https://github.com/bwu1234/llm-training/blob/main/docs/architecture-explorer.md): the
  [interactive page](https://bwu1234.github.io/llm-training/) that shows every module and tensor
  step of both models, and how it is generated

**Project records**: [the backlog](https://github.com/bwu1234/llm-training/blob/main/docs/backlog.md) (the only tracker for pending work), the
[GPT-2 parity gap register](https://github.com/bwu1234/llm-training/blob/main/docs/gpt2-parity-gaps.md),
[upstream provenance](https://github.com/bwu1234/llm-training/blob/main/docs/upstream-provenance.md), the
[experiment reports](https://github.com/bwu1234/llm-training/tree/main/docs/experiments), and
[`IMPLEMENTATION_PLAN.md`](https://github.com/bwu1234/llm-training/blob/main/IMPLEMENTATION_PLAN.md), the original plan kept for history.

## Source and license

See [upstream provenance](https://github.com/bwu1234/llm-training/blob/main/docs/upstream-provenance.md) for the source commit,
symbol mapping, changes, and hashes. Derived code retains Sebastian Raschka's
attribution, and each upstream repository's license is shipped unchanged under
[`LICENSES/`](https://github.com/bwu1234/llm-training/tree/main/LICENSES):

- [`LLMs-from-scratch.txt`](https://github.com/bwu1234/llm-training/blob/main/LICENSES/LLMs-from-scratch.txt) covers the GPT-2 model, the
  next-token data, loss, monitoring and generation code, and the chapter 7 instruction format.
  It is Apache 2.0 with the books and their images excluded from "Source".
- [`reasoning-from-scratch.txt`](https://github.com/bwu1234/llm-training/blob/main/LICENSES/reasoning-from-scratch.txt) covers the Qwen3 model
  and the GRPO, grading and distillation code in `reasoning/`. It is plain Apache 2.0.

Each derived file's header names its license. Both files are installed with the wheel. The
package does not depend on Jupyter, TensorFlow, or a sibling source directory.
