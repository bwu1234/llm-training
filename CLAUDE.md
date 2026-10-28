# CLAUDE.md

A Python package that trains GPT-style models with the implementations from Sebastian Raschka's
*LLMs-from-scratch* (GPT-2) and *reasoning-from-scratch* (Qwen3, GRPO). The README is the
landing page. User workflows are in `docs/how-to/` and run artifacts in
`docs/reference/run-artifacts.md`. Read them before changing CLI behavior. Measured costs live
in one place each: `docs/finetuning.md` (Recipes), `docs/grpo.md` (Cost on an M2 Max) and
`docs/how-to/train-tinystories.md`; other pages link to them.
`docs/backlog.md` is the **only** tracker for pending work, open decisions and known issues.

## Commands

Python 3.14.9 is pinned in `.python-version`, and `uv` manages everything.

```bash
uv sync --frozen                     # if uv lacks 3.14.9: uv venv --python "$(pyenv root)/versions/3.14.9/bin/python3.14" first
uv run ruff check .
uv run ruff format --check .         # CI enforces formatting; `uv run ruff format .` to fix
uv run pytest tests/unit -q          # ~6 s, synthetic tokens, no downloads
uv run pytest tests/unit/test_families.py -q   # one file
uv run python -m tests.reference.compare   # numerical parity against the frozen upstream code
uv run python -m llm_training.explain      # after editing model code or a preset recipe; the unit tests fail until spec.js is regenerated
```

The full suite, as CI runs it, needs the GPT-2 tokenizer prepared first:

```bash
uv run llm-training prepare --config configs/smoke.yaml
LLM_TEST_TOKENIZER_CACHE="$PWD/data/tokenizer" uv run pytest -q
```

**Integration tests skip, not fail, when their assets are missing.** A green run without
`LLM_TEST_TOKENIZER_CACHE` proves little, so check the `-ra` skip summary. Other gates:

- `LLM_QWEN3_DIR`: Qwen3 tokenizer tests. Defaults to `data/qwen3-0.6b-base`.
- `LLM_ORACLE=1`: tests in `tests/oracle/` against public weights and HF Transformers. Run
  `uv sync --group oracle` first. The first oracle run downloads the weights: about 550 MB
  for GPT-2 and 1.5 GB for Qwen3.
- `LLM_GPT2_HF_DIR`: also needed for `tests/oracle/test_gpt2_lineage.py`.

Paths inside a config resolve relative to that config file.

## Architecture

Code is under `src/llm_training/`. Pretraining, continued pretraining (`model.init_from`) and
SFT (`data.format: instructions`) all run through the one next-token trainer in `lm/`.
`reasoning/` (GRPO, MATH eval) is built on Qwen3 directly and does **not** go through `Family`.

**The family rule.** Nothing outside `model/<name>/` branches on a family name. The trainer,
`prepare`, `predict`, `generate` and `respond` go through the `Family` interface.
`model/registry.py` imports a family lazily by name, so a run never imports another family's
code. To add a family:
1. Write `model/<name>/family.py`.
2. Add the name to `FAMILIES`, and its pinned releases to `PRETRAINED`.
3. Check it with `tests/unit/test_families.py` and `tests/unit/test_fingerprint_scope.py`.

**The resume code fingerprint.** Read this before editing training code.
- `training/provenance.py` `implementation_fingerprint` hashes the normalized AST of the code a
  run executes. Comments, docstrings and formatting don't change the hash.
- The hashed paths are `lm_training_code(cfg)` in `lm/trainer.py` for next-token runs, and
  `GRPO_CODE` in `reasoning/trainer.py` for GRPO.
- `__init__.py`, `__main__.py`, `cli.py` and `prediction.py` are exempt.
- Any executable edit under the hashed paths makes existing checkpoints refuse
  `train --resume` unless `--allow-code-drift` is passed. Mention that when an edit has this
  effect.
- A new module the trainer runs from outside the listed paths must be added to the list, or
  it escapes the guard.
- A renamed listed path raises an error rather than silently dropping out.

**Determinism contract.**
- Exact resume is tested bitwise on CPU FP32. On MPS it matches to about 1e-7.
- The runtime is single-process with `num_workers=0`.
- Evaluation, `output.sample_every` samples and the initial/final reports use isolated or
  private RNG, so they never change training order or training RNG state.
- The exception is deliberate: `training.recipe: sampled_train_validation` consumes RNG
  exactly as the book's source loop does, which keeps `pretrain_gpt_162m.yaml` at parity.
  Don't "fix" it. The resume and parity tests compare optimizer, RNG and batch history
  exactly.
- The learning rate is a pure function of the update index.
- `--stop-after` is an absolute count of completed updates.

**Book parity.**
- Every config field defaults to the book's chapter-5 behavior: PyTorch init, untied head,
  explicit attention, constant LR, no clipping. The GPT-2 recipe pieces are opt-in fields
  (table in `docs/gpt2.md`).
- `configs/pretrain_gpt_162m.yaml` is frozen as the source-parity reference.
- `tests/reference/` is a frozen, attributed copy of the upstream definitions and training
  loop. Keep it independent of production refactors: the parity tests depend on it not
  changing.
- `.claude/settings.json` denies edits to both. If one is needed, stop and ask; the user
  edits it by hand.
- The chapter-5 reproduction matches the source in FP32 at `atol=1e-6`, `rtol=1e-5`, so a
  divergence there is a regression, not precision drift.
- Fine-tuning and GRPO run in float32, while the books ran them in bfloat16. Their reported
  results (chapter 7/8 SFT losses, MATH-500 15.2% and 47.4%) can differ, possibly by a lot:
  many lr-1e-5 updates vanish in bfloat16.

**Upstream provenance.** Before porting new code from Raschka's repositories, read
`.claude/rules/upstream-provenance.md`. It loads by itself when you touch an already-derived file.

## Docs conventions

- The docs follow [Diátaxis](https://diataxis.fr/). Each page is one type: a task goes in
  `docs/how-to/`, lookup material (files, fields, tables) in `docs/reference/`, and how and why
  in the topic guides (`docs/gpt2.md`, `qwen3.md`, `training-loop.md`, `finetuning.md`,
  `grpo.md`, `architecture-explorer.md`). The README stays a landing page; link new pages from
  its Documentation section. The README is also the package's long description, so its links to
  repo files are absolute `https://github.com/bwu1234/llm-training/blob/main/...` URLs. The backlog, the gap register, `upstream-provenance.md` and
  `experiments/` are project records, outside the four types.
- `docs/backlog.md` tracks only what is pending. When an item lands, delete it there and
  describe it where its type belongs (above), or in the gap register `gpt2-parity-gaps.md`. When work is found or deferred, add it to the backlog, not
  to a guide.
- `IMPLEMENTATION_PLAN.md` is the original plan. Where it conflicts with the gap register, the
  register's priorities win.

## Long runs

The backlog's R1–R7 runs are built but have not been launched, and each needs an explicit
go-ahead (decision D3). They share one MPS GPU on a 64 GB machine, with peaks of 15–41 GB each,
so run them **one at a time**. `data/` and `runs/` are git-ignored. A Qwen3 `last.pt` is about
9 GB and there is no checkpoint retention yet, so watch disk space.
