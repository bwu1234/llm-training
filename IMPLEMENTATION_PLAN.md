# LLM training from scratch: implementation plan

Date: 2026-10-24  
Status: M0/M1 completed, including a fresh ten-epoch source comparison. The M3
full-state recovery slice is implemented and verified across fresh processes;
M2, the remaining M3 training features, and M4 onward remain pending. See
[chapter/resume evidence](docs/experiments/chapter-reproduction.md),
[initial evidence](docs/experiments/initial-baseline.md), and [README](README.md).

**GPT-2 parity audit (2026-10-24):** the implemented slice reproduces the book's
chapter-5 recipe, not GPT-2 pretraining. Initialization, weight-decay groups, AdamW
betas, LR schedule, clipping, accumulation, weight tying, context length, SDPA/AMP,
and token shards differ from GPT-2 practice. [The gap register](docs/gpt2-parity-gaps.md)
lists each gap with evidence, priority, fix, and acceptance gate; its priorities
supersede the milestone ordering below where they conflict. The repository also has
no Git commits yet, so run provenance records `commit: null`.

**Update 2026-10-25:** architecture parity (L1) is met, verified against the public GPT-2
weights. The recipe mechanisms (L2) are implemented and unit/integration tested: GPT-2 init with
a residual-scaling flag, tied head, QKV bias, SDPA, weight-decay groups, betas, warmup/cosine,
clipping, accumulation, and a code-drift-aware resume guard, in `configs/pretrain_gpt2_124m.yaml`.
The recipe's values are provisional (tokens per update, warmup, schedule length) until the
throughput pilot and corpus choice, so L2 is not frozen.
Decisions recorded: the pretrained-weight oracle is allowed as a test only; the target is GPT-2
small on a reduced budget. Open work is tracked in [the backlog](docs/backlog.md).

## 1. Goal and scope

Build an installable Python project in `llm-training` that reuses Sebastian Raschka's **LLMs-from-scratch** implementations and supports a reproducible workflow:

**prepare data → initialize a GPT model → pretrain → checkpoint/resume → evaluate → generate → export**.

The objective is to preserve the transparent, educational model implementation while adding the engineering practices needed for dependable training experiments: explicit configuration, versioned data, recovery, measured evaluation, tests, and documented commands.

For the first release, “from scratch” means randomly initialized model weights, next-token prediction, and training on a declared corpus. Reusing the GPT-2 tokenizer is allowed and must be recorded. Training a new tokenizer is a later, separate experiment. Loading pretrained GPT-2 weights is outside the initial pretraining workflow.

Start with a small GPT model that can complete a useful local run. Support a book-sized model as a configuration after measuring memory and throughput. This project should demonstrate sound training engineering; a small-corpus run does not establish foundation-model quality or large-scale production readiness.

## 2. What already exists

Inspected local source: `/Users/bwu19/LLMs-from-scratch`, remote `https://github.com/rasbt/LLMs-from-scratch`, HEAD `cdbd33e6d57a71e20ea31434841095fac50cecd8`.

At inspection, `llm-training` was empty and was not a Git repository. The source checkout had a local change to `requirements.txt`; preserve it and do not use that unrecorded environment as a reproducibility baseline.

**The source is already partly modularized.** It contains an installable `llms-from-scratch` package under `pkg/llms_from_scratch/`, standalone chapter scripts, and tests. Most foundational code can be reused without converting notebooks. Notebook extraction should be limited to a specific feature absent from the Python sources.

Paths below are relative to the source repository:

| Capability | Existing source | Proposed use |
| --- | --- | --- |
| Token windows and next-token targets | `pkg/llms_from_scratch/ch02.py`: `GPTDatasetV1`, `create_dataloader_v1` | Preserve as the reference data path; add pretokenized shards for larger runs. |
| Causal attention | `pkg/llms_from_scratch/ch03.py`: `MultiHeadAttention` | Reuse the explicit implementation for the baseline. |
| GPT architecture | `pkg/llms_from_scratch/ch04.py`: normalization, GELU, feed-forward, transformer block, `GPTModel` | Reorganize into named modules without changing initial behavior. |
| Loss, training, generation | `pkg/llms_from_scratch/ch05.py`: `calc_loss_batch`, `train_model_simple`, `generate` | Use as the numerical reference; separate computation from orchestration and plotting. |
| Runnable book experiment | `ch05/01_main-chapter-code/gpt_train.py` | Reproduce its configuration and data split before introducing training changes. |
| Warmup, cosine decay, clipping | `pkg/llms_from_scratch/appendix_d.py` | Adapt the ideas into independently testable optimizer-step utilities. |
| Larger corpus preparation | `ch05/03_bonus_pretraining_on_gutenberg/prepare_dataset.py`, `pretraining_simple.py` | Reference document handling and pretraining flow; add manifests, deterministic splits, and token shards. |
| Faster architecture/training | `GPTModelFast` in `ch04.py`; `ch05/10_llm-training-speed/01_opt_single_gpu.py` | Optional optimization track after the reference run passes. |
| Distributed training | `ch05/10_llm-training-speed/02_opt_multi_gpu_ddp.py` | Later reference for DDP, rather than an initial infrastructure requirement. |
| Instruction tuning | `pkg/llms_from_scratch/ch07.py`: `InstructionDataset`, `custom_collate_fn`, `format_input` | Optional extension after pretraining is complete. |
| Existing checks | `pkg/llms_from_scratch/tests/`, chapter-specific tests | Reuse relevant assertions and attribution; add small tests for this project's contracts. |

The existing chapter-5 loop is intentionally simple: it updates once per batch, prints progress and samples, and returns lists. This project needs explicit run state, recovery, structured metrics, and separation of training from evaluation and generation.

## 3. Reuse strategy

**Recommendation: vendor a small, attributed subset of the Python implementations, then refactor behind stable project APIs.** This makes the new repository self-contained, keeps the model readable, and avoids coupling it to a sibling checkout.

The alternative is a pinned upstream package dependency with thin adapters. That is useful for reference comparisons, but the inspected upstream dependency list also includes notebook, plotting, TensorFlow, and other tutorial tooling. It is broader than the intended pretraining runtime. Do not install the entire upstream dependency set just to import the model.

Implementation rules:

1. Copy only the required classes/functions from the pinned source; do not copy every `previous_chapters.py` or automatically export all notebook cells.
2. Preserve source copyright headers and `LICENSE.txt`; retain applicable notices, and mark modified files. Record the source license as shipped, including its stated exclusions for books and related images. Keep copied tutorial prose and images out of the code migration.
3. Add `docs/upstream-provenance.md` mapping each derived module to source commit, original path, symbols, and local changes. Capture hashes of copied source files and any relevant local source diff.
4. Build a small, attributed test reference under `tests/reference/`; it must work without the sibling checkout. Keep the production implementation in one place.
5. Refactor imports and file boundaries first. Make behavioral changes in separate, measured steps.
6. No `sys.path` hacks, imports from absolute user paths, runtime notebook execution, or downloads during module import.
7. Handle future upstream updates as explicit changes with provenance and parity checks.

## 4. Proposed package and tooling

Use Python 3.14.9 (requested by the user), PyTorch, `tiktoken`, NumPy, and a small YAML parser. Validate dependency compatibility when bootstrapping and commit a lockfile. Use `uv` for environment management, `pytest` for tests, and Ruff for formatting/linting. A standard-library CLI and typed dataclasses are sufficient initially; a large training framework is unnecessary for the first workflow.

Keep notebooks, plotting, experiment-service integrations, and instruction-tuning dependencies optional. Local JSONL metrics are the initial record; TensorBoard or MLflow can be added when there is a concrete need.

Proposed layout, with later-stage modules added when their milestone begins:

```text
llm-training/
├── pyproject.toml
├── uv.lock
├── README.md
├── IMPLEMENTATION_PLAN.md
├── LICENSE.txt
├── configs/
│   ├── smoke.yaml
│   ├── pretrain_gpt_162m.yaml
│   └── pretrain_small.yaml
├── src/llm_training/
│   ├── __init__.py
│   ├── __main__.py
│   ├── cli.py
│   ├── config.py
│   ├── model/
│   │   ├── attention.py
│   │   ├── layers.py
│   │   └── gpt.py
│   ├── data/
│   │   ├── tokenizer.py
│   │   ├── prepare.py
│   │   ├── dataset.py
│   │   └── manifest.py
│   ├── training/
│   │   ├── trainer.py
│   │   ├── optim.py
│   │   ├── checkpoint.py
│   │   └── state.py
│   ├── evaluation.py
│   ├── generation.py
│   └── artifacts.py
├── tests/
│   ├── reference/
│   ├── unit/
│   └── integration/
├── docs/
│   ├── upstream-provenance.md
│   ├── training-guide.md
│   └── experiments/
├── .github/workflows/ci.yml
├── data/                         # generated corpus artifacts; Git-ignored
└── runs/                         # generated run artifacts; Git-ignored
```

This was the starting proposal. As built, model code lives per family (`model/gpt2/`,
`model/qwen3/`, each with its tokenizer), and the GRPO fine-tune is in `reasoning/`; see the
[README](README.md).

Public interfaces should remain small: `GPTModel(config)`, a dataset yielding `(inputs, targets)`, a trainer accepting explicit dependencies, and separate evaluation/generation functions. CLI commands orchestrate these APIs; notebooks may call them later.

## 5. Configuration and operating contracts

### Configuration

Use a validated YAML configuration with explicit sections:

| Section | Required decisions |
| --- | --- |
| `model` | Vocabulary size, context length, embedding width, layers, heads, dropout, QKV bias, implementation variant. |
| `data` | Input manifest, tokenizer identity, split policy/seed, stride or packing policy, document separator, shard format. |
| `training` | Initialization seed, microbatch size, accumulation steps, maximum optimizer steps, AdamW settings, schedule, clipping, evaluation/checkpoint intervals. |
| `runtime` | Requested device, precision, worker count, deterministic mode, compilation setting. |
| `output` | Run root, checkpoint retention, sample prompts and generation settings. |

Reject unknown fields and invalid combinations, including width not divisible by head count, insufficient tokens for a batch, incompatible vocabulary, and unsupported precision/device requests. Resolve relative input paths consistently against the config file; record resolved paths and content hashes. Save the fully resolved configuration before training.

Use three distinct presets:

- **Smoke:** tiny dimensions and fixed small data, FP32, short step limit; proves execution and recovery.
- **GPT 162M pretraining:** match the inspected chapter-5 script's model settings, seed, text split, batching, optimizer, evaluation subset, and schedule. Preserve its behavior even where later workflows improve it.
- **Small pretraining:** document-level splits, prepared tokens, a declared compute budget, and robust training features. Start around 4 layers, width 256, 4 heads, and context 256; measure the actual parameter count because embeddings and the untied output head are substantial.

Keep GPT-2 tokenization in all initial presets. Record that it is reused rather than trained. Preserve the source model's untied embeddings/output head in the reference; weight tying or a different initialization is an explicit experiment.

### Data preparation

- Separate acquisition from preprocessing and training. Cache inputs explicitly; training must work offline after preparation, including tokenizer assets.
- Begin with the source's `the-verdict.txt` for reference reproduction. Pin bytes/checksum instead of relying on a mutable `main` URL.
- For the meaningful pretraining run, select a larger, documented corpus snapshot. Record origin, version, usage terms, document IDs/hashes, exclusions, and preprocessing version. Dataset selection remains open until capacity and budget are known.
- For that corpus, normalize and deduplicate before assigning whole documents to train/validation/test with a stable policy. Check overlap and preserve the split manifest. Never create overlapping windows before splitting.
- Pretokenize into fixed-width integer shards with document boundaries and metadata. GPT-2 IDs (vocabulary 50,257) fit in 16 bits, so `uint16` shards are sufficient and halve storage; a larger or padded vocabulary would need rechecking. Cast batch IDs to `int64` for the embedding lookup.
- Make EOS insertion and crossing document boundaries explicit. Standard causal attention plus EOS permits cross-document attention; isolated document attention would be a separate change.
- Produce shifted targets from `context_length + 1` tokens. Define handling of incomplete tails and padding, and record counts of dropped or masked tokens. Fail clearly on empty splits/loaders.
- Retain the in-memory source-compatible loader for reference runs; use memory-mapped shards for larger data.

### Training correctness and recovery

- Keep the first path single-process FP32 with plain AdamW and source-compatible behavior.
- Add accumulation, warmup/cosine decay, gradient clipping, and documented weight-decay parameter groups afterward. Report changes from the book recipe.
- Define `global_step` as completed optimizer updates; track microsteps separately. Log valid target tokens, not just padded input positions.
- For full unmasked batches, effective tokens/update are `microbatch × context_length × accumulation × world_size`. Handle partial accumulation groups explicitly; normalize gradients by the actual valid-token total when batch sizes or masks vary.
- Step the schedule at the documented optimizer-update boundary. With AMP, account for skipped optimizer updates; unscale before clipping. Use PyTorch's current AMP guidance when implementing this path. [PyTorch AMP examples](https://docs.pytorch.org/docs/stable/notes/amp_examples.html)
- Seed Python, NumPy, Torch, and the data-order generator. Evaluation and sampling must not alter subsequent training order or training RNG state.
- Save versioned checkpoints at optimizer boundaries: model, optimizer, scheduler, scaler if used, completed steps, tokens seen, RNG states, sampler permutation/cursor, best validation metric, config hash, and dataset/tokenizer identities.
- Initially use `num_workers=0` and a resumable sampler so the data cursor is unambiguous. Introduce worker prefetching only after recovery behavior is defined and tested.
- Write checkpoints atomically; maintain `last` and `best` plus bounded retention. Validate compatibility before resuming. A warm start from weights must be labeled separately from full-state resume.
- Record nonfinite loss/gradients as failures, checkpoint at the last safe boundary where possible, and return a nonzero exit code. Handle interruption at a safe boundary with a recoverable checkpoint.

PyTorch's general checkpoint guidance includes optimizer state beyond model weights; this project additionally needs dataset position and experiment identity to meet its recovery contract. [Saving and loading models](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html)

### Evaluation and artifacts

Evaluate in inference mode with dropout disabled and restore the prior model mode afterward. Compute summed negative log-likelihood divided by valid target-token count; report perplexity as `exp(mean NLL)` with overflow handling. Record the exact split and evaluated token count. A fixed evaluation subset is acceptable for monitoring if clearly labeled; use the declared full validation split for final comparisons.

Select checkpoints using validation data. Keep a test split untouched until the recipe is selected. Fixed prompts and samples help inspect behavior but do not substitute for held-out metrics. Compare loss/perplexity only under matching tokenization, data, context, and evaluation policies.

Each run should contain:

```text
runs/<run-id>/
  config.resolved.yaml
  provenance.json          # Git SHA/dirty state, dependency lock hash, source lineage
  environment.json         # Python/Torch, hardware, backend, precision
  data-manifest.json       # split/shard hashes and tokenizer identity
  status.json              # running/completed/interrupted/failed and reason
  metrics.jsonl            # steps, tokens, loss, LR, gradient norm, timing, memory
  checkpoints/             # last and best full training states
  evaluation.json
  samples.jsonl            # checkpoint, prompt, token IDs, sampling settings/seed
```

An exported inference artifact contains weights, model configuration, tokenizer identity/assets, provenance, and evaluation metadata. It must reload in a fresh process without an optimizer or training dataset.

## 6. Delivery milestones and acceptance gates

Complete milestones in order. Each should be a reviewable change with documented commands and retained evidence; do not combine all phases into one large refactor.

| Milestone | Work | Acceptance gate |
| --- | --- | --- |
| **M0: bootstrap and provenance** | Initialize this repository; add package metadata, lockfile, minimal CLI, attribution, source map, and reference fixtures. Inventory local CPU/MPS/CUDA capacity. | Clean install and import work without sibling checkout or Jupyter; no import-time network/device initialization. Source lineage and local modifications are recorded. |
| **M1: faithful modular baseline** | Port model/data/loss/generation; add the reference config and minimal single-process trainer. | With identical state dicts and token batches, compare FP32 logits, loss, gradients, and one AdamW update against the reference. Declare tolerances before running (initial CPU target: `atol=1e-6`, `rtol=1e-5`); investigate discrepancies. Complete a fresh chapter-5-style run and retain a reference-versus-port report. |
| **M2: prepared data** | Add manifests, deterministic document splits, deduplication, token shards, and offline loading. | Repeating preparation yields matching split/token hashes; no document/hash overlap; shifted targets and boundary handling are correct; empty/short input errors are actionable. |
| **M3: reliable trainer** | Add resolved configs, structured metrics, accumulation, scheduling, clipping, full checkpoints, and safe interruption. | On fixed CPU FP32 data, compare an uninterrupted run with a fresh-process stop/resume run: same subsequent batches, steps, scheduler/optimizer state, and matching final parameters within declared tolerance. Accumulation agrees with an equivalent full batch with dropout disabled. Complete prepare → train → resume → evaluate → generate → export/reload. |
| **M4: measured small pretraining** | Choose the larger corpus snapshot, model size, and token budget from a pilot; freeze the recipe; run pretraining. | Finite losses, complete provenance/artifacts, held-out NLL improvement over initialization on the same split, a validated resume, and an independently reloaded export. Publish the budget, failures, train/validation gap, and limitations. Do not infer broad capability from this gate. |
| **M5: measured performance options** | Profile; optionally add SDPA, AMP, fused optimizer, compile, pinned loading, and activation checkpointing one at a time. | For each option, report numerical/convergence checks and measured throughput/memory under a fixed workload. Record warmup/compile cost; preserve the eager reference path and flag unsupported devices. |
| **M6: optional scale/post-training** | Add DDP if multiple GPUs are available; separately add instruction tuning from chapter 7 if desired. | DDP: matched global-batch update test, correct data sharding/epoch reseeding, globally reduced token-weighted metrics, coordinated checkpoints with per-rank RNG/cursors. SFT: explicit starting checkpoint, padding/prompt masking policy, separate data lineage and evaluation. |

The minimum useful release is **M0–M4**. Optimizations, multi-GPU execution, and instruction tuning are later extensions.

## 7. Hardware and experiment budget

Do not assume a CUDA GPU from the repository path. Use CPU FP32 for portable tests; probe MPS for local experimentation and CUDA for accelerator runs. Log the resolved backend and fail if an explicitly requested backend is unavailable. MPS numerical behavior must be validated independently.

Before committing to M4, run a short pilot after warmup and measure parameter count, peak memory, training tokens/second, evaluation time, and checkpoint size/time. Include parameters, gradients, Adam states, activations, and logits in memory planning. Reduce model/context/microbatch size if needed; accumulation increases effective batch size but does not remove model-state memory requirements.

Estimate wall time from **planned token presentations / measured tokens per second**, adding evaluation, checkpoint, and startup costs. Set maximum steps/tokens, runtime budget, and disk retention before the larger run. Distinguish unique corpus tokens from repeated token presentations. Do not borrow the upstream A100 speed numbers as estimates for this machine.

Seeding does not guarantee identical results across devices, PyTorch releases, or platforms. Limit exact recovery claims to the tested environment and deterministic configuration; record tolerances for accelerator comparisons. [PyTorch reproducibility guidance](https://docs.pytorch.org/docs/stable/notes/randomness.html)

## 8. Intended command workflow

The full pipeline below remains a target. Preparation, training, and resume are
implemented for the smoke/GPT 162M presets documented in the README. The larger-corpus
configs and standalone evaluate/generate/export commands below remain planned;
evaluation and generation run as part of training completion. A standalone
`predict` command also exposes stepwise greedy generation with candidate
probabilities; see the README. The standalone evaluate/export commands remain
planned.

```bash
uv sync --frozen
uv run llm-training prepare --config configs/data/small_corpus.yaml
uv run llm-training train --config configs/pretrain_small.yaml
uv run llm-training train --resume runs/<run-id>/checkpoints/last.pt
uv run llm-training evaluate --checkpoint runs/<run-id>/checkpoints/best.pt --split validation
uv run llm-training generate --checkpoint runs/<run-id>/checkpoints/best.pt --prompt "Every effort moves you"
uv run llm-training export --checkpoint runs/<run-id>/checkpoints/best.pt --output artifacts/<run-id>
```

The CLI should print the run directory, resolved device, parameter count, data identity, and budget before training. Resume restores the saved recipe and records parent checkpoint lineage; only documented operational overrides are allowed. README and the training guide must contain a complete, tested smoke workflow, including data preparation and tokenizer caching.

## 9. Verification and documentation

Run fast, offline CPU checks in CI: formatting/lint, package installation/import, tiny model forward/backward, causal masking (changing future tokens cannot affect earlier logits), target shifting, weighted evaluation, config validation, checkpoint compatibility, and a tiny end-to-end resume/export run. Cache or provide the tokenizer fixture so tests do not download assets.

Keep chapter-scale reproduction and longer training as explicit integration commands with saved reports. Accelerator/DDP checks must run on actual hardware before being described as supported; skipping them on CPU CI is not evidence they work.

Write user documentation alongside each milestone: setup, source lineage, data preparation, training/resume, evaluation interpretation, artifact format, and failure recovery. Record experiment results separately from future plans. A passing unit test or decreasing training loss alone does not prove successful pretraining.

## 10. Completed slices and next implementation

The original **M0 and smallest M1 slice** is complete:

1. Create the package, locked environment, Git ignore rules, and source provenance/license files.
2. Port the explicit GPT components and source-compatible token dataset/loss without architectural changes.
3. Add a tiny offline parity test and a faithful `pretrain_gpt_162m.yaml` preset.
4. Expose a minimal `train` command with bounded steps, resolved configuration, metrics, and saved model state.
5. Run a fresh CPU smoke experiment and a source-versus-port comparison; document observed results before proceeding to corpus scaling.

The subsequent ten-epoch source comparison and full-state recovery slice are also
complete: the run stopped at update 46 and resumed through update 90 matched the
upstream weights, optimizer/RNG state, monitored losses, and epoch samples exactly.
CPU FP32 was used on the local 64 GiB arm64 host; accelerator execution remains
unverified. See the experiment report for the exact runtime and measured scope.

Later slices (the GPT-2 recipe, TinyStories, Qwen3, GRPO, continued pretraining and SFT) are
described in the README and guides. What remains, in recommended order, is in
[the backlog](docs/backlog.md).

