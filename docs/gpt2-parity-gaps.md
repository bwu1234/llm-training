# GPT-2 parity gap register

Opened 2026-10-24; status updated 2026-10-25 (TinyStories additions 2026-10-26). Scope: what separates this repository from a
proper from-scratch GPT-2 (124M) pretraining setup.

**Reading this document:** this register is the specification for each gap: its evidence,
fix and acceptance gate, *as found on 2026-10-24*, plus the record of what was done. Open
work, its order, and the decisions it waits on are tracked only in
[the backlog](backlog.md). The [status table](#status) is authoritative for what has been
built. File paths follow
the current layout (GPT-2 modules moved into `model/gpt2/` and `docs/model-architecture.md`
became `docs/gpt2.md` on 2026-10-26, and the next-token trainer modules moved into `lm/`
later that day); line numbers are as of 2026-10-24.

**State on 2026-10-25:** **L1 (architecture) is met**, verified against the public GPT-2
weights. **L2 (recipe) has its mechanisms implemented and tested** (init, decay groups, betas,
schedule, clipping, accumulation), but the recipe is not frozen: the preset's values are
provisional (32,768 tokens per update against nanoGPT's ~0.5M scale, placeholder warmup and
schedule length, no corpus), and it still differs from nanoGPT's procedure in sampling (epoch
sliding windows, not random offsets) and precision (FP32, not bf16). The data pipeline,
throughput features (bf16, compile, DDP), benchmarks, a chosen corpus, and a measured budget
are missing, so no GPT-2 pretraining run exists and L3 is untouched.
The book's 162M recipe is preserved unchanged as the source-parity reference. Items marked
**Defect** were wrong or fragile when found.

## Reference points

Every "GPT-2 / reference" claim below comes from one of these primary sources, read
on 2026-10-24. Nothing is quoted from memory unless marked *unverified*.

| Tag | Source |
| --- | --- |
| **OAI** | [openai/gpt-2 `src/model.py`](https://github.com/openai/gpt-2/blob/master/src/model.py), the original architecture |
| **nGPT-m** | [karpathy/nanoGPT `model.py`](https://github.com/karpathy/nanoGPT/blob/master/model.py), a widely used reproduction (init, tying, decay groups, SDPA) |
| **nGPT-t** | [karpathy/nanoGPT `train.py`](https://github.com/karpathy/nanoGPT/blob/master/train.py), the training recipe (defaults, schedule, clipping, accumulation, memmap) |
| **nGPT-r** | [karpathy/nanoGPT README](https://github.com/karpathy/nanoGPT/blob/master/README.md), the reproduction target |

nanoGPT is a reproduction, not the original training code. OpenAI never released
its training recipe or WebText. The recipe items are therefore "community-standard",
not "what OpenAI did".

## What "parity" should mean

Three levels, each with a testable gate. Today only a fragment of level 1 exists.

| Level | Meaning | Gate |
| --- | --- | --- |
| **L1 Architecture** | Same function as GPT-2 small | A config of 12 layers, width 768, 12 heads, context 1024, QKV bias on, tied head has **124,439,808** parameters. Loading the public GPT-2 124M weights gives logits matching a reference implementation at FP32 `atol=1e-4`. |
| **L2 Recipe** | Same training procedure as the reference | Unit-tested init statistics, weight-decay groups, LR schedule values, clipping, and accumulation equivalence, under one frozen `gpt2_124m` recipe config. |
| **L3 Outcome** | Same quality at a stated budget | Held-out loss on a documented corpus split, compared with the reference numbers: nGPT-r reports "~2.85" for its reproduction (about 4 days on 8×A100) and about 3.11 (README text) or 3.12 (README table) for OpenAI's pretrained weights evaluated on OpenWebText. A reduced budget needs its own pre-declared target. |

The L1 gate requires loading pretrained weights **as a test oracle only**. That
conflicts with the plan's "pretrained weights are outside the initial pretraining
workflow" (`IMPLEMENTATION_PLAN.md` §1). The two are compatible if the loader lives
under `tests/` or an optional extra and the training path never calls it. This is a
decision for the maintainer; see "Decisions needed".

Parameter arithmetic for the L1 gate (checked against the current preset):
`162,419,712 − 38,597,376` (untied head) `+ 12×3×768` (QKV biases)
`+ (1024−256)×768` (position table) `= 124,439,808`.

## Status

Verified 2026-10-25 with the full suite (69 passed, plus the opt-in oracle, 2 passed).

| Status | Items |
| --- | --- |
| **Done** (mechanism built and tested; preset values provisional) | A1 tie option; A2/A3 via the `pretrain_gpt2_124m.yaml` preset; A4 init (with residual-scaling flag); A5 SDPA (`attn_impl`); A6 GELU constant; B1 decay groups; B2 betas; B3 warmup/cosine; B4 clipping; B5 accumulation; B6 dropout 0 in the preset; C3 `uint16` plan correction; C2 `uint16` token files with a memory-mapped loader, for registered corpora (`data/corpus.py`); C5 EOS policy for token corpora (prepended `<|endoftext|>`, attention crosses documents, recorded in the manifest); E1 pretrained-weight oracle; F1 repository committed (2026-10-25); F2 resume guard |
| **Open or partial** | Tracked in [the backlog](backlog.md#gpt-2-pretraining-track), which holds their order and the decisions they wait on |

### What was verified for the Done items

- **L1 architecture:** `configs/pretrain_gpt2_124m.yaml` has exactly 124,439,808 parameters
  (checked on the `meta` device), and with the public GPT-2 weights loaded, logits match
  Hugging Face `GPT2LMHeadModel` at the declared `rtol=atol=1e-4` for both attention paths:
  maximum absolute difference 1.4e-4 at logit magnitudes up to about 100 (about 1.4e-6
  relative), identical argmax at every position, loss within 1e-4. A deliberately broken
  weight mapping differs by 61.5, so the check can fail.
- **L2 recipe:** unit tests cover init statistics (including residual-scaled projections),
  initial loss near ln(50257), tied-weight round trip, decay groups (no 1-D tensor decays; a
  tied tensor appears once), betas, the schedule table, clipping, and SDPA versus explicit
  attention (`atol=1e-5`, `rtol=1e-4`, FP32, dropout off). Integration tests cover exact
  fresh-process resume with every feature active (accumulation with and without a partial final
  group, clipping, cosine, tied head, SDPA, dropout 0.1) and accumulation equal to the
  equivalent full batch at `atol=1e-6`, `rtol=1e-5`. The accumulation test was
  mutation-checked: removing the `1/accum` scaling is invisible in the weights (Adam is nearly
  scale-invariant) and was caught only after the recorded loss and gradient norm were added to
  the comparison.
- **Book path unchanged:** the 30 pre-existing tests, including the frozen-source comparison
  that asserts zero difference, pass unmodified.

### Limits of what is verified

- All checks are CPU FP32 on one machine. SDPA, bf16, MPS, and CUDA behavior are unvalidated.
- The resume guard is exact only for unchanged executable code; with `--allow-code-drift`
  nothing is guaranteed, and the child run records the override.
- `train()` is still one large function (F4). The new logic lives in `training/optim.py`,
  `lm/schedule.py`, and `model/gpt2/init.py`, but the accumulation loop is inside `train()`.
- The preset's data path, warmup, evaluation cadence, and token budget are placeholders.

## Priority key

- **P0**: blocks any meaningful claim of GPT-2-style training; small to fix.
- **P1**: needed for a real run (tokens, throughput, or recoverability).
- **P2**: needed to call the result a reproduction (evaluation and benchmarks).
- **P3**: hardening or convenience.

---

## A. Architecture (L1)

| ID | Pri | Gap | Evidence in this repo | GPT-2 / reference | Fix | Gate |
| --- | --- | --- | --- | --- | --- | --- |
| A1 | P0 | **No weight-tying option.** The head is always a separate matrix, giving 162M instead of 124M. | `model/gpt2/model.py:56`; `docs/gpt2.md` | OAI computes logits with `matmul(h, wte, transpose_b=True)`; nGPT-m sets `wte.weight = lm_head.weight`. | Add `model.tie_embeddings: bool`. Keep `false` for the book-parity preset. | Param count 124,439,808 with tying; the tied tensor appears once in `state_dict` handling. |
| A2 | P0 | **QKV bias is off in every preset** (the option exists in config). | `configs/*.yaml`: `qkv_bias: false` | OAI's `c_attn` conv1d has a zero-initialized bias. | Set `true` in the GPT-2 preset. No code change. | L1 parameter count. |
| A3 | P0 | **Context 256, not 1024.** | `configs/pretrain_gpt_162m.yaml` | GPT-2 uses 1024 (`docs/gpt2.md` table); nGPT-t `block_size = 1024`. | New preset. Also revisit memory: the explicit attention matrix is `B×H×T×T`. | L1 parameter count. |
| A4 | P0 | **No weight initialization.** Embeddings use PyTorch's default N(0,1). | `model/gpt2/model.py` has no init; `docs/gpt2.md:7` says "PyTorch's defaults" | OAI: `wte` std 0.02, `wpe` **0.01**, all conv1d weights 0.02, zero biases, LN gain 1/bias 0, no residual scaling. nGPT-m: N(0, 0.02) for Linear and Embedding, then residual `c_proj.weight` at `0.02/√(2·n_layer)`. | Add `init_weights()`. Make the residual scaling a flag, since the original code does not use it and nanoGPT does. | Statistical test of per-tensor std after init; initial loss close to `ln(50257) ≈ 10.82`. Currently initial NLL is 10.98 on the chapter text. |
| A5 | P1 | **Explicit O(T²) attention**; no fused kernel. | `model/gpt2/attention.py:46-54` | nGPT-m calls `scaled_dot_product_attention(..., is_causal=True)` when available. | Add an `attention_impl: sdpa|explicit` switch. Keep `explicit` as the parity reference. | Logits equal to the explicit path at `atol=1e-6` (FP32); dropout-off comparison. |
| A6 | P1 | **GELU rebuilds a tensor on every call**: `torch.sqrt(torch.tensor(2.0/torch.pi))` is created on CPU each forward. On CUDA/MPS this forces a host-to-device copy per layer per step. | `model/gpt2/layers.py:34` | OAI uses the same tanh formula (`np.sqrt(2/np.pi)` as a constant). | Use `math.sqrt(2/math.pi)` or `nn.GELU(approximate="tanh")`. | Numerical equality with the current output. |
| A7 | P3 | **Causal mask is a persistent buffer** (T×T float32 per layer), so it is saved in every `state_dict` and tied to the context length. | `model/gpt2/attention.py:23` | nGPT-m registers it as a non-parameter buffer for the fallback path only. | `persistent=False`; for SDPA, no mask at all. | Old checkpoints still load (`strict` handling) or the format version is bumped. |
| A8 | P3 | **Hand-written LayerNorm** instead of `F.layer_norm`. Numerically equivalent but slower, and blocks `torch.compile` fusion on some backends. | `model/gpt2/layers.py:9-20` | OAI `eps=1e-5`, same as here. | Optional `F.layer_norm` path behind parity tests. | Equality at `atol=1e-6`. |
| A9 | P3 | **Vocabulary fixed to 50,257.** Many reproductions pad to a multiple of 64 for GEMM efficiency. | `config.py:105` | *unverified* (nanoGPT pads to 50,304 via `vocab_size` override; not re-fetched). | Allow `vocab_size >= 50257` with padded rows never targeted. | Loss unchanged. |
| A10 | P3 | **Greedy generation only; no KV cache; no temperature/top-k/top-p sampling.** `predict --top-k` only changes what is displayed. | `generation.py`; `README.md` | GPT-2 samples with temperature and top-k. | Add `generate` with sampling and a KV cache (the plan's planned `generate` command). | Cache path equals the uncached path. |

## B. Optimization recipe (L2)

| ID | Pri | Gap | Evidence | Reference | Fix | Gate |
| --- | --- | --- | --- | --- | --- | --- |
| B1 | P0 | **Weight decay applies to every parameter**, including LayerNorm scale/shift and biases. | `lm/trainer.py:192` passes `model.parameters()` | nGPT-m: decay only tensors with `dim() >= 2`; biases and LayerNorm get 0. | Two param groups. | Unit test that no 1-D tensor is in the decay group. |
| B2 | P0 | **Default AdamW betas** (0.9, 0.999). | `trainer.py:192-194` | nGPT-t: `beta1 = 0.9`, `beta2 = 0.95`. | Config fields `beta1`, `beta2`. | Optimizer `param_groups` inspection. |
| B3 | P0 | **Constant learning rate.** `learning_rate` in the metrics record is just the config value. | `trainer.py:363`; `scheduler_state_dict` is always `None` | nGPT-t: linear warmup 2000 steps, cosine decay to `min_lr`; peak 6e-4, `weight_decay` 0.1 (verified defaults). | Schedule function stepped per optimizer update; checkpoint its state; log the actual LR. | Table-driven test of LR at warmup start, end, midpoint, and tail; resume equality. |
| B4 | P0 | **No gradient clipping.** Only a non-finite check exists. | `trainer.py:343-352` | nGPT-t: `clip_grad_norm_(..., 1.0)` after unscaling. | Clip, and log the pre-clip norm. | Test with a synthetic large gradient. |
| B5 | P0 | **No gradient accumulation.** Effective batch is 2 × 256 = 512 tokens. | `trainer.py:339-353` | nGPT-t: `gradient_accumulation_steps = 5*8 = 40`; tokens/update = micro-batch × 1024 × 40 (a micro-batch of 12, from memory, gives ≈0.49M — *unverified*). | Accumulate with loss scaling by the valid-token total. | Accumulated step equals an equivalent full batch with dropout off (the plan's M3 gate). |
| B6 | P0 | **Dropout 0.1** in the only pretraining preset. | `configs/pretrain_gpt_162m.yaml` | nGPT-t: `dropout = 0.0 # for pretraining 0 is good`. | The GPT-2 preset sets 0.0. | Config. |
| B7 | P1 | **Per-step host syncs.** `loss.item()` plus a `.all()` finite check on every parameter's gradient each update. | `trainer.py:343-361` | n/a | Check the finite-ness of the clip norm once; log every N steps. | Throughput measurement before/after. |
| B8 | P1 | **Thin metrics.** No gradient norm, tokens/s, step time, memory, or MFU. The plan's `metrics.jsonl` spec lists them. | `trainer.py:358-365` | n/a | Add the fields; document units. | Fields present in `metrics.jsonl`. |
| B9 | P2 | **No fused/foreach AdamW option.** | `trainer.py:192` | nGPT-t uses a fused AdamW when available *(unverified detail)*. | `fused=True` on CUDA. | Step equality at tolerance. |

## C. Data pipeline

| ID | Pri | Gap | Evidence | Reference | Fix | Gate |
| --- | --- | --- | --- | --- | --- | --- |
| C1 | P1 | **No real corpus.** The only dataset is a ~20 KB story. | `data/the-verdict.txt`; `README.md` | nGPT-r uses OpenWebText, "an open reproduction of OpenAI's (private) WebText". | Choose a documented snapshot (plan M2/M4 decision). | Recorded origin, version, terms, hashes. |
| C2 | P1 | **No token shards.** The dataset tokenizes the whole text in memory and stores one tensor per window. | `data/dataset.py:12-23` | nGPT-t reads a `np.memmap` of `uint16`. | `prepare` writes `.bin` shards plus a manifest; loader memory-maps them. | Repeated preparation gives identical hashes. |
| C3 | P1 | **Plan error on dtype.** `IMPLEMENTATION_PLAN.md` said token IDs need more than 16 bits and chose `uint32`. | (fixed in this change) | 50,257 < 65,536; nGPT-t uses `uint16`. | `uint16` shards; cast to `int64` for the embedding. Halves disk and page-cache use. | Round-trip test over IDs 0 and 50256. |
| C4 | P1 | **Split is by character offset before tokenization**, with 90/10 and no document boundaries or deduplication. | `trainer.py:176-182` | n/a | Document-level hash split plus deduplication (plan M2). | No train/validation document or hash overlap. |
| C5 | P1 | **No `<|endoftext|>` separation policy** in the training stream. | `data/dataset.py:16` allows the token but nothing inserts it | nGPT-t's prepare step appends EOT per document *(unverified; not re-fetched)*. | Explicit EOS insertion; record whether attention crosses documents. | Test of boundary handling. |
| C6 | P1 | **Epoch/cursor resume depends on replaying an in-memory `DataLoader`.** This will not scale to random-offset sampling over shards. | `trainer.py:326-337` | nGPT-t samples random offsets each step. | Counter-based per-step sampling (seed, step) so resume needs no replay. | Resume equality at several steps, as now. |
| C7 | P2 | **No held-out test split.** | `config.py` has `train_ratio` only | n/a | Train/validation/test manifest; test untouched until the recipe is frozen. | Manifest check. |
| C8 | P1 | **No token budget.** | `training.max_steps` only | nGPT-r: about 4 days on 8×A100 for ~2.85. A Chinchilla-style rule of thumb is ~20 tokens/parameter (≈2.5B for 124M) *(unverified here)*. | Set the budget from a measured pilot (plan §7). | Budget recorded in the run manifest. |

## D. Precision and throughput

| ID | Pri | Gap | Evidence | Reference | Fix | Gate |
| --- | --- | --- | --- | --- | --- | --- |
| D1 | P1 | **FP32 only.** `precision` is hard-coded and no config knob exists. | `trainer.py:145`; `config.py` `RuntimeConfig` | nGPT-t: `dtype = bfloat16` if supported, else float16. | `runtime.precision`: autocast bf16 (and fp16 + GradScaler); populate `scaler_state_dict`. | Loss within tolerance of FP32 over a short run. |
| D2 | P1 | **No `torch.compile`.** | none | nGPT-t: `compile = True`. | `runtime.compile`; record compile time. | Same loss as eager at tolerance. |
| D3 | P1 | **No distributed training.** Single process. | `README.md` | nGPT-r uses DDP on 8 GPUs. | DDP with per-rank RNG/cursor and reduced token-weighted metrics (plan M6). | Matched global-batch update test. |
| D4 | P1 | **No accelerator validation.** CPU FP32 is the only tested path; MPS/CUDA resume is unvalidated. | `README.md` "Stop and resume" | n/a | Run the parity suite on real hardware. | Recorded results per backend. |
| D5 | P2 | **No `auto` device.** | `config.py:124` allows only cpu/mps/cuda | n/a | `device: auto` resolving to a recorded backend. | Logged resolved device. |
| D6 | P2 | **No pilot measurement** of tokens/s, peak memory, checkpoint time. | `IMPLEMENTATION_PLAN.md` §7 | n/a | Run before committing to a budget. | Report under `docs/experiments/`. |

## E. Evaluation and benchmarks (L2/L3)

| ID | Pri | Gap | Evidence | Fix | Gate |
| --- | --- | --- | --- | --- | --- |
| E1 | P1 | **No pretrained-weight oracle.** Nothing checks that this model *is* GPT-2. | n/a | Test-only loader mapping the public 124M checkpoint (Conv1D weights are transposed relative to `nn.Linear`) into this model. | L1 logits gate. |
| E2 | P1 | **Periodic monitoring is a batch mean** in the book-parity mode, not token-weighted. The token-weighted path exists in `evaluation.py` and is used for the final report. | `lm/monitoring.py:12-27`; `evaluation.py` | Use `evaluate()` for monitoring in the GPT-2 recipe, with a fixed evaluation subset. | Both agree on equal-length batches. |
| E3 | P2 | **No standard benchmark evals.** Validation loss on the training distribution is the only metric. | n/a | Add held-out corpus loss plus at least one zero-shot task; fix and document tokenization/detokenization conventions. Pick tasks after C1. | Pretrained GPT-2 scores reproduced within tolerance before scoring own models. |
| E4 | P2 | **No reference-loss calibration.** Without evaluating pretrained GPT-2 on the *same* split, a loss number has no anchor (nGPT-r itself reports ~3.11 vs ~3.12 for the same baseline). | n/a | Evaluate the oracle weights on the project validation split and store the number. | Report in `docs/experiments/`. |
| E5 | P3 | **No `evaluate` / `generate` / `export` commands.** Planned, absent. | `cli.py` | Implement (plan §8). | Fresh-process export reload. |

## F. Checkpointing, resume, and operations

| ID | Pri | Gap | Evidence | Fix | Gate |
| --- | --- | --- | --- | --- | --- |
| F1 | P0 | **Defect: no commits.** `main` has no commits, so `provenance.json` records `commit: null`. | `git log` fails; `trainer.py:61-75` | Commit the work; keep `runs/`, `data/`, `dist/` ignored (already). | Provenance shows a SHA. |
| F2 | P0 | **Defect: resume is locked to exact code bytes.** `implementation_sha256` hashes every `.py` in the package; any mismatch is rejected. A docstring edit makes a long run unresumable. | `trainer.py:150-155`, `trainer.py:103-122` | Warn on drift; hash only model/data/trainer code; or require an explicit `--allow-code-drift`. Exactness claims must stay limited to unchanged code. | Resume succeeds after a comment-only edit; fails after a model edit. |
| F3 | P1 | **Defect: unbounded checkpoint size.** Each full checkpoint is about 1.8 GB (weights + two Adam moments); every improvement writes `last.pt` and copies it to `best.pt`. `runs/` is already about 10 GB. | `trainer.py:283-306` | Retention (keep last N); write `best` as weights-only; optional bf16 weights export. | Disk bound documented and tested. |
| F4 | P1 | **`train()` is one ~340-line function of closures.** Accumulation, schedules, AMP, and DDP all have to be threaded through it. | `trainer.py:124-464` | Split into a `Trainer` with state, `step`, `evaluate`, and `checkpoint` methods before adding features. | Existing resume and parity tests still pass unchanged. |
| F5 | P2 | **Absolute paths are baked into the saved config and its hash**, so a checkpoint moved to another checkout or machine points at the old data path. *Reasoned from `config.py:131` and `trainer.py:134`; not tested.* | `config.py:126-131` | Store repo-relative paths plus content hashes; resolve at load. | Resume from a relocated directory. |
| F6 | P2 | **Resume replays `batch_cursor` batches** to rebuild the iterator. Cheap here, but linear in progress for large epochs. | `trainer.py:326-337` | Obsoleted by C6. | n/a |
| F7 | P3 | **Scheduler/scaler state fields are always `null`.** Correct today; must be populated with B3/D1. | `trainer.py:292-293` | Populate and validate on load. | Resume equality with schedule and scaler active. |
| F8 | P3 | **No experiment-tracker integration.** | n/a | Optional TensorBoard/MLflow once metrics are richer (plan §4 defers it). | n/a |
| F9 | P3 | **Hangs are invisible.** No heartbeat or watchdog. | `status.json` written at start and end only | Periodic heartbeat file. | Stale heartbeat detectable. |

## G. Test coverage

Before 2026-10-25 (30 tests): numerical parity with upstream, causal mask, shifted
targets, weighted evaluation, config validation, offline tokenizer failure, RNG round trip,
atomic-save failure, exact multi-process resume, SIGTERM, prediction traces.

Added 2026-10-25 (39 tests plus the 2-test opt-in oracle): init statistics, initial loss,
tie validation and round trip, decay groups, betas, schedule table, config validation of the
new fields, clipping (scaled, untouched, small, non-finite), SDPA versus explicit and causality,
GELU exact equality, resume-guard fingerprint, preset parameter count, exact resume with every
feature active (two accumulation shapes), accumulation versus full batch, and the drift guard.

Added 2026-10-26: `uint16` token-file round trips (C2/C3) and `uint32` files for Qwen3. Tests
still missing belong to open items in the backlog (F5 relocated checkpoint, F7 active scaler,
D4 accelerators). CI is CPU-only and does not run the oracle; skipping accelerators or the
oracle there is not evidence they work.

## Order of work

The order of the remaining items is kept in [the backlog](backlog.md#5-recommended-next-steps).
Done so far: the recipe core (A1–A6, B1–B6) and the resume guard (F2) behind
`pretrain_gpt2_124m.yaml`, with `pretrain_gpt_162m.yaml` frozen (2026-10-25); the pretrained
oracle (E1); `uint16` token files for registered corpora (C2, C5); and the repository's first
commit (F1, 2026-10-25).

## Decisions

Resolved 2026-10-25:

- **Pretrained GPT-2 weights as a test oracle only (E1): yes.** Implemented in
  `tests/reference/gpt2_weights.py`, behind the optional `oracle` dependency group
  (`uv sync --group oracle`) and an opt-in environment variable. Training code does not
  import it. (Since 2026-10-26, fine-tuning starts from released GPT-2 weights through
  `model/gpt2/pretrained.py`; this mapping checks that those files are OpenAI's.)
- **Residual-scaled init (A4): a flag, defaulting to nanoGPT's scaling.**
  `model.residual_scaled_init` (default `true` under `init: gpt2`). nanoGPT also uses 0.02 for
  the position table, where OpenAI's released code uses 0.01; this follows nanoGPT.
- **Target (L3): GPT-2 small architecture on a smaller budget**, not a match to ~2.85 loss
  (about 4 days on 8×A100 per nGPT-r). The budget is not yet chosen; it must be set from the
  throughput pilot (D6) and declared, together with the target loss, before the run.

The open decision, which corpus to pretrain on, is D1 in [the backlog](backlog.md#4-open-decisions).

