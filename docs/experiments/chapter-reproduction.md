# Chapter-sized reproduction and checkpoint recovery

This report covers the ten-epoch chapter-5 recipe and full-state resume extension,
executed 2026-10-24 America/Detroit (2026-10-25 UTC).

## Result

**Passed**, on Python **3.14.9**, Torch **2.14.1**, macOS arm64, CPU FP32 with
four intra-op threads. The uninterrupted upstream loop and the packaged run,
stopped at update 46 and resumed to update 90 in a fresh process, had:

- **0.0** maximum absolute difference in final model weights and optimizer state.
- Identical Python, NumPy, and Torch RNG states.
- Identical values for all **18** monitored training/validation loss pairs and
  token counters, and identical text for all **10** epoch samples.
- Final full validation NLL **6.598802**, versus initialization NLL **10.981107**,
  on **512** target tokens; final perplexity **734.214788**. The lowest periodically
  monitored validation loss was **6.161668** at update **56**, so the last model
  is not the best monitored checkpoint. The child run retained that best model.

The independent upstream training loop took **81.63 seconds**. Checkpoint writes,
restart, startup, and reporting add overhead to the packaged workflow; these
timings are not an optimization benchmark. The test suite passed **21 tests**,
including offline training/reload, repeated exact resume, source-loop parity,
SIGTERM, and failed-save recovery. CI configuration was updated; no remote CI
result is claimed.

A clean wheel installation with the locked runtime dependencies also resumed a
two-update smoke checkpoint through update four from outside the repository.
Network connections were blocked during that resume. The installed wheel's code
identity matched the source environment, so the same compatibility guard passed.

See the [retained comparison JSON](chapter-reproduction.json). The complete local
artifacts are under `runs/chapter-reproduction-20261025/`, with the stopped child
at `port-runs/20261025T020635Z-41adad03/` and the completed resumed child at
`port-runs/20261025T020735Z-8779a178/`. These directories contain weights, full
checkpoints, source logs, data hashes, metrics, samples, and parent lineage and
are Git-ignored because of their size.

## Contract

Compare a fresh run of the frozen upstream `train_model_simple` and model
definitions against the packaged trainer, stopped at update 46 and resumed in a
new process. Use the pinned `the-verdict.txt`, GPT-2 tokenization, the original
90/10 character split, context 256, width 768, 12 layers/heads, dropout 0.1, batch
size 2, seed 123, plain AdamW at 0.0005 with weight decay 0.1, ten epochs, monitoring
every five zero-based steps, and 50 greedy tokens per epoch.

The source architecture has **162,419,712 parameters**, including its untied
output head. There are 18 training windows and two validation windows, giving
nine updates/epoch, **90 updates**, and **46,080 token presentations**. These are
repeated presentations of a tiny text, not that many unique corpus tokens.

The declared acceptance tolerances, set before running, are `atol=1e-6`,
`rtol=1e-5` for weights and optimizer tensors. Compare all 18 monitored
training/validation loss pairs and token counters, all ten epoch samples, and
final Python/NumPy/Torch RNG state. The literal source helper averages batch
losses; the additional final project evaluation reports token-weighted NLL.

## Recovery checks

Small CPU integration runs verify repeated fresh-process resume at a mid-epoch
cursor, the exact epoch boundary, and another mid-epoch cursor. They compare
final weights, AdamW state, live/epoch-start RNG, all batch hashes, metrics, and
epoch samples against uninterrupted runs. Both baseline monitoring and source
book monitoring are covered. SIGTERM recovery and failed atomic writes are
tested separately.

Checkpoints store versioned model/optimizer state, explicit null scheduler/scaler
states (neither feature is active), resolved recipe/hash, data/tokenizer identity,
runtime/code identity, best validation metric/step, progress, and RNG/history.
Resume reconstructs the workers=0 deterministic iterator and restores live RNG
after replaying the saved data cursor. Runtime/code/data drift is rejected.

## Reproduce

```bash
uv sync --frozen
uv run llm-training prepare --config configs/pretrain_gpt_162m.yaml
LLM_TEST_TOKENIZER_CACHE="$PWD/data/tokenizer" uv run pytest -q
uv run python -m scripts.compare_training_runs \
  --config configs/pretrain_gpt_162m.yaml \
  --output runs/chapter-reproduction-new \
  --stop-after 46
```

Use a fresh output directory. The complete raw artifacts live in that directory;
the smaller comparison JSON can be retained with an experiment report. Each
phase can also be run with `--phase reference`, `--phase port`, or `--phase compare`
against the same output directory to inspect/retry a failed phase.

## Limits

Exact recovery is a CPU FP32 result for the recorded software/hardware and thread
configuration. It does not establish cross-platform or accelerator equivalence.
The recipe retains the book's tiny corpus and character split; lower training
loss or matching samples does not establish broad language capability. Larger
corpora, document-level deduplication/splits, accumulation/scheduling, AMP, and
distributed training remain separate milestones.

The checkpoint design follows PyTorch's recommendation to save optimizer state
alongside model weights, and additionally preserves experiment/data/RNG/cursor
state for this recovery contract. See [PyTorch saving/loading guidance](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html).
