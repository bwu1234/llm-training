# Initial package verification

Performed 2026-10-24 America/Detroit (2026-10-25 UTC).

## Scope

M0 and the initial M1 slice: package/environment setup, selected upstream Python
implementations reorganized into modules, bounded training, artifacts, and tiny
numerical comparison. The full ten-epoch chapter-sized reproduction was not run.

Environment: Python **3.14.9**, Torch **2.14.1**, macOS arm64. Host memory is 64 GiB;
MPS is available and CUDA is unavailable. All numerical and training verification
here used **CPU FP32**. Accelerator support is not validated by these results.

## Results

- Numerical parity passed at declared `atol=1e-6`, `rtol=1e-5`: maximum absolute
  differences were **0.0** for logits, loss, gradients, and state after one AdamW
  update. The tiny comparison uses identical weights, synthetic token batches,
  and matching nonzero dropout draws. See [raw parity report](baseline-parity.json).
- The 3,242,240-parameter smoke model completed four optimizer updates and 128
  target-token presentations. Validation on the same first 32 target tokens
  changed from **10.905166** to **10.783382** NLL. This checks execution on a
  repetitive synthetic fixture, not generalization. See [raw smoke result](smoke-result.json).
- **15 tests passed**, including offline training and a fresh-process model reload
  with identical greedy token IDs. Network connections were blocked in the
  training/reload subprocesses after explicit tokenizer preparation.
- Ruff lint and format checks passed. Wheel and source distribution builds passed.
- A wheel installed offline into a separate Python 3.14.9 environment imported
  successfully outside the repository, including the data modules and bundled
  source-provenance manifest. A broad ignore rule initially excluded the data
  package; it was restricted to root-level generated data, and the clean install
  was repeated successfully. CI includes an outside-checkout wheel import check.

The retained local run is `runs/20261025T015329Z-813fc8cc/`, containing resolved
configuration, data/tokenizer hashes, source provenance, metrics, weights, and
sample output. Large/generated artifacts are Git-ignored; the small result JSONs
in this directory preserve the reported measurements.

## Reproduce

Run from the repository root:

```bash
uv sync --frozen
uv run llm-training prepare --config configs/smoke.yaml
uv run python -m tests.reference.compare
LLM_TEST_TOKENIZER_CACHE="$PWD/data/tokenizer" uv run pytest -q
uv run llm-training train --config configs/smoke.yaml
```

## Remaining gates

The GPT 162M preset preserves architecture/hyperparameters and the character split;
the trainer deliberately changes evaluation RNG handling and sampling cadence.
One-update parity does not establish full-run trajectory equivalence. Full-state
resume, document-level splitting/deduplication, larger-corpus pretraining, and
accelerator/distributed validation remain pending. No pretrained weights were
loaded, and the GPT-2 tokenizer was reused rather than trained.
