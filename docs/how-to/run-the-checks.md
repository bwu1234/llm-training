# Run the checks

```bash
uv run ruff check .
uv run ruff format --check .
uv run pytest tests/unit -q
uv run python -m tests.reference.compare
LLM_TEST_TOKENIZER_CACHE="$PWD/data/tokenizer" uv run pytest -q
uv build
# Opt-in: compare against the public GPT-2 124M weights (downloads ~550 MB once)
# and run Qwen3-0.6B against Hugging Face Transformers (~1.5 GB once; see ../qwen3.md).
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
See [initial evidence](../experiments/initial-baseline.md) for the original slice
and [chapter/resume evidence](../experiments/chapter-reproduction.md) for this extension.
