# Inspect next-token predictions

Open the [saved eight-step view](../experiments/next-token-predictions.md) for
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

The [saved JSON trace](../experiments/next-token-predictions.json) records the
same eight-step example. For its first step, the model assigns about 22.89% to
the punctuation token `?"` and 18.50% to ` know`; it appends `?"`.
