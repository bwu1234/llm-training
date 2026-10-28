# Train a model with readable output (TinyStories, about a day)

`configs/tinystories_51m.yaml` trains a **51,213,824-parameter** GPT (8 layers, width 512,
8 heads, context 512, the [GPT-2 small recipe](../gpt2.md#gpt-2-small-recipe)) on
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
- **Samples.** `output.sample_every: N` adds a sampled continuation to the records of updates
  1, N + 1, 2N + 1, ... in `metrics.jsonl`, using `output.temperature` / `output.top_k`.
  Sampling draws from a private generator seeded by (seed, update), so samples never change
  training. A resumed run records the same text on CPU (tested bitwise); on MPS the ~1e-7
  weight drift could in principle flip a draw (the 6-update MPS check matched).
- **`generate`** samples from any `model.pt` or full checkpoint (temperature 0 is greedy) and
  stops at `<|endoftext|>`. Use `--device cpu` (the default) during training so it does not
  compete for the GPU.

Measured on an M2 Max (FP32): **4.41 s per update** (32,768 tokens, about 7,430 tokens/s) over
the first 60 updates, so the 14,155-update epoch takes about 17–18 hours. Stop and resume work
on MPS, but not bitwise: MPS kernels are not bitwise deterministic. Two identical fresh runs
differ by up to 1.2e-7 in the weights, and a resumed run differs from an uninterrupted one by
the same amount, with identical batches.

To pause and continue the run, see [Stop and resume](stop-and-resume.md). To watch its loss
and samples, see [View a run](view-a-run.md).
