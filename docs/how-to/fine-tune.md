# Fine-tune from released weights (continued pretraining and SFT)

`model.init_from` starts the same trainer from OpenAI's GPT-2 small or medium, Qwen3-0.6B
base, or any finished run's `model.pt`. Raw text continues pretraining; `data.format:
instructions` is supervised fine-tuning, following chapter 7 of *Build a Large Language
Model (From Scratch)* and chapter 8 of *Build a Reasoning Model (From Scratch)*.

```bash
uv run llm-training prepare --config configs/sft_gpt2_medium_alpaca.yaml   # ~1.7 GB once
uv run llm-training train --config configs/sft_gpt2_medium_alpaca.yaml     # ~8-10 min on MPS
```

`configs/sft_qwen3_0_6b_math_distill.yaml` is chapter 8's reasoning distillation (an
estimated 18–22 hours). The two `cpt_*_tinystories.yaml` recipes are plumbing tests only: a
long run on children's stories would erode the models' general skills. See
[the fine-tuning guide](../finetuning.md) for measured costs and the book parity checks.

## Use the result

```bash
# Chapter 7: responses to the 110 held-out instructions, one JSONL record each.
uv run llm-training respond --checkpoint runs/sft/<run>/model.pt \
  --config runs/sft/<run>/config.resolved.yaml --device mps

# Chapter 8: MATH-500 accuracy of a Qwen3 model.pt (or a full checkpoint), same scorer as GRPO.
uv run llm-training math-eval --model runs/sft/<run>/model.pt --limit 100

# Free-form text; the tokenizer defaults to the checkpoint's family.
uv run llm-training generate --checkpoint runs/cpt/<run>/model.pt --prompt "Once upon a time"
```

`respond` follows chapter 7. It decodes greedily from `format_input` until end-of-text,
writes the model's response next to the reference, and skips recorded examples when rerun.
Chapter 7 then has a larger model grade the responses; that grading step is not built
([backlog](../backlog.md)).
