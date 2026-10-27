"""A fine-tuned model's responses to held-out instructions, as chapter 7 collects them.

Greedy decoding, stopping at end-of-text, from the same prompt the chapter generates from:
`format_input` for Alpaca rows (the model writes the `### Response:` header itself), and the
math prompt for MATH rows. Records append to a JSONL file one example at a time and a rerun
skips recorded examples. Scoring the responses (chapter 7 asks a larger model to grade them)
is left to the reader of that file.
"""

import json
from pathlib import Path

import torch

from .data.instructions import (
    DATASETS,
    format_input,
    load_instructions,
    prompt_text,
    reference_response,
    split_instructions,
)
from .lm.config import load_config, window_length
from .lm.generation import generate_sampled
from .prediction import load_inference_model


def respond(checkpoint, config, *, split, limit=None, max_new_tokens=256, device="cpu", output):
    cfg = load_config(config)
    if cfg.data.format != "instructions":
        raise ValueError("respond needs a recipe with data.format: instructions")
    name = cfg.data.dataset
    model, tokenizer, _, optimizer_steps, _ = load_inference_model(
        checkpoint, cfg.data.tokenizer_cache, device
    )
    model.eval()
    rows, digest = load_instructions(name, cfg.data.path)
    splits, _ = split_instructions(name, rows, tokenizer, window_length(cfg), cfg.data.limit)
    if split not in splits:
        raise ValueError(f"{name} has no {split} split; choose from {sorted(splits)}")
    examples = splits[split][:limit]
    output = Path(output)
    done = set()
    if output.is_file():
        for line in output.read_text().splitlines():
            if not line:
                continue
            record = json.loads(line)
            if (record["max_new_tokens"], record["checkpoint"], record["split"]) != (
                max_new_tokens,
                str(Path(checkpoint).resolve()),
                split,
            ):
                raise ValueError(f"{output} holds responses from other settings; choose --output")
            done.add(record["index"])
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8") as out:
        for example in examples:
            if example["index"] in done:
                continue
            entry = rows[example["index"]]
            alpaca = DATASETS[name]["prompt"] == "alpaca"
            prompt = format_input(entry) if alpaca else prompt_text(name, entry)
            ids = tokenizer.encode_ordinary(prompt)
            generated = generate_sampled(
                model,
                torch.tensor([ids], device=device),
                max_new_tokens,
                window_length(cfg),
                eos_id=tokenizer.eot_token,
            )[0].tolist()
            # Decode only the new tokens: slicing the decoded text by the prompt's length, as
            # chapter 7 does, would misalign if the tokenizer normalized the prompt.
            text = tokenizer.decode(generated[len(ids) :])
            if alpaca:
                text = text.replace("### Response:", "").strip()
            record = {
                "index": example["index"],
                "split": split,
                "prompt": prompt,
                "reference": reference_response(name, entry),
                "model_response": text,
                "response_tokens": len(generated) - len(ids),
                "max_new_tokens": max_new_tokens,
                "checkpoint": str(Path(checkpoint).resolve()),
                "optimizer_steps": optimizer_steps,
                "dataset_sha256": digest,
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            out.flush()
    return output
