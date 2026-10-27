# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt at the repository root).
# Source: https://github.com/rasbt/LLMs-from-scratch (ch07/01_main-chapter-code)
# Modified: `format_input` is copied; the rest is new. Batching reproduces chapter 7's
# `custom_collate_fn` (and its exercise 7.2 prompt masking) from examples that already end
# in the end-of-text token; tests/unit/test_instructions.py checks the tensors are equal.

"""Instruction data for supervised fine-tuning: pinned datasets, prompts, splits, batching.

Each registered dataset follows one book recipe:

- `alpaca-1100`: *Build a Large Language Model (From Scratch)* chapter 7. 1,100 Alpaca-style
  instruction/input/output rows, the Alpaca prompt, and an 85/10/5 train/test/validation
  split in file order.
- `math-distill-r1`: *Build a Reasoning Model (From Scratch)* chapter 8. 12,000 MATH problems
  (MATH-500 excluded) with DeepSeek-R1's reasoning and answer, the chapter 3 math prompt, and
  25 validation examples after a `random.Random(123)` shuffle.

Every example ends with the tokenizer's end-of-text token, so the model learns to stop.
Examples longer than `max_length` tokens are dropped before splitting (chapter 8's filter).
The loss covers the response, plus the prompt unless `mask_prompt` is set.
"""

import hashlib
import json
import random
import urllib.request
from pathlib import Path

import torch
from torch.utils.data import Dataset

IGNORE_INDEX = -100  # cross_entropy's default ignore_index

DATASETS = {
    "alpaca-1100": {
        "description": "LLMs-from-scratch chapter 7 instruction data (Alpaca format)",
        "license": "LLMs-from-scratch repository license (Apache 2.0; see LICENSE.txt)",
        "url": "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/"
        "cdbd33e6d57a71e20ea31434841095fac50cecd8/ch07/01_main-chapter-code/instruction-data.json",
        "file": "instruction-data.json",
        "sha256": "7e09a9af353722ca15c18bc580f093f9c39d3178335a75177319cb1e1d163944",
        "prompt": "alpaca",
        "split": "ch07",
    },
    "math-distill-r1": {
        "description": "rasbt/math_distill: DeepSeek-R1 answers to 12,000 MATH training problems",
        "license": "Apache-2.0",
        "url": "https://huggingface.co/datasets/rasbt/math_distill/resolve/"
        "f48286d924d7568010c9e2ea7f2f94dac703b99e/data/deepseek-r1-math-train.json",
        "file": "deepseek-r1-math-train.json",
        "sha256": "0493f2ac5b553c73fb722741414502bf047e79cc2f5b7e4f5b373e592f4a9e87",
        "prompt": "math",
        "split": "ch08",
    },
}


def format_input(entry):
    instruction_text = (
        f"Below is an instruction that describes a task. "
        f"Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{entry['instruction']}"
    )

    input_text = f"\n\n### Input:\n{entry['input']}" if entry["input"] else ""

    return instruction_text + input_text


def prompt_text(name, entry):
    """The text a model sees before its response; generation starts right after it."""
    if DATASETS[name]["prompt"] == "alpaca":
        return format_input(entry) + "\n\n### Response:\n"
    from ..reasoning.grading import render_prompt

    return render_prompt(entry["problem"])


def reference_response(name, entry):
    if DATASETS[name]["prompt"] == "alpaca":
        return entry["output"]
    from ..reasoning.distill import format_distilled_answer

    return format_distilled_answer(entry)


def prepare_instructions(name, path):
    """Download the pinned file into `path` unless a verified copy is already there."""
    spec = DATASETS[name]
    target = Path(path) / spec["file"]
    if target.is_file() and _sha256(target.read_bytes()) == spec["sha256"]:
        return
    print(f"Downloading {spec['url']}", flush=True)
    with urllib.request.urlopen(spec["url"], timeout=60) as response:
        data = response.read()
    if _sha256(data) != spec["sha256"]:
        raise ValueError(f"Downloaded {spec['file']} does not match its pinned SHA-256")
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".partial")
    partial.write_bytes(data)
    partial.replace(target)


def load_instructions(name, path):
    """Return (rows, sha256) of the verified local file."""
    spec = DATASETS[name]
    target = Path(path) / spec["file"]
    try:
        data = target.read_bytes()
    except OSError as exc:
        raise ValueError(f"{target} is missing; run `llm-training prepare` first") from exc
    digest = _sha256(data)
    if digest != spec["sha256"]:
        raise ValueError(f"{target} differs from the pinned file; run `llm-training prepare`")
    return json.loads(data), digest


def encode_example(name, entry, tokenizer):
    """Token ids ending in end-of-text, and how many leading ids are prompt.

    Chapter 7 encodes the whole text at once and measures its prompt as `format_input` alone,
    so the `### Response:` header counts as response. Chapter 8 encodes prompt and answer
    separately and concatenates them.
    """
    if DATASETS[name]["prompt"] == "alpaca":
        ids = tokenizer.encode_ordinary(prompt_text(name, entry) + entry["output"])
        prompt_len = len(tokenizer.encode_ordinary(format_input(entry)))
    else:
        prompt_ids = tokenizer.encode_ordinary(prompt_text(name, entry))
        ids = prompt_ids + tokenizer.encode_ordinary(reference_response(name, entry))
        prompt_len = len(prompt_ids)
    ids.append(tokenizer.eot_token)
    return {"token_ids": ids, "prompt_len": min(prompt_len, len(ids) - 1)}


def split_instructions(name, rows, tokenizer, max_length, limit=None):
    """Return ({split: examples}, counts of dropped rows). Splits follow the book.

    Like chapter 8, rows with missing or malformed fields are skipped rather than fatal.
    """
    rows = rows[:limit] if limit is not None else rows
    examples = []
    dropped = {"malformed": 0, "too_long": 0}
    for index, row in enumerate(rows):
        try:
            example = encode_example(name, row, tokenizer) | {"index": index}
        except KeyError, TypeError, ValueError:
            dropped["malformed"] += 1
            continue
        if len(example["token_ids"]) > max_length:
            dropped["too_long"] += 1
            continue
        examples.append(example)
    if DATASETS[name]["split"] == "ch07":
        train_portion = int(len(examples) * 0.85)
        test_portion = int(len(examples) * 0.1)
        splits = {
            "train": examples[:train_portion],
            "test": examples[train_portion : train_portion + test_portion],
            "validation": examples[train_portion + test_portion :],
        }
    else:
        random.Random(123).shuffle(examples)
        splits = {"train": examples[25:], "validation": examples[:25]}
    if not splits["train"] or not splits["validation"]:
        raise ValueError(f"Too few {name} examples for a train/validation split")
    return splits, dropped


class InstructionDataset(Dataset):
    def __init__(self, examples):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]


def collate_instructions(batch, pad_token_id, mask_prompt=False):
    """Pad to the longest example; targets are inputs shifted by one, -100 where ignored.

    Equal to chapter 7's `custom_collate_fn` on the same examples without their final
    end-of-text token (it appends that token as the first pad, then ignores the other pads).
    """
    length = max(len(example["token_ids"]) for example in batch) - 1
    inputs = torch.full((len(batch), length), pad_token_id, dtype=torch.long)
    targets = torch.full((len(batch), length), IGNORE_INDEX, dtype=torch.long)
    for row, example in enumerate(batch):
        ids = torch.tensor(example["token_ids"], dtype=torch.long)
        inputs[row, : len(ids) - 1] = ids[:-1]
        targets[row, : len(ids) - 1] = ids[1:]
        if mask_prompt:
            targets[row, : max(example["prompt_len"] - 1, 0)] = IGNORE_INDEX
    return inputs, targets


def _sha256(data):
    return hashlib.sha256(data).hexdigest()
