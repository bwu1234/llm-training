"""Instruction data: book parity for prompts, batching and loss; splits; masked evaluation."""

import os
import random
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader

from llm_training.data import instructions
from llm_training.data.instructions import (
    IGNORE_INDEX,
    InstructionDataset,
    collate_instructions,
    encode_example,
    split_instructions,
)
from llm_training.evaluation import evaluate
from llm_training.model.qwen3.model import Qwen3Model
from llm_training.training.loss import next_token_cross_entropy
from tests.reference import instruction_baseline as book
from tests.unit.test_qwen3 import TINY

EOT = 50256


class CharTokenizer:
    """One id per character, with both the source (`encode`) and trainer interfaces."""

    eot_token = eos_token_id = EOT
    n_vocab = 0x110000

    def encode_ordinary(self, text):
        return [ord(c) for c in text]

    def encode(self, text, chat_wrapped=False):
        return self.encode_ordinary(text)


ALPACA = [
    {"instruction": "Rewrite in passive voice.", "input": "The cat ate the fish.", "output": "x"},
    {"instruction": "Name a color.", "input": "", "output": "Blue."},
    {"instruction": "Translate to French.", "input": "Thank you", "output": "Merci beaucoup."},
]
MATH = [
    {"problem": "1+1?", "message_thinking": "<think>Add.</think>", "message_content": "\\boxed{2}"},
    {"problem": "2*3?", "message_content": "<think></think>\\boxed{6}"},
    {"problem": "Empty?", "message_thinking": "x", "message_content": "  "},  # malformed
    {"problem": "4-1?", "message_thinking": "Subtract one.", "message_content": "\\boxed{3}"},
]


def random_items(seed, count=6):
    generator = random.Random(seed)
    return [
        [generator.randrange(EOT) for _ in range(generator.randint(1, 9))] for _ in range(count)
    ]


# --- chapter 7 parity --------------------------------------------------------------------


def test_alpaca_examples_match_chapter_7_encoding_and_instruction_length():
    tokenizer = CharTokenizer()
    texts = book.InstructionDataset(ALPACA, tokenizer).encoded_texts
    lengths = book.InstructionDatasetWithMasking(ALPACA, tokenizer).instruction_lengths
    for entry, text, length in zip(ALPACA, texts, lengths, strict=True):
        example = encode_example("alpaca-1100", entry, tokenizer)
        assert example["token_ids"] == text + [EOT]
        assert example["prompt_len"] == length


@pytest.mark.parametrize("seed", range(5))
def test_collate_equals_chapter_7_collate(seed):
    items = random_items(seed)
    expected = book.custom_collate_fn(items)
    actual = collate_instructions(
        [{"token_ids": item + [EOT], "prompt_len": 1} for item in items], EOT
    )
    for got, want in zip(actual, expected, strict=True):
        assert torch.equal(got, want)


@pytest.mark.parametrize("seed", range(5))
def test_prompt_masking_equals_exercise_7_2(seed):
    items = random_items(seed)
    lengths = [random.Random(seed + i).randint(1, len(item)) for i, item in enumerate(items)]
    expected = book.custom_collate_with_masking_fn(list(zip(lengths, items, strict=True)))
    examples = [
        {"token_ids": item + [EOT], "prompt_len": length}
        for item, length in zip(items, lengths, strict=True)
    ]
    actual = collate_instructions(examples, EOT, mask_prompt=True)
    for got, want in zip(actual, expected, strict=True):
        assert torch.equal(got, want)


def test_gpt2_alpaca_examples_match_chapter_7_with_the_real_tokenizer():
    cache = os.environ.get("LLM_TEST_TOKENIZER_CACHE")
    if not cache:
        pytest.skip("Set LLM_TEST_TOKENIZER_CACHE to a prepared GPT-2 cache")
    from llm_training.model.registry import load_tokenizer

    tokenizer, _ = load_tokenizer("gpt2", Path(cache))
    texts = book.InstructionDataset(ALPACA, tokenizer).encoded_texts
    lengths = book.InstructionDatasetWithMasking(ALPACA, tokenizer).instruction_lengths
    for entry, text, length in zip(ALPACA, texts, lengths, strict=True):
        example = encode_example("alpaca-1100", entry, tokenizer)
        assert (example["token_ids"], example["prompt_len"]) == (text + [EOT], length)


# --- chapter 8 parity --------------------------------------------------------------------


def test_math_examples_and_split_match_chapter_8():
    tokenizer = CharTokenizer()
    rows = MATH * 20  # enough rows for chapter 8's 25 validation examples
    expected, skipped = book.build_examples(rows, tokenizer)
    # A limit that drops the longest row, so the length filter is exercised too.
    limit = max(len(e["token_ids"]) for e in expected) - 1
    expected, removed = book.filter_examples_by_max_len(expected, max_len=limit)
    splits, dropped = split_instructions("math-distill-r1", rows, tokenizer, max_length=limit)
    assert dropped == {"malformed": skipped, "too_long": removed}
    assert removed > 0
    random.Random(123).shuffle(expected)

    def strip(examples):
        return [{k: e[k] for k in ("token_ids", "prompt_len")} for e in examples]

    assert strip(splits["train"]) == strip(expected[25:])
    assert strip(splits["validation"]) == strip(expected[:25])


def test_masked_batch_loss_and_gradient_equal_chapter_8_per_example_loss():
    torch.manual_seed(0)
    model = Qwen3Model(TINY)
    generator = torch.Generator().manual_seed(1)
    ids = torch.randint(0, int(TINY["vocab_size"]), (12,), generator=generator).tolist()
    example = {"token_ids": ids, "prompt_len": 5}

    expected = book.compute_example_loss(model, example, "cpu")
    expected_grads = torch.autograd.grad(expected, list(model.parameters()))
    inputs, targets = collate_instructions([example], pad_token_id=0, mask_prompt=True)
    actual = next_token_cross_entropy(inputs, targets, model, "cpu")
    actual_grads = torch.autograd.grad(actual, list(model.parameters()))
    torch.testing.assert_close(actual, expected, rtol=0, atol=1e-6)
    for got, want in zip(actual_grads, expected_grads, strict=True):
        torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-7)


QWEN3_DIR = Path(os.environ.get("LLM_QWEN3_DIR", "data/qwen3-0.6b-base"))


def test_math_examples_match_chapter_8_with_the_real_qwen3_tokenizer():
    if not (QWEN3_DIR / "tokenizer-base.json").is_file():
        pytest.skip("Set LLM_QWEN3_DIR to a directory holding the pinned Qwen3 tokenizer")
    from llm_training.model.qwen3.tokenizer import Qwen3TextTokenizer, Qwen3Tokenizer

    source = Qwen3Tokenizer(QWEN3_DIR / "tokenizer-base.json")
    ours = Qwen3TextTokenizer(QWEN3_DIR / "tokenizer-base.json")
    rows = [r for r in MATH if r["message_content"].strip()]
    expected, _ = book.build_examples(rows, source)
    for row, want in zip(rows, expected, strict=True):
        got = encode_example("math-distill-r1", row, ours)
        assert (got["token_ids"], got["prompt_len"]) == (want["token_ids"], want["prompt_len"])
    # The model pads its vocabulary past the tokenizer's; those ids decode to nothing.
    assert ours.decode_single_token_bytes(151_700) == b""


# --- splits, evaluation, data files ------------------------------------------------------


def test_alpaca_split_is_85_10_5_in_file_order_and_limit_applies_first():
    rows = [{"instruction": f"Task {i}", "input": "", "output": "ok"} for i in range(100)]
    splits, dropped = split_instructions("alpaca-1100", rows, CharTokenizer(), max_length=1024)
    assert {k: len(v) for k, v in splits.items()} == {"train": 85, "test": 10, "validation": 5}
    assert [e["index"] for e in splits["train"]] == list(range(85))
    assert dropped == {"malformed": 0, "too_long": 0}
    splits, _ = split_instructions("alpaca-1100", rows, CharTokenizer(), 1024, limit=40)
    assert {k: len(v) for k, v in splits.items()} == {"train": 34, "test": 4, "validation": 2}


def test_every_example_has_a_supervised_target_even_with_the_prompt_masked():
    for entry in ALPACA:
        example = encode_example("alpaca-1100", entry, CharTokenizer())
        _, targets = collate_instructions([example], EOT, mask_prompt=True)
        assert (targets != IGNORE_INDEX).sum() >= 1
        assert targets[0, -1] == EOT  # the model learns to stop


def test_evaluation_counts_only_supervised_targets():
    torch.manual_seed(0)
    model = Qwen3Model(TINY)
    examples = [
        {"token_ids": list(range(1, 4)), "prompt_len": 2},
        {"token_ids": list(range(1, 10)), "prompt_len": 3},
    ]
    loader = DataLoader(
        InstructionDataset(examples),
        batch_size=2,
        collate_fn=lambda b: collate_instructions(b, 0, mask_prompt=True),
    )
    inputs, targets = next(iter(loader))
    supervised = int((targets != IGNORE_INDEX).sum())
    assert supervised == (3 - 2) + (9 - 3)
    with torch.no_grad():
        nll = torch.nn.functional.cross_entropy(
            model(inputs).flatten(0, 1), targets.flatten(), reduction="sum"
        )
    result = evaluate(model, loader, "cpu")
    assert result["tokens"] == supervised
    assert result["loss"] == pytest.approx(nll.item() / supervised)


def test_load_rejects_a_file_that_differs_from_the_pin(tmp_path):
    spec = instructions.DATASETS["alpaca-1100"]
    (tmp_path / spec["file"]).write_text("[]")
    with pytest.raises(ValueError, match="differs from the pinned file"):
        instructions.load_instructions("alpaca-1100", tmp_path)
    with pytest.raises(ValueError, match="missing"):
        instructions.load_instructions("alpaca-1100", tmp_path / "absent")
