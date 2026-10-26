import math

import pytest
import torch
from torch import nn

from llm_training.generation import generate_greedy
from llm_training.model.gpt import GPTModel
from llm_training.prediction import render_markdown, trace_next_tokens
from tests.reference.compare import TINY_CONFIG


class Tokenizer:
    def encode(self, text):
        return [0, 1, 2, 1, 0] if text else []

    def decode(self, ids):
        return "".join(self.decode_single_token_bytes(i).decode(errors="replace") for i in ids)

    def decode_single_token_bytes(self, token_id):
        return {0: b"\xc3", 1: b" |`", 2: b"\n", 3: b" winner"}.get(token_id, b"x")


class FixedModel(nn.Module):
    def __init__(self, logits):
        super().__init__()
        self.register_buffer("logits", torch.tensor(logits))
        self.contexts = []

    def forward(self, inputs):
        assert not self.training
        self.contexts.append(inputs[0].tolist())
        return self.logits[None, None, :].expand(inputs.shape[0], inputs.shape[1], -1)


def test_probabilities_are_full_vocabulary_and_context_tracks_winner():
    model = FixedModel([math.log(p) for p in (0.1, 0.2, 0.3, 0.4)])
    report = trace_next_tokens(
        model, Tokenizer(), "prompt", context_size=3, max_new_tokens=2, top_k=2
    )
    assert model.training
    assert model.contexts == [[2, 1, 0], [1, 0, 3]]
    assert report["generated_token_ids"] == [3, 3]
    assert [s["context_tokens_dropped"] for s in report["steps"]] == [2, 3]
    for step in report["steps"]:
        assert [r["token_id"] for r in step["candidates"]] == [3, 2]
        assert [r["probability"] for r in step["candidates"]] == pytest.approx([0.4, 0.3])
        assert step["displayed_probability_mass"] == pytest.approx(0.7)
        assert step["other_probability_mass"] == pytest.approx(0.3)


def test_ties_raw_bytes_and_markdown_escaping():
    report = trace_next_tokens(
        FixedModel([0.0] * 4), Tokenizer(), "prompt", context_size=8, max_new_tokens=1, top_k=4
    )
    assert report["generated_token_ids"] == [0]
    first = report["steps"][0]["candidates"][0]
    assert first["bytes_hex"] == "c3"
    assert first["complete_utf8"] is False
    markdown = render_markdown(report)
    assert "&#124;" in markdown
    assert "\\n" in markdown
    assert "bytes: c3" in markdown
    assert "25.0000%" in markdown


def test_trace_matches_original_greedy_generation_without_rng_or_mode_changes():
    torch.manual_seed(123)
    model = GPTModel(TINY_CONFIG)
    before = torch.get_rng_state()
    report = trace_next_tokens(
        model, Tokenizer(), "prompt", context_size=8, max_new_tokens=6, top_k=3
    )
    assert model.training
    assert torch.equal(before, torch.get_rng_state())
    model.eval()
    expected = generate_greedy(model, torch.tensor([[0, 1, 2, 1, 0]]), 6, 8)
    assert report["all_token_ids"] == expected[0].tolist()
    assert all(row["candidates"][0]["selected"] for row in report["steps"])


@pytest.mark.parametrize(
    "options",
    [
        {"top_k": 0},
        {"top_k": 5},
        {"max_new_tokens": 0},
        {"context_size": 0},
        {"prompt": ""},
    ],
)
def test_invalid_trace_arguments(options):
    arguments = {"context_size": 3, "max_new_tokens": 1, "top_k": 2, "prompt": "prompt"}
    arguments.update(options)
    model = FixedModel([0.0] * 4)
    with pytest.raises(ValueError):
        trace_next_tokens(model, Tokenizer(), **arguments)
    assert model.training


def test_nonfinite_model_output_restores_mode():
    model = FixedModel([float("nan"), 0.0])
    with pytest.raises(ValueError, match="nonfinite"):
        trace_next_tokens(model, Tokenizer(), "prompt", context_size=3, top_k=1)
    assert model.training
