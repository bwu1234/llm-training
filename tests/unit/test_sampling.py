"""Sampled generation, sampling config validation, and the TinyStories preset."""

from pathlib import Path

import pytest
import torch
import yaml
from torch import nn

from llm_training.lm.config import config_from_dict, load_config
from llm_training.lm.generation import generate_greedy, generate_sampled
from llm_training.model.gpt2.model import GPTModel
from tests.reference.compare import TINY_CONFIG


def tiny_model():
    torch.manual_seed(0)
    return GPTModel(TINY_CONFIG | {"drop_rate": 0.0}).eval()


def prompt():
    return torch.tensor([[1, 2, 3]])


def sampled(model, seed, **options):
    generator = torch.Generator().manual_seed(seed)
    return generate_sampled(
        model, prompt(), 12, TINY_CONFIG["context_length"], **options, generator=generator
    )


def test_temperature_zero_and_top_one_are_greedy():
    model = tiny_model()
    greedy = generate_greedy(model, prompt(), 12, TINY_CONFIG["context_length"])
    assert torch.equal(sampled(model, 0), greedy)
    assert torch.equal(sampled(model, 0, temperature=1.0, top_k=1), greedy)
    # A top_k at temperature 0 keeps the argmax, so the seed `_sample` passes changes nothing.
    assert all(torch.equal(sampled(model, seed, top_k=3), greedy) for seed in range(4))


def test_seeded_sampling_reproduces_and_leaves_global_rng_untouched():
    model = tiny_model()
    state = torch.get_rng_state()
    first = sampled(model, 7, temperature=1.5)
    assert torch.equal(torch.get_rng_state(), state)
    assert torch.equal(sampled(model, 7, temperature=1.5), first)
    assert any(not torch.equal(sampled(model, s, temperature=1.5), first) for s in range(1, 5))


class FixedLogits(nn.Module):
    def __init__(self, logits):
        super().__init__()
        self.register_buffer("logits", torch.tensor(logits))

    def forward(self, inputs):
        return self.logits[None, None, :].expand(inputs.shape[0], inputs.shape[1], -1)


def test_draws_stay_inside_top_k_and_stop_at_end_of_text():
    model = FixedLogits([0.0, 3.0, 2.9, 0.1, 2.8, 0.2])
    out = generate_sampled(
        model,
        prompt(),
        300,
        8,
        temperature=5.0,
        top_k=3,
        generator=torch.Generator().manual_seed(0),
    )
    drawn = set(out[0, 3:].tolist())
    assert drawn == {1, 2, 4}
    stopped = generate_sampled(model, prompt(), 50, 8, temperature=0.0, eos_id=1)
    assert torch.equal(stopped, prompt())


def raw_smoke():
    return yaml.safe_load(Path("configs/smoke.yaml").read_text())


@pytest.mark.parametrize(
    "section,values",
    [
        ("output", {"temperature": -0.1}),
        ("output", {"temperature": float("inf")}),
        ("output", {"top_k": 0}),
        ("output", {"top_k": 50258}),
        ("output", {"sample_every": 0}),
        ("data", {"format": "parquet"}),
        ("data", {"format": "tokens"}),  # no corpus
        ("data", {"format": "tokens", "corpus": "unknown"}),
        ("data", {"format": "tokens", "corpus": "tinystories", "train_ratio": 0.5}),
        ("data", {"format": "tokens", "corpus": "tinystories", "sha256": "0" * 64}),
        ("data", {"corpus": "tinystories"}),  # text format
    ],
)
def test_invalid_sampling_and_data_format_values_are_rejected(section, values):
    raw = raw_smoke()
    raw[section] |= values
    with pytest.raises(ValueError):
        config_from_dict(raw, Path("configs").resolve())


def test_defaults_keep_book_behaviour():
    cfg = load_config("configs/smoke.yaml")
    assert (cfg.data.format, cfg.data.corpus) == ("text", None)
    assert (cfg.output.temperature, cfg.output.top_k, cfg.output.sample_every) == (0.0, None, None)


def test_tinystories_preset_is_valid_and_has_51m_parameters():
    cfg = load_config("configs/tinystories_51m.yaml")
    assert (cfg.data.format, cfg.data.corpus, cfg.runtime.device) == (
        "tokens",
        "tinystories",
        "mps",
    )
    with torch.device("meta"):
        model = GPTModel(vars(cfg.model))
    assert sum(p.numel() for p in model.parameters()) == 51_213_824
    tokens_per_update = (
        cfg.training.batch_size * cfg.training.grad_accum_steps * cfg.model.context_length
    )
    assert tokens_per_update == 32_768
