import socket
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import pytest
import torch
import yaml
from torch.utils.data import DataLoader, TensorDataset

from llm_training.config import load_config
from llm_training.data.dataset import SlidingWindowTokenDataset
from llm_training.evaluation import evaluate
from llm_training.model.gpt2.model import GPTModel
from llm_training.model.gpt2.tokenizer import load_tokenizer
from tests.reference import baseline
from tests.reference.compare import TINY_CONFIG, compare


def test_upstream_numerical_parity():
    assert compare()["passed"]


def test_causal_mask():
    torch.manual_seed(123)
    model = GPTModel(TINY_CONFIG).eval()
    a = torch.randint(0, 64, (2, 8))
    b = a.clone()
    b[:, 4:] = (b[:, 4:] + 1) % 64
    with torch.inference_mode():
        torch.testing.assert_close(model(a)[:, :4], model(b)[:, :4], atol=0, rtol=0)


class FakeTokenizer:
    def encode(self, text, allowed_special):
        return list(range(len(text)))


def test_shifted_targets_match_reference():
    for cls in (SlidingWindowTokenDataset, baseline.GPTDatasetV1):
        data = cls("abcdefghijk", FakeTokenizer(), max_length=4, stride=4)
        assert len(data) == 2
        x, y = data[1]
        assert x.tolist() == [4, 5, 6, 7]
        assert y.tolist() == [5, 6, 7, 8]


def test_weighted_evaluation_and_mode_restore():
    torch.manual_seed(1)
    model = GPTModel(TINY_CONFIG)
    x, y = torch.randint(0, 64, (3, 8)), torch.randint(0, 64, (3, 8))
    model.eval()
    with torch.inference_mode():
        expected = torch.nn.functional.cross_entropy(model(x).flatten(0, 1), y.flatten()).item()
    model.train()
    actual = evaluate(model, DataLoader(TensorDataset(x, y), batch_size=2), "cpu")
    assert actual["tokens"] == 24
    assert actual["loss"] == pytest.approx(expected, abs=1e-6)
    assert model.training


@pytest.mark.parametrize(
    "section,key,value",
    [
        ("model", "emb_dim", 31),
        ("model", "vocab_size", 64),
        ("training", "max_steps", 0),
        ("training", "batch_size", True),
        ("runtime", "device", "typo"),
        ("model", "drop_rate", float("nan")),
        ("model", "qkv_bias", "false"),
        ("data", "unknown", 1),
    ],
)
def test_invalid_configuration(tmp_path, section, key, value):
    raw = deepcopy(yaml.safe_load(Path("configs/smoke.yaml").read_text()))
    raw[section][key] = value
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError):
        load_config(path)


def test_missing_tokenizer_fails_without_network(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Unexpected network access")

    monkeypatch.setattr(socket, "create_connection", forbidden)
    with pytest.raises(ValueError, match="prepare"):
        load_tokenizer(tmp_path)


def test_package_import_has_no_heavy_side_effects():
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import llm_training; import sys; "
            "assert 'torch' not in sys.modules; assert 'tiktoken' not in sys.modules",
        ],
        check=True,
    )
