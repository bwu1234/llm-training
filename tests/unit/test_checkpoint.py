import random

import numpy as np
import pytest
import torch

from llm_training.training.checkpoint import atomic_save, capture_rng, isolated_rng, restore_rng


def test_rng_round_trip_and_isolation():
    state = capture_rng("cpu")
    expected = (random.random(), np.random.rand(), torch.rand(3))
    restore_rng(state, "cpu")
    actual = (random.random(), np.random.rand(), torch.rand(3))
    assert expected[:2] == actual[:2]
    assert torch.equal(expected[2], actual[2])
    before = torch.get_rng_state()
    with isolated_rng("cpu"):
        torch.rand(10)
    assert torch.equal(before, torch.get_rng_state())


def test_failed_save_preserves_last_good_state(tmp_path, monkeypatch):
    path = tmp_path / "last.pt"
    atomic_save(path, {"step": 1})

    def fail(*args, **kwargs):
        raise OSError("simulated disk failure")

    monkeypatch.setattr(torch, "save", fail)
    with pytest.raises(OSError, match="disk failure"):
        atomic_save(path, {"step": 2})
    assert torch.load(path, weights_only=True) == {"step": 1}
    assert not path.with_suffix(".pt.tmp").exists()
