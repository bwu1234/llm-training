"""Versioned atomic training state and explicit RNG capture for exact local resume."""

import hashlib
import json
import os
import random
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch

FORMAT_VERSION = 2


def config_hash(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()


def capture_rng(device):
    name, keys, pos, has_gauss, cached = np.random.get_state()
    state = {
        "python": random.getstate(),
        "numpy": [name, keys.tolist(), pos, has_gauss, cached],
        "torch_cpu": torch.get_rng_state(),
    }
    if device == "cuda":
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    elif device == "mps":
        state["torch_mps"] = torch.mps.get_rng_state()
    return state


def restore_rng(state, device):
    random.setstate(state["python"])
    name, keys, pos, has_gauss, cached = state["numpy"]
    np.random.set_state((name, np.array(keys, dtype=np.uint32), pos, has_gauss, cached))
    torch.set_rng_state(state["torch_cpu"])
    if device == "cuda":
        torch.cuda.set_rng_state_all(state["torch_cuda"])
    elif device == "mps":
        torch.mps.set_rng_state(state["torch_mps"])


@contextmanager
def isolated_rng(device):
    state = capture_rng(device)
    try:
        yield
    finally:
        restore_rng(state, device)


def atomic_save(path: Path, state):
    """Replace only after a successful, flushed write. The previous file survives failure."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        with temporary.open("wb") as stream:
            torch.save(state, stream)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def load_checkpoint(path):
    state = torch.load(path, map_location="cpu", weights_only=True)
    required = {
        "format_version",
        "config",
        "config_hash",
        "environment",
        "data_identity",
        "model_state_dict",
        "optimizer_state_dict",
        "rng",
        "progress",
        "history",
        "epoch_samples",
        "initial_evaluation",
        "best_validation",
        "best_step",
        "scheduler_state_dict",
        "scaler_state_dict",
    }
    if not isinstance(state, dict) or not required <= state.keys():
        raise ValueError("Not a full training checkpoint; model.pt weights cannot resume training")
    if state["format_version"] != FORMAT_VERSION:
        raise ValueError("Unsupported training checkpoint version")
    if state["config_hash"] != config_hash(state["config"]):
        raise ValueError("Checkpoint configuration hash mismatch")
    return state
