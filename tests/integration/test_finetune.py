"""Supervised fine-tuning from a saved model.pt: exact resume and stage hand-off, both families.

Runs offline on CPU with tiny random models. The instruction file is a fixture, so each
subprocess re-pins the `alpaca-1100` entry to the fixture's SHA-256 before running the CLI.
"""

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import yaml

from llm_training.model.registry import build_model
from llm_training.training.checkpoint import atomic_save, load_checkpoint
from scripts.compare_training_runs import compare_states

QWEN3_DIR = Path(os.environ.get("LLM_QWEN3_DIR", "data/qwen3-0.6b-base"))

FAMILIES = {
    "gpt2": {
        "vocab_size": 50257,
        "context_length": 64,
        "emb_dim": 16,
        "n_heads": 2,
        "n_layers": 1,
        "drop_rate": 0.1,
        "qkv_bias": False,
    },
    "qwen3": {
        "vocab_size": 151_936,
        "context_length": 64,
        "emb_dim": 16,
        "n_heads": 2,
        "n_layers": 1,
        "hidden_dim": 24,
        "head_dim": 8,
        "qk_norm": True,
        "n_kv_groups": 1,
        "rope_base": 1_000_000.0,
    },
}
ROWS = [
    {"instruction": f"Repeat the word {w}.", "input": "" if i % 2 else w, "output": f"{w} {w}"}
    for i, w in enumerate(["red", "blue", "green", "cat", "dog", "sun", "moon", "tree"] * 3)
]


def setup(tmp_path, arch):
    if arch == "gpt2":
        cache = os.environ.get("LLM_TEST_TOKENIZER_CACHE")
        if not cache:
            pytest.skip("Set LLM_TEST_TOKENIZER_CACHE to a prepared GPT-2 cache")
    else:
        if not (QWEN3_DIR / "tokenizer-base.json").is_file():
            pytest.skip("Set LLM_QWEN3_DIR to a directory holding the pinned Qwen3 tokenizer")
        cache = QWEN3_DIR
    torch.manual_seed(0)
    config = FAMILIES[arch] | {"arch": arch}
    atomic_save(
        tmp_path / "start.pt",
        {
            "format_version": 1,
            "arch": arch,
            "model_config": config,
            "model_state_dict": build_model(config).state_dict(),
        },
    )
    data = tmp_path / "instructions"
    data.mkdir()
    (data / "instruction-data.json").write_text(json.dumps(ROWS))
    raw = {
        "model": {"arch": arch, "init_from": str(tmp_path / "start.pt")},
        "data": {
            "format": "instructions",
            "dataset": "alpaca-1100",
            "path": str(data),
            "tokenizer_cache": str(Path(cache).resolve()),
            "max_length": 64,
            "mask_prompt": arch == "qwen3",
        },
        "training": {
            "seed": 7,
            "batch_size": 4,
            "num_epochs": 2,
            "max_steps": None,
            "learning_rate": 0.001,
            "weight_decay": 0.1,
            "eval_freq": 2,
            "eval_iter": 2,
            "checkpoint_every": 2,
            "grad_clip": 1.0,
        },
        "runtime": {"device": "cpu", "num_threads": 1},
        "output": {
            "root": str(tmp_path / "runs"),
            "prompt": "Repeat the word red.",
            "sample_tokens": 6,
            "temperature": 0.8,
            "top_k": 20,
            "sample_every": 3,
        },
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(raw))
    digest = hashlib.sha256((data / "instruction-data.json").read_bytes()).hexdigest()
    return path, Path(raw["output"]["root"]), digest


def cli(digest, command, arguments, check=True):
    script = f"""
import runpy, socket, sys
def forbidden(*args, **kwargs):
    raise AssertionError("Unexpected network access")
socket.socket.connect = forbidden
from llm_training.data import instructions
instructions.DATASETS["alpaca-1100"]["sha256"] = "{digest}"
sys.argv = ["llm-training", "{command}", *sys.argv[1:]]
runpy.run_module("llm_training", run_name="__main__")
"""
    return subprocess.run(
        [sys.executable, "-c", script, *map(str, arguments)],
        check=check,
        capture_output=True,
        text=True,
        timeout=180,
    )


def run(root, digest, arguments):
    before = set(root.iterdir()) if root.exists() else set()
    script = f"""
import runpy, socket, sys
def forbidden(*args, **kwargs):
    raise AssertionError("Unexpected network access")
socket.socket.connect = forbidden
from llm_training.data import instructions
instructions.DATASETS["alpaca-1100"]["sha256"] = "{digest}"
sys.argv = ["llm-training", "train", *sys.argv[1:]]
runpy.run_module("llm_training", run_name="__main__")
"""
    subprocess.run(
        [sys.executable, "-c", script, *map(str, arguments)],
        check=True,
        stdout=subprocess.DEVNULL,
        timeout=180,
    )
    (new,) = set(root.iterdir()) - before
    return new


@pytest.mark.parametrize("arch", ["gpt2", "qwen3"])
def test_fine_tuning_resumes_exactly_and_hands_off_its_model(tmp_path, arch):
    config, root, digest = setup(tmp_path, arch)
    uninterrupted = run(root, digest, ["--config", config])
    stopped = run(root, digest, ["--config", config, "--stop-after", "3"])
    resumed = run(root, digest, ["--resume", stopped / "checkpoints/last.pt"])
    a, b = [load_checkpoint(p / "checkpoints/last.pt") for p in (uninterrupted, resumed)]
    for field in ("model_state_dict", "optimizer_state_dict", "rng", "progress", "history"):
        assert compare_states(a[field], b[field], field) == 0
    # 24 rows -> 20 train / 2 test / 2 validation; batches of 4 for two epochs.
    manifest = json.loads((uninterrupted / "data-manifest.json").read_text())
    assert manifest["examples"] == {"train": 20, "test": 2, "validation": 2}
    assert a["progress"]["optimizer_steps"] == 10
    # Masked prompts and padding never count as trained tokens.
    first = a["history"][0]
    assert 0 < first["tokens_seen"] < 4 * 64
    provenance = json.loads((uninterrupted / "provenance.json").read_text())
    assert provenance["initial_weights"]["model_file"] == str(tmp_path / "start.pt")
    resumed_provenance = json.loads((resumed / "provenance.json").read_text())
    assert resumed_provenance["initial_weights"] == provenance["initial_weights"]
    # Responses to the held-out split; a rerun with other settings refuses to mix them in.
    responses = tmp_path / "responses.jsonl"
    common = ["--checkpoint", uninterrupted / "model.pt", "--config", config]
    cli(digest, "respond", [*common, "--max-new-tokens", "3", "--output", responses])
    (record,) = [json.loads(line) for line in responses.read_text().splitlines()][:1]
    assert record["split"] == "test" and record["response_tokens"] <= 3
    assert len(responses.read_text().splitlines()) == 2  # the test split holds 2 rows
    mixed = cli(digest, "respond", [*common, "--max-new-tokens", "4", "--output", responses], False)
    assert mixed.returncode != 0 and "other settings" in mixed.stderr
    # The finished model starts the next stage.
    saved = torch.load(uninterrupted / "model.pt", weights_only=True)
    assert saved["arch"] == arch and saved["tokenizer"] == manifest["tokenizer_manifest"]
    raw = yaml.safe_load(config.read_text())
    raw["model"] = {"arch": arch, "init_from": str(uninterrupted / "model.pt")}
    raw["training"]["num_epochs"] = 1
    config.write_text(yaml.safe_dump(raw))
    next_stage = run(root, digest, ["--config", config])
    initial = load_checkpoint(next_stage / "checkpoints/last.pt")
    assert initial["progress"]["optimizer_steps"] == 5
    assert json.loads((next_stage / "status.json").read_text())["status"] == "completed"
