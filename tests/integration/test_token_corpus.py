"""End-to-end token-corpus training: periodic samples, exact resume, and offline generation."""

import json
import subprocess
import sys
from pathlib import Path

import yaml

from llm_training.data.corpus import build_token_corpus
from llm_training.data.tokenizer import load_tokenizer
from llm_training.training.checkpoint import load_checkpoint
from scripts.compare_training_runs import compare_states
from tests.integration.test_gpt2_recipe import GPT2_MODEL, RECIPE
from tests.integration.test_resume import command, configuration

NAMES = ["Lily", "Tom", "Sue", "Max", "Ben"]
THINGS = ["ball", "cat", "tree", "car", "cake"]


def stories(count, offset=0):
    return "\n<|endoftext|>\n".join(
        f"Once upon a time {NAMES[i % 5]} saw a {THINGS[(i + offset) % 5]}. It was happy."
        for i in range(count)
    )


def token_configuration(tmp_path):
    config, root = configuration(tmp_path, "validation_only")
    raw = yaml.safe_load(config.read_text())
    cache = Path(raw["data"]["tokenizer_cache"])
    sources = {"train": tmp_path / "train.txt", "val": tmp_path / "val.txt"}
    sources["train"].write_text(stories(24))
    sources["val"].write_text(stories(6, offset=2))
    build_token_corpus(
        sources,
        tmp_path / "corpus",
        load_tokenizer(cache),
        json.loads((cache / "manifest.json").read_text()),
        {"corpus": "tinystories"},
    )
    raw["model"] |= GPT2_MODEL
    raw["data"] = {
        "format": "tokens",
        "corpus": "tinystories",
        "path": str(tmp_path / "corpus"),
        "tokenizer_cache": str(cache),
    }
    raw["training"] |= RECIPE | {"num_epochs": 2, "eval_freq": 2, "checkpoint_every": 2}
    raw["output"] |= {
        "prompt": "Once upon a time",
        "sample_tokens": 12,
        "temperature": 0.8,
        "top_k": 20,
        "sample_every": 3,
    }
    config.write_text(yaml.safe_dump(raw))
    return config, root


def test_token_corpus_training_resumes_exactly_with_samples(tmp_path):
    config, root = token_configuration(tmp_path)
    uninterrupted = command(root, ["--config", config])
    manifest = json.loads((uninterrupted / "data-manifest.json").read_text())
    assert manifest["format"] == "tokens"
    assert manifest["eot_policy"].startswith("prepended")
    assert manifest["windows"]["train"] >= 8 and manifest["windows"]["validation"] >= 1

    stopped = command(root, ["--config", config, "--stop-after", "2"])
    resumed = command(root, ["--resume", stopped / "checkpoints/last.pt"])
    a, b = [load_checkpoint(p / "checkpoints/last.pt") for p in (uninterrupted, resumed)]
    for field in ("model_state_dict", "optimizer_state_dict", "rng", "progress", "history"):
        assert compare_states(a[field], b[field], field) == 0
    samples = [r for r in a["history"] if "sample" in r]
    assert [r["optimizer_step"] for r in samples] == [
        s for s in range(1, len(a["history"]) + 1) if (s - 1) % 3 == 0
    ]
    assert all(r["sample"].startswith("Once upon a time") for r in samples)
    # Samples draw from a private generator, so switching them off leaves training identical.
    raw = yaml.safe_load(config.read_text())
    raw["output"]["sample_every"] = None
    quiet_config = tmp_path / "quiet.yaml"
    quiet_config.write_text(yaml.safe_dump(raw))
    quiet = load_checkpoint(command(root, ["--config", quiet_config]) / "checkpoints/last.pt")
    for field in ("model_state_dict", "rng"):
        assert compare_states(a[field], quiet[field], field) == 0
    final = json.loads((uninterrupted / "sample.json").read_text())
    assert final["method"] == "sampled" and final["temperature"] == 0.8
    assert (uninterrupted / "sample.json").read_bytes() == (resumed / "sample.json").read_bytes()

    script = """
import runpy, socket, sys
def forbidden(*args, **kwargs):
    raise AssertionError("Network access during generation")
socket.socket.connect = forbidden
sys.argv = ["llm-training", "generate", *sys.argv[1:]]
runpy.run_module("llm_training", run_name="__main__")
"""
    arguments = [
        "--checkpoint",
        str(uninterrupted / "model.pt"),
        "--tokenizer-cache",
        yaml.safe_load(config.read_text())["data"]["tokenizer_cache"],
        "--max-new-tokens",
        "10",
        "--num-samples",
        "2",
    ]
    outputs = [
        subprocess.run(
            [sys.executable, "-c", script, *arguments],
            check=True,
            capture_output=True,
            text=True,
            timeout=60,
        ).stdout
        for _ in range(2)
    ]
    assert outputs[0] == outputs[1]  # seeded: the same command prints the same stories
    assert outputs[0].count("--- seed") == 2
    assert outputs[0].count("Once upon a time") == 2
