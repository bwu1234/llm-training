"""Fresh-process exact recovery, including source evaluations and epoch boundaries."""

import json
import os
import signal
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from llm_training.training.checkpoint import atomic_save, load_checkpoint
from scripts.compare_training_runs import compare_states


def configuration(tmp_path, recipe):
    cache = os.environ.get("LLM_TEST_TOKENIZER_CACHE")
    if not cache:
        pytest.skip("Set LLM_TEST_TOKENIZER_CACHE to a prepared GPT-2 cache")
    text = tmp_path / "text.txt"
    text.write_text("a b c d e f g h i j k l m n o p " * 8)
    raw = yaml.safe_load(Path("configs/smoke.yaml").read_text())
    raw["model"] |= {"context_length": 16, "emb_dim": 16, "n_layers": 1, "drop_rate": 0.2}
    raw["data"] |= {
        "path": str(text),
        "tokenizer_cache": str(Path(cache).resolve()),
        "train_ratio": 0.75,
    }
    raw["training"] |= {
        "recipe": recipe,
        "num_epochs": 3,
        "max_steps": None,
        "eval_freq": 2,
        "eval_iter": 2,
        "checkpoint_every": 2,
    }
    raw["output"] |= {"root": str(tmp_path / "runs"), "sample_tokens": 50}
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump(raw))
    return config, Path(raw["output"]["root"])


def command(root, arguments):
    before = set(root.iterdir()) if root.exists() else set()
    script = """
import runpy, socket, sys
def forbidden(*args, **kwargs):
    raise AssertionError("Unexpected network access")
socket.socket.connect = forbidden
sys.argv = ["llm-training", "train", *sys.argv[1:]]
runpy.run_module("llm_training", run_name="__main__")
"""
    subprocess.run(
        [sys.executable, "-c", script, *map(str, arguments)],
        check=True,
        stdout=subprocess.DEVNULL,
        timeout=90,
    )
    (run,) = set(root.iterdir()) - before
    return run


@pytest.mark.parametrize("recipe", ["validation_only", "sampled_train_validation"])
def test_exact_resume_across_mid_epoch_and_epoch_boundary(tmp_path, recipe):
    config, root = configuration(tmp_path, recipe)
    uninterrupted = command(root, ["--config", config])
    stopped = command(root, ["--config", config, "--stop-after", "2"])
    first = load_checkpoint(stopped / "checkpoints/last.pt")
    assert first["progress"]["batch_cursor"] == 2
    assert first["optimizer_state_dict"]["state"]
    # Derive the boundary from tokenized windows rather than character counts.
    windows = json.loads((stopped / "data-manifest.json").read_text())["windows"]["train"]
    boundary = windows // 2
    assert boundary > 2
    for step in (boundary, boundary + 1):
        stopped = command(root, ["--resume", stopped / "checkpoints/last.pt", "--stop-after", step])
        state = load_checkpoint(stopped / "checkpoints/last.pt")
        assert state["progress"]["optimizer_steps"] == step
        if step == boundary:
            assert state["progress"]["epoch"] == 1
            assert state["progress"]["batch_cursor"] == 0
    resumed = command(root, ["--resume", stopped / "checkpoints/last.pt"])
    a, b = [load_checkpoint(p / "checkpoints/last.pt") for p in (uninterrupted, resumed)]
    for field in (
        "model_state_dict",
        "optimizer_state_dict",
        "rng",
        "progress",
        "history",
        "epoch_samples",
        "best_validation",
    ):
        assert compare_states(a[field], b[field], field) == 0
    assert len(a["history"]) == boundary * 3
    for run in (uninterrupted, resumed):
        selected = load_checkpoint(run / "checkpoints/best.pt")
        assert selected["progress"]["optimizer_steps"] == b["best_step"]
        assert selected["best_validation"] == b["best_validation"]
    left_best, right_best = [
        load_checkpoint(p / "checkpoints/best.pt") for p in (uninterrupted, resumed)
    ]
    assert compare_states(left_best["model_state_dict"], right_best["model_state_dict"]) == 0
    assert (uninterrupted / "sample.json").read_bytes() == (resumed / "sample.json").read_bytes()
    assert json.loads((resumed / "status.json").read_text())["status"] == "completed"
    # Embedding weights are not an optimizer/RNG/data checkpoint.
    with pytest.raises(ValueError, match="full training checkpoint"):
        load_checkpoint(resumed / "model.pt")
    corrupted = first | {"config": first["config"] | {"unexpected": True}}
    atomic_save(tmp_path / "corrupted.pt", corrupted)
    with pytest.raises(ValueError, match="configuration hash"):
        load_checkpoint(tmp_path / "corrupted.pt")
    wrong_environment = first | {"environment": first["environment"] | {"num_threads": 999}}
    atomic_save(tmp_path / "wrong-environment.pt", wrong_environment)
    rejected = subprocess.run(
        [
            sys.executable,
            "-m",
            "llm_training",
            "train",
            "--resume",
            str(tmp_path / "wrong-environment.pt"),
        ],
        capture_output=True,
        text=True,
    )
    assert rejected.returncode != 0
    assert "environment" in rejected.stderr
    # A saved config and a changed input cannot silently continue the run.
    raw = yaml.safe_load(config.read_text())
    Path(raw["data"]["path"]).write_text("changed data")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "llm_training",
            "train",
            "--resume",
            str(stopped / "checkpoints/last.pt"),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "identity differs" in result.stderr


def test_training_loop_matches_frozen_source(tmp_path):
    config, _ = configuration(tmp_path, "sampled_train_validation")
    output = tmp_path / "reproduction"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.compare_training_runs",
            "--config",
            str(config),
            "--output",
            str(output),
            "--stop-after",
            "2",
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        timeout=90,
    )
    report = json.loads((output / "comparison.json").read_text())
    assert report["passed"]
    assert report["epoch_samples_compared"] == 3
    assert all(value == 0 for value in report["max_absolute_differences"].values())


def test_signal_saves_at_safe_boundary(tmp_path):
    config, root = configuration(tmp_path, "validation_only")
    raw = yaml.safe_load(config.read_text())
    raw["training"]["num_epochs"] = 100
    config.write_text(yaml.safe_dump(raw))
    process = subprocess.Popen(
        [sys.executable, "-u", "-m", "llm_training", "train", "--config", str(config)],
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        # Wait for an actual completed update, not an arbitrary wall-clock delay.
        for line in process.stdout:
            if '"optimizer_step": 1,' in line:
                process.send_signal(signal.SIGTERM)
                break
        process.communicate(timeout=60)
        assert process.returncode == 0
        (run,) = root.iterdir()
        state = load_checkpoint(run / "checkpoints/last.pt")
        status = json.loads((run / "status.json").read_text())
        assert status["status"] == "interrupted"
        assert status["signal"] == signal.SIGTERM
        assert status["optimizer_steps"] == state["progress"]["optimizer_steps"]
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
