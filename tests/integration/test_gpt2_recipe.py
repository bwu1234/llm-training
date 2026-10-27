"""End-to-end checks of the GPT-2 recipe: accumulation, exact resume, and the resume guard."""

import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import yaml

from llm_training.lm.config import load_config
from llm_training.lm.schedule import learning_rate_at
from llm_training.lm.trainer import train
from llm_training.training.checkpoint import atomic_save, load_checkpoint
from scripts.compare_training_runs import compare_states
from tests.integration.test_resume import command, configuration

RECIPE = {
    "grad_accum_steps": 2,
    "grad_clip": 1.0,
    "lr_schedule": "cosine",
    "warmup_steps": 2,
    "min_lr": 0.00005,
    "beta2": 0.95,
    "weight_decay_policy": "matrices",
}
GPT2_MODEL = {
    "qkv_bias": True,
    "tie_embeddings": True,
    "init": "gpt2",
    "attn_impl": "sdpa",
    "drop_rate": 0.1,
}


def gpt2_configuration(tmp_path, accum):
    config, root = configuration(tmp_path, "validation_only")
    raw = yaml.safe_load(config.read_text())
    raw["model"] |= GPT2_MODEL
    raw["training"] |= RECIPE | {
        "grad_accum_steps": accum,
        "num_epochs": 3,
        "eval_freq": 2,
        "checkpoint_every": 2,
    }
    Path(raw["data"]["path"]).write_text("a b c d e f g h i j k l m n o p " * 24)
    config.write_text(yaml.safe_dump(raw))
    return config, root


# The fixture yields 9 micro-batches per epoch. accum=2 ends each epoch with a partial group of
# one (5 updates per epoch; stop 5 lands exactly on that boundary); accum=3 divides evenly
# (3 updates per epoch; stop 3 is the boundary).
@pytest.mark.parametrize("accum,stops", [(2, (3, 5)), (3, (3, 4))])
def test_exact_resume_with_every_recipe_feature_active(tmp_path, accum, stops):
    config, root = gpt2_configuration(tmp_path, accum)
    uninterrupted = command(root, ["--config", config])
    stopped = command(root, ["--config", config, "--stop-after", "1"])
    first = load_checkpoint(stopped / "checkpoints/last.pt")
    windows = json.loads((stopped / "data-manifest.json").read_text())["windows"]["train"]
    micro_batches = windows // 2
    assert (micro_batches % accum == 0) == (accum == 3)
    assert first["progress"]["batch_cursor"] == accum  # one update consumed a full group
    assert first["scheduler_state_dict"] == {"lr_schedule": "cosine", "last_step": 1}
    for step in stops:
        stopped = command(root, ["--resume", stopped / "checkpoints/last.pt", "--stop-after", step])
    resumed = command(root, ["--resume", stopped / "checkpoints/last.pt"])
    a, b = [load_checkpoint(p / "checkpoints/last.pt") for p in (uninterrupted, resumed)]
    for field in (
        "model_state_dict",
        "optimizer_state_dict",
        "rng",
        "progress",
        "history",
        "scheduler_state_dict",
        "best_validation",
    ):
        assert compare_states(a[field], b[field], field) == 0
    # The recorded rate is the schedule's value, and the optimizer really used two groups.
    cfg = load_config(resumed / "config.resolved.yaml")
    total = b["progress"]["optimizer_steps"]
    assert [r["learning_rate"] for r in b["history"]] == pytest.approx(
        [learning_rate_at(cfg.training, s, total) for s in range(total)]
    )
    assert len(b["optimizer_state_dict"]["param_groups"]) == 2
    assert all(r["grad_norm"] > 0 for r in b["history"])
    assert b["model_state_dict"]["out_head.weight"].equal(b["model_state_dict"]["tok_emb.weight"])


def test_accumulated_update_equals_the_equivalent_full_batch(tmp_path):
    # With dropout off, two micro-batches of 2 must equal one batch of 4 (plan gate for M3).
    config, _ = configuration(tmp_path, "validation_only")
    raw = yaml.safe_load(config.read_text())
    Path(raw["data"]["path"]).write_text("a b c d e f g h i j k l m n o p " * 24)
    raw["model"]["drop_rate"] = 0.0
    raw["training"] |= {"max_steps": 2, "num_epochs": 1, "grad_clip": 1.0}
    finals = []
    for batch, accum in ((4, 1), (2, 2)):
        raw["training"] |= {"batch_size": batch, "grad_accum_steps": accum}
        raw["output"]["root"] = str(tmp_path / f"runs-{batch}x{accum}")
        path = tmp_path / f"config-{batch}x{accum}.yaml"
        path.write_text(yaml.safe_dump(raw))
        run = train(load_config(path))
        finals.append(load_checkpoint(run / "checkpoints/last.pt"))
    # Declared before measuring: the repository's standard FP32 tolerance.
    full, accumulated = finals
    torch.testing.assert_close(
        full["model_state_dict"], accumulated["model_state_dict"], atol=1e-6, rtol=1e-5
    )
    # Adam's update is nearly invariant to gradient scale, so weights alone would not notice a
    # missing 1/accum factor; the recorded loss and pre-clip gradient norm are scale-sensitive.
    for left, right in zip(full["history"], accumulated["history"], strict=True):
        assert right["train_loss"] == pytest.approx(left["train_loss"], rel=1e-5)
        assert right["grad_norm"] == pytest.approx(left["grad_norm"], rel=1e-4)


def run_cli(arguments):
    return subprocess.run(
        [sys.executable, "-m", "llm_training", "train", *map(str, arguments)],
        capture_output=True,
        text=True,
    )


def test_code_drift_is_refused_unless_explicitly_allowed_and_is_recorded(tmp_path):
    config, root = configuration(tmp_path, "validation_only")
    stopped = command(root, ["--config", config, "--stop-after", "1"])
    saved = load_checkpoint(stopped / "checkpoints/last.pt")
    env = saved["environment"] | {"implementation_sha256": "0" * 64}
    tampered = tmp_path / "drifted.pt"
    atomic_save(tampered, saved | {"environment": env})

    refused = run_cli(["--resume", tampered])
    assert refused.returncode != 0
    assert "--allow-code-drift" in refused.stderr

    before = set(root.iterdir())
    allowed = run_cli(["--resume", tampered, "--allow-code-drift"])
    assert allowed.returncode == 0, allowed.stderr
    (child,) = set(root.iterdir()) - before
    provenance = json.loads((child / "provenance.json").read_text())
    assert provenance["code_drift_allowed"] is True
    assert provenance["parent_implementation_sha256"] == "0" * 64


def test_unchanged_code_resumes_without_the_override_and_records_no_drift(tmp_path):
    config, root = configuration(tmp_path, "validation_only")
    stopped = command(root, ["--config", config, "--stop-after", "1"])
    before = set(root.iterdir())
    result = run_cli(["--resume", stopped / "checkpoints/last.pt"])
    assert result.returncode == 0, result.stderr
    (child,) = set(root.iterdir()) - before
    assert json.loads((child / "provenance.json").read_text())["code_drift_allowed"] is False
