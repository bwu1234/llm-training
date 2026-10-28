"""The run viewer reads a run directory as `Trainer` wrote it and embeds it safely."""

import json
import re

import pytest
import yaml

from llm_training.explain import runs as viewer
from llm_training.lm.config import TrainingConfig
from llm_training.lm.schedule import learning_rate_at

TRAINING = {
    "batch_size": 2,
    "num_epochs": 1,
    "max_steps": None,
    "learning_rate": 1e-3,
    "lr_schedule": "cosine",
    "warmup_steps": 2,
    "min_lr": 1e-4,
    "eval_freq": 3,
    "eval_iter": 1,
    "checkpoint_every": 4,
    "recipe": "validation_only",
    "field_from_a_later_version": 1,
}
# Closing the data block early, and the "double-escaped" state that would extend it.
HOSTILE = "</script><script>alert(1)</script> <!--<script> & more"


def write_run(path, training=TRAINING, windows=20, steps=10, status="completed"):
    """A run directory with `steps` updates; validation after updates 1, 4, 7, 10."""
    path.mkdir()
    config = {"model": {"arch": "gpt2", "context_length": 8}, "data": {}, "training": training}
    (path / "config.resolved.yaml").write_text(yaml.safe_dump(config))
    (path / "data-manifest.json").write_text(
        json.dumps({"tokenizer_manifest": {"encoding": "gpt2"}, "windows": {"train": windows}})
    )
    (path / "status.json").write_text(json.dumps({"status": status}))
    (path / "provenance.json").write_text(
        json.dumps(
            {"parent_checkpoint": "/elsewhere/checkpoints/last.pt", "parent_optimizer_step": 3}
        )
    )
    cfg = viewer._training_config(training)
    losses = {1: 5.0, 4: 4.0, 7: 4.5, 10: 3.0}
    lines = []
    for step in range(1, steps + 1):
        record = {
            "optimizer_step": step,
            "epoch": 1,
            "train_loss": 6.0 - step * 0.1,
            "tokens_seen": step * 16,
            "learning_rate": learning_rate_at(cfg, step - 1, 10),
            "batch_sha256": "0" * 64,
        }
        if step > 1:  # the first record predates grad_norm, as old runs do
            record["grad_norm"] = 0.5
        if step in losses:
            record["validation"] = {"loss": losses[step], "perplexity": 1.0, "tokens": 16}
        if step == 4:
            record["sample"] = HOSTILE
        lines.append(json.dumps(record))
    # A live run can leave a partial last line.
    (path / "metrics.jsonl").write_text("\n".join(lines) + '\n{"optimizer_st')
    return path


def embedded(html):
    (block,) = re.findall(
        r'<script type="application/json" id="run-data">(.*?)</script>', html, re.S
    )
    return json.loads(block)


def test_budget_schedule_and_events(tmp_path):
    run = viewer.load_run(write_run(tmp_path / "run"))
    assert run["total_steps"] == 10  # 20 windows / batch 2
    assert run["schedule_check"] == {"checked": 10, "mismatched": []}
    assert run["updates"]["step"] == list(range(1, 11))  # partial line skipped
    assert run["updates"]["grad_norm"][0] is None
    events = run["events"]
    assert events["evaluations"] == [1, 4, 7, 10]
    assert events["improvements"] == [1, 4, 10]
    assert (events["best_step"], events["best_loss"]) == (10, 3.0)
    # every 4th update, each new best, the end, and the state the child resumed from
    assert events["checkpoints"] == [1, 3, 4, 8, 10]
    assert events["resume_step"] == 3
    # The planned curve is the trainer's schedule, warmup boundary included.
    planned = dict(run["planned"])
    cfg = TrainingConfig(**{k: v for k, v in TRAINING.items() if k != "field_from_a_later_version"})
    assert planned[3] == viewer._round(learning_rate_at(cfg, 2, 10))
    assert {1, 2, 3, 10} <= planned.keys()


def test_schedule_mismatch_is_reported(tmp_path):
    path = write_run(tmp_path / "run")
    lines = (path / "metrics.jsonl").read_text().splitlines()
    record = json.loads(lines[5])
    record["learning_rate"] *= 2
    lines[5] = json.dumps(record)
    (path / "metrics.jsonl").write_text("\n".join(lines))
    assert viewer.load_run(path)["schedule_check"]["mismatched"] == [6]


def test_budget_with_accumulation_and_cap():
    training = viewer._training_config(
        TRAINING | {"grad_accum_steps": 3, "num_epochs": 2, "max_steps": 5}
    )
    # 21 windows -> 10 batches -> ceil(10 / 3) = 4 updates per epoch, 8 in all, capped at 5
    assert viewer.planned_total_steps(training, {"windows": {"train": 21}}) == 5
    assert viewer.planned_total_steps(training, {}) is None


def test_book_recipe_detected_from_records(tmp_path):
    path = tmp_path / "book"
    path.mkdir()
    training = {k: v for k, v in TRAINING.items() if k != "recipe"}
    (path / "config.resolved.yaml").write_text(
        yaml.safe_dump({"model": {}, "data": {}, "training": training})
    )
    record = {
        "optimizer_step": 1,
        "epoch": 1,
        "train_loss": 11.0,
        "tokens_seen": 512,
        "learning_rate": 1e-3,
        "source_train_loss": 9.8,
        "source_val_loss": 9.9,
    }
    (path / "metrics.jsonl").write_text(json.dumps(record) + "\n")
    (path / "epoch-samples.jsonl").write_text(
        json.dumps({"epoch": 1, "optimizer_step": 1, "text": "Every effort"}) + "\n"
    )
    run = viewer.load_run(path)
    assert run["training"]["recipe"] == "sampled_train_validation"
    assert run["evals"] == [{"step": 1, "tokens": 512, "loss": 9.9, "train": 9.8}]
    assert run["samples"][0]["kind"] == "end of epoch 1"
    assert run["total_steps"] is None  # no manifest window count, and not completed


def test_page_embeds_untrusted_text_inertly(tmp_path):
    run = viewer.load_run(write_run(tmp_path / "run"))
    html = viewer.render([run])
    assert HOSTILE not in html
    block = html.split('id="run-data">', 1)[1].split("</script>", 1)[0]
    assert not set("<>&") & set(block)
    data = embedded(html)
    assert data["runs"][0]["samples"][0]["text"] == HOSTILE
    assert data["warnings"] == []
    # Offline: nothing loaded from the network (the SVG namespace is an identifier, not a fetch).
    assert not re.search(r'(src|href)\s*=\s*["\']?https?:', html)
    assert "@import" not in html


def test_overlay_warnings(tmp_path):
    a = viewer.load_run(write_run(tmp_path / "a"))
    b = viewer.load_run(write_run(tmp_path / "b", training=TRAINING | {"recipe": "x"}))
    b["parent_checkpoint"] = str(tmp_path / "a" / "checkpoints" / "last.pt")
    warnings = embedded(viewer.render([a, b]))["warnings"]
    assert any("different recipes" in w for w in warnings)
    assert any("resumed from a" in w for w in warnings)


def test_cli_writes_next_to_the_run(tmp_path, capsys):
    path = write_run(tmp_path / "run")
    viewer.main([str(path)])
    assert (path / "run-view.html").is_file()
    assert "wrote" in capsys.readouterr().out


def test_grpo_runs_are_rejected(tmp_path):
    path = tmp_path / "grpo"
    path.mkdir()
    (path / "config.resolved.yaml").write_text(
        yaml.safe_dump({"model": {}, "data": {}, "training": {"steps": 100, "num_rollouts": 8}})
    )
    with pytest.raises(ValueError, match="GRPO"):
        viewer.load_run(path)
