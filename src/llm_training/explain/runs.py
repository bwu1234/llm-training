"""`python -m llm_training.explain.runs RUN_DIR [RUN_DIR ...]`: a local run viewer.

Reads only a next-token run directory's JSON and YAML records (never a checkpoint, which can
be 9 GB) and writes one self-contained HTML page: loss, learning rate and gradient norm on a
shared axis, evaluation and checkpoint events, samples, and the end-of-run evaluation. The
planned schedule comes from the trainer's own `learning_rate_at`, checked against the rate
each update recorded. Nothing is loaded from the network, so the page works offline.
"""

import argparse
import json
import math
from dataclasses import fields
from pathlib import Path

import yaml

from llm_training.lm.config import TrainingConfig
from llm_training.lm.schedule import learning_rate_at

TEMPLATE = Path(__file__).with_name("run_view.html")
PLANNED_POINTS = 800  # enough for a smooth curve at any page width
MAX_RUNS = 8  # one categorical color per run


def _read_json(path):
    return json.loads(path.read_text()) if path.is_file() else None


def _read_jsonl(path):
    """Records in a JSONL file, skipping a partial last line that a live run may leave."""
    if not path.is_file():
        return []
    records = []
    for line in path.read_text().splitlines():
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records


def _round(value, digits=7):
    """Shorter JSON: significant digits beyond float32 training noise carry nothing."""
    if value is None or not math.isfinite(value):
        return None
    return float(f"{value:.{digits}g}")


def _training_config(raw):
    """The run's `TrainingConfig`, built without `config_from_dict`'s path checks.

    Validating the whole recipe would resolve model weights and data paths, which a viewer
    should not need: they may have moved since the run.
    """
    known = {f.name for f in fields(TrainingConfig)}
    return TrainingConfig(**{k: v for k, v in raw.items() if k in known})


def planned_total_steps(training, manifest):
    """The update budget `Trainer` computed, from the recorded training-window count."""
    windows = (manifest or {}).get("windows", {}).get("train")
    if windows is None:
        return None
    batches = windows // training.batch_size  # the training loader drops the last partial batch
    total = -(-batches // training.grad_accum_steps) * training.num_epochs
    if training.max_steps is not None:
        total = min(total, training.max_steps)
    return total


def _events(updates, evals, training, status, parent_step):
    """Evaluation, checkpoint, new-best, resume and end points, as `Trainer` produced them."""
    best, improvements = math.inf, []
    for point in evals:
        if point["loss"] < best:
            best = point["loss"]
            improvements.append(point["step"])
    last = updates[-1]["optimizer_step"] if updates else 0
    checkpoints = set(range(training.checkpoint_every, last + 1, training.checkpoint_every))
    checkpoints |= set(improvements)
    if parent_step:  # a child run checkpoints the state it resumed from before its first update
        checkpoints.add(parent_step)
    if status and status.get("status") in ("completed", "interrupted") and last:
        checkpoints.add(last)
    return {
        "evaluations": [p["step"] for p in evals],
        "checkpoints": sorted(checkpoints),
        "improvements": improvements,
        "best_step": improvements[-1] if improvements else None,
        "best_loss": _round(best) if improvements else None,
        "resume_step": parent_step,
        "last_step": last,
    }


def load_run(run_dir):
    """Everything the viewer shows for one run, as plain JSON-ready data."""
    run_dir = Path(run_dir)
    config_path = run_dir / "config.resolved.yaml"
    if not config_path.is_file():
        raise ValueError(f"{run_dir} is not a next-token run directory (no config.resolved.yaml)")
    config = yaml.safe_load(config_path.read_text())
    if "num_rollouts" in config["training"]:
        raise ValueError(f"{run_dir} is a GRPO run; the run viewer covers next-token runs only")
    training = _training_config(config["training"])
    manifest = _read_json(run_dir / "data-manifest.json")
    status = _read_json(run_dir / "status.json")
    provenance = _read_json(run_dir / "provenance.json") or {}
    environment = _read_json(run_dir / "environment.json") or {}
    records = _read_jsonl(run_dir / "metrics.jsonl")
    # Runs older than `training.recipe` recorded the book's monitoring without naming it.
    sampled = any("source_val_loss" in r for r in records)
    recipe = "sampled_train_validation" if sampled else training.recipe

    updates, evals, samples = [], [], []
    for r in records:
        updates.append(r)
        if "source_val_loss" in r:
            evals.append(
                {
                    "step": r["optimizer_step"],
                    "tokens": r["tokens_seen"],
                    "loss": r["source_val_loss"],
                    "train": r.get("source_train_loss"),
                }
            )
        elif "validation" in r:
            v = r["validation"]
            evals.append(
                {
                    "step": r["optimizer_step"],
                    "tokens": r["tokens_seen"],
                    "loss": v["loss"],
                    "perplexity": v.get("perplexity"),
                    "eval_tokens": v.get("tokens"),
                }
            )
        if "sample" in r:
            samples.append({"step": r["optimizer_step"], "text": r["sample"], "kind": "update"})
    for s in _read_jsonl(run_dir / "epoch-samples.jsonl"):
        samples.append(
            {"step": s["optimizer_step"], "text": s["text"], "kind": f"end of epoch {s['epoch']}"}
        )
    samples.sort(key=lambda s: s["step"])

    total = planned_total_steps(training, manifest)
    if total is None and status and status.get("status") == "completed" and updates:
        total = updates[-1]["optimizer_step"]
    planned, schedule_check = [], None
    if total:
        # Rate of update `step` (1-based) is learning_rate_at(k = step - 1).
        def rate(step):
            return learning_rate_at(training, step - 1, total)

        stride = max(1, total // PLANNED_POINTS)
        steps = sorted(
            set(range(1, total + 1, stride))
            | {total, min(training.warmup_steps, total), min(training.warmup_steps + 1, total)}
        )
        planned = [[s, _round(rate(s))] for s in steps if s >= 1]
        recorded = [r for r in updates if "learning_rate" in r]
        if recorded:
            mismatched = [
                r["optimizer_step"]
                for r in recorded
                if r["optimizer_step"] > total or rate(r["optimizer_step"]) != r["learning_rate"]
            ]
            schedule_check = {"checked": len(recorded), "mismatched": mismatched[:20]}

    model = config.get("model", {})
    data = config.get("data", {})
    window = data.get("max_length") or model.get("context_length")
    evaluation = _read_json(run_dir / "evaluation.json")
    return {
        "name": run_dir.name,
        "path": str(run_dir),
        "status": status,
        "arch": model.get("arch", "gpt2"),
        "encoding": ((manifest or {}).get("tokenizer_manifest") or {}).get("encoding"),
        "data_format": data.get("format", "text"),
        "mask_prompt": data.get("mask_prompt", False),
        "device": environment.get("device"),
        "parent_checkpoint": provenance.get("parent_checkpoint"),
        "training": {
            "recipe": recipe,
            "learning_rate": training.learning_rate,
            "lr_schedule": training.lr_schedule,
            "warmup_steps": training.warmup_steps,
            "min_lr": training.min_lr,
            "grad_clip": training.grad_clip,
            "batch_size": training.batch_size,
            "grad_accum_steps": training.grad_accum_steps,
            "window": window,
            "eval_freq": training.eval_freq,
            "eval_iter": training.eval_iter,
            "checkpoint_every": training.checkpoint_every,
            "num_epochs": training.num_epochs,
            "max_steps": training.max_steps,
        },
        "total_steps": total,
        "updates": {
            "step": [r["optimizer_step"] for r in updates],
            "tokens": [r["tokens_seen"] for r in updates],
            "epoch": [r["epoch"] for r in updates],
            "train_loss": [_round(r["train_loss"]) for r in updates],
            "learning_rate": [_round(r.get("learning_rate")) for r in updates],
            "grad_norm": [_round(r.get("grad_norm")) for r in updates],
        },
        "evals": [
            {k: _round(v) if isinstance(v, float) else v for k, v in e.items()} for e in evals
        ],
        "samples": samples,
        "planned": planned,
        "schedule_check": schedule_check,
        "events": _events(
            updates, evals, training, status, provenance.get("parent_optimizer_step")
        ),
        "evaluation": evaluation,
    }


def _warnings(runs):
    """Reasons the overlaid runs' numbers may not compare, as the training-loop guide lists."""
    notes = []
    for key, label in (
        ("encoding", "tokenizers"),
        ("mask_prompt", "prompt masking"),
    ):
        if len({json.dumps(r[key]) for r in runs}) > 1:
            notes.append(f"These runs use different {label}, so their losses are not comparable.")
    if len({r["training"]["recipe"] for r in runs}) > 1:
        notes.append(
            "These runs monitor with different recipes (validation_only is token-weighted; "
            "sampled_train_validation averages batch means), so validation losses differ in "
            "definition."
        )
    paths = {str(Path(r["path"]).resolve()): r["name"] for r in runs}
    for r in runs:
        parent = r["parent_checkpoint"]
        if parent and str(Path(parent).parent.parent) in paths:
            notes.append(
                f"{r['name']} resumed from {paths[str(Path(parent).parent.parent)]} and already "
                "contains its history; pass only the child."
            )
    return notes


def render(runs):
    """The viewer page with `runs` embedded as inert JSON."""
    payload = json.dumps({"runs": runs, "warnings": _warnings(runs)}, allow_nan=False)
    # Model samples are arbitrary text. With no literal <, > or & left, neither "</script>"
    # nor "<!--<script>" can end the data block early or extend it past its closing tag.
    # These characters occur only inside JSON strings, where \u escapes parse back unchanged.
    for char, escape in (("<", "\\u003c"), (">", "\\u003e"), ("&", "\\u0026")):
        payload = payload.replace(char, escape)
    return TEMPLATE.read_text().replace("__RUN_DATA__", payload)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Write a local HTML view of training runs.")
    parser.add_argument("runs", nargs="+", type=Path, help="next-token run directories")
    parser.add_argument(
        "--output",
        type=Path,
        help="HTML file to write (default: RUN_DIR/run-view.html for a single run)",
    )
    args = parser.parse_args(argv)
    if len(args.runs) > MAX_RUNS:
        parser.error(f"at most {MAX_RUNS} runs can be overlaid")
    if args.output is None and len(args.runs) > 1:
        parser.error("--output is required when overlaying several runs")
    runs = [load_run(path) for path in args.runs]
    output = args.output or args.runs[0] / "run-view.html"
    output.write_text(render(runs))
    for run in runs:
        check = run["schedule_check"]
        if check and check["mismatched"]:
            print(
                f"warning: {run['name']}: recorded learning rates differ from the schedule at "
                f"updates {check['mismatched']}"
            )
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
