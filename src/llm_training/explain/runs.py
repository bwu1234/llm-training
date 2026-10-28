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
import sys
from dataclasses import fields
from pathlib import Path

import yaml

from llm_training.lm.config import TrainingConfig
from llm_training.lm.schedule import learning_rate_at

TEMPLATE = Path(__file__).with_name("run_view.html")
PLANNED_POINTS = 800  # enough for a smooth curve at any page width
MAX_RUNS = 8  # one categorical color per run


def _artifact(run_dir, name):
    """`run_dir / name` if it exists, refusing a symlink.

    The page embeds what it reads, so a planted `evaluation.json -> ~/secret.json` would
    publish that file to anyone who can read the page.
    """
    path = run_dir / name
    if not path.exists() and not path.is_symlink():
        return None
    if path.is_symlink():
        raise ValueError(f"{path} is a link; the run viewer reads only files inside the run")
    if not path.is_file():
        raise ValueError(f"{path} is not a file")
    return path


def _read_json(run_dir, name):
    path = _artifact(run_dir, name)
    return json.loads(path.read_text()) if path else None


def _read_jsonl(run_dir, name):
    """Records in a JSONL file, skipping a partial last line that a live run may leave."""
    path = _artifact(run_dir, name)
    if path is None:
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
    if not run_dir.is_dir():
        raise ValueError(f"{run_dir} is not a directory")
    config_path = _artifact(run_dir, "config.resolved.yaml")
    if config_path is None:
        raise ValueError(f"{run_dir} is not a next-token run directory (no config.resolved.yaml)")
    config = yaml.safe_load(config_path.read_text())
    if not isinstance(config, dict) or not isinstance(config.get("training"), dict):
        raise ValueError(f"{config_path} has no training section")
    if "num_rollouts" in config["training"]:
        raise ValueError(f"{run_dir} is a GRPO run; the run viewer covers next-token runs only")
    training = _training_config(config["training"])
    manifest = _read_json(run_dir, "data-manifest.json")
    status = _read_json(run_dir, "status.json")
    provenance = _read_json(run_dir, "provenance.json") or {}
    environment = _read_json(run_dir, "environment.json") or {}
    records = _read_jsonl(run_dir, "metrics.jsonl")
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
    for s in _read_jsonl(run_dir, "epoch-samples.jsonl"):
        samples.append(
            {"step": s["optimizer_step"], "text": s["text"], "kind": f"end of epoch {s['epoch']}"}
        )
    final = _read_json(run_dir, "sample.json")
    if final and status and status.get("status") == "completed":
        # sample.json has no step; the final reports follow the last update.
        step = status.get("optimizer_steps") or (updates[-1]["optimizer_step"] if updates else 0)
        # With the same settings and seed, a sample due at the last update has the same text.
        if not any(s["step"] == step and s["text"] == final["text"] for s in samples):
            samples.append({"step": step, "text": final["text"], "kind": "final"})
    samples.sort(key=lambda s: s["step"])  # stable: the final sample stays last at its step

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
    evaluation = _read_json(run_dir, "evaluation.json")
    return {
        "name": run_dir.name,
        "path": str(run_dir),
        "status": status,
        "arch": model.get("arch", "gpt2"),
        "encoding": ((manifest or {}).get("tokenizer_manifest") or {}).get("encoding"),
        "data_format": data.get("format", "text"),
        "mask_prompt": data.get("mask_prompt", False),
        "data_sha256": (manifest or {}).get("sha256"),
        "validation_identity": _validation_identity(manifest, window),
        "monitored_subset": _monitored_subset(evals, training, manifest),
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


def _validation_identity(manifest, window):
    """What fixes the validation tokens a run scores, or None when an old run lacks a field.

    A token corpus hashes its validation split on its own (its top-level digest covers the
    training split too). Text and instruction runs split one input file, so its digest plus
    the split recorded for that format identifies the validation part. The window belongs
    too: the same tokens scored with a shorter context give a higher loss.
    """
    m = manifest or {}
    if m.get("format") == "tokens":
        parts = [(m.get("splits") or {}).get("val", {}).get("sha256")]
    elif m.get("format") == "instructions":
        parts = [m.get("sha256"), m.get("dataset"), (m.get("examples") or {}).get("validation")]
    else:
        parts = [m.get("sha256"), m.get("split_character")]
    parts.append(window)
    if None in parts:  # an old run that didn't record it
        return None
    if m.get("format") == "instructions":
        parts.append(m.get("limit"))  # None here means no limit, not unknown
    return json.dumps(parts)


def _monitored_subset(evals, training, manifest):
    """How many validation targets each monitored loss averages, or None when unrecorded.

    `validation_only` records the count; the book recipe doesn't, so its subset is given in
    windows from `eval_iter` and the batch size instead.
    """
    counts = {e["eval_tokens"] for e in evals if e.get("eval_tokens") is not None}
    if counts:
        return f"{max(counts)} tokens"
    windows = ((manifest or {}).get("windows") or {}).get("validation")
    if not evals or windows is None:
        return None
    if training.eval_iter is None:
        return f"{windows} windows"
    return f"{min(training.eval_iter * training.batch_size, windows)} windows"


def _differ(runs, key):
    """Whether the runs that recorded `key` disagree; runs too old to record it are skipped."""
    return len({json.dumps(r[key]) for r in runs if r[key] is not None}) > 1


def _warnings(runs):
    """Reasons the overlaid runs' numbers may not compare, as the training-loop guide lists."""
    notes = []
    for key, label in (
        ("encoding", "tokenizers"),
        ("mask_prompt", "prompt masking"),
    ):
        if _differ(runs, key):
            notes.append(f"These runs use different {label}, so their losses are not comparable.")
    if _differ(runs, "validation_identity"):
        notes.append(
            "These runs validate on different tokens (split or context window), so their "
            "validation losses are not comparable."
        )
    elif _differ(runs, "data_sha256"):
        if all(r["validation_identity"] is not None for r in runs):
            notes.append(
                "These runs train on different data. Their validation losses share a split and "
                "compare; their training losses do not."
            )
        else:
            notes.append(
                "These runs use different data, and an older run doesn't record its validation "
                "split, so their losses may not compare."
            )
    if len({r["training"]["recipe"] for r in runs}) > 1:
        notes.append(
            "These runs monitor with different recipes (validation_only is token-weighted; "
            "sampled_train_validation averages batch means), so validation losses differ in "
            "definition."
        )
    elif _differ(runs, "monitored_subset"):
        notes.append(
            "These runs monitor validation subsets of different sizes, so their validation "
            "curves differ in coverage. Compare full_validation in the end-of-run evaluation."
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
    output = args.output or args.runs[0] / "run-view.html"
    try:
        runs = [load_run(path) for path in args.runs]
        output.write_text(render(runs))
    except (ValueError, OSError, yaml.YAMLError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
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
