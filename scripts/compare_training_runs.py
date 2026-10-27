"""Compare a full reference training run with a stopped/resumed packaged run.

Run with `uv run python -m scripts.compare_training_runs --config ... --output ...`.
Each phase uses a fresh process. No sibling repository or notebook is required.
"""

import argparse
import contextlib
import io
import json
import random
import subprocess
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from llm_training.lm.config import load_config
from llm_training.model.gpt2.tokenizer import load_tokenizer
from llm_training.training.checkpoint import atomic_save, capture_rng, load_checkpoint
from llm_training.training.provenance import collect_runtime_metadata, write_json
from tests.reference import baseline, training_loop

ATOL, RTOL = 1e-6, 1e-5


def run_reference_training(cfg, output):
    torch.set_num_threads(cfg.runtime.num_threads)
    torch.manual_seed(cfg.training.seed)
    random.seed(cfg.training.seed)
    np.random.seed(cfg.training.seed)
    tokenizer = load_tokenizer(Path(cfg.data.tokenizer_cache))
    text = Path(cfg.data.path).read_text()
    model = baseline.GPTModel(asdict(cfg.model)).to(cfg.runtime.device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.training.learning_rate, weight_decay=cfg.training.weight_decay
    )
    split = int(len(text) * cfg.data.train_ratio)
    datasets = [
        baseline.GPTDatasetV1(part, tokenizer, cfg.model.context_length, cfg.model.context_length)
        for part in (text[:split], text[split:])
    ]
    train = DataLoader(
        datasets[0], batch_size=cfg.training.batch_size, shuffle=True, drop_last=True
    )
    val = DataLoader(datasets[1], batch_size=cfg.training.batch_size, shuffle=False)
    if cfg.output.sample_tokens != 50:
        raise ValueError("The literal upstream loop generates 50 tokens per epoch")
    buffer = io.StringIO()
    started = time.perf_counter()
    with contextlib.redirect_stdout(buffer):
        train_losses, val_losses, tokens_seen = training_loop.train_model_simple(
            model,
            train,
            val,
            optimizer,
            cfg.runtime.device,
            cfg.training.num_epochs,
            cfg.training.eval_freq,
            cfg.training.eval_iter,
            cfg.output.prompt,
            tokenizer,
        )
    elapsed = time.perf_counter() - started
    (output / "reference.log").write_text(buffer.getvalue())
    samples = [line for line in buffer.getvalue().splitlines() if not line.startswith("Ep ")]
    atomic_save(
        output / "reference.pt",
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "rng": capture_rng(cfg.runtime.device),
        },
    )
    write_json(
        output / "reference-metrics.json",
        {
            "train_losses": train_losses,
            "val_losses": val_losses,
            "tokens_seen": tokens_seen,
            "epoch_samples": samples,
            "optimizer_steps": len(train) * cfg.training.num_epochs,
            "elapsed_seconds": elapsed,
            "parameters": sum(p.numel() for p in model.parameters()),
            "environment": collect_runtime_metadata(),
        },
    )


def run_resumed_training(cfg, output, stop_after):
    cfg = replace(cfg, output=replace(cfg.output, root=str((output / "port-runs").resolve())))
    path = output / "port-config.yaml"
    path.write_text(yaml.safe_dump(cfg.to_dict(), sort_keys=False))

    def run_cli(arguments, log):
        with (output / log).open("w") as stream:
            subprocess.run(
                [sys.executable, "-m", "llm_training", "train", *arguments],
                check=True,
                stdout=stream,
                stderr=subprocess.STDOUT,
            )

    run_cli(["--config", str(path), "--stop-after", str(stop_after)], "port-stop.log")
    runs = sorted(Path(cfg.output.root).iterdir())
    stopped = runs[-1]
    assert json.loads((stopped / "status.json").read_text())["status"] == "interrupted"
    run_cli(["--resume", str(stopped / "checkpoints/last.pt")], "port-resume.log")
    completed = sorted(set(Path(cfg.output.root).iterdir()) - set(runs))
    if len(completed) != 1:
        raise ValueError("Could not identify the resumed child run")
    write_json(
        output / "port-result.json",
        {"stopped_run": str(stopped), "resumed_run": str(completed[0]), "stop_after": stop_after},
    )


def compare_states(left, right, path="state"):
    """Compare nested tensor/optimizer/RNG state; report the largest absolute error."""
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, atol=ATOL, rtol=RTOL, msg=path)
        return (left - right).abs().max().item() if left.numel() else 0
    if isinstance(left, dict):
        assert left.keys() == right.keys(), path
        return max((compare_states(left[k], right[k], f"{path}.{k}") for k in left), default=0)
    if isinstance(left, (list, tuple)):
        assert len(left) == len(right), path
        return max(
            (
                compare_states(a, b, f"{path}.{i}")
                for i, (a, b) in enumerate(zip(left, right, strict=True))
            ),
            default=0,
        )
    assert left == right, (path, left, right)
    return 0


def comparison(output):
    result = json.loads((output / "port-result.json").read_text())
    run = Path(result["resumed_run"])
    ref = torch.load(output / "reference.pt", weights_only=True, map_location="cpu")
    candidate = load_checkpoint(run / "checkpoints/last.pt")
    source = json.loads((output / "reference-metrics.json").read_text())
    records = [r for r in candidate["history"] if "source_val_loss" in r]
    assert len(records) == len(source["train_losses"])
    for key, field in [("train_losses", "source_train_loss"), ("val_losses", "source_val_loss")]:
        torch.testing.assert_close(
            torch.tensor(source[key], dtype=torch.float64),
            torch.tensor([r[field] for r in records], dtype=torch.float64),
            atol=ATOL,
            rtol=RTOL,
        )
    assert source["tokens_seen"] == [r["tokens_seen"] for r in records]
    assert source["epoch_samples"] == [
        s["text"].replace("\n", " ") for s in candidate["epoch_samples"]
    ]
    assert source["optimizer_steps"] == candidate["progress"]["optimizer_steps"]
    differences = {
        name: compare_states(ref[name], candidate[name], name)
        for name in ("model_state_dict", "optimizer_state_dict", "rng")
    }
    write_json(
        output / "comparison.json",
        {
            "passed": True,
            "atol": ATOL,
            "rtol": RTOL,
            "max_absolute_differences": differences,
            "optimizer_steps": source["optimizer_steps"],
            "parameters": source["parameters"],
            "stop_after": result["stop_after"],
            "evaluations_compared": len(records),
            "epoch_samples_compared": len(source["epoch_samples"]),
            "environment": source["environment"],
            "config": candidate["config"],
            "reference_elapsed_seconds": source["elapsed_seconds"],
            "final_evaluation": json.loads((run / "evaluation.json").read_text()),
            "port_runs": result,
        },
    )
    print((output / "comparison.json").read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase", choices=["all", "reference", "port", "compare"], default="all")
    parser.add_argument("--stop-after", type=int, default=46)
    args = parser.parse_args()
    cfg = load_config(args.config)
    if cfg.training.recipe != "sampled_train_validation" or cfg.training.max_steps is not None:
        raise ValueError("Reproduction requires recipe=sampled_train_validation and max_steps=null")
    output = args.output.resolve()
    if args.phase == "all":
        output.mkdir(parents=True, exist_ok=False)
        for phase in ("reference", "port", "compare"):
            print(f"Running {phase}: {output}", flush=True)
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "scripts.compare_training_runs",
                    "--config",
                    str(args.config.resolve()),
                    "--output",
                    str(output),
                    "--phase",
                    phase,
                    "--stop-after",
                    str(args.stop_after),
                ],
                check=True,
            )
    elif args.phase == "reference":
        run_reference_training(cfg, output)
    elif args.phase == "port":
        run_resumed_training(cfg, output, args.stop_after)
    else:
        comparison(output)


if __name__ == "__main__":
    main()
