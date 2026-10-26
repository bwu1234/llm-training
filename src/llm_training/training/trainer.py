"""Single-process FP32 training with periodic loss monitoring and full-state recovery."""

import hashlib
import json
import math
import random
import shutil
import signal
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from llm_training.config import Config, config_from_dict
from llm_training.data.corpus import SPLITS, load_token_corpus
from llm_training.data.dataset import SlidingWindowTokenDataset
from llm_training.data.token_dataset import TokenWindowDataset
from llm_training.evaluation import evaluate
from llm_training.generation import generate_greedy, generate_sampled
from llm_training.model.gpt2.model import GPTModel
from llm_training.model.gpt2.tokenizer import load_tokenizer
from llm_training.training.checkpoint import (
    FORMAT_VERSION,
    atomic_save,
    capture_rng,
    config_hash,
    isolated_rng,
    load_checkpoint,
    restore_rng,
)
from llm_training.training.loss import next_token_cross_entropy
from llm_training.training.monitoring import evaluate_train_validation_loss
from llm_training.training.optim import build_optimizer, clip_and_measure
from llm_training.training.provenance import (
    collect_runtime_metadata,
    git_provenance,
    implementation_fingerprint,
    write_json,
)
from llm_training.training.schedule import is_constant, learning_rate_at


def _sample(model, tokenizer, cfg, device, step=0):
    was_training = model.training
    model.eval()
    try:
        prompt = tokenizer.encode(cfg.output.prompt) or [tokenizer.eot_token]
        idx = torch.tensor([prompt], device=device)
        if cfg.output.temperature == 0 and cfg.output.top_k is None:
            ids = generate_greedy(model, idx, cfg.output.sample_tokens, cfg.model.context_length)[
                0
            ].tolist()
            return {
                "prompt": cfg.output.prompt,
                "text": tokenizer.decode(ids),
                "token_ids": ids,
                "method": "greedy",
            }
        # A private generator keyed by (seed, update) makes each sample a pure function of the
        # weights and never draws from the training RNG, so turning samples on or off leaves
        # training (dropout masks, shuffling) unchanged.
        seed = (cfg.training.seed << 31) + step
        ids = generate_sampled(
            model,
            idx,
            cfg.output.sample_tokens,
            cfg.model.context_length,
            temperature=cfg.output.temperature,
            top_k=cfg.output.top_k,
            eos_id=tokenizer.eot_token,
            generator=torch.Generator().manual_seed(seed),
        )[0].tolist()
        return {
            "prompt": cfg.output.prompt,
            "text": tokenizer.decode(ids),
            "token_ids": ids,
            "method": "sampled",
            "temperature": cfg.output.temperature,
            "top_k": cfg.output.top_k,
            "seed": seed,
        }
    finally:
        model.train(was_training)


def _load_datasets(cfg, tokenizer, tokenizer_manifest):
    """Return (train, validation) datasets, the data digest, and data-manifest fields."""
    length = cfg.model.context_length
    if cfg.data.format == "tokens":
        path = Path(cfg.data.path)
        manifest, digest = load_token_corpus(path, cfg.data.corpus, tokenizer_manifest)
        datasets = [TokenWindowDataset(path / f"{split}.bin", length, length) for split in SPLITS]
        fields = {
            "format": "tokens",
            "corpus": manifest["provenance"],
            "split_policy": "corpus-defined train/val files",
            "document_policy": manifest["document_policy"],
            "eot_policy": manifest["eot_policy"],
            "cross_document_attention": manifest["cross_document_attention"],
            "splits": manifest["splits"],
        }
        return datasets, digest, fields
    data_bytes = Path(cfg.data.path).read_bytes()
    digest = hashlib.sha256(data_bytes).hexdigest()
    if cfg.data.sha256 is not None and digest != cfg.data.sha256:
        raise ValueError("Input data SHA256 does not match config")
    text = data_bytes.decode("utf-8")
    split = int(len(text) * cfg.data.train_ratio)
    datasets = [
        SlidingWindowTokenDataset(part, tokenizer, length, length)
        for part in (text[:split], text[split:])
    ]
    fields = {
        "split_policy": "character boundary before independent tokenization",
        "split_character": split,
    }
    return datasets, digest, fields


def _batch_hash(batches):
    # One micro-batch hashes its own bytes (the book path); a group hashes their concatenation.
    return hashlib.sha256(b"".join(b.numpy().tobytes() for b in batches)).hexdigest()


# The code a GPT-2 language-model run executes; see implementation_fingerprint.
LM_TRAINING_CODE = (
    "config.py",
    "evaluation.py",
    "generation.py",
    "data",
    "model/gpt2",
    "training",
)


def _same_environment(saved, current, allow_code_drift=False):
    keys = (
        "python",
        "torch",
        "numpy",
        "tiktoken",
        "platform",
        "machine",
        "device",
        "precision",
        "num_threads",
        "deterministic_algorithms",
        "matmul_precision",
    )
    if any(saved.get(k) != current.get(k) for k in keys):
        raise ValueError(
            "Resume requires the original Python/Torch/platform/device/thread environment"
        )
    drifted = saved.get("implementation_sha256") != current.get("implementation_sha256")
    if drifted and not allow_code_drift:
        raise ValueError(
            "Resume environment mismatch: training code changed since this checkpoint was written; "
            "exact recovery is only guaranteed for identical code. "
            "Pass --allow-code-drift to resume anyway; the run records the override."
        )
    return drifted


def train(
    cfg: Config | None = None,
    *,
    resume: Path | None = None,
    stop_after: int | None = None,
    allow_code_drift: bool = False,
) -> Path:
    """Resume creates a child run; recipe and total budget remain immutable.

    `stop_after` is an absolute completed-update count, useful for planned stops.
    SIGINT/SIGTERM request the same stop after the current safe update boundary.
    """
    saved = load_checkpoint(resume) if resume else None
    if saved:
        restored_config = config_from_dict(saved["config"], Path.cwd())
        if cfg is not None and cfg.to_dict() != restored_config.to_dict():
            raise ValueError("Cannot change the saved training recipe on resume")
        cfg = restored_config
    if cfg is None:
        raise ValueError("Provide a config or a full training checkpoint")
    if stop_after is not None and (type(stop_after) is not int or stop_after <= 0):
        raise ValueError("stop_after must be a positive absolute optimizer-step count")
    device = cfg.runtime.device
    env = collect_runtime_metadata() | {
        "device": device,
        "precision": "float32",
        "num_threads": cfg.runtime.num_threads,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "matmul_precision": torch.get_float32_matmul_precision(),
    }
    package = Path(__file__).parent.parent
    env["implementation_sha256"] = implementation_fingerprint(package, LM_TRAINING_CODE)
    if device == "cuda" and not env["cuda_available"]:
        raise ValueError("CUDA was requested but is unavailable")
    if device == "mps" and not env["mps_available"]:
        raise ValueError("MPS was requested but is unavailable")
    code_drift = _same_environment(saved["environment"], env, allow_code_drift) if saved else False
    torch.set_num_threads(cfg.runtime.num_threads)
    torch.manual_seed(cfg.training.seed)
    random.seed(cfg.training.seed)
    np.random.seed(cfg.training.seed)
    tokenizer = load_tokenizer(Path(cfg.data.tokenizer_cache))
    tokenizer_manifest = json.loads((Path(cfg.data.tokenizer_cache) / "manifest.json").read_text())
    datasets, digest, data_fields = _load_datasets(cfg, tokenizer, tokenizer_manifest)
    identity = {"sha256": digest, "tokenizer_manifest": tokenizer_manifest}
    if saved and saved["data_identity"] != identity:
        raise ValueError("Dataset/tokenizer identity differs from the checkpoint")
    if len(datasets[0]) < cfg.training.batch_size or not len(datasets[1]):
        raise ValueError("Insufficient data: need a full training batch and a validation window")
    train_loader = DataLoader(
        datasets[0], batch_size=cfg.training.batch_size, shuffle=True, drop_last=True, num_workers=0
    )
    val_loader = DataLoader(
        datasets[1], batch_size=cfg.training.batch_size, shuffle=False, num_workers=0
    )
    model = GPTModel(asdict(cfg.model)).to(device)
    optimizer = build_optimizer(model, cfg.training)
    accum = cfg.training.grad_accum_steps
    # A final partial group (epoch length not divisible by `accum`) still updates, averaged
    # over the micro-batches it actually has; batches are equal-sized (drop_last), so the
    # mean of per-batch means is the token-weighted mean.
    steps_per_epoch = -(-len(train_loader) // accum)
    total_steps = steps_per_epoch * cfg.training.num_epochs
    if cfg.training.max_steps is not None:
        total_steps = min(total_steps, cfg.training.max_steps)
    if cfg.training.warmup_steps >= total_steps and not is_constant(cfg.training):
        raise ValueError("training.warmup_steps must be smaller than the total optimizer steps")
    progress = {
        "epoch": 0,
        "batch_cursor": 0,
        "optimizer_steps": 0,
        "tokens_seen": 0,
        "epoch_start_rng": None,
    }
    history, epoch_samples = [], []
    best_validation = float("inf")
    best_step = None
    if saved:
        progress = saved["progress"]
        if progress["optimizer_steps"] >= total_steps:
            raise ValueError("Checkpoint has already completed the configured training budget")
        model.load_state_dict(saved["model_state_dict"])
        optimizer.load_state_dict(saved["optimizer_state_dict"])
        history, epoch_samples = saved["history"], saved["epoch_samples"]
        best_validation = saved["best_validation"]
        best_step = saved["best_step"]
        restore_rng(saved["rng"], device)
    if stop_after is not None and stop_after <= progress["optimizer_steps"]:
        raise ValueError("stop_after must exceed the checkpoint's completed updates")

    def validation(max_batches):
        with isolated_rng(device):
            return evaluate(model, val_loader, device, max_batches)

    initial = saved["initial_evaluation"] if saved else validation(cfg.training.eval_iter)
    root = Path(__file__).resolve().parents[3]
    lockfile = root / "uv.lock"
    run = Path(cfg.output.root) / (
        datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]
    )
    run.mkdir(parents=True, exist_ok=False)
    (run / "config.resolved.yaml").write_text(yaml.safe_dump(cfg.to_dict(), sort_keys=False))
    write_json(run / "environment.json", env)
    write_json(
        run / "provenance.json",
        git_provenance()
        | {
            "upstream": json.loads(
                (Path(__file__).parent.parent / "upstream-provenance.json").read_text()
            ),
            "config_sha256": config_hash(cfg.to_dict()),
            "lock_sha256": hashlib.sha256(lockfile.read_bytes()).hexdigest()
            if lockfile.is_file()
            else None,
            "parent_checkpoint": str(resume.resolve()) if resume else None,
            "parent_optimizer_step": progress["optimizer_steps"] if resume else None,
            "code_drift_allowed": code_drift,
            "parent_implementation_sha256": saved["environment"].get("implementation_sha256")
            if saved
            else None,
        },
    )
    write_json(
        run / "data-manifest.json",
        identity
        | {
            "path": cfg.data.path,
            "tokenizer": "gpt2 (reused, not trained)",
            **data_fields,
            "windows": {"train": len(datasets[0]), "validation": len(datasets[1])},
        },
    )
    print(
        json.dumps(
            {
                "run": str(run),
                "parameters": sum(p.numel() for p in model.parameters()),
                "device": device,
                "data_sha256": digest,
                "total_optimizer_steps": total_steps,
                "resume_step": progress["optimizer_steps"],
            }
        ),
        flush=True,
    )
    write_json(run / "status.json", {"status": "running"})
    requested_signal = None

    def request_stop(signum, frame):
        nonlocal requested_signal
        requested_signal = signum

    handlers = {s: signal.signal(s, request_stop) for s in (signal.SIGINT, signal.SIGTERM)}
    started = time.perf_counter()

    def checkpoint(improved=False):
        state = {
            "format_version": FORMAT_VERSION,
            "config": cfg.to_dict(),
            "config_hash": config_hash(cfg.to_dict()),
            "environment": env,
            "data_identity": identity,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": None
            if is_constant(cfg.training)
            else {
                "lr_schedule": cfg.training.lr_schedule,
                "last_step": progress["optimizer_steps"],
            },
            "scaler_state_dict": None,
            "rng": capture_rng(device),
            "progress": progress,
            "history": history,
            "epoch_samples": epoch_samples,
            "initial_evaluation": initial,
            "best_validation": best_validation,
            "best_step": best_step,
        }
        atomic_save(run / "checkpoints/last.pt", state)
        if improved:
            temporary = run / "checkpoints/best.pt.tmp"
            shutil.copyfile(run / "checkpoints/last.pt", temporary)
            temporary.replace(run / "checkpoints/best.pt")

    try:
        # A valid initial state survives even if the first optimizer update fails.
        checkpoint()
        if saved and best_step is not None:
            parent_best = resume.parent / "best.pt"
            if parent_best.is_file():
                inherited = load_checkpoint(parent_best)
                if (
                    inherited["progress"]["optimizer_steps"] == best_step
                    and inherited["best_validation"] == best_validation
                ):
                    shutil.copyfile(parent_best, run / "checkpoints/best.pt")
                del inherited
        with (run / "metrics.jsonl").open("w") as metrics:
            for record in history:
                metrics.write(json.dumps(record, allow_nan=False) + "\n")
            metrics.flush()
            while progress["optimizer_steps"] < total_steps:
                if progress["epoch_start_rng"] is None:
                    progress["epoch_start_rng"] = capture_rng(device)
                    iterator = iter(train_loader)
                else:
                    # Replay only deterministic data access, never model updates.
                    # Restore live RNG afterward: evaluation also consumes seeds.
                    live_rng = capture_rng(device)
                    restore_rng(progress["epoch_start_rng"], device)
                    iterator = iter(train_loader)
                    for _ in range(progress["batch_cursor"]):
                        next(iterator)
                    restore_rng(live_rng, device)
                model.train()
                while progress["batch_cursor"] < len(train_loader):
                    group = min(accum, len(train_loader) - progress["batch_cursor"])
                    if not is_constant(cfg.training):
                        # Pure function of the update index: never trust a restored optimizer lr.
                        lr = learning_rate_at(
                            cfg.training, progress["optimizer_steps"], total_steps
                        )
                        for param_group in optimizer.param_groups:
                            param_group["lr"] = lr
                    else:
                        lr = cfg.training.learning_rate
                    optimizer.zero_grad()
                    batches, loss_sum, group_tokens = [], 0.0, 0
                    for _ in range(group):
                        inputs, targets = next(iterator)
                        loss = next_token_cross_entropy(inputs, targets, model, device)
                        if not torch.isfinite(loss):
                            raise ValueError(
                                "Nonfinite training loss; resume the previous safe checkpoint"
                            )
                        (loss / group if group > 1 else loss).backward()
                        loss_sum += loss.item()
                        batches.append(inputs)
                        group_tokens += targets.numel()
                    grad_norm = clip_and_measure(model, cfg.training.grad_clip)
                    if not torch.isfinite(grad_norm):
                        raise ValueError("Nonfinite gradients; resume the previous safe checkpoint")
                    optimizer.step()
                    progress["optimizer_steps"] += 1
                    progress["batch_cursor"] += group
                    progress["tokens_seen"] += group_tokens
                    steps = progress["optimizer_steps"]
                    record = {
                        "optimizer_step": steps,
                        "epoch": progress["epoch"] + 1,
                        "train_loss": loss_sum / group,
                        "tokens_seen": progress["tokens_seen"],
                        "learning_rate": lr,
                        "grad_norm": grad_norm.item(),
                        "batch_sha256": _batch_hash(batches),
                    }
                    improved = False
                    if (steps - 1) % cfg.training.eval_freq == 0:
                        if cfg.training.recipe == "sampled_train_validation":
                            train_loss, val_loss = evaluate_train_validation_loss(
                                model, train_loader, val_loader, device, cfg.training.eval_iter
                            )
                            record |= {"source_train_loss": train_loss, "source_val_loss": val_loss}
                        else:
                            record["validation"] = validation(cfg.training.eval_iter)
                            val_loss = record["validation"]["loss"]
                        if not math.isfinite(val_loss):
                            raise ValueError("Nonfinite validation loss")
                        improved = val_loss < best_validation
                        best_validation = min(best_validation, val_loss)
                        if improved:
                            best_step = steps
                    sample_every = cfg.output.sample_every
                    if sample_every is not None and (steps - 1) % sample_every == 0:
                        record["sample"] = _sample(model, tokenizer, cfg, device, steps)["text"]
                    history.append(record)
                    metrics.write(json.dumps(record, allow_nan=False) + "\n")
                    metrics.flush()
                    if progress["batch_cursor"] == len(train_loader):
                        if cfg.training.recipe == "sampled_train_validation":
                            epoch_samples.append(
                                _sample(model, tokenizer, cfg, device, steps)
                                | {"epoch": progress["epoch"] + 1, "optimizer_step": steps}
                            )
                        progress["epoch"] += 1
                        progress["batch_cursor"] = 0
                        progress["epoch_start_rng"] = None
                    stopped = requested_signal is not None or (
                        stop_after is not None and steps >= stop_after
                    )
                    done = steps >= total_steps
                    if improved or steps % cfg.training.checkpoint_every == 0 or stopped or done:
                        checkpoint(improved)
                    print(json.dumps(record), flush=True)
                    if stopped or done:
                        break
                    if progress["batch_cursor"] == 0:
                        break
                if stopped or done:
                    break
        done = progress["optimizer_steps"] >= total_steps
        status = "completed" if done else "interrupted"
        # Final reports never perturb the state already saved for the next update.
        if done:
            final = validation(cfg.training.eval_iter)
            full_validation = validation(None)
            write_json(
                run / "evaluation.json",
                {
                    "initial": initial,
                    "final": final,
                    "full_validation": full_validation,
                    "split": "validation",
                    "policy": "initial/final monitor subset plus final full split",
                },
            )
            atomic_save(
                run / "model.pt",
                {
                    "format_version": 1,
                    "model_config": asdict(cfg.model),
                    "model_state_dict": model.state_dict(),
                    "optimizer_steps": progress["optimizer_steps"],
                },
            )
            write_json(
                run / "sample.json",
                _sample(model, tokenizer, cfg, device, progress["optimizer_steps"]),
            )
        (run / "epoch-samples.jsonl").write_text(
            "".join(json.dumps(s) + "\n" for s in epoch_samples)
        )
        write_json(
            run / "status.json",
            {
                "status": status,
                "optimizer_steps": progress["optimizer_steps"],
                "tokens_seen": progress["tokens_seen"],
                "completed_epochs": progress["epoch"],
                "batch_cursor": progress["batch_cursor"],
                "signal": requested_signal,
                "elapsed_seconds": time.perf_counter() - started,
            },
        )
    except BaseException as exc:
        # Do not serialize potentially half-applied updates; last.pt remains safe.
        write_json(
            run / "status.json",
            {
                "status": "failed",
                "error": str(exc),
                "optimizer_steps": progress["optimizer_steps"],
                "recovery_checkpoint": "checkpoints/last.pt",
            },
        )
        raise
    finally:
        for signum, handler in handlers.items():
            signal.signal(signum, handler)
    print(f"{status.capitalize()}: {run}", flush=True)
    return run
