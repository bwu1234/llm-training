"""Single-process FP32 training with periodic loss monitoring and full-state recovery."""

import hashlib
import importlib.metadata
import json
import math
import random
import shutil
import signal
import time
from dataclasses import asdict
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from uuid import uuid4

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from llm_training.data.corpus import SPLITS, load_token_corpus
from llm_training.data.dataset import SlidingWindowTokenDataset
from llm_training.data.token_dataset import TokenWindowDataset
from llm_training.lm.config import Config, config_from_dict, window_length
from llm_training.lm.evaluation import evaluate
from llm_training.lm.generation import generate_greedy, generate_sampled
from llm_training.lm.loss import next_token_cross_entropy
from llm_training.lm.monitoring import evaluate_train_validation_loss
from llm_training.lm.optim import build_optimizer
from llm_training.lm.schedule import is_constant, learning_rate_at
from llm_training.model.registry import MODEL_FILE_FORMAT, PRETRAINED, family, read_model_file
from llm_training.training.checkpoint import (
    FORMAT_VERSION,
    atomic_save,
    capture_rng,
    config_hash,
    isolated_rng,
    load_checkpoint,
    restore_rng,
)
from llm_training.training.optim import clip_and_measure
from llm_training.training.provenance import (
    collect_runtime_metadata,
    git_provenance,
    implementation_fingerprint,
    write_json,
)


def _sample(model, tokenizer, cfg, device, step=0):
    was_training = model.training
    model.eval()
    try:
        prompt = tokenizer.encode(cfg.output.prompt) or [tokenizer.eot_token]
        idx = torch.tensor([prompt], device=device)
        if cfg.output.temperature == 0 and cfg.output.top_k is None:
            if cfg.data.format == "instructions":
                # A fine-tuned response ends at end-of-text; the book's text-only greedy loop
                # (kept below for its parity test) would keep generating past it.
                ids = generate_sampled(
                    model,
                    idx,
                    cfg.output.sample_tokens,
                    window_length(cfg),
                    eos_id=tokenizer.eot_token,
                )
            else:
                ids = generate_greedy(model, idx, cfg.output.sample_tokens, window_length(cfg))
            ids = ids[0].tolist()
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
            window_length(cfg),
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
    length = window_length(cfg)
    if cfg.data.format == "instructions":
        from llm_training.data.instructions import (
            DATASETS,
            InstructionDataset,
            load_instructions,
            split_instructions,
        )

        rows, digest = load_instructions(cfg.data.dataset, cfg.data.path)
        splits, dropped = split_instructions(
            cfg.data.dataset, rows, tokenizer, length, cfg.data.limit
        )
        spec = DATASETS[cfg.data.dataset]
        fields = {
            "format": "instructions",
            "dataset": cfg.data.dataset,
            "source": {k: spec[k] for k in ("description", "license", "url", "sha256")},
            "split_policy": spec["split"],
            "examples": {split: len(examples) for split, examples in splits.items()},
            "dropped": dropped,
            "max_length": length,
            "mask_prompt": cfg.data.mask_prompt,
            "limit": cfg.data.limit,
        }
        datasets = [InstructionDataset(splits[s]) for s in ("train", "validation")]
        return datasets, digest, fields
    if cfg.data.format == "tokens":
        path = Path(cfg.data.path)
        manifest, digest = load_token_corpus(path, cfg.data.corpus, tokenizer_manifest)
        datasets = [
            TokenWindowDataset(path / f"{split}.bin", length, length, manifest["dtype"])
            for split in SPLITS
        ]
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


def lm_training_code(cfg):
    """The package paths a next-token run executes, for `implementation_fingerprint`.

    Only the run's own model family counts, so editing Qwen3 never blocks resuming a GPT-2
    run, and the reasoning modules count only when the math-distill prompt uses them.
    """
    code = [
        "data",
        "lm",
        "model/assets.py",
        "model/family.py",
        "model/registry.py",
        f"model/{cfg.model.arch}",
        "training",
    ]
    if cfg.data.format == "instructions":
        from llm_training.data.instructions import DATASETS

        if DATASETS[cfg.data.dataset]["prompt"] == "math":
            code += ["reasoning/distill.py", "reasoning/grading.py"]
    return tuple(code)


def _same_environment(saved, current, allow_code_drift=False, packages=()):
    """`packages` are the family's `resume_packages`, recorded under their names."""
    keys = (
        "python",
        "torch",
        "numpy",
        "tiktoken",
        *packages,
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


def _load_initial_weights(model, model_cfg, model_family):
    """Load the run's starting weights; return a provenance record of their source."""
    source = model_cfg.init_from
    if source in PRETRAINED:
        state, origin = model_family.release_state(source, model_cfg.weights_dir)
        record = {"pretrained": source, **origin}
    else:
        saved = read_model_file(source)
        if saved["arch"] != model_cfg.arch:
            raise ValueError(f"{source} holds a {saved['arch']} model, not {model_cfg.arch}")
        for key in model_family.architecture_keys:
            if saved["model_config"][key] != getattr(model_cfg, key):
                raise ValueError(f"model.{key} differs from the weights in {source}")
        state = saved["model_state_dict"]
        with open(source, "rb") as file:
            record = {
                "model_file": source,
                "sha256": hashlib.file_digest(file, "sha256").hexdigest(),
            }
    model_family.load_weights(model, state)
    return record


def train(
    cfg: Config | None = None,
    *,
    resume: Path | None = None,
    stop_after: int | None = None,
    allow_code_drift: bool = False,
) -> Path:
    """Run one next-token training run and return its directory; see `Trainer`."""
    return Trainer(
        cfg, resume=resume, stop_after=stop_after, allow_code_drift=allow_code_drift
    ).run()


class Trainer:
    """One next-token run. Resume creates a child run; recipe and total budget remain immutable.

    Construction does everything that can fail before a run directory exists: it restores the
    recipe and environment from a checkpoint, seeds, loads data, and builds the model and
    optimizer in a fixed order, since model initialization and data shuffling consume the
    seeded RNG streams. `run` then trains to the budget, `stop_after`, or a signal.

    `stop_after` is an absolute completed-update count, useful for planned stops.
    SIGINT/SIGTERM request the same stop after the current safe update boundary.
    """

    def __init__(
        self,
        cfg: Config | None = None,
        *,
        resume: Path | None = None,
        stop_after: int | None = None,
        allow_code_drift: bool = False,
    ):
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
        self.cfg, self.resume, self.saved, self.stop_after = cfg, resume, saved, stop_after
        self.device = device = cfg.runtime.device
        env = collect_runtime_metadata() | {
            "device": device,
            "precision": "float32",
            "num_threads": cfg.runtime.num_threads,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "matmul_precision": torch.get_float32_matmul_precision(),
        }
        self.family = model_family = family(cfg.model.arch)
        for package_name in model_family.resume_packages:
            env[package_name] = importlib.metadata.version(package_name)
        package = Path(__file__).parent.parent
        env["implementation_sha256"] = implementation_fingerprint(package, lm_training_code(cfg))
        if device == "cuda" and not env["cuda_available"]:
            raise ValueError("CUDA was requested but is unavailable")
        if device == "mps" and not env["mps_available"]:
            raise ValueError("MPS was requested but is unavailable")
        self.env = env
        self.code_drift = (
            _same_environment(
                saved["environment"], env, allow_code_drift, model_family.resume_packages
            )
            if saved
            else False
        )
        torch.set_num_threads(cfg.runtime.num_threads)
        torch.manual_seed(cfg.training.seed)
        random.seed(cfg.training.seed)
        np.random.seed(cfg.training.seed)
        self.tokenizer, self.tokenizer_manifest = model_family.load_tokenizer(
            cfg.data.tokenizer_cache
        )
        self.datasets, self.digest, self.data_fields = _load_datasets(
            cfg, self.tokenizer, self.tokenizer_manifest
        )
        self.identity = {"sha256": self.digest, "tokenizer_manifest": self.tokenizer_manifest}
        if saved and saved["data_identity"] != self.identity:
            raise ValueError("Dataset/tokenizer identity differs from the checkpoint")
        if len(self.datasets[0]) < cfg.training.batch_size or not len(self.datasets[1]):
            raise ValueError(
                "Insufficient data: need a full training batch and a validation window"
            )
        self.train_loader, self.val_loader = self._loaders()
        self.model = model_family.build(asdict(cfg.model))
        # A resumed run reports the weights its lineage started from, kept in the checkpoint.
        self.initial_weights = saved.get("initial_weights") if saved else None
        if cfg.model.init_from is not None and not saved:
            self.initial_weights = _load_initial_weights(self.model, cfg.model, model_family)
        self.model.to(device)
        self.optimizer = build_optimizer(self.model, cfg.training)
        # A final partial group (epoch length not divisible by `accum`) still updates, averaged
        # over the micro-batches it actually has; batches are equal-sized (drop_last), so the
        # mean of per-batch means is the token-weighted mean.
        accum = cfg.training.grad_accum_steps
        steps_per_epoch = -(-len(self.train_loader) // accum)
        self.total_steps = steps_per_epoch * cfg.training.num_epochs
        if cfg.training.max_steps is not None:
            self.total_steps = min(self.total_steps, cfg.training.max_steps)
        if cfg.training.warmup_steps >= self.total_steps and not is_constant(cfg.training):
            raise ValueError("training.warmup_steps must be smaller than the total optimizer steps")
        self.progress = {
            "epoch": 0,
            "batch_cursor": 0,
            "optimizer_steps": 0,
            "tokens_seen": 0,
            "epoch_start_rng": None,
        }
        self.history, self.epoch_samples = [], []
        self.best_validation = float("inf")
        self.best_step = None
        if saved:
            self.progress = saved["progress"]
            if self.progress["optimizer_steps"] >= self.total_steps:
                raise ValueError("Checkpoint has already completed the configured training budget")
            self.model.load_state_dict(saved["model_state_dict"])
            self.optimizer.load_state_dict(saved["optimizer_state_dict"])
            self.history, self.epoch_samples = saved["history"], saved["epoch_samples"]
            self.best_validation = saved["best_validation"]
            self.best_step = saved["best_step"]
            restore_rng(saved["rng"], device)
        if stop_after is not None and stop_after <= self.progress["optimizer_steps"]:
            raise ValueError("stop_after must exceed the checkpoint's completed updates")
        self.requested_signal = None

    def _loaders(self):
        cfg = self.cfg
        collate = None
        if cfg.data.format == "instructions":
            from llm_training.data.instructions import collate_instructions

            collate = partial(
                collate_instructions,
                pad_token_id=self.tokenizer.eot_token,
                mask_prompt=cfg.data.mask_prompt,
            )
        train_loader = DataLoader(
            self.datasets[0],
            batch_size=cfg.training.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=0,
            collate_fn=collate,
        )
        val_loader = DataLoader(
            self.datasets[1],
            batch_size=cfg.training.batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=collate,
        )
        return train_loader, val_loader

    def evaluate(self, max_batches):
        """Token-weighted validation loss, drawing nothing from the training RNG streams."""
        with isolated_rng(self.device):
            return evaluate(self.model, self.val_loader, self.device, max_batches)

    def run(self) -> Path:
        cfg, saved, progress = self.cfg, self.saved, self.progress
        self.initial = (
            saved["initial_evaluation"] if saved else self.evaluate(cfg.training.eval_iter)
        )
        self.run_dir = run = self._create_run_dir()
        print(
            json.dumps(
                {
                    "run": str(run),
                    "parameters": sum(p.numel() for p in self.model.parameters()),
                    "device": self.device,
                    "data_sha256": self.digest,
                    "total_optimizer_steps": self.total_steps,
                    "resume_step": progress["optimizer_steps"],
                }
            ),
            flush=True,
        )
        write_json(run / "status.json", {"status": "running"})
        handlers = {
            s: signal.signal(s, self._request_stop) for s in (signal.SIGINT, signal.SIGTERM)
        }
        started = time.perf_counter()
        try:
            # A valid initial state survives even if the first optimizer update fails.
            self.checkpoint()
            self._inherit_best()
            with (run / "metrics.jsonl").open("w") as metrics:
                for record in self.history:
                    metrics.write(json.dumps(record, allow_nan=False) + "\n")
                metrics.flush()
                while progress["optimizer_steps"] < self.total_steps:
                    if self._train_epoch(metrics):
                        break
            done = progress["optimizer_steps"] >= self.total_steps
            status = "completed" if done else "interrupted"
            # Final reports never perturb the state already saved for the next update.
            if done:
                self._write_final_reports()
            (run / "epoch-samples.jsonl").write_text(
                "".join(json.dumps(s) + "\n" for s in self.epoch_samples)
            )
            write_json(
                run / "status.json",
                {
                    "status": status,
                    "optimizer_steps": progress["optimizer_steps"],
                    "tokens_seen": progress["tokens_seen"],
                    "completed_epochs": progress["epoch"],
                    "batch_cursor": progress["batch_cursor"],
                    "signal": self.requested_signal,
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

    def _request_stop(self, signum, frame):
        self.requested_signal = signum

    def _create_run_dir(self):
        """Make the run directory and write the records that describe the run before training."""
        cfg, saved, resume = self.cfg, self.saved, self.resume
        root = Path(__file__).resolve().parents[3]
        lockfile = root / "uv.lock"
        run = Path(cfg.output.root) / (
            datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]
        )
        run.mkdir(parents=True, exist_ok=False)
        (run / "config.resolved.yaml").write_text(yaml.safe_dump(cfg.to_dict(), sort_keys=False))
        write_json(run / "environment.json", self.env)
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
                "parent_optimizer_step": self.progress["optimizer_steps"] if resume else None,
                "code_drift_allowed": self.code_drift,
                "parent_implementation_sha256": saved["environment"].get("implementation_sha256")
                if saved
                else None,
                "initial_weights": self.initial_weights,
            },
        )
        write_json(
            run / "data-manifest.json",
            self.identity
            | {
                "path": cfg.data.path,
                "tokenizer": f"{cfg.model.arch} (reused, not trained)",
                **self.data_fields,
                "windows": {"train": len(self.datasets[0]), "validation": len(self.datasets[1])},
            },
        )
        return run

    def checkpoint(self, improved=False):
        """Write the full state to `last.pt`, and copy it to `best.pt` when `improved`."""
        cfg, run = self.cfg, self.run_dir
        state = {
            "format_version": FORMAT_VERSION,
            "config": cfg.to_dict(),
            "config_hash": config_hash(cfg.to_dict()),
            "environment": self.env,
            "data_identity": self.identity,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": None
            if is_constant(cfg.training)
            else {
                "lr_schedule": cfg.training.lr_schedule,
                "last_step": self.progress["optimizer_steps"],
            },
            "scaler_state_dict": None,
            "rng": capture_rng(self.device),
            "progress": self.progress,
            "history": self.history,
            "epoch_samples": self.epoch_samples,
            "initial_evaluation": self.initial,
            "best_validation": self.best_validation,
            "best_step": self.best_step,
            "initial_weights": self.initial_weights,
        }
        atomic_save(run / "checkpoints/last.pt", state)
        if improved:
            temporary = run / "checkpoints/best.pt.tmp"
            shutil.copyfile(run / "checkpoints/last.pt", temporary)
            temporary.replace(run / "checkpoints/best.pt")

    def _inherit_best(self):
        """A child run starts with its parent's `best.pt` when that is still the best state."""
        if not (self.saved and self.best_step is not None):
            return
        parent_best = self.resume.parent / "best.pt"
        if parent_best.is_file():
            inherited = load_checkpoint(parent_best)
            if (
                inherited["progress"]["optimizer_steps"] == self.best_step
                and inherited["best_validation"] == self.best_validation
            ):
                shutil.copyfile(parent_best, self.run_dir / "checkpoints/best.pt")
            del inherited

    def _epoch_iterator(self):
        """The training-data iterator, positioned at the epoch's `batch_cursor`."""
        progress, device = self.progress, self.device
        if progress["epoch_start_rng"] is None:
            progress["epoch_start_rng"] = capture_rng(device)
            return iter(self.train_loader)
        # Replay only deterministic data access, never model updates.
        # Restore live RNG afterward: evaluation also consumes seeds.
        live_rng = capture_rng(device)
        restore_rng(progress["epoch_start_rng"], device)
        iterator = iter(self.train_loader)
        for _ in range(progress["batch_cursor"]):
            next(iterator)
        restore_rng(live_rng, device)
        return iterator

    def _train_epoch(self, metrics):
        """Train through the rest of the current epoch; return True when the run should stop."""
        cfg, progress = self.cfg, self.progress
        iterator = self._epoch_iterator()
        self.model.train()
        while progress["batch_cursor"] < len(self.train_loader):
            record = self.step(iterator)
            steps = record["optimizer_step"]
            improved = self._monitor(record)
            self.history.append(record)
            metrics.write(json.dumps(record, allow_nan=False) + "\n")
            metrics.flush()
            if progress["batch_cursor"] == len(self.train_loader):
                self._end_epoch(steps)
            stopped = self.requested_signal is not None or (
                self.stop_after is not None and steps >= self.stop_after
            )
            done = steps >= self.total_steps
            if improved or steps % cfg.training.checkpoint_every == 0 or stopped or done:
                self.checkpoint(improved)
            print(json.dumps(record), flush=True)
            if stopped or done:
                return True
            if progress["batch_cursor"] == 0:
                return False
        return False

    def step(self, iterator):
        """One optimizer update over the next group of micro-batches; return its record."""
        cfg, progress, model, optimizer = self.cfg, self.progress, self.model, self.optimizer
        group = min(
            cfg.training.grad_accum_steps, len(self.train_loader) - progress["batch_cursor"]
        )
        if not is_constant(cfg.training):
            # Pure function of the update index: never trust a restored optimizer lr.
            lr = learning_rate_at(cfg.training, progress["optimizer_steps"], self.total_steps)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr
        else:
            lr = cfg.training.learning_rate
        optimizer.zero_grad()
        batches, loss_sum, group_tokens = [], 0.0, 0
        for _ in range(group):
            inputs, targets = next(iterator)
            loss = next_token_cross_entropy(inputs, targets, model, self.device)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss; resume the previous safe checkpoint")
            (loss / group if group > 1 else loss).backward()
            loss_sum += loss.item()
            batches.append(inputs)
            group_tokens += (targets != -100).sum().item()
        grad_norm = clip_and_measure(model, cfg.training.grad_clip)
        if not torch.isfinite(grad_norm):
            raise ValueError("Nonfinite gradients; resume the previous safe checkpoint")
        optimizer.step()
        progress["optimizer_steps"] += 1
        progress["batch_cursor"] += group
        progress["tokens_seen"] += group_tokens
        return {
            "optimizer_step": progress["optimizer_steps"],
            "epoch": progress["epoch"] + 1,
            "train_loss": loss_sum / group,
            "tokens_seen": progress["tokens_seen"],
            "learning_rate": lr,
            "grad_norm": grad_norm.item(),
            "batch_sha256": _batch_hash(batches),
        }

    def _monitor(self, record):
        """Add the scheduled evaluation and sample to `record`; return True on a new best."""
        cfg, steps = self.cfg, record["optimizer_step"]
        improved = False
        if (steps - 1) % cfg.training.eval_freq == 0:
            if cfg.training.recipe == "sampled_train_validation":
                train_loss, val_loss = evaluate_train_validation_loss(
                    self.model,
                    self.train_loader,
                    self.val_loader,
                    self.device,
                    cfg.training.eval_iter,
                )
                record |= {"source_train_loss": train_loss, "source_val_loss": val_loss}
            else:
                record["validation"] = self.evaluate(cfg.training.eval_iter)
                val_loss = record["validation"]["loss"]
            if not math.isfinite(val_loss):
                raise ValueError("Nonfinite validation loss")
            improved = val_loss < self.best_validation
            self.best_validation = min(self.best_validation, val_loss)
            if improved:
                self.best_step = steps
        sample_every = cfg.output.sample_every
        if sample_every is not None and (steps - 1) % sample_every == 0:
            record["sample"] = _sample(self.model, self.tokenizer, cfg, self.device, steps)["text"]
        return improved

    def _end_epoch(self, steps):
        cfg, progress = self.cfg, self.progress
        if cfg.training.recipe == "sampled_train_validation":
            self.epoch_samples.append(
                _sample(self.model, self.tokenizer, cfg, self.device, steps)
                | {"epoch": progress["epoch"] + 1, "optimizer_step": steps}
            )
        progress["epoch"] += 1
        progress["batch_cursor"] = 0
        progress["epoch_start_rng"] = None

    def _write_final_reports(self):
        cfg, run, steps = self.cfg, self.run_dir, self.progress["optimizer_steps"]
        final = self.evaluate(cfg.training.eval_iter)
        full_validation = self.evaluate(None)
        write_json(
            run / "evaluation.json",
            {
                "initial": self.initial,
                "final": final,
                "full_validation": full_validation,
                "split": "validation",
                "policy": "initial/final monitor subset plus final full split",
            },
        )
        atomic_save(
            run / "model.pt",
            {
                "format_version": MODEL_FILE_FORMAT,
                "arch": cfg.model.arch,
                "model_config": asdict(cfg.model),
                "model_state_dict": self.model.state_dict(),
                "optimizer_steps": steps,
                "tokenizer": self.tokenizer_manifest,
            },
        )
        write_json(
            run / "sample.json", _sample(self.model, self.tokenizer, cfg, self.device, steps)
        )
