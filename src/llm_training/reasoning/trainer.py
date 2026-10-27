"""GRPO fine-tuning of the Qwen3-0.6B base model on MATH with verifiable rewards.

Chapter 6 of *Build a Reasoning Model (From Scratch)* as a resumable run: one problem per
step in file order, a group of sampled rollouts, a reward of 1 for a `\\boxed{}` answer that
grades equal to the reference, group-normalized advantages, and the policy-gradient loss
without a KL term. Each run directory records its recipe, environment, provenance, per-step
metrics, every rollout's text, and two kinds of weights:

- `checkpoints/last.pt`: full state (weights, AdamW, sampler RNG), overwritten in place;
- `snapshots/stepNNNNN.pt`: weights only, kept, for scoring with `llm-training math-eval`.
  GRPO without a KL term can get worse with more steps, so pick a snapshot by its score.
"""

import hashlib
import importlib.metadata
import json
import signal
import time
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import torch
import yaml

from llm_training.model.qwen3 import pretrained
from llm_training.model.qwen3.pretrained import load_qwen3_base, load_qwen3_state
from llm_training.training.checkpoint import atomic_save, config_hash
from llm_training.training.optim import clip_and_measure
from llm_training.training.provenance import (
    collect_runtime_metadata,
    git_provenance,
    implementation_fingerprint,
    write_json,
)

from .config import GRPOConfig, grpo_config_from_dict
from .evaluate import SNAPSHOT_FORMAT
from .grading import TimeLimitedGrader
from .grpo import grpo_step
from .math_data import dataset_identity, load_math_data

FORMAT_VERSION = 1

# The code a GRPO run executes; see implementation_fingerprint.
GRPO_CODE = (
    "model/qwen3",
    "reasoning",
    "training/checkpoint.py",
    "training/optim.py",
    "training/provenance.py",
)

_RESUME_ENVIRONMENT = (
    "python",
    "torch",
    "numpy",
    "tokenizers",
    "sympy",
    "platform",
    "machine",
    "device",
    "precision",
    "matmul_precision",
)


def grpo_environment(device):
    env = collect_runtime_metadata() | {
        "tokenizers": importlib.metadata.version("tokenizers"),
        "sympy": importlib.metadata.version("sympy"),
        "device": device,
        "precision": "float32",
        "matmul_precision": torch.get_float32_matmul_precision(),
    }
    env["implementation_sha256"] = implementation_fingerprint(
        Path(__file__).parent.parent, GRPO_CODE
    )
    return env


def load_grpo_checkpoint(path):
    state = torch.load(path, map_location="cpu", weights_only=True)
    required = {
        "format_version",
        "config",
        "config_hash",
        "environment",
        "model_state_dict",
        "optimizer_state_dict",
        "generator_state",
        "step",
        "history",
    }
    if not isinstance(state, dict) or not required <= state.keys():
        raise ValueError("Not a full GRPO checkpoint; snapshots hold weights only")
    if state["format_version"] != FORMAT_VERSION:
        raise ValueError("Unsupported GRPO checkpoint version")
    if state["config_hash"] != config_hash(state["config"]):
        raise ValueError("Checkpoint configuration hash mismatch")
    return state


def train_grpo(
    cfg: GRPOConfig | None = None,
    *,
    resume: Path | None = None,
    stop_after: int | None = None,
    allow_code_drift: bool = False,
) -> Path:
    """Resume creates a child run; the recipe and step budget stay fixed.

    `stop_after` is an absolute completed-step count. SIGINT/SIGTERM stop after the current
    step and save the full state.
    """
    saved = load_grpo_checkpoint(resume) if resume else None
    if saved:
        restored = grpo_config_from_dict(saved["config"], Path.cwd())
        if cfg is not None and cfg.to_dict() != restored.to_dict():
            raise ValueError("Cannot change the saved GRPO recipe on resume")
        cfg = restored
    if cfg is None:
        raise ValueError("Provide a config or a full GRPO checkpoint")
    t = cfg.training
    if stop_after is not None and (type(stop_after) is not int or stop_after <= 0):
        raise ValueError("stop_after must be a positive absolute step count")
    device = cfg.runtime.device
    env = grpo_environment(device)
    if device == "cuda" and not env["cuda_available"]:
        raise ValueError("CUDA was requested but is unavailable")
    if device == "mps" and not env["mps_available"]:
        raise ValueError("MPS was requested but is unavailable")
    code_drift = False
    if saved:
        if any(saved["environment"].get(k) != env.get(k) for k in _RESUME_ENVIRONMENT):
            raise ValueError(
                "Resume requires the original Python/Torch/platform/device environment"
            )
        code_drift = saved["environment"]["implementation_sha256"] != env["implementation_sha256"]
        if code_drift and not allow_code_drift:
            raise ValueError(
                "GRPO code changed since this checkpoint was written; pass --allow-code-drift "
                "to resume anyway (the run records the override)"
            )
        if saved["step"] >= t.steps:
            raise ValueError("Checkpoint has already completed the configured steps")
    step = saved["step"] if saved else 0
    if stop_after is not None and stop_after <= step:
        raise ValueError("stop_after must exceed the checkpoint's completed steps")

    problems = load_math_data(cfg.data.path, "train")
    model, tokenizer = load_qwen3_base(
        cfg.model.weights,
        device=device,
        attn_impl=cfg.model.attn_impl,
        tie_embeddings=cfg.model.tie_embeddings,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=t.learning_rate, weight_decay=t.weight_decay
    )
    # Rollout sampling is the run's only randomness. A dedicated generator, saved in every
    # checkpoint, makes a resumed run draw the same rollouts as an uninterrupted one.
    generator = torch.Generator(device=device).manual_seed(t.seed)
    history = []
    if saved:
        load_qwen3_state(model, saved["model_state_dict"])
        optimizer.load_state_dict(saved["optimizer_state_dict"])
        generator.set_state(saved["generator_state"])
        history = saved["history"]

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
                (root / "src/llm_training/upstream-provenance.json").read_text()
            ),
            "base_weights": {
                "repository": pretrained.REPOSITORY,
                "revision": pretrained.REVISION,
                "files": pretrained.FILES,
            },
            "datasets": dataset_identity(),
            "config_sha256": config_hash(cfg.to_dict()),
            "lock_sha256": hashlib.sha256(lockfile.read_bytes()).hexdigest()
            if lockfile.is_file()
            else None,
            "parent_checkpoint": str(Path(resume).resolve()) if resume else None,
            "parent_step": step if resume else None,
            "code_drift_allowed": code_drift,
            "parent_implementation_sha256": saved["environment"]["implementation_sha256"]
            if saved
            else None,
        },
    )
    parameters = sum(p.numel() for p in model.parameters())
    print(
        json.dumps(
            {
                "run": str(run),
                "parameters": parameters,
                "device": device,
                "steps": t.steps,
                "resume_step": step,
            }
        ),
        flush=True,
    )
    write_json(run / "status.json", {"status": "running", "step": step})
    requested_signal = None

    def request_stop(signum, frame):
        nonlocal requested_signal
        requested_signal = signum

    handlers = {s: signal.signal(s, request_stop) for s in (signal.SIGINT, signal.SIGTERM)}
    started = time.perf_counter()

    def checkpoint():
        state = {
            "format_version": FORMAT_VERSION,
            "config": cfg.to_dict(),
            "config_hash": config_hash(cfg.to_dict()),
            "environment": env,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "generator_state": generator.get_state(),
            "step": step,
            "history": history,
        }
        atomic_save(run / "checkpoints/last.pt", state)

    def snapshot():
        atomic_save(
            run / f"snapshots/step{step:05d}.pt",
            {
                "format_version": SNAPSHOT_FORMAT,
                "model_config": {"tie_embeddings": cfg.model.tie_embeddings},
                "model_state_dict": model.state_dict(),
                "step": step,
                "run": str(run),
                "config_sha256": config_hash(cfg.to_dict()),
            },
        )

    try:
        with (
            TimeLimitedGrader(t.grade_timeout) as grader,
            (run / "metrics.jsonl").open("w") as metrics,
            (run / "rollouts.jsonl").open("w") as rollouts,
        ):
            for record in history:
                metrics.write(json.dumps(record, allow_nan=False) + "\n")
            metrics.flush()
            while step < t.steps:
                step_started = time.perf_counter()
                index = step % len(problems)
                example = problems[index]
                model.train()
                optimizer.zero_grad()
                stats = grpo_step(
                    model,
                    tokenizer,
                    example,
                    grader,
                    device,
                    num_rollouts=t.num_rollouts,
                    max_new_tokens=t.max_new_tokens,
                    temperature=t.temperature,
                    top_p=t.top_p,
                    generator=generator,
                )
                updated = not (stats["zero_advantage"] and t.skip_zero_advantage_updates)
                grad_norm = None
                if updated:
                    norm = clip_and_measure(model, t.grad_clip)
                    if not torch.isfinite(norm):
                        raise ValueError("Nonfinite gradients; resume the previous checkpoint")
                    optimizer.step()
                    grad_norm = norm.item()
                optimizer.zero_grad()
                step += 1
                samples = stats["samples"]
                gen_lens = [s["gen_len"] for s in samples]
                seconds = time.perf_counter() - step_started
                record = {
                    "step": step,
                    "problem_index": index,
                    "unique_id": example.get("unique_id"),
                    "reward_mean": sum(stats["rewards"]) / len(stats["rewards"]),
                    "rewards": stats["rewards"],
                    "loss": stats["loss"],
                    "zero_advantage": stats["zero_advantage"],
                    "updated": updated,
                    "grad_norm": grad_norm,
                    "mean_gen_len": sum(gen_lens) / len(gen_lens),
                    "hit_token_limit": sum(n >= t.max_new_tokens for n in gen_lens),
                    "grader_issues": [s["grader_issue"] for s in samples if s["grader_issue"]],
                    "seconds": seconds,
                    "gen_tokens_per_second": sum(gen_lens) / seconds,
                }
                history.append(record)
                metrics.write(json.dumps(record, allow_nan=False) + "\n")
                metrics.flush()
                rollouts.write(
                    json.dumps({"step": step, "problem_index": index, "samples": samples}) + "\n"
                )
                rollouts.flush()
                stopped = requested_signal is not None or (
                    stop_after is not None and step >= stop_after
                )
                done = step >= t.steps
                if step % t.snapshot_every == 0 or done:
                    snapshot()
                if step % t.checkpoint_every == 0 or stopped or done:
                    checkpoint()
                print(json.dumps(record), flush=True)
                if stopped:
                    break
        status = "completed" if step >= t.steps else "interrupted"
        write_json(
            run / "status.json",
            {
                "status": status,
                "step": step,
                "signal": requested_signal,
                "elapsed_seconds": time.perf_counter() - started,
            },
        )
    except BaseException as exc:
        if (run / "checkpoints/last.pt").is_file():
            recovery = "checkpoints/last.pt"
        else:  # nothing saved yet: restart, or resume the parent again
            recovery = str(Path(resume).resolve()) if resume else None
        write_json(
            run / "status.json",
            {
                "status": "failed",
                "error": str(exc),
                "step": step,
                "recovery_checkpoint": recovery,
            },
        )
        raise
    finally:
        for signum, handler in handlers.items():
            signal.signal(signum, handler)
    print(f"{status.capitalize()}: {run}", flush=True)
    return run
