"""Strict configuration for GRPO fine-tuning of the Qwen3-0.6B base model on MATH.

Defaults follow the run behind chapter 6's 47.4% MATH-500 result in *Build a Reasoning Model
(From Scratch)* (`rlvr_grpo_original_no_kl.py --num_rollouts 8 --max_new_tokens 512`), except
that parameters and optimizer state are float32 here, where upstream holds both in bfloat16.
"""

import math
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import yaml


@dataclass(frozen=True)
class GRPOModelConfig:
    weights: str = "../data/qwen3-0.6b-base"  # directory holding the pinned base files
    # The source model trains its head and token embedding as separate parameters.
    tie_embeddings: bool = False
    attn_impl: str = "sdpa"


@dataclass(frozen=True)
class GRPODataConfig:
    path: str = "../data/math"


@dataclass(frozen=True)
class GRPOTrainingConfig:
    seed: int = 42
    steps: int = 100  # one problem per step, in file order
    num_rollouts: int = 8
    max_new_tokens: int = 512
    temperature: float = 0.8
    top_p: float = 0.9
    learning_rate: float = 1e-5
    weight_decay: float = 0.01  # torch.optim.AdamW's default, which upstream uses
    grad_clip: float = 1.0
    # Upstream's --skip-zero-advantage-updates: no optimizer step when all rewards are equal.
    # Off, a zero-gradient step still applies AdamW momentum and weight decay, as upstream's
    # default (and the 47.4% run) does.
    skip_zero_advantage_updates: bool = False
    checkpoint_every: int = 10  # full resumable state, overwritten in place
    snapshot_every: int = 10  # weights-only copies kept for checkpoint selection
    grade_timeout: float = 30.0  # seconds per answer before it is graded incorrect


@dataclass(frozen=True)
class GRPORuntimeConfig:
    device: str = "mps"


@dataclass(frozen=True)
class GRPOOutputConfig:
    root: str = "../runs/grpo"


@dataclass(frozen=True)
class GRPOConfig:
    model: GRPOModelConfig
    data: GRPODataConfig
    training: GRPOTrainingConfig
    runtime: GRPORuntimeConfig
    output: GRPOOutputConfig

    def to_dict(self):
        return asdict(self)


def load_grpo_config(path: str | Path) -> GRPOConfig:
    path = Path(path).resolve()
    return grpo_config_from_dict(yaml.safe_load(path.read_text()), path.parent)


def grpo_config_from_dict(raw: dict, base_dir: Path) -> GRPOConfig:
    """Validate a YAML recipe or the embedded recipe of a checkpoint."""
    sections = {
        "model": GRPOModelConfig,
        "data": GRPODataConfig,
        "training": GRPOTrainingConfig,
        "runtime": GRPORuntimeConfig,
        "output": GRPOOutputConfig,
    }
    if not isinstance(raw, dict) or set(raw) != set(sections):
        raise ValueError(f"Config must contain exactly these sections: {list(sections)}")
    values = {}
    for name, cls in sections.items():
        if not isinstance(raw[name], dict):
            raise ValueError(f"{name} must be a mapping")
        if unknown := set(raw[name]) - {f.name for f in fields(cls)}:
            raise ValueError(f"Unknown fields in {name}: {sorted(unknown)}")
        values[name] = cls(**raw[name])
    cfg = GRPOConfig(**values)
    t = cfg.training
    for name in ("steps", "num_rollouts", "max_new_tokens", "checkpoint_every", "snapshot_every"):
        if type(getattr(t, name)) is not int or getattr(t, name) <= 0:
            raise ValueError(f"training.{name} must be a positive integer")
    if t.num_rollouts < 2:
        raise ValueError("training.num_rollouts must be at least 2: advantages compare rollouts")
    if type(t.seed) is not int or not 0 <= t.seed < 2**63:
        raise ValueError("training.seed must be an integer in [0, 2**63)")
    for value, name, lo, hi, inclusive in [
        (t.temperature, "training.temperature", 0, float("inf"), False),
        (t.top_p, "training.top_p", 0, 1, False),
        (t.learning_rate, "training.learning_rate", 0, float("inf"), False),
        (t.weight_decay, "training.weight_decay", 0, float("inf"), True),
        (t.grad_clip, "training.grad_clip", 0, float("inf"), False),
        (t.grade_timeout, "training.grade_timeout", 0, float("inf"), False),
    ]:
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError(f"{name} must be a finite number")
        if not ((value >= lo if inclusive else value > lo) and value <= hi):
            raise ValueError(f"{name} is outside its supported range")
    for section, name in [(cfg.model, "tie_embeddings"), (t, "skip_zero_advantage_updates")]:
        if type(getattr(section, name)) is not bool:
            raise ValueError(f"{name} must be boolean")
    if cfg.model.attn_impl not in ("explicit", "sdpa"):
        raise ValueError("model.attn_impl must be explicit or sdpa")
    if cfg.runtime.device not in {"cpu", "mps", "cuda"}:
        raise ValueError("runtime.device must be cpu, mps, or cuda")
    for section, name in [(cfg.model, "weights"), (cfg.data, "path"), (cfg.output, "root")]:
        value = getattr(section, name)
        if not isinstance(value, str) or not value:
            raise ValueError(f"{name} must be a nonempty path string")
        object.__setattr__(section, name, str((base_dir / value).resolve()))
    return cfg
