"""Strict configuration for the initial single-process, FP32 training path."""

import math
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import yaml


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 50257
    context_length: int = 16
    emb_dim: int = 32
    n_heads: int = 4
    n_layers: int = 2
    drop_rate: float = 0.1
    qkv_bias: bool = False
    # Defaults reproduce the book model; the gpt2_124m preset overrides all four.
    tie_embeddings: bool = False
    init: str = (
        "pytorch"  # "pytorch" keeps module defaults; "gpt2" is N(0, 0.02), see model/gpt2/init.py
    )
    residual_scaled_init: bool = True  # only read when init == "gpt2"
    attn_impl: str = "explicit"  # "sdpa" uses the fused kernel; dropout RNG then differs


@dataclass(frozen=True)
class DataConfig:
    path: str = "../tests/fixtures/smoke.txt"
    train_ratio: float = 0.9
    tokenizer_cache: str = "../data/tokenizer"
    source_url: str | None = None
    sha256: str | None = None
    # "text": `path` is one UTF-8 file, split at `train_ratio` of its characters (book path).
    # "tokens": `path` is a directory prepared from the registered `corpus` (data/corpus.py),
    # with its own train/val files; `train_ratio`, `source_url` and `sha256` do not apply.
    format: str = "text"
    corpus: str | None = None


@dataclass(frozen=True)
class TrainingConfig:
    seed: int = 123
    batch_size: int = 2
    num_epochs: int = 1
    max_steps: int | None = 4
    learning_rate: float = 0.0005
    weight_decay: float = 0.1
    eval_freq: int = 5
    eval_iter: int = 1
    recipe: str = "validation_only"
    checkpoint_every: int = 10
    # Optimizer recipe. Defaults reproduce the book loop bit-for-bit.
    beta1: float = 0.9
    beta2: float = 0.999
    weight_decay_policy: str = "all"  # "matrices" exempts biases and LayerNorm (dim < 2)
    grad_accum_steps: int = 1  # micro-batches per optimizer update
    grad_clip: float | None = None  # max global gradient norm
    lr_schedule: str = "constant"  # or "cosine"
    warmup_steps: int = 0
    min_lr: float = 0.0  # cosine floor


@dataclass(frozen=True)
class RuntimeConfig:
    device: str = "cpu"
    num_threads: int = 1


@dataclass(frozen=True)
class OutputConfig:
    root: str = "../runs"
    prompt: str = "Every effort moves you"
    sample_tokens: int = 20
    # Sampling for training-time samples. temperature 0 keeps the book's greedy decoding.
    temperature: float = 0.0
    top_k: int | None = None
    sample_every: int | None = None  # also add a "sample" to every Nth update's record


@dataclass(frozen=True)
class Config:
    model: ModelConfig
    data: DataConfig
    training: TrainingConfig
    runtime: RuntimeConfig
    output: OutputConfig

    def to_dict(self):
        return asdict(self)


def load_config(path: str | Path) -> Config:
    path = Path(path).resolve()
    raw = yaml.safe_load(path.read_text())
    return config_from_dict(raw, path.parent)


def config_from_dict(raw: dict, base_dir: Path) -> Config:
    """Validate either a YAML recipe or the embedded recipe of a checkpoint."""
    sections = {
        "model": ModelConfig,
        "data": DataConfig,
        "training": TrainingConfig,
        "runtime": RuntimeConfig,
        "output": OutputConfig,
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
    cfg = Config(**values)
    # Validate types as well as ranges: YAML strings/bools must not become sizes.
    for name in ("vocab_size", "context_length", "emb_dim", "n_heads", "n_layers"):
        _positive_int(getattr(cfg.model, name), f"model.{name}")
    for name in (
        "batch_size",
        "num_epochs",
        "eval_freq",
        "eval_iter",
        "checkpoint_every",
        "grad_accum_steps",
    ):
        _positive_int(getattr(cfg.training, name), f"training.{name}")
    if type(cfg.training.warmup_steps) is not int or cfg.training.warmup_steps < 0:
        raise ValueError("training.warmup_steps must be a non-negative integer")
    _positive_int(cfg.runtime.num_threads, "runtime.num_threads")
    _positive_int(cfg.output.sample_tokens, "output.sample_tokens")
    if cfg.training.max_steps is not None:
        _positive_int(cfg.training.max_steps, "training.max_steps")
    if type(cfg.training.seed) is not int or not 0 <= cfg.training.seed < 2**32:
        raise ValueError("training.seed must be an integer in [0, 2**32)")
    if cfg.model.vocab_size != 50257:
        raise ValueError("The initial GPT-2 tokenizer requires model.vocab_size=50257")
    if cfg.model.emb_dim % cfg.model.n_heads:
        raise ValueError("model.emb_dim must be divisible by model.n_heads")
    for value, name, lo, hi in [
        (cfg.model.drop_rate, "model.drop_rate", 0, 1),
        (cfg.data.train_ratio, "data.train_ratio", 0, 1),
        (cfg.training.learning_rate, "training.learning_rate", 0, float("inf")),
        (cfg.training.weight_decay, "training.weight_decay", 0, float("inf")),
        (cfg.training.beta1, "training.beta1", 0, 1),
        (cfg.training.beta2, "training.beta2", 0, 1),
        (cfg.training.min_lr, "training.min_lr", 0, float("inf")),
    ]:
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError(f"{name} must be a finite number")
        inclusive = name in {
            "model.drop_rate",
            "training.weight_decay",
            "training.beta1",
            "training.beta2",
            "training.min_lr",
        }
        if not ((value >= lo if inclusive else value > lo) and value < hi):
            raise ValueError(f"{name} is outside its supported range")
    for name in ("qkv_bias", "tie_embeddings", "residual_scaled_init"):
        if type(getattr(cfg.model, name)) is not bool:
            raise ValueError(f"model.{name} must be boolean")
    if cfg.model.init not in ("pytorch", "gpt2"):
        raise ValueError("model.init must be pytorch or gpt2")
    if cfg.model.attn_impl not in ("explicit", "sdpa"):
        raise ValueError("model.attn_impl must be explicit or sdpa")
    if cfg.model.tie_embeddings and cfg.model.init != "gpt2":
        # Sharing a matrix between PyTorch's N(0,1) embedding and a Kaiming-scaled head
        # leaves one of the two badly scaled and the initial loss far above ln(vocab).
        raise ValueError("model.tie_embeddings requires model.init: gpt2")
    if cfg.training.grad_clip is not None and (
        type(cfg.training.grad_clip) not in (int, float)
        or not math.isfinite(cfg.training.grad_clip)
        or cfg.training.grad_clip <= 0
    ):
        raise ValueError("training.grad_clip must be a positive finite number or null")
    if cfg.training.weight_decay_policy not in ("all", "matrices"):
        raise ValueError("training.weight_decay_policy must be all or matrices")
    if cfg.training.lr_schedule not in ("constant", "cosine"):
        raise ValueError("training.lr_schedule must be constant or cosine")
    if cfg.training.min_lr > cfg.training.learning_rate:
        raise ValueError("training.min_lr must not exceed training.learning_rate")
    if cfg.training.recipe not in ("validation_only", "sampled_train_validation"):
        raise ValueError("training.recipe must be validation_only or sampled_train_validation")
    if not isinstance(cfg.runtime.device, str) or cfg.runtime.device not in {"cpu", "mps", "cuda"}:
        raise ValueError("runtime.device must be cpu, mps, or cuda")
    for section, names in [(cfg.data, ("path", "tokenizer_cache")), (cfg.output, ("root",))]:
        for name in names:
            value = getattr(section, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a nonempty path string")
            object.__setattr__(section, name, str((base_dir / value).resolve()))
    if not isinstance(cfg.output.prompt, str):
        raise ValueError("output.prompt must be text")
    temperature = cfg.output.temperature
    if type(temperature) not in (int, float) or not math.isfinite(temperature) or temperature < 0:
        raise ValueError("output.temperature must be a non-negative finite number")
    for name in ("top_k", "sample_every"):
        if getattr(cfg.output, name) is not None:
            _positive_int(getattr(cfg.output, name), f"output.{name}")
    if cfg.output.top_k is not None and cfg.output.top_k > cfg.model.vocab_size:
        raise ValueError("output.top_k must not exceed model.vocab_size")
    if cfg.data.format not in ("text", "tokens"):
        raise ValueError("data.format must be text or tokens")
    if cfg.data.format == "tokens":
        from .data.corpus import CORPORA

        if cfg.data.corpus not in CORPORA:
            raise ValueError(f"data.corpus must be one of {sorted(CORPORA)} for token data")
        if cfg.data.source_url is not None or cfg.data.sha256 is not None:
            raise ValueError("Token corpora are pinned by data.corpus; omit source_url/sha256")
        if cfg.data.train_ratio != DataConfig.train_ratio:
            raise ValueError("Token corpora have their own split; omit data.train_ratio")
    elif cfg.data.corpus is not None:
        raise ValueError("data.corpus applies only to data.format: tokens")
    if cfg.data.sha256 is not None:
        if not isinstance(cfg.data.sha256, str) or len(cfg.data.sha256) != 64:
            raise ValueError("data.sha256 must be a 64-character hex digest")
        try:
            int(cfg.data.sha256, 16)
        except ValueError as exc:
            raise ValueError("data.sha256 must be a 64-character hex digest") from exc
    if cfg.data.source_url is not None:
        if not isinstance(cfg.data.source_url, str) or not cfg.data.source_url.startswith(
            "https://"
        ):
            raise ValueError("data.source_url must use HTTPS")
        if not isinstance(cfg.data.sha256, str) or len(cfg.data.sha256) != 64:
            raise ValueError("Downloads require data.sha256")
    return cfg


def _positive_int(value, name):
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
