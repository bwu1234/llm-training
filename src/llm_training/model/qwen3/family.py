"""The Qwen3 family as the next-token trainer sees it (see `model/family.py`)."""

import math
from dataclasses import dataclass
from pathlib import Path

import torch

from ..family import Family, positive_int
from . import pretrained
from .model import QWEN3_0_6B, Qwen3Model


@dataclass(frozen=True)
class Qwen3ModelConfig:
    """The Qwen3 family (`model/qwen3`); defaults are the released 0.6B architecture."""

    arch: str = "qwen3"
    init_from: str | None = None
    weights_dir: str | None = None
    vocab_size: int = 151_936
    context_length: int = 40_960  # sizes the RoPE tables; data.max_length sets the window
    emb_dim: int = 1024
    n_heads: int = 16
    n_layers: int = 28
    hidden_dim: int = 3072
    head_dim: int = 128
    qk_norm: bool = True
    n_kv_groups: int = 8
    rope_base: float = 1_000_000.0
    # The source model trains its head and embedding as separate parameters (chapters 6, 8).
    tie_embeddings: bool = False
    attn_impl: str = "sdpa"


ARCHITECTURE_KEYS = (
    "vocab_size",
    "context_length",
    "emb_dim",
    "n_heads",
    "n_layers",
    "hidden_dim",
    "head_dim",
    "qk_norm",
    "n_kv_groups",
    "rope_base",
)


def validate(cfg):
    m = cfg.model
    if m.init_from is None:
        raise ValueError(
            "Qwen3 starts from released or saved weights: set model.init_from "
            "(training Qwen3 from random weights is not built)"
        )
    for name in ("hidden_dim", "head_dim", "n_kv_groups"):
        positive_int(getattr(m, name), f"model.{name}")
    if m.vocab_size != 151_936:
        raise ValueError("The Qwen3 tokenizer requires model.vocab_size=151936")
    if m.n_heads % m.n_kv_groups:
        raise ValueError("model.n_heads must be divisible by model.n_kv_groups")
    if type(m.qk_norm) is not bool:
        raise ValueError("model.qk_norm must be boolean")
    if type(m.rope_base) not in (int, float) or not math.isfinite(m.rope_base) or m.rope_base <= 0:
        raise ValueError("model.rope_base must be a positive finite number")
    if cfg.data.max_length is None:
        raise ValueError("data.max_length is required for Qwen3 (its context only sizes RoPE)")


def release_architecture(name):
    return {key: QWEN3_0_6B[key] for key in ARCHITECTURE_KEYS}


def release_state(name, path):
    pretrained.verify_qwen3_base(path)
    weights = Path(path) / pretrained.WEIGHTS
    state = torch.load(weights, map_location="cpu", weights_only=True, mmap=True)
    url = (
        f"https://huggingface.co/{pretrained.REPOSITORY}/resolve/"
        f"{pretrained.REVISION}/{pretrained.WEIGHTS}"
    )
    return state, {"url": url, "sha256": pretrained.FILES[pretrained.WEIGHTS]}


FAMILY = Family(
    name="qwen3",
    config=Qwen3ModelConfig,
    architecture_keys=ARCHITECTURE_KEYS,
    inherited_build_keys=("tie_embeddings", "attn_impl"),
    default_tokenizer_cache="data/qwen3-0.6b-base",
    build=Qwen3Model,
    load_weights=pretrained.load_qwen3_state,
    prepare_tokenizer=pretrained.prepare_qwen3_tokenizer,
    load_tokenizer=lambda path: (
        pretrained.load_qwen3_text_tokenizer(Path(path)),
        pretrained.TOKENIZER_MANIFEST,
    ),
    validate=validate,
    release_architecture=release_architecture,
    prepare_release=lambda name, path: pretrained.prepare_qwen3_base(path),
    release_state=release_state,
    resume_packages=("tokenizers",),
)
