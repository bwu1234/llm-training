"""The GPT-2 family as the next-token trainer sees it (see `model/family.py`)."""

import json
from dataclasses import dataclass
from pathlib import Path

from ..family import Family
from . import pretrained, tokenizer
from .model import GPTModel


@dataclass(frozen=True)
class GPT2ModelConfig:
    """The GPT-2 family (`model/gpt2`)."""

    arch: str = "gpt2"
    # A pinned release (model/registry.py PRETRAINED) or a project model.pt; null trains from
    # scratch. Weights supply the shape fields, which a recipe may then omit.
    init_from: str | None = None
    weights_dir: str | None = None  # where a pinned release's files live
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


ARCHITECTURE_KEYS = ("vocab_size", "context_length", "emb_dim", "n_heads", "n_layers", "qkv_bias")


def validate(cfg):
    m = cfg.model
    if m.vocab_size != 50257:
        raise ValueError("The GPT-2 tokenizer requires model.vocab_size=50257")
    if m.emb_dim % m.n_heads:
        raise ValueError("model.emb_dim must be divisible by model.n_heads")
    if type(m.drop_rate) not in (int, float) or not 0 <= m.drop_rate < 1:
        raise ValueError("model.drop_rate is outside its supported range")
    for name in ("qkv_bias", "residual_scaled_init"):
        if type(getattr(m, name)) is not bool:
            raise ValueError(f"model.{name} must be boolean")
    if m.init not in ("pytorch", "gpt2"):
        raise ValueError("model.init must be pytorch or gpt2")
    if m.tie_embeddings and m.init != "gpt2" and m.init_from is None:
        # Sharing a matrix between PyTorch's N(0,1) embedding and a Kaiming-scaled head
        # leaves one of the two badly scaled and the initial loss far above ln(vocab).
        raise ValueError("model.tie_embeddings requires model.init: gpt2")


def load_tokenizer(path):
    path = Path(path)
    return tokenizer.load_tokenizer(path), json.loads((path / "manifest.json").read_text())


def release_architecture(name):
    config = pretrained.PRESETS[name]["config"]
    return {key: config[key] for key in ARCHITECTURE_KEYS}


def release_state(name, path):
    preset = pretrained.PRESETS[name]
    source = {"url": f"{pretrained.BASE_URL}/{preset['file']}", "sha256": preset["sha256"]}
    return pretrained.load_gpt2_state(name, path), source


FAMILY = Family(
    name="gpt2",
    config=GPT2ModelConfig,
    architecture_keys=ARCHITECTURE_KEYS,
    inherited_build_keys=(
        "drop_rate",
        "tie_embeddings",
        "attn_impl",
        "init",
        "residual_scaled_init",
    ),
    default_tokenizer_cache="data/tokenizer",
    build=GPTModel,
    load_weights=pretrained.load_gpt2_into,
    prepare_tokenizer=lambda path: tokenizer.prepare_tokenizer(Path(path)),
    load_tokenizer=load_tokenizer,
    validate=validate,
    release_architecture=release_architecture,
    prepare_release=pretrained.prepare_gpt2,
    release_state=release_state,
    # Chapter 7 fine-tunes the releases with dropout off (see pretrained.py).
    release_defaults={"drop_rate": 0.0},
)
