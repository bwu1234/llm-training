"""The model families the next-token trainer builds, and the released weights it starts from.

A family is an architecture (`model/<family>/model.py`) plus its tokenizer. Both sit behind
small shared interfaces: a model maps token ids to logits, and a tokenizer offers the subset
of tiktoken's `Encoding` the trainer uses (`encode`, `encode_ordinary_batch`, `decode`,
`decode_single_token_bytes`, `eot_token`, `n_vocab`).

`init_from` names either a pinned release below or a project `model.pt`, which carries its
family, configuration and tokenizer identity, so one stage's output starts the next.
"""

import json
import pickle
from pathlib import Path

import torch

# Configuration keys that fix parameter shapes; the rest (dropout, attention kernel, tying)
# may differ between the weights' source and the run that loads them.
ARCHITECTURE_KEYS = {
    "gpt2": ("vocab_size", "context_length", "emb_dim", "n_heads", "n_layers", "qkv_bias"),
    "qwen3": (
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
    ),
}
PRETRAINED = {"gpt2-small-124m": "gpt2", "gpt2-medium-355m": "gpt2", "qwen3-0.6b-base": "qwen3"}
MODEL_FILE_FORMAT = 1


def pretrained_architecture(name):
    if PRETRAINED[name] == "gpt2":
        from .gpt2.pretrained import PRESETS

        config = PRESETS[name]["config"]
    else:
        from .qwen3.model import QWEN3_0_6B as config
    return {key: config[key] for key in ARCHITECTURE_KEYS[PRETRAINED[name]]}


def prepare_pretrained(name, path):
    if PRETRAINED[name] == "gpt2":
        from .gpt2.pretrained import prepare_gpt2

        prepare_gpt2(name, path)
    else:
        from .qwen3.pretrained import prepare_qwen3_base

        prepare_qwen3_base(path)


def pretrained_state(name, path):
    """Return the verified released state dict and a record of where it came from."""
    if PRETRAINED[name] == "gpt2":
        from .gpt2.pretrained import BASE_URL, PRESETS, load_gpt2_state

        state = load_gpt2_state(name, path)
        source = {"url": f"{BASE_URL}/{PRESETS[name]['file']}", "sha256": PRESETS[name]["sha256"]}
    else:
        from .qwen3.pretrained import FILES, REPOSITORY, REVISION, WEIGHTS, verify_qwen3_base

        verify_qwen3_base(path)
        state = torch.load(Path(path) / WEIGHTS, map_location="cpu", weights_only=True, mmap=True)
        source = {
            "url": f"https://huggingface.co/{REPOSITORY}/resolve/{REVISION}/{WEIGHTS}",
            "sha256": FILES[WEIGHTS],
        }
    return state, {"pretrained": name, **source}


def build_model(config):
    """Construct the family's model from a model-config mapping (`arch` defaults to gpt2)."""
    if config.get("arch", "gpt2") == "gpt2":
        from .gpt2.model import GPTModel

        return GPTModel(config)
    from .qwen3.model import Qwen3Model

    return Qwen3Model(config)


def load_weights(model, arch, state):
    """Strictly load `state`, refusing a tied model when the state's head differs."""
    if arch == "gpt2":
        from .gpt2.pretrained import load_gpt2_into

        return load_gpt2_into(model, state)
    from .qwen3.pretrained import load_qwen3_state

    return load_qwen3_state(model, state)


def read_model_file(path):
    """Load a project `model.pt`: weights plus the family and config needed to rebuild them."""
    try:
        saved = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    except (RuntimeError, EOFError, pickle.UnpicklingError) as exc:
        raise ValueError(f"{path} is not a project model.pt: {exc}") from exc
    if (
        not isinstance(saved, dict)
        or saved.get("format_version") != MODEL_FILE_FORMAT
        or not isinstance(saved.get("model_config"), dict)
        or "model_state_dict" not in saved
    ):
        raise ValueError(f"{path} is not a project model.pt")
    # Files written before Qwen3 support carry no family and are GPT-2.
    saved["arch"] = saved.get("arch") or saved["model_config"].get("arch", "gpt2")
    return saved


def prepare_tokenizer(arch, path):
    if arch == "gpt2":
        from .gpt2.tokenizer import prepare_tokenizer as prepare

        prepare(Path(path))
    else:
        from .qwen3.pretrained import prepare_qwen3_tokenizer

        prepare_qwen3_tokenizer(path)


def load_tokenizer(arch, path):
    """Return (tokenizer, identity manifest) for the family's verified local assets."""
    path = Path(path)
    if arch == "gpt2":
        from .gpt2.tokenizer import load_tokenizer as load

        tokenizer = load(path)
        return tokenizer, json.loads((path / "manifest.json").read_text())
    from .qwen3.pretrained import TOKENIZER_MANIFEST, load_qwen3_text_tokenizer

    return load_qwen3_text_tokenizer(path), TOKENIZER_MANIFEST
