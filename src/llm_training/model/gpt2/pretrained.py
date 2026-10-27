"""Pinned OpenAI GPT-2 weights: explicit acquisition and verified loading.

The files are `rasbt/gpt2-from-scratch-pytorch`, the *LLMs-from-scratch* author's conversion
of OpenAI's released GPT-2 checkpoints into the book's `GPTModel` parameter names (the
alternative weight loading of chapter 5), so they load with a strict `load_state_dict` and
need no name mapping. They are float32. The file stores the output head as a separate copy
that equals the token embedding, as OpenAI's model ties them.

Only `prepare_gpt2` touches the network; loading reads and verifies local files.
"""

from pathlib import Path

import torch

from ..assets import download_pinned, verify_pinned

REPOSITORY = "rasbt/gpt2-from-scratch-pytorch"
REVISION = "402728fa68de0d3bb91e3e148ae351511eb7f4db"
BASE_URL = f"https://huggingface.co/{REPOSITORY}/resolve/{REVISION}"

# Shared by every released size; dropout is off, as chapter 7 fine-tunes with drop_rate 0.0.
_RELEASED = {"vocab_size": 50257, "context_length": 1024, "qkv_bias": True, "drop_rate": 0.0}
PRESETS = {
    "gpt2-small-124m": {
        "file": "gpt2-small-124M.pth",
        "sha256": "24a9078c5b27137fb2706d2206349c759952a7de8f72ef8e305dc02511bcabf8",
        "config": _RELEASED | {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
    },
    "gpt2-medium-355m": {
        "file": "gpt2-medium-355M.pth",
        "sha256": "9b6b658dc97b2f7a029990036638c7219619c6933eada7fb506c554561f73c60",
        "config": _RELEASED | {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    },
}


def _files(name):
    preset = PRESETS[name]
    return {preset["file"]: preset["sha256"]}


def prepare_gpt2(name, path):
    """Download the pinned weights for `name` into `path` unless already verified there."""
    download_pinned(BASE_URL, _files(name), path)


def load_gpt2_state(name, path):
    """Return the verified state dict, memory-mapped from disk rather than copied into RAM."""
    verify_pinned(_files(name), path, f"`llm-training prepare` for {name}")
    return torch.load(
        Path(path) / PRESETS[name]["file"], map_location="cpu", weights_only=True, mmap=True
    )


def load_gpt2_into(model, state):
    """Strictly load `state` into `model`; refuses to silently drop an untied output head."""
    if model.out_head.weight is model.tok_emb.weight and not torch.equal(
        state["out_head.weight"], state["tok_emb.weight"]
    ):
        # Both keys load into the one shared tensor, so the head would overwrite the embedding.
        raise ValueError("State has an untied output head but the model ties it")
    model.load_state_dict(state)
    return model
