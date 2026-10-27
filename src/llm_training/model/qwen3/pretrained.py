"""Pinned Qwen3-0.6B base weights and tokenizer: explicit acquisition, verified loading.

The weights are the `rasbt/qwen3-from-scratch` conversion of the official
`Qwen/Qwen3-0.6B-Base` checkpoint, already keyed by this package's parameter names, so they
load with a strict `load_state_dict` and need no name mapping. They are stored in bfloat16;
loading into the float32 model widens them exactly. The tokenizer file is byte-identical to
the official `Qwen/Qwen3-0.6B-Base` `tokenizer.json` (same git blob).

Only `prepare_qwen3_base` touches the network; loading reads and verifies local files.
"""

import hashlib
import urllib.request
from pathlib import Path

import torch

from .model import QWEN3_0_6B, Qwen3Model
from .tokenizer import Qwen3Tokenizer

REPOSITORY = "rasbt/qwen3-from-scratch"
REVISION = "8a3862bde98adf1c10e551bc4a400bf836f11e78"
WEIGHTS = "qwen3-0.6B-base.pth"
TOKENIZER = "tokenizer-base.json"  # the source tokenizer picks EOS <|endoftext|> from "base"
FILES = {
    WEIGHTS: "ff38e5cc1530161ca5d68f6284f9343e58d637358b2bee9cc37e781b1c440443",
    TOKENIZER: "c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539",
}


def prepare_qwen3_base(path):
    """Download any missing or mismatched pinned file into `path` (~1.5 GB of weights)."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    for name, expected in FILES.items():
        target = path / name
        if target.is_file() and _sha256(target) == expected:
            continue
        url = f"https://huggingface.co/{REPOSITORY}/resolve/{REVISION}/{name}"
        partial = target.with_name(target.name + ".partial")
        digest = hashlib.sha256()
        # Stream in 1 MiB chunks, hashing as we write, so the 1.5 GB file never sits in memory.
        with urllib.request.urlopen(url, timeout=60) as response, partial.open("wb") as out:
            while chunk := response.read(1 << 20):
                digest.update(chunk)
                out.write(chunk)
        if digest.hexdigest() != expected:
            partial.unlink()
            raise ValueError(f"Downloaded {name} SHA256 mismatch")
        partial.replace(target)


def verify_qwen3_base(path):
    for name, expected in FILES.items():
        target = Path(path) / name
        if not target.is_file() or _sha256(target) != expected:
            raise ValueError(
                f"{target} is missing or differs from the pinned file; run prepare_qwen3_base"
            )


def load_qwen3_base(path, device="cpu", attn_impl="explicit", tie_embeddings=True):
    """Return (model, tokenizer): the float32 Qwen3-0.6B base model in eval mode.

    The pinned file stores the head and the embedding as equal tensors, so it loads either
    way. `tie_embeddings=False` builds them as separate parameters, as the source's
    `Qwen3Model` does, which lets fine-tuning move them apart.
    """
    path = Path(path)
    verify_qwen3_base(path)
    # mmap reads tensors from the file on demand instead of first copying all 1.5 GB to RAM.
    state = torch.load(path / WEIGHTS, map_location="cpu", weights_only=True, mmap=True)
    model = Qwen3Model(QWEN3_0_6B | {"attn_impl": attn_impl, "tie_embeddings": tie_embeddings})
    load_qwen3_state(model, state)
    return model.to(device).eval(), Qwen3Tokenizer(str(path / TOKENIZER))


def load_qwen3_state(model, state):
    """Strictly load `state` into `model`; refuses to silently drop an untied output head."""
    if model.out_head.weight is model.tok_emb.weight and not torch.equal(
        state["out_head.weight"], state["tok_emb.weight"]
    ):
        # Both keys load into the one shared tensor, so the head would overwrite the embedding.
        raise ValueError("State has an untied output head but the model ties it")
    model.load_state_dict(state)
    return model


def _sha256(path):
    with open(path, "rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()
