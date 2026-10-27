"""Pinned Qwen3-0.6B base weights and tokenizer: explicit acquisition, verified loading.

The weights are the `rasbt/qwen3-from-scratch` conversion of the official
`Qwen/Qwen3-0.6B-Base` checkpoint, already keyed by this package's parameter names, so they
load with a strict `load_state_dict` and need no name mapping. They are stored in bfloat16;
loading into the float32 model widens them exactly. The tokenizer file is byte-identical to
the official `Qwen/Qwen3-0.6B-Base` `tokenizer.json` (same git blob).

Only `prepare_qwen3_base` touches the network; loading reads and verifies local files.
"""

from pathlib import Path

import torch

from ..assets import download_pinned, verify_pinned
from .model import QWEN3_0_6B, Qwen3Model
from .tokenizer import Qwen3TextTokenizer, Qwen3Tokenizer

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
    download_pinned(f"https://huggingface.co/{REPOSITORY}/resolve/{REVISION}", FILES, path)


def verify_qwen3_base(path):
    verify_pinned(FILES, path, "prepare_qwen3_base")


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


def prepare_qwen3_tokenizer(path):
    """Download only the pinned tokenizer file into `path`."""
    download_pinned(
        f"https://huggingface.co/{REPOSITORY}/resolve/{REVISION}",
        {TOKENIZER: FILES[TOKENIZER]},
        path,
    )


def load_qwen3_text_tokenizer(path):
    """Return the verified pinned tokenizer behind the next-token trainer's interface."""
    verify_pinned({TOKENIZER: FILES[TOKENIZER]}, path, "`llm-training prepare`")
    return Qwen3TextTokenizer(Path(path) / TOKENIZER)


TOKENIZER_MANIFEST = {"encoding": "qwen3", "files": {TOKENIZER: FILES[TOKENIZER]}}
