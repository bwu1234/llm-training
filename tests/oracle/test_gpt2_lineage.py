"""Lineage gate: the converted GPT-2 small file that fine-tuning loads is OpenAI's release.

`rasbt/gpt2-from-scratch-pytorch` stores GPT-2 under this package's parameter names. This
test maps the official `openai-community/gpt2` `model.safetensors` into the model with the
oracle mapping and requires every tensor to be bit-identical. Opt-in: it reads ~550 MB from
`LLM_GPT2_HF_DIR` (a directory holding the official `model.safetensors`) and the pinned
converted file from `LLM_GPT2_DIR` (default `data/gpt2-small-124m`).

    LLM_ORACLE=1 LLM_GPT2_HF_DIR=~/.cache/huggingface/hub/models--openai-community--gpt2/snapshots/607a30d783dfa663caf39e06633721c8d4cfcd7e \
      uv run pytest tests/oracle/test_gpt2_lineage.py -q
"""

import json
import os
import struct
from pathlib import Path

import numpy as np
import pytest
import torch

from llm_training.model.gpt2.model import GPTModel
from llm_training.model.gpt2.pretrained import PRESETS, load_gpt2_state
from tests.reference.gpt2_weights import load_hf_gpt2_into

pytestmark = pytest.mark.skipif(
    not (os.environ.get("LLM_ORACLE") and os.environ.get("LLM_GPT2_HF_DIR")),
    reason="Set LLM_ORACLE=1 and LLM_GPT2_HF_DIR to compare GPT-2 small with OpenAI's release",
)


def read_safetensors(path):
    """Minimal float32 safetensors reader: an 8-byte header length, a JSON header, raw data."""
    with open(path, "rb") as file:
        (length,) = struct.unpack("<Q", file.read(8))
        header = json.loads(file.read(length))
        data = file.read()
    tensors = {}
    for name, spec in header.items():
        if name == "__metadata__":
            continue
        assert spec["dtype"] == "F32", (name, spec["dtype"])
        start, end = spec["data_offsets"]
        array = np.frombuffer(data[start:end], dtype="<f4").reshape(spec["shape"])
        # The oracle mapping expects GPT2LMHeadModel's "transformer." prefix.
        key = name if name.startswith("transformer.") else f"transformer.{name}"
        tensors[key] = torch.from_numpy(array.copy())
    return tensors


def test_converted_small_weights_equal_openai_release():
    official = read_safetensors(
        Path(os.environ["LLM_GPT2_HF_DIR"]).expanduser() / "model.safetensors"
    )
    # Tied, as OpenAI's model is: the mapping writes the embedding, which is also the head.
    model = GPTModel(PRESETS["gpt2-small-124m"]["config"] | {"tie_embeddings": True})
    load_hf_gpt2_into(model, official)
    converted = load_gpt2_state(
        "gpt2-small-124m", os.environ.get("LLM_GPT2_DIR", "data/gpt2-small-124m")
    )
    expected = model.state_dict()
    assert set(converted) == set(expected)
    for name, tensor in expected.items():
        assert torch.equal(converted[name], tensor), name
    assert torch.equal(converted["out_head.weight"], official["transformer.wte.weight"])
