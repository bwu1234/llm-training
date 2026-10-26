"""L1 architecture gate: the ported Qwen3 modules compute Qwen3-0.6B, weight for weight.

Opt-in because it reads the ~1.5 GB pinned base weights (downloading them into
`data/qwen3-0.6b-base` unless `LLM_QWEN3_DIR` names a directory that already has them) and needs
the `oracle` dependency group:

    uv sync --group oracle
    LLM_ORACLE=1 uv run pytest tests/oracle/test_qwen3_oracle.py -q

Hugging Face's `Qwen3ForCausalLM`, built from the official config, runs the same weights as the
independent implementation. Tolerances were declared before the first run: FP32 logits at
rtol=1e-4, atol=1e-3 (28 layers, against GPT-2's 12, and larger logits), identical argmax at
every position, and mean cross-entropy within 1e-4. A narrow miss is reported as a finding, not
absorbed by loosening these.
"""

import os
from pathlib import Path

import pytest
import torch

from llm_training.model.qwen3.pretrained import (
    TOKENIZER,
    load_qwen3_base,
    prepare_qwen3_base,
    verify_qwen3_base,
)
from tests.reference.qwen3_weights import OFFICIAL_CONFIG, to_hf_state

pytestmark = pytest.mark.skipif(
    not os.environ.get("LLM_ORACLE"), reason="Set LLM_ORACLE=1 to load Qwen3-0.6B and run"
)

RTOL, ATOL = 1e-4, 1e-3
# Ids of real tokens; rows above 151,668 are unused padding of the embedding matrix.
TEXT_VOCAB = 151_643


@pytest.fixture(scope="module")
def asset_dir():
    if "LLM_QWEN3_DIR" in os.environ:
        path = Path(os.environ["LLM_QWEN3_DIR"])
        verify_qwen3_base(path)  # never writes into a directory we did not create
        return path
    path = Path(__file__).resolve().parents[2] / "data" / "qwen3-0.6b-base"
    prepare_qwen3_base(path)
    return path


@pytest.fixture(scope="module")
def ours(asset_dir):
    return load_qwen3_base(asset_dir)


@pytest.fixture(scope="module")
def hf_model(ours):
    # LLM_ORACLE is set here, so a missing dependency must fail loudly, not skip.
    import transformers

    config = transformers.Qwen3Config(**OFFICIAL_CONFIG)
    model = transformers.Qwen3ForCausalLM._from_config(
        config, attn_implementation="eager", dtype=torch.float32
    ).eval()
    # A bfloat16 RoPE table would be rounded before any upcast could recover it.
    assert model.model.rotary_emb.inv_freq.dtype == torch.float32
    model.load_state_dict(to_hf_state(ours[0].state_dict()), strict=True)
    assert model.lm_head.weight is model.model.embed_tokens.weight
    assert all(p.dtype == torch.float32 for p in model.parameters())
    return model


def token_batches():
    generator = torch.Generator().manual_seed(2024)
    return [
        torch.randint(0, TEXT_VOCAB, (2, 64), generator=generator),
        torch.randint(0, TEXT_VOCAB, (1, 1024), generator=generator),
    ]


@pytest.mark.parametrize("attn_impl", ["explicit", "sdpa"])
def test_logits_match_hugging_face_qwen3(ours, hf_model, attn_impl):
    model = ours[0]
    for block in model.trf_blocks:
        block.att.attn_impl = attn_impl
    assert sum(p.numel() for p in model.parameters()) == 596_049_920
    for tokens in token_batches():
        with torch.inference_mode():
            expected = hf_model(tokens).logits
            actual = model(tokens)
        diff = (actual - expected).abs().max().item()
        print(f"{attn_impl} {tuple(tokens.shape)}: max abs logit difference {diff:.3e}")
        torch.testing.assert_close(
            actual, expected, rtol=RTOL, atol=ATOL, msg=f"max abs logit difference {diff:.3e}"
        )
        assert torch.equal(actual.argmax(-1), expected.argmax(-1))

        def loss(logits):
            return torch.nn.functional.cross_entropy(
                logits[:, :-1].flatten(0, 1), tokens[:, 1:].flatten()
            )

        assert loss(actual).item() == pytest.approx(loss(expected).item(), abs=1e-4)


def test_tokenizer_matches_hugging_face_tokenizers(asset_dir, ours):
    from tokenizers import Tokenizer

    reference = Tokenizer.from_file(str(asset_dir / TOKENIZER))
    tokenizer = ours[1]
    assert tokenizer.eos_token == "<|endoftext|>"
    assert tokenizer.eos_token_id == OFFICIAL_CONFIG["eos_token_id"]
    for text in [
        "Every effort moves you",
        "  leading and trailing spaces  ",
        "line one\nline two\n\n\ttabbed",
        "naïve café — 東京 🚀",
        "def f(x):\n    return x**2  # code",
        "before<|endoftext|>after",
        "<|im_start|>user\nHi<|im_end|>\n",
    ]:
        ids = tokenizer.encode(text)
        assert ids == reference.encode(text).ids, text
        assert tokenizer.decode(ids) == text
