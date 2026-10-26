"""L1 architecture gate: this model reproduces OpenAI's GPT-2 small, weight for weight.

Opt-in because it downloads the ~550 MB public `openai-community/gpt2` weights and needs the
`oracle` dependency group:

    uv sync --group oracle
    LLM_ORACLE=1 uv run pytest tests/oracle -q

Tolerances were declared before the first run: FP32 logits at rtol=1e-4, atol=1e-4, identical
argmax at every position, and mean cross-entropy within 1e-4. A narrow miss is reported as a
finding, not absorbed by loosening these.
"""

import os

import pytest
import torch

from llm_training.model.gpt import GPTModel
from tests.reference.gpt2_weights import GPT2_SMALL, load_hf_gpt2_into

pytestmark = pytest.mark.skipif(
    not os.environ.get("LLM_ORACLE"), reason="Set LLM_ORACLE=1 to download GPT-2 and run"
)

RTOL, ATOL = 1e-4, 1e-4


@pytest.fixture(scope="module")
def hf_model():
    # LLM_ORACLE is set here, so a missing dependency must fail loudly, not skip.
    import transformers

    model = transformers.GPT2LMHeadModel.from_pretrained(
        "openai-community/gpt2", attn_implementation="eager"
    ).eval()
    return model.to(torch.float32)


def token_batches():
    generator = torch.Generator().manual_seed(2024)
    return [
        torch.randint(0, 50257, (2, 64), generator=generator),
        torch.randint(0, 50257, (1, 1024), generator=generator),  # the full context window
    ]


@pytest.mark.parametrize("attn_impl", ["explicit", "sdpa"])
def test_logits_match_hugging_face_gpt2(hf_model, attn_impl):
    ours = GPTModel(GPT2_SMALL | {"attn_impl": attn_impl}).eval()
    load_hf_gpt2_into(ours, hf_model.state_dict())
    assert sum(p.numel() for p in ours.parameters()) == 124_439_808
    for tokens in token_batches():
        with torch.inference_mode():
            expected = hf_model(tokens).logits
            actual = ours(tokens)
        diff = (actual - expected).abs().max().item()
        torch.testing.assert_close(
            actual, expected, rtol=RTOL, atol=ATOL, msg=f"max abs logit difference {diff:.3e}"
        )
        assert torch.equal(actual.argmax(-1), expected.argmax(-1))

        def loss(logits):
            return torch.nn.functional.cross_entropy(
                logits[:, :-1].flatten(0, 1), tokens[:, 1:].flatten()
            )

        assert loss(actual).item() == pytest.approx(loss(expected).item(), abs=1e-4)
