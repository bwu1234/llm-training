"""Unit checks for the Qwen3 port: frozen-source parity, tying, SDPA, KV cache, loading."""

import pytest
import torch

from llm_training.model.qwen3 import pretrained
from llm_training.model.qwen3.model import QWEN3_0_6B, KVCache, Qwen3Model
from tests.reference import qwen3_baseline
from tests.reference.qwen3_weights import OFFICIAL_CONFIG, hf_config_kwargs

# head_dim (24) differs from emb_dim / n_heads (16), as in the 0.6B model (128 vs 64).
TINY = dict(
    vocab_size=101,
    context_length=32,
    emb_dim=64,
    n_heads=4,
    n_layers=2,
    hidden_dim=96,
    head_dim=24,
    qk_norm=True,
    n_kv_groups=2,
    rope_base=1_000_000.0,
)


def tokens(*shape, seed=0):
    return torch.randint(
        0, int(TINY["vocab_size"]), shape, generator=torch.Generator().manual_seed(seed)
    )


def decode_with_cache(model, cache_cls, ids, prefill):
    """Logits from a prefill of `prefill` tokens, then one cached step per remaining token."""
    cache = cache_cls(n_layers=len(model.trf_blocks))
    model.reset_kv_cache()
    with torch.no_grad():
        steps = [model(ids[:, :prefill], cache=cache)]
        steps += [model(ids[:, t : t + 1], cache=cache) for t in range(prefill, ids.shape[1])]
    return torch.cat(steps, dim=1)


# --- frozen-source parity ----------------------------------------------------------------


def test_default_model_is_the_frozen_source_model():
    torch.manual_seed(123)
    reference = qwen3_baseline.Qwen3Model(TINY | {"dtype": torch.float32}).eval()
    torch.manual_seed(123)
    ours = Qwen3Model(TINY).eval()
    # Same construction order under the same seed: the initial weights are identical too.
    ref_state, our_state = reference.state_dict(), ours.state_dict()
    assert ref_state.keys() == our_state.keys()
    assert all(torch.equal(ref_state[k], our_state[k]) for k in ref_state)

    ids = tokens(2, 20)
    with torch.no_grad():
        assert torch.equal(ours(ids), reference(ids))
    assert torch.equal(
        decode_with_cache(ours, KVCache, ids, prefill=12),
        decode_with_cache(reference, qwen3_baseline.KVCache, ids, prefill=12),
    )


def test_preset_matches_official_config():
    assert hf_config_kwargs(QWEN3_0_6B) == {
        k: v
        for k, v in OFFICIAL_CONFIG.items()
        if k not in ("max_position_embeddings", "eos_token_id")
    }
    # Our RoPE table is longer than HF's 32,768 positions; entries depend only on position.
    assert QWEN3_0_6B["context_length"] >= OFFICIAL_CONFIG["max_position_embeddings"]
    source = dict(qwen3_baseline.QWEN_CONFIG_06_B)
    source.pop("dtype")
    assert {k: v for k, v in QWEN3_0_6B.items() if k != "tie_embeddings"} == source


def test_preset_has_qwen3_0_6b_parameter_count():
    with torch.device("meta"):  # shapes only: no 2.4 GB allocation
        model = Qwen3Model(QWEN3_0_6B)
    # Hugging Face reports 596.0M for Qwen/Qwen3-0.6B-Base; the tied head adds nothing.
    assert sum(p.numel() for p in model.parameters()) == 596_049_920


# --- options -----------------------------------------------------------------------------


def test_tied_head_shares_the_embedding_and_loads_tied_state():
    untied = Qwen3Model(TINY)
    tied = Qwen3Model(TINY | {"tie_embeddings": True})
    assert tied.out_head.weight is tied.tok_emb.weight
    vocab_matrix = TINY["vocab_size"] * TINY["emb_dim"]
    count = sum(p.numel() for p in untied.parameters()) - sum(p.numel() for p in tied.parameters())
    assert count == vocab_matrix

    state = untied.state_dict()
    with pytest.raises(ValueError, match="untied output head"):
        pretrained.load_qwen3_state(tied, state)
    state["out_head.weight"] = state["tok_emb.weight"].clone()
    pretrained.load_qwen3_state(tied, state)
    assert tied.out_head.weight is tied.tok_emb.weight
    assert torch.equal(tied.tok_emb.weight, untied.tok_emb.weight)


def test_loading_rejects_missing_and_unexpected_keys():
    model = Qwen3Model(TINY)
    state = model.state_dict()
    with pytest.raises(RuntimeError, match="Missing key"):
        pretrained.load_qwen3_state(
            model, {k: v for k, v in state.items() if k != "final_norm.scale"}
        )
    with pytest.raises(RuntimeError, match="Unexpected key"):
        pretrained.load_qwen3_state(model, state | {"pos_emb.weight": torch.zeros(1)})


def test_bfloat16_weights_widen_exactly_into_float32():
    source = Qwen3Model(TINY).to(torch.bfloat16)
    model = pretrained.load_qwen3_state(Qwen3Model(TINY), source.state_dict())
    assert all(p.dtype == torch.float32 for p in model.parameters())
    for name, tensor in source.state_dict().items():
        assert torch.equal(model.state_dict()[name], tensor.float())


def test_sdpa_matches_explicit_attention():
    torch.manual_seed(0)
    explicit = Qwen3Model(TINY).eval()
    sdpa = Qwen3Model(TINY | {"attn_impl": "sdpa"}).eval()
    sdpa.load_state_dict(explicit.state_dict())
    ids = tokens(2, 20, seed=1)
    with torch.no_grad():
        torch.testing.assert_close(sdpa(ids), explicit(ids), rtol=1e-5, atol=1e-5)
    # Cached steps exercise the non-square (explicit-mask) branch.
    torch.testing.assert_close(
        decode_with_cache(sdpa, KVCache, ids, prefill=7),
        decode_with_cache(explicit, KVCache, ids, prefill=7),
        rtol=1e-5,
        atol=1e-5,
    )


def test_kv_cache_matches_full_forward():
    model = Qwen3Model(TINY).eval()
    ids = tokens(2, 20, seed=2)
    with torch.no_grad():
        full = model(ids)
    torch.testing.assert_close(
        decode_with_cache(model, KVCache, ids, prefill=5), full, rtol=1e-5, atol=1e-5
    )


def test_input_longer_than_context_raises():
    model = Qwen3Model(TINY)
    with pytest.raises(ValueError, match="exceeds the model's context length"):
        model(tokens(1, TINY["context_length"] + 1))


# --- pinned assets -----------------------------------------------------------------------


def test_verify_rejects_missing_and_altered_files(tmp_path, monkeypatch):
    files = {"weights.pth": b"weights", "tokenizer.json": b"{}"}
    for name, data in files.items():
        (tmp_path / name).write_bytes(data)
    monkeypatch.setattr(pretrained, "FILES", {n: pretrained._sha256(tmp_path / n) for n in files})
    pretrained.verify_qwen3_base(tmp_path)
    (tmp_path / "tokenizer.json").write_bytes(b"[]")
    with pytest.raises(ValueError, match="differs from the pinned file"):
        pretrained.verify_qwen3_base(tmp_path)
    (tmp_path / "tokenizer.json").unlink()
    with pytest.raises(ValueError, match="missing"):
        pretrained.verify_qwen3_base(tmp_path)
