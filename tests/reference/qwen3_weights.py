"""Map `llm_training` Qwen3 weights into Hugging Face `Qwen3ForCausalLM` names.

Test-only oracle: the inverse of the source's `load_hf_weights_into_qwen`. It lets an
independent implementation (Hugging Face Transformers) run the same weights, so matching logits
show that the ported modules compute Qwen3. Training code must never import this module.
"""

from typing import Any

# `Qwen/Qwen3-0.6B-Base` config.json at revision da87bfb608c14b7cf20ba1ce41287e8de496c0cd,
# restricted to the fields that shape the computation.
OFFICIAL_CONFIG: dict[str, Any] = dict(
    vocab_size=151936,
    hidden_size=1024,
    intermediate_size=3072,
    num_hidden_layers=28,
    num_attention_heads=16,
    num_key_value_heads=8,
    head_dim=128,
    rms_norm_eps=1e-06,
    rope_theta=1000000,
    attention_bias=False,
    hidden_act="silu",
    tie_word_embeddings=True,
    max_position_embeddings=32768,
    eos_token_id=151643,
)

_BLOCK = {
    "att.W_query.weight": "self_attn.q_proj.weight",
    "att.W_key.weight": "self_attn.k_proj.weight",
    "att.W_value.weight": "self_attn.v_proj.weight",
    "att.out_proj.weight": "self_attn.o_proj.weight",
    "att.q_norm.scale": "self_attn.q_norm.weight",
    "att.k_norm.scale": "self_attn.k_norm.weight",
    "ff.fc1.weight": "mlp.gate_proj.weight",
    "ff.fc2.weight": "mlp.up_proj.weight",
    "ff.fc3.weight": "mlp.down_proj.weight",
    "norm1.scale": "input_layernorm.weight",
    "norm2.scale": "post_attention_layernorm.weight",
}


def hf_config_kwargs(cfg):
    """Hugging Face `Qwen3Config` arguments for one of our model config dicts."""
    return dict(
        vocab_size=cfg["vocab_size"],
        hidden_size=cfg["emb_dim"],
        intermediate_size=cfg["hidden_dim"],
        num_hidden_layers=cfg["n_layers"],
        num_attention_heads=cfg["n_heads"],
        num_key_value_heads=cfg["n_kv_groups"],
        head_dim=cfg["head_dim"],
        rms_norm_eps=1e-06,
        rope_theta=cfg["rope_base"],
        attention_bias=False,
        hidden_act="silu",
        tie_word_embeddings=cfg.get("tie_embeddings", False),
    )


def to_hf_state(state):
    """Rename every tensor of our `state_dict` to its Hugging Face name; unknown keys fail."""
    renamed = {}
    for name, tensor in state.items():
        if name == "tok_emb.weight":
            hf = "model.embed_tokens.weight"
        elif name == "final_norm.scale":
            hf = "model.norm.weight"
        elif name == "out_head.weight":
            hf = "lm_head.weight"
        else:
            prefix, index, rest = name.split(".", 2)
            assert prefix == "trf_blocks", name
            hf = f"model.layers.{index}.{_BLOCK[rest]}"
        renamed[hf] = tensor
    return renamed
