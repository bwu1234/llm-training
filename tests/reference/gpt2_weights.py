"""Map public GPT-2 (Hugging Face `GPT2LMHeadModel`) weights into `llm_training.GPTModel`.

Test-only oracle: it exists to show that the model *is* GPT-2, by checking logits against an
independent implementation loaded with OpenAI's trained weights. Training code must never
import this module, and the pretrained weights are never a starting point for a run.

Hugging Face stores projections as `Conv1D`, whose weight is (in, out); `nn.Linear` is
(out, in), so every projection is transposed. The fused `c_attn` holds Q, K, V side by side.
"""

import torch

GPT2_SMALL = dict(
    vocab_size=50257,
    context_length=1024,
    emb_dim=768,
    n_heads=12,
    n_layers=12,
    drop_rate=0.0,
    qkv_bias=True,
    tie_embeddings=True,
    init="gpt2",
)


def load_hf_gpt2_into(model, hf_state):
    """Copy a Hugging Face GPT-2 state dict into `model` in place; returns `model`."""
    d = model.tok_emb.weight.shape[1]

    def get(name):
        return hf_state[name].detach().clone()

    with torch.no_grad():
        model.tok_emb.weight.copy_(get("transformer.wte.weight"))
        model.pos_emb.weight.copy_(get("transformer.wpe.weight"))
        for i, block in enumerate(model.trf_blocks):
            h = f"transformer.h.{i}"
            q_w, k_w, v_w = get(f"{h}.attn.c_attn.weight").split(d, dim=1)
            q_b, k_b, v_b = get(f"{h}.attn.c_attn.bias").split(d, dim=0)
            for layer, w, b in (
                (block.att.W_query, q_w, q_b),
                (block.att.W_key, k_w, k_b),
                (block.att.W_value, v_w, v_b),
            ):
                layer.weight.copy_(w.T)
                layer.bias.copy_(b)
            block.att.out_proj.weight.copy_(get(f"{h}.attn.c_proj.weight").T)
            block.att.out_proj.bias.copy_(get(f"{h}.attn.c_proj.bias"))
            block.ff.layers[0].weight.copy_(get(f"{h}.mlp.c_fc.weight").T)
            block.ff.layers[0].bias.copy_(get(f"{h}.mlp.c_fc.bias"))
            block.ff.layers[2].weight.copy_(get(f"{h}.mlp.c_proj.weight").T)
            block.ff.layers[2].bias.copy_(get(f"{h}.mlp.c_proj.bias"))
            block.norm1.scale.copy_(get(f"{h}.ln_1.weight"))
            block.norm1.shift.copy_(get(f"{h}.ln_1.bias"))
            block.norm2.scale.copy_(get(f"{h}.ln_2.weight"))
            block.norm2.shift.copy_(get(f"{h}.ln_2.bias"))
        model.final_norm.scale.copy_(get("transformer.ln_f.weight"))
        model.final_norm.shift.copy_(get("transformer.ln_f.bias"))
    return model
