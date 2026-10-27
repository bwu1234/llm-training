"""Recipes that start from released or saved weights: arch, init_from, windows, instructions."""

from pathlib import Path

import pytest
import torch
import yaml

from llm_training.lm.config import config_from_dict, window_length
from llm_training.model.gpt2.model import GPTModel
from llm_training.model.gpt2.pretrained import load_gpt2_into
from llm_training.model.registry import MODEL_FILE_FORMAT, build_model, read_model_file
from llm_training.training.checkpoint import atomic_save

BASE = Path(".").resolve()


def recipe(model=None, data=None, training=None):
    raw = yaml.safe_load(Path("configs/smoke.yaml").read_text())
    if model is not None:
        raw["model"] = model
    raw["data"] |= data or {}
    raw["training"] |= training or {}
    return raw


def qwen3_model(**overrides):
    return {
        "arch": "qwen3",
        "init_from": "qwen3-0.6b-base",
        "weights_dir": "data/qwen3-0.6b-base",
    } | overrides


def test_pinned_release_supplies_the_shape_and_chapter_7_dropout():
    model = {"init_from": "gpt2-small-124m", "weights_dir": "data/gpt2-small-124m"}
    cfg = config_from_dict(recipe(model), BASE)
    assert (cfg.model.emb_dim, cfg.model.n_layers, cfg.model.context_length) == (768, 12, 1024)
    assert cfg.model.qkv_bias and cfg.model.drop_rate == 0.0
    assert cfg.model.weights_dir == str(BASE / "data/gpt2-small-124m")
    # Tying a released model is allowed without GPT-2 init: the weights replace the init.
    config_from_dict(recipe(model | {"tie_embeddings": True}), BASE)
    with pytest.raises(ValueError, match="model.emb_dim is 768"):
        config_from_dict(recipe(model | {"emb_dim": 1024}), BASE)
    with pytest.raises(ValueError, match="weights_dir"):
        config_from_dict(recipe({"init_from": "gpt2-small-124m"}), BASE)


def test_qwen3_needs_released_weights_and_a_training_window():
    cfg = config_from_dict(recipe(qwen3_model(), {"max_length": 512}), BASE)
    assert cfg.model.context_length == 40_960 and window_length(cfg) == 512
    with pytest.raises(ValueError, match="from random weights is not built"):
        config_from_dict(recipe({"arch": "qwen3"}, {"max_length": 512}), BASE)
    with pytest.raises(ValueError, match="max_length is required"):
        config_from_dict(recipe(qwen3_model()), BASE)
    with pytest.raises(ValueError, match="is a gpt2 model"):
        config_from_dict(recipe(qwen3_model(init_from="gpt2-small-124m")), BASE)
    with pytest.raises(ValueError, match="must not exceed"):
        config_from_dict(recipe(None, {"max_length": 17}), BASE)  # smoke context is 16


def test_instruction_recipes_are_pinned_and_unaccumulated():
    data = {"format": "instructions", "dataset": "alpaca-1100", "path": "data/instructions"}
    cfg = config_from_dict(recipe(None, data), BASE)
    assert cfg.data.dataset == "alpaca-1100" and not cfg.data.mask_prompt
    with pytest.raises(ValueError, match="grad_accum_steps must be 1"):
        config_from_dict(recipe(None, data, {"grad_accum_steps": 2}), BASE)
    with pytest.raises(ValueError, match="data.dataset must be one of"):
        config_from_dict(recipe(None, data | {"dataset": "other"}), BASE)
    with pytest.raises(ValueError, match="apply only to instructions"):
        config_from_dict(recipe(None, {"mask_prompt": True}), BASE)


def test_model_file_supplies_the_shape_and_old_files_read_as_gpt2(tmp_path):
    config = {
        "vocab_size": 50257,
        "context_length": 16,
        "emb_dim": 32,
        "n_heads": 4,
        "n_layers": 2,
        "drop_rate": 0.25,
        "qkv_bias": False,
    }
    model = GPTModel(config)
    # A model.pt from before Qwen3 support: no family, no tokenizer identity.
    atomic_save(
        tmp_path / "model.pt",
        {"format_version": 1, "model_config": config, "model_state_dict": model.state_dict()},
    )
    assert read_model_file(tmp_path / "model.pt")["arch"] == "gpt2"
    cfg = config_from_dict(recipe({"init_from": str(tmp_path / "model.pt")}), BASE)
    assert (cfg.model.emb_dim, cfg.model.drop_rate) == (32, 0.25)
    assert cfg.model.init_from == str(tmp_path / "model.pt")
    with pytest.raises(ValueError, match="holds a gpt2 model"):
        config_from_dict(
            recipe({"arch": "qwen3", "init_from": str(tmp_path / "model.pt")}, {"max_length": 8}),
            BASE,
        )
    # A resumed recipe names every shape field, so the source file need not exist any more.
    (tmp_path / "model.pt").unlink()
    complete = config_from_dict(recipe(config | {"init_from": str(tmp_path / "model.pt")}), BASE)
    assert complete.model.emb_dim == 32
    assert MODEL_FILE_FORMAT == 1


def test_model_file_passes_on_tying_and_kernel_unless_the_recipe_overrides(tmp_path):
    config = {
        "vocab_size": 50257,
        "context_length": 16,
        "emb_dim": 32,
        "n_heads": 4,
        "n_layers": 2,
        "drop_rate": 0.0,
        "qkv_bias": True,
        "tie_embeddings": True,
        "init": "gpt2",
        "attn_impl": "sdpa",
    }
    atomic_save(
        tmp_path / "model.pt",
        {
            "format_version": 1,
            "arch": "gpt2",
            "model_config": config,
            "model_state_dict": GPTModel(config).state_dict(),
        },
    )
    source = {"init_from": str(tmp_path / "model.pt")}
    cfg = config_from_dict(recipe(source), BASE)
    assert (cfg.model.tie_embeddings, cfg.model.attn_impl, cfg.model.init) == (True, "sdpa", "gpt2")
    cfg = config_from_dict(recipe(source | {"tie_embeddings": False}), BASE)
    assert not cfg.model.tie_embeddings
    (tmp_path / "notes.txt").write_text("not a checkpoint")
    with pytest.raises(ValueError, match="not a project model.pt"):
        config_from_dict(recipe({"init_from": str(tmp_path / "notes.txt")}), BASE)


def test_a_tied_model_refuses_a_state_whose_head_differs():
    config = {
        "vocab_size": 50257,
        "context_length": 8,
        "emb_dim": 16,
        "n_heads": 2,
        "n_layers": 1,
        "drop_rate": 0.0,
        "qkv_bias": False,
    }
    state = GPTModel(config).state_dict()
    tied = build_model(config | {"tie_embeddings": True})
    with pytest.raises(ValueError, match="untied output head"):
        load_gpt2_into(tied, state)
    state["out_head.weight"] = state["tok_emb.weight"].clone()
    load_gpt2_into(tied, state)
    assert torch.equal(tied.out_head.weight, state["tok_emb.weight"])
