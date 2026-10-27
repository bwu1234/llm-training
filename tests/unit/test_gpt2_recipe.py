"""Unit checks for the GPT-2 recipe: init, tying, decay groups, schedule, clipping, attention."""

import math
import textwrap
from dataclasses import asdict, replace
from pathlib import Path

import pytest
import torch
import yaml

from llm_training.lm import trainer
from llm_training.lm.config import config_from_dict, load_config
from llm_training.lm.schedule import is_constant, learning_rate_at
from llm_training.model.gpt2.layers import GELU
from llm_training.model.gpt2.model import GPTModel
from llm_training.training.optim import build_optimizer, clip_and_measure
from llm_training.training.provenance import implementation_fingerprint
from tests.reference import baseline
from tests.reference.compare import TINY_CONFIG

GPT2_TINY = TINY_CONFIG | {
    "qkv_bias": True,
    "tie_embeddings": True,
    "init": "gpt2",
    "residual_scaled_init": True,
}


def training_config(**overrides):
    raw = yaml.safe_load(Path("configs/smoke.yaml").read_text())
    raw["training"] |= overrides
    return config_from_dict(raw, Path("configs").resolve()).training


# --- preset ------------------------------------------------------------------------------


def test_gpt2_preset_has_gpt2_small_parameter_count():
    cfg = load_config("configs/pretrain_gpt2_124m.yaml")
    with torch.device("meta"):  # shapes only: no 500 MB allocation
        model = GPTModel(asdict(cfg.model))
    assert sum(p.numel() for p in model.parameters()) == 124_439_808
    assert model.out_head.weight is model.tok_emb.weight


def test_book_preset_is_unchanged():
    cfg = load_config("configs/pretrain_gpt_162m.yaml")
    assert (cfg.model.init, cfg.model.tie_embeddings, cfg.model.attn_impl) == (
        "pytorch",
        False,
        "explicit",
    )
    t = cfg.training
    assert (t.beta1, t.beta2, t.weight_decay_policy, t.grad_accum_steps, t.grad_clip) == (
        0.9,
        0.999,
        "all",
        1,
        None,
    )
    assert is_constant(t)


# --- initialization ----------------------------------------------------------------------


def test_gpt2_init_statistics():
    cfg = TINY_CONFIG | {"vocab_size": 4096, "emb_dim": 256, "n_layers": 4, "init": "gpt2"}
    model = GPTModel(cfg)
    residual = {f"trf_blocks.{i}.{n}" for i in range(4) for n in ("att.out_proj", "ff.layers.2")}
    scaled = 0.02 / math.sqrt(2 * 4)
    for name, p in model.named_parameters():
        if name.endswith("bias") or name.endswith("shift"):
            assert p.abs().max() == 0, name
        elif name.endswith("scale"):
            assert (p == 1).all(), name
        else:
            expected = scaled if name.removesuffix(".weight") in residual else 0.02
            assert p.std().item() == pytest.approx(expected, rel=0.05), name


def test_residual_scaling_can_be_disabled():
    cfg = TINY_CONFIG | {"vocab_size": 4096, "emb_dim": 256, "n_layers": 4, "init": "gpt2"}
    model = GPTModel(cfg | {"residual_scaled_init": False})
    assert model.trf_blocks[0].att.out_proj.weight.std().item() == pytest.approx(0.02, rel=0.05)


def test_initial_loss_is_near_log_vocab():
    torch.manual_seed(0)
    cfg = TINY_CONFIG | {"vocab_size": 50257, "init": "gpt2", "tie_embeddings": True}
    model = GPTModel(cfg).eval()
    x = torch.randint(0, 50257, (4, 8))
    y = torch.randint(0, 50257, (4, 8))
    with torch.inference_mode():
        loss = torch.nn.functional.cross_entropy(model(x).flatten(0, 1), y.flatten()).item()
    assert math.log(50257) - 0.05 < loss < math.log(50257) + 0.2


def test_tied_head_requires_gpt2_init():
    raw = yaml.safe_load(Path("configs/smoke.yaml").read_text())
    raw["model"]["tie_embeddings"] = True
    with pytest.raises(ValueError, match="requires model.init"):
        config_from_dict(raw, Path("configs").resolve())


def test_tied_weights_survive_state_dict_round_trip():
    a, b = GPTModel(GPT2_TINY), GPTModel(GPT2_TINY)
    b.load_state_dict(a.state_dict())
    assert b.out_head.weight is b.tok_emb.weight
    torch.testing.assert_close(b.tok_emb.weight, a.tok_emb.weight, atol=0, rtol=0)


# --- optimizer groups --------------------------------------------------------------------


def test_matrices_policy_exempts_one_dimensional_tensors_and_lists_tied_once():
    model = GPTModel(GPT2_TINY)
    optimizer = build_optimizer(model, training_config(weight_decay_policy="matrices"))
    decay, no_decay = optimizer.param_groups
    assert decay["weight_decay"] == 0.1 and no_decay["weight_decay"] == 0.0
    assert all(p.dim() >= 2 for p in decay["params"])
    assert all(p.dim() < 2 for p in no_decay["params"])
    listed = [id(p) for group in optimizer.param_groups for p in group["params"]]
    assert len(listed) == len(set(listed)) == len(list(model.parameters()))


def test_all_policy_is_a_single_group_with_default_betas():
    model = GPTModel(TINY_CONFIG)
    optimizer = build_optimizer(model, training_config())
    assert len(optimizer.param_groups) == 1
    assert optimizer.param_groups[0]["betas"] == (0.9, 0.999)


def test_betas_are_applied():
    optimizer = build_optimizer(
        GPTModel(TINY_CONFIG), training_config(beta2=0.95, weight_decay_policy="matrices")
    )
    assert all(g["betas"] == (0.9, 0.95) for g in optimizer.param_groups)


# --- schedule ----------------------------------------------------------------------------


def test_cosine_schedule_table():
    t = training_config(learning_rate=1.0, min_lr=0.1, lr_schedule="cosine", warmup_steps=4)
    total = 14
    at = lambda step: learning_rate_at(t, step, total)  # noqa: E731
    assert at(0) == pytest.approx(1 / 5)  # nanoGPT: lr * (it + 1) / (warmup + 1)
    assert at(3) == pytest.approx(4 / 5)
    assert at(4) == pytest.approx(1.0)  # decay starts at the peak
    assert at(9) == pytest.approx(0.1 + 0.5 * (1 + math.cos(math.pi * 0.5)) * 0.9)
    assert at(total) == pytest.approx(0.1)
    assert at(total + 5) == 0.1
    rates = [at(s) for s in range(4, total + 1)]
    assert rates == sorted(rates, reverse=True)


def test_constant_schedule_is_a_noop_and_warmup_alone_is_not():
    assert is_constant(training_config())
    assert not is_constant(training_config(warmup_steps=2))
    t = training_config(learning_rate=0.5, warmup_steps=2)
    assert [learning_rate_at(t, s, 10) for s in (0, 1, 2, 9)] == [
        pytest.approx(0.5 / 3),
        pytest.approx(1.0 / 3),
        0.5,
        0.5,
    ]


@pytest.mark.parametrize(
    "overrides",
    [
        {"beta2": 1.0},
        {"beta1": -0.1},
        {"grad_clip": 0},
        {"grad_clip": float("nan")},
        {"grad_accum_steps": 0},
        {"warmup_steps": -1},
        {"lr_schedule": "linear"},
        {"weight_decay_policy": "none"},
        {"min_lr": 1.0},
    ],
)
def test_invalid_recipe_values_are_rejected(overrides):
    with pytest.raises(ValueError):
        training_config(**overrides)


# --- gradient clipping -------------------------------------------------------------------


def grads_set_to(model, value):
    for p in model.parameters():
        p.grad = torch.full_like(p, value)


def total_norm(model):
    return torch.linalg.vector_norm(torch.stack([p.grad.norm() for p in model.parameters()]))


def test_clipping_scales_to_max_norm_and_reports_the_pre_clip_norm():
    model = GPTModel(TINY_CONFIG)
    grads_set_to(model, 10.0)
    before = total_norm(model).item()
    reported = clip_and_measure(model, 1.0)
    assert reported.item() == pytest.approx(before, rel=1e-5)
    assert total_norm(model).item() == pytest.approx(1.0, rel=1e-4)


def test_no_clip_leaves_gradients_untouched():
    model = GPTModel(TINY_CONFIG)
    grads_set_to(model, 10.0)
    clip_and_measure(model, None)
    assert all((p.grad == 10.0).all() for p in model.parameters())


def test_small_gradients_are_not_clipped():
    model = GPTModel(TINY_CONFIG)
    grads_set_to(model, 1e-4)
    clip_and_measure(model, 1.0)
    assert all((p.grad == 1e-4).all() for p in model.parameters())


def test_nonfinite_gradient_is_reported_not_clipped():
    model = GPTModel(TINY_CONFIG)
    grads_set_to(model, 1.0)
    next(model.parameters()).grad.view(-1)[0] = float("nan")
    assert not torch.isfinite(clip_and_measure(model, 1.0))


# --- attention and GELU ------------------------------------------------------------------


@pytest.mark.parametrize("tokens", [1, 8, 64])
def test_sdpa_matches_explicit_attention_without_dropout(tokens):
    # Tolerance declared before measuring: FP32, eval mode (no dropout draws).
    torch.manual_seed(7)
    cfg = TINY_CONFIG | {"context_length": 64, "drop_rate": 0.0, "qkv_bias": True}
    explicit = GPTModel(cfg).eval()
    fused = GPTModel(cfg | {"attn_impl": "sdpa"}).eval()
    fused.load_state_dict(explicit.state_dict())
    x = torch.randint(0, 64, (3, tokens))
    with torch.inference_mode():
        torch.testing.assert_close(fused(x), explicit(x), atol=1e-5, rtol=1e-4)


def test_sdpa_is_causal():
    torch.manual_seed(1)
    model = GPTModel(TINY_CONFIG | {"attn_impl": "sdpa"}).eval()
    a = torch.randint(0, 64, (2, 8))
    b = a.clone()
    b[:, 4:] = (b[:, 4:] + 1) % 64
    with torch.inference_mode():
        torch.testing.assert_close(model(a)[:, :4], model(b)[:, :4], atol=0, rtol=0)


def test_gelu_matches_the_frozen_reference_exactly_and_adds_no_state():
    x = torch.randn(4, 33) * 3
    torch.testing.assert_close(GELU()(x), baseline.GELU()(x), atol=0, rtol=0)
    assert not GELU().state_dict()  # buffer is non-persistent: checkpoints are unchanged


# --- resume guard fingerprint ------------------------------------------------------------


def package_copy(tmp_path, source):
    package = tmp_path / "pkg"
    package.mkdir(exist_ok=True)
    (package / "mod.py").write_text(textwrap.dedent(source))
    return package


def test_fingerprint_ignores_comments_docstrings_and_formatting(tmp_path):
    package = package_copy(
        tmp_path,
        '''
        """Module docs."""
        def f(x):
            """Doc."""
            return x + 1
        ''',
    )
    before = implementation_fingerprint(package)
    package_copy(
        tmp_path,
        '''
        """Rewritten module docs."""
        # a new comment
        def f( x ):
            """Different doc."""
            return (x+1)
        ''',
    )
    assert implementation_fingerprint(package) == before


def test_fingerprint_changes_with_executable_code(tmp_path):
    package = package_copy(tmp_path, "def f(x):\n    return x + 1\n")
    before = implementation_fingerprint(package)
    package_copy(tmp_path, "def f(x):\n    return x + 2\n")
    assert implementation_fingerprint(package) != before


def test_fingerprint_ignores_cli_and_inspection_modules(tmp_path):
    package = package_copy(tmp_path, "X = 1\n")
    before = implementation_fingerprint(package)
    (package / "cli.py").write_text("print('changed')\n")
    (package / "prediction.py").write_text("print('changed')\n")
    assert implementation_fingerprint(package) == before


def test_scoped_fingerprint_ignores_code_outside_its_paths(tmp_path):
    package = package_copy(tmp_path, "X = 1\n")
    (package / "other").mkdir()
    (package / "other" / "mod.py").write_text("Y = 1\n")
    before = implementation_fingerprint(package, ["mod.py"])
    (package / "other" / "mod.py").write_text("Y = 2\n")
    assert implementation_fingerprint(package, ["mod.py"]) == before
    assert implementation_fingerprint(package, ["mod.py", "other"]) != before


def test_scoped_fingerprint_rejects_missing_paths(tmp_path):
    package = package_copy(tmp_path, "X = 1\n")
    with pytest.raises(ValueError, match="do not exist"):
        implementation_fingerprint(package, ["renamed"])


def test_lm_fingerprint_covers_only_the_run_family():
    package = Path(trainer.__file__).parent.parent
    gpt2 = trainer.lm_training_code(load_config("configs/smoke.yaml"))
    implementation_fingerprint(package, gpt2)  # every listed path exists
    assert "model/gpt2" in gpt2
    assert not any(entry.startswith(("model/qwen3", "reasoning")) for entry in gpt2)


def test_replace_helper_keeps_frozen_config_valid():
    # Guards the `replace` import used by scripts: configs stay frozen dataclasses.
    cfg = load_config("configs/smoke.yaml")
    assert replace(cfg.model, init="gpt2").init == "gpt2"
