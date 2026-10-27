"""Unit checks for GRPO on MATH: frozen-source parity of grading, sampling, and the GRPO
gradient; the time-limited grader; data pinning; evaluation resume; exact training resume."""

import copy
import json
from pathlib import Path

import pytest
import torch
import yaml

from llm_training.model.qwen3.model import Qwen3Model
from llm_training.reasoning import evaluate, grading, grpo, math_data, trainer
from llm_training.reasoning.config import grpo_config_from_dict, load_grpo_config
from tests.reference import reasoning_baseline
from tests.unit.test_qwen3 import TINY as TINY_QWEN3

# The book's prompt template alone is ~140 characters, so this tiny model needs more positions.
TINY = TINY_QWEN3 | {"context_length": 256}
EOS = 3


class CharTokenizer:
    """A deterministic stand-in for Qwen3Tokenizer over the tiny model's 101-token vocabulary."""

    eos_token_id = EOS

    def encode(self, text):
        return [ord(c) % int(TINY["vocab_size"]) for c in text]

    def decode(self, ids):
        return "".join(chr(ord("a") + i % 26) for i in ids)


def tiny_model(seed=0):
    torch.manual_seed(seed)
    return Qwen3Model(TINY)


def varying_reward(text, ground_truth, *grader):
    # Varies across rollouts (lengths often hit the cap, so not length parity), making the
    # advantages nonzero; the gradient test asserts that it does.
    return float(text[:1] < "n")


# --- grading -----------------------------------------------------------------------------

# The chapter 3 demo table: (prediction, ground truth, expected).
CHAPTER_CASES = [
    ("3/4", r"\frac{3}{4}", True),
    ("(3)/(4)", r"3/4", True),
    (r"\frac{\sqrt{8}}{2}", "sqrt(2)", True),
    (r"\( \frac{1}{2} + \frac{1}{6} \)", "2/3", True),
    ("(1, 2)", r"(1,2)", True),
    ("(2, 1)", "(1, 2)", False),
    ("(1, 2, 3)", "(1, 2)", False),
    ("0.5", "1/2", True),
    ("0.3333333333", "1/3", False),
    ("1,234/2", "617", True),
    (r"\text{2/3}", "2/3", True),
    ("50%", "1/2", False),
    (r"2\cdot 3/4", "3/2", True),
    (r"90^\circ", "90", True),
    (r"\left(\frac{3}{4}\right)", "3/4", True),
    (r"2²", "2**2", True),
]


@pytest.mark.parametrize(("pred", "truth", "expected"), CHAPTER_CASES)
def test_grading_matches_chapter_table_and_frozen_source(pred, truth, expected):
    assert grading.grade_answer(pred, truth) is expected
    assert reasoning_baseline.grade_answer(pred, truth) is expected


@pytest.mark.parametrize(
    "text",
    [
        r"so \boxed{\frac{1}{2}} and later \boxed{3}",
        r"\boxed{ {a}{b} }",
        r"\boxed{unbalanced",
        "The answer is 42 or maybe -3/4",
        "no digits at all",
        "",
    ],
)
@pytest.mark.parametrize("fallback", [None, "number_only", "number_then_full"])
def test_answer_extraction_matches_frozen_source(text, fallback):
    expected = reasoning_baseline.extract_final_candidate(text, fallback=fallback)
    assert grading.extract_final_candidate(text, fallback=fallback) == expected


def test_prompt_matches_frozen_source():
    assert grading.render_prompt("1+1?") == reasoning_baseline.render_prompt("1+1?")


def test_time_limited_grader_grades_times_out_and_recovers():
    with grading.TimeLimitedGrader(timeout=2) as grader:
        assert grader.grade("0.5", "1/2") == (True, None)
        # Upstream would compute this power inline for an unbounded time.
        assert grader.grade("9^9^9^9", "1") == (False, "timeout")
        assert grader.grade("(1, 2)", "(1,2)") == (True, None)


# --- sampling and the GRPO gradient ------------------------------------------------------


def test_top_p_filter_matches_frozen_source():
    probas = torch.softmax(torch.randn(3, 50, generator=torch.Generator().manual_seed(0)), -1)
    for top_p in (None, 0.0, 0.5, 0.9, 1.0):
        assert torch.equal(
            grpo.top_p_filter(probas, top_p), reasoning_baseline.top_p_filter(probas, top_p)
        )


def test_sampling_matches_frozen_source_on_cpu():
    model, tokenizer = tiny_model().eval(), CharTokenizer()
    kwargs = dict(prompt="Question: 2+2?", device="cpu", max_new_tokens=12, top_p=0.9)
    torch.manual_seed(7)
    ours = grpo.sample_response(model, tokenizer, **kwargs)
    torch.manual_seed(7)
    theirs = reasoning_baseline.sample_response(model, tokenizer, **kwargs)
    assert torch.equal(ours[0], theirs[0]) and ours[1:] == theirs[1:]


def test_explicit_generator_makes_sampling_independent_of_global_rng():
    model, tokenizer = tiny_model().eval(), CharTokenizer()

    def sample(global_seed):
        torch.manual_seed(global_seed)
        gen = torch.Generator().manual_seed(5)
        return grpo.sample_response(model, tokenizer, "Q?", "cpu", 12, generator=gen)[0]

    assert torch.equal(sample(1), sample(2))


def test_greedy_generation_matches_frozen_source():
    model, tokenizer = tiny_model(), CharTokenizer()
    prompt = "Question: 3*3?"
    ids = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0)
    expected = [
        t.item() for t in reasoning_baseline.generate_text_basic_stream_cache(model, ids, 20, EOS)
    ]
    assert grpo.generate_greedy_ids(model, tokenizer, prompt, "cpu", 20) == expected


@pytest.mark.parametrize("reward", [varying_reward, lambda *a: 1.0])
def test_grpo_gradient_matches_frozen_source(monkeypatch, reward):
    """Per-rollout backward (and the zero-advantage shortcut) gives upstream's gradient."""
    monkeypatch.setattr(reasoning_baseline, "reward_rlvr", reward)
    monkeypatch.setattr(
        grpo, "reward_rlvr", lambda text, truth, grader: (reward(text, truth), None)
    )
    example = {"problem": "What is 2+2?", "answer": "4"}
    settings = dict(num_rollouts=4, max_new_tokens=10, temperature=0.8, top_p=0.9)
    reference, ours = tiny_model(), tiny_model()

    torch.manual_seed(11)
    stats_ref = reasoning_baseline.compute_grpo_loss(
        reference, CharTokenizer(), example, "cpu", **settings
    )
    stats_ref["loss_tensor"].backward()
    torch.manual_seed(11)
    stats = grpo.grpo_step(ours, CharTokenizer(), example, None, "cpu", **settings)

    assert [s["text"] for s in stats["samples"]] == [s["text"] for s in stats_ref["samples"]]
    assert stats["rewards"] == stats_ref["rewards"]
    # The varying reward must really take the gradient path, not the zero-advantage shortcut.
    assert stats["zero_advantage"] is (reward is not varying_reward)
    assert stats["loss"] == pytest.approx(stats_ref["loss"], abs=1e-6)
    for (name, p_ref), p in zip(reference.named_parameters(), ours.parameters()):
        torch.testing.assert_close(p.grad, p_ref.grad, rtol=1e-5, atol=1e-7, msg=name)


def test_grpo_step_rejects_a_single_rollout():
    with pytest.raises(ValueError, match="at least two"):
        grpo.grpo_step(tiny_model(), CharTokenizer(), {"problem": "", "answer": ""}, None, "cpu", 1)


# --- data and evaluation -----------------------------------------------------------------


def test_math_data_rejects_unpinned_files(tmp_path):
    with pytest.raises(ValueError, match="missing"):
        math_data.load_math_data(tmp_path, "math500")
    (tmp_path / math_data.DATASETS["math500"]["file"]).write_text("[]")
    with pytest.raises(ValueError, match="differs from the pinned file"):
        math_data.load_math_data(tmp_path, "math500")


def test_evaluation_resumes_without_regenerating(monkeypatch, tmp_path):
    problems = [{"problem": f"p{i}", "answer": str(i)} for i in range(4)]
    calls = []

    def fake_generate(model, tokenizer, prompt, device, max_new_tokens):
        calls.append(prompt)
        return [0, 1, 2]  # decodes to "abc": no number, so graded against the full text

    monkeypatch.setattr(evaluate, "generate_greedy_ids", fake_generate)
    out = tmp_path / "eval.jsonl"
    with grading.TimeLimitedGrader(timeout=10) as grader:
        evaluate.evaluate_math500(None, CharTokenizer(), problems[:2], grader, "cpu", out, 8)
        summary = evaluate.evaluate_math500(None, CharTokenizer(), problems, grader, "cpu", out, 8)
        assert len(calls) == 4  # the rerun generated only the two new problems
        assert summary["problems"] == 4 and summary["correct"] == 0
        with pytest.raises(ValueError, match="max_new_tokens"):
            evaluate.evaluate_math500(None, CharTokenizer(), problems, grader, "cpu", out, 16)
        with pytest.raises(ValueError, match="different problem list"):
            evaluate.evaluate_math500(None, CharTokenizer(), problems[::-1], grader, "cpu", out, 8)


# --- config and training -----------------------------------------------------------------


def test_shipped_config_is_the_chapter_recipe():
    cfg = load_grpo_config("configs/grpo_qwen3_0_6b_math.yaml")
    t = cfg.training
    assert (t.num_rollouts, t.max_new_tokens, t.temperature, t.top_p) == (8, 512, 0.8, 0.9)
    assert (t.learning_rate, t.weight_decay, t.grad_clip, t.seed) == (1e-5, 0.01, 1.0, 42)
    assert not t.skip_zero_advantage_updates and not cfg.model.tie_embeddings


@pytest.mark.parametrize(
    ("section", "field", "value", "message"),
    [
        ("training", "num_rollouts", 1, "at least 2"),
        ("training", "top_p", 0, "outside"),
        ("training", "temperature", "hot", "finite number"),
        ("model", "tie_embeddings", "no", "boolean"),
        ("model", "attn_impl", "flash", "explicit or sdpa"),
        ("training", "kl_coef", 0.1, "Unknown fields"),
    ],
)
def test_config_rejects_invalid_values(section, field, value, message):
    raw = yaml.safe_load(Path("configs/grpo_qwen3_0_6b_math.yaml").read_text())
    raw[section][field] = value
    with pytest.raises(ValueError, match=message):
        grpo_config_from_dict(raw, Path("configs").resolve())


def grpo_test_config(tmp_path, **training):
    raw = yaml.safe_load(Path("configs/grpo_qwen3_0_6b_math.yaml").read_text())
    raw["training"] |= {
        "steps": 4,
        "num_rollouts": 3,
        "max_new_tokens": 6,
        "checkpoint_every": 2,
        "snapshot_every": 2,
    } | training
    raw["runtime"]["device"] = "cpu"
    raw["output"]["root"] = str(tmp_path / "runs")
    return grpo_config_from_dict(raw, Path("configs").resolve())


@pytest.fixture
def tiny_training(monkeypatch):
    problems = [{"problem": f"What is {i}+{i}?", "answer": str(2 * i)} for i in range(3)]
    monkeypatch.setattr(trainer, "load_math_data", lambda path, name: problems)

    def load_tiny(path, device, attn_impl, tie_embeddings):
        return Qwen3Model(TINY | {"tie_embeddings": tie_embeddings}).to(device), CharTokenizer()

    monkeypatch.setattr(trainer, "load_qwen3_base", load_tiny)
    monkeypatch.setattr(grpo, "reward_rlvr", lambda t, a, g: (varying_reward(t, a), None))


def final_weights(run):
    return torch.load(sorted((run / "snapshots").glob("*.pt"))[-1], weights_only=True)[
        "model_state_dict"
    ]


def test_training_resume_reproduces_an_uninterrupted_run(tmp_path, tiny_training):
    cfg = grpo_test_config(tmp_path)
    torch.manual_seed(0)  # identical initial tiny weights for both runs
    full = trainer.train_grpo(cfg)
    torch.manual_seed(0)
    first = trainer.train_grpo(cfg, stop_after=2)
    assert json.loads((first / "status.json").read_text())["status"] == "interrupted"
    resumed = trainer.train_grpo(resume=first / "checkpoints/last.pt")

    records = [json.loads(line) for line in (full / "metrics.jsonl").read_text().splitlines()]
    resumed_records = [
        json.loads(line) for line in (resumed / "metrics.jsonl").read_text().splitlines()
    ]
    assert [r["rewards"] for r in records] == [r["rewards"] for r in resumed_records]
    assert [r["problem_index"] for r in records] == [0, 1, 2, 0]
    for name, tensor in final_weights(full).items():
        assert torch.equal(tensor, final_weights(resumed)[name]), name
    assert sorted(p.name for p in (full / "snapshots").glob("*.pt")) == [
        "step00002.pt",
        "step00004.pt",
    ]
    provenance = json.loads((resumed / "provenance.json").read_text())
    assert provenance["parent_step"] == 2 and provenance["code_drift_allowed"] is False


def test_skip_zero_advantage_updates_leaves_weights_unchanged(tmp_path, monkeypatch, tiny_training):
    monkeypatch.setattr(grpo, "reward_rlvr", lambda t, a, g: (0.0, None))
    torch.manual_seed(0)
    initial = copy.deepcopy(Qwen3Model(TINY).state_dict())
    torch.manual_seed(0)
    run = trainer.train_grpo(grpo_test_config(tmp_path, skip_zero_advantage_updates=True))
    records = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines()]
    assert all(r["zero_advantage"] and not r["updated"] for r in records)
    for name, tensor in final_weights(run).items():
        assert torch.equal(tensor, initial[name]), name


def test_zero_advantage_steps_still_apply_weight_decay_by_default(
    tmp_path, monkeypatch, tiny_training
):
    monkeypatch.setattr(grpo, "reward_rlvr", lambda t, a, g: (0.0, None))
    torch.manual_seed(0)
    initial = copy.deepcopy(Qwen3Model(TINY).state_dict())
    torch.manual_seed(0)
    run = trainer.train_grpo(grpo_test_config(tmp_path, steps=2))
    weights = final_weights(run)
    # AdamW with a zero gradient only decays, w <- w * (1 - lr * wd), once per step. The
    # factor is within float32 rounding of 1, so compare bitwise against the same two ops.
    expected = initial["tok_emb.weight"].clone().mul_(1 - 1e-5 * 0.01).mul_(1 - 1e-5 * 0.01)
    assert not torch.equal(weights["tok_emb.weight"], initial["tok_emb.weight"])
    assert torch.equal(weights["tok_emb.weight"], expected)


def test_grpo_fingerprint_covers_qwen3_but_not_gpt2():
    package = Path(trainer.__file__).parent.parent
    trainer.implementation_fingerprint(package, trainer.GRPO_CODE)  # every listed path exists
    assert "model/qwen3" in trainer.GRPO_CODE
    assert not any(entry.startswith(("model/gpt2", "data")) for entry in trainer.GRPO_CODE)
