"""The architecture explainer's data is current, adds up, and explains every step it shows."""

import json
import re

import pytest
import torch
from torch import nn

from llm_training.explain import spec as explain

# Documented in the README and the model guides.
DOCUMENTED = {
    "smoke": 3_242_240,
    "gpt-162m": 162_419_712,
    "gpt2-124m": 124_439_808,
    "tinystories-51m": 51_213_824,
    "qwen3-0.6b": 596_049_920,
}


@pytest.fixture(scope="module")
def spec():
    return explain.build_spec()


def walk(node, parent=None):
    yield node, parent
    for child in node["children"]:
        yield from walk(child, node)


def modules(tree):
    """Module calls with their enclosing module call."""
    for node, parent in walk(tree):
        if "module" in node and (parent is None or "module" in parent):
            yield node, parent


def preset(spec, name):
    (found,) = [p for p in spec["presets"] if p["id"] == name]
    return found


def steps(node, times=1):
    """Each operation with the number of times it runs, after unfolding repeated blocks."""
    times *= node.get("repeat", 1)
    if "op" in node:
        yield node, times
    for child in node["children"]:
        yield from steps(child, times)


def test_committed_spec_is_current(spec):
    committed = explain.SPEC.read_text()
    assert explain.render(spec) == committed, (
        f"docs/architecture/spec.js is stale: run `{explain.COMMAND}` and commit the result"
    )


@pytest.mark.parametrize("preset, total", DOCUMENTED.items())
def test_parameter_counts_match_the_docs(spec, preset, total):
    assert globals()["preset"](spec, preset)["total"] == total


def test_components_add_up_to_the_total(spec):
    """Tied parameters count once, at their first owner, so the parts sum to the whole."""
    for preset in spec["presets"]:
        tree = preset["tree"]
        assert tree["total"] == preset["total"]
        for node, _ in modules(tree):
            own = sum(p["count"] for p in node["params"] if "tied_to" not in p)
            inside = sum(c["total"] * c["repeat"] for c in node["children"])
            assert node["total"] == own + inside, (preset["id"], node["module"])


def test_repeated_blocks_fold_into_one(spec):
    for preset in spec["presets"]:
        layers = preset["config"]["n_layers"]
        assert [n["repeat"] for n, _ in walk(preset["tree"]) if n.get("repeat", 1) > 1] == [layers]


def test_every_class_function_op_and_role_is_explained(spec):
    """A new family's modules, functions and operations show up here until someone writes prose."""
    content = json.loads(explain.CONTENT.read_text())
    for kind in ("classes", "functions", "ops"):
        assert not [key for key, info in spec[kind].items() if "note" not in info], kind
        assert not set(content[kind]) - set(spec[kind]), f"{kind} notes for nothing that runs"
        for info in spec[kind].values():
            for target in info["note"].get("see", []):
                assert (explain.REPO / target.split("#")[0]).exists(), target

    roles = set()
    for preset in spec["presets"]:
        for node, parent in modules(preset["tree"]):
            if parent is not None:
                attr = node["module"].removeprefix(parent["module"]).lstrip(".")
                if attr := re.sub(r"(^|\.)\d+$", "", attr):
                    roles.add(f"{parent['class']}.{attr}")
    assert not set(spec["roles"]) - roles, "notes for modules that no longer exist"


def test_source_excerpts_keep_their_license_headers(spec):
    for key, info in [*spec["classes"].items(), *spec["functions"].items()]:
        if "path" in info and info["path"].startswith("src/llm_training/model/"):
            assert any("Copyright" in line for line in info["header"]), key


@pytest.mark.parametrize("name", ["smoke", "gpt2-124m", "qwen3-0.6b", "qwen3-0.6b-explicit"])
def test_folding_keeps_every_step(spec, name):
    """The page's tree, unfolded, runs exactly the operations of one unfolded forward pass."""
    entry = preset(spec, name)
    model, _, _ = explain.build_model(entry)
    traced = [n["op"] for n, _ in walk(explain.trace(model, 2, 5)) if "op" in n]
    shown = sum(times for _, times in steps(entry["tree"]))
    assert shown == len(traced)


def test_the_steps_that_are_not_modules_are_shown(spec):
    gpt = preset(spec, "gpt-162m")["tree"]
    block = next(n for n, _ in walk(gpt) if n.get("repeat", 1) > 1)
    residuals = [n for n in block["children"] if n.get("op") == "torch.Tensor.add"]
    assert [n["code"].split("#")[0].strip() for n in residuals] == ["x = x + shortcut"] * 2
    assert {"torch.Tensor.masked_fill_", "torch.softmax"} <= {n["op"] for n, _ in steps(block)}

    qwen = preset(spec, "qwen3-0.6b")["tree"]
    assert [b["name"] for b in qwen["buffers"]] == ["cos", "sin"]
    rope = [n for n, _ in walk(qwen) if n.get("function", "").endswith(".apply_rope")]
    assert [r["out"] for r in rope] == [["B", 16, "T", 128], ["B", 8, "T", 128]]  # queries, keys
    for call in rope:
        uses = [u for n, _ in steps(call) for u in n["uses"]]
        assert uses == ["cos", "sin"]
    gate = [n for n, _ in walk(qwen) if n.get("code", "").startswith("x = nn.functional.silu")]
    assert [n["op"] for n in gate] == ["torch.nn.functional.silu", "torch.Tensor.mul"]


def test_computation_outside_any_step_fails_the_trace(monkeypatch):
    """The dispatch check: a kernel that no recorded step or PyTorch module ran is an error."""

    class Hidden(nn.Module):
        def forward(self, x):
            with torch._C.DisableTorchFunction():
                return x + 1

    model = nn.Sequential(nn.Embedding(4, 3), Hidden())
    here = Hidden.forward.__code__.co_filename
    monkeypatch.setattr(explain, "_in_package", lambda filename: filename == here)
    with pytest.raises(ValueError, match="outside every recorded step"):
        explain.trace(model, 2, 5)
