"""The architecture explainer's data is current, adds up, and explains every step it shows."""

import json

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

    roles = {n["role"] for p in spec["presets"] for n, parent in modules(p["tree"]) if parent}
    assert roles <= set(spec["roles"]), f"modules without a role: {roles - set(spec['roles'])}"
    assert not set(spec["roles"]) - roles, "notes for modules that no longer exist"

    buffers = {
        f"{node['class']}.{b['name']}"
        for preset in spec["presets"]
        for node, _ in modules(preset["tree"])
        for b in node["buffers"]
    }
    assert buffers == set(spec["buffers"]), "every buffer needs a note, and every note a buffer"
    assert all(role["label"] for role in spec["roles"].values())


def test_every_step_is_labeled_and_every_label_is_used(spec):
    """Each operation names what it does; each labeled source line still runs."""
    labels = json.loads(explain.CONTENT.read_text())["steps"]
    used = set()

    def visit(node, container):
        if "op" in node:
            assert node.get("label"), (container, node["op"], node["code"])
            used.update((container, f) for f in labels.get(container, {}) if f in node["code"])
        inner = node["class"] if "module" in node else node.get("function", container)
        for child in node["children"]:
            visit(child, inner)

    for preset in spec["presets"]:
        visit(preset["tree"], None)
    fragments = {(container, f) for container, lines in labels.items() for f in lines}
    assert not fragments - used, "labels for source lines that no longer run"


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


def by_id(tree):
    return {n["id"]: n for n, _ in walk(tree)}


def test_every_input_has_a_traced_source(spec):
    """Each tensor a node reads came from a node shown on the page, a parameter or buffer, or
    the token IDs; folding points references at the nodes left standing."""
    for entry in spec["presets"]:
        nodes = by_id(entry["tree"])
        for node in nodes.values():
            for source in node["inputs"]:
                assert source.keys() & {"from", "tensor", "input"}, (entry["id"], node["id"])
                if "from" in source:
                    assert source["from"] in nodes, (entry["id"], source)
                    assert source["from"] < node["id"], "a node reads a later node"


def test_qwen3_passes_mask_and_rope_tables_beside_the_residual_stream(spec):
    """Nothing is added to Qwen3's token embedding: the mask and RoPE tables go to every block
    as separate arguments, and only attention reads them."""
    for name, mask_readers in [("qwen3-0.6b", []), ("qwen3-0.6b-explicit", ["masked_fill"])]:
        tree = preset(spec, name)["tree"]
        nodes = by_id(tree)
        block = next(n for n in tree["children"] if n.get("repeat", 1) > 1)
        source = {s["name"]: s for s in block["inputs"]}
        assert list(source) == ["x", "mask", "cos", "sin"]
        assert source["x"].get("first_copy") and not source["mask"].get("first_copy")
        assert nodes[source["x"]["from"]]["module"] == "tok_emb"
        assert nodes[source["mask"]["from"]]["label"] == "Causal mask"
        assert (source["cos"]["tensor"], source["sin"]["tensor"]) == ("cos", "sin")

        mask = source["mask"]["from"]
        readers = [
            n["op"].rsplit(".", 1)[1]
            for n in nodes.values()
            if "op" in n and any(s.get("from") == mask for s in n["inputs"])
        ]
        assert readers == mask_readers  # with SDPA, `is_causal=True` replaces it

        first_residual = next(
            n for n in block["children"] if n.get("label") == "Residual connection"
        )
        assert [nodes[s["from"]].get("module") for s in first_residual["inputs"]] == [
            "trf_blocks.0.att",
            "tok_emb",
        ]


def test_gpt2_adds_positions_to_the_token_embedding(spec):
    tree = preset(spec, "gpt-162m")["tree"]
    nodes = by_id(tree)
    (add,) = [n for n in tree["children"] if n.get("label") == "Add position to token embeddings"]
    assert [nodes[s["from"]]["module"] for s in add["inputs"]] == ["tok_emb", "pos_emb"]


def test_operations_that_do_one_thing_are_one_row(spec):
    """Consecutive operations with one label become a group; its inputs are what it reads from
    outside, and a step labeled on its own stays a row of its own."""
    tree = preset(spec, "gpt2-124m")["tree"]
    nodes = by_id(tree)
    norm = next(n for n, _ in walk(tree) if n.get("module") == "final_norm")
    assert [(c.get("op", "").rsplit(".", 1)[-1], c["label"]) for c in norm["children"]] == [
        ("mean", "Mean over channels"),
        ("var", "Variance over channels"),
        ("", "Normalize"),
        ("", "Learned scale and shift"),
    ]
    normalize = norm["children"][2]
    assert [c["op"].rsplit(".", 1)[1] for c in normalize["children"]] == [
        "sub",
        "add",
        "sqrt",
        "div",
    ]
    assert [nodes[s["from"]].get("op") for s in normalize["inputs"]] == [
        None,  # x, the module's input
        "torch.Tensor.mean",
        "torch.Tensor.var",
    ]
    assert normalize["out"] == normalize["children"][-1]["out"] == ["B", "T", 768]

    qwen = preset(spec, "qwen3-0.6b")["tree"]
    (mask,) = [n for n in qwen["children"] if n.get("group")]
    assert mask["label"] == "Causal mask" and mask["inputs"] == []
    assert [c["op"].rsplit(".", 1)[1] for c in mask["children"]] == ["ones", "triu", "__getitem__"]
    residuals = [n for n, _ in walk(qwen) if n.get("label") == "Residual connection"]
    assert residuals and all("op" in n for n in residuals)
