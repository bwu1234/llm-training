"""The architecture explainer's data is current, adds up, and explains every module it shows."""

import re

import pytest

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


def test_committed_spec_is_current(spec):
    committed = explain.SPEC.read_text()
    assert explain.render(spec) == committed, (
        f"docs/architecture/spec.js is stale: run `{explain.COMMAND}` and commit the result"
    )


@pytest.mark.parametrize("preset, total", DOCUMENTED.items())
def test_parameter_counts_match_the_docs(spec, preset, total):
    (found,) = [p for p in spec["presets"] if p["id"] == preset]
    assert found["total"] == total


def test_components_add_up_to_the_total(spec):
    """Tied parameters count once, at their first owner, so the parts sum to the whole."""
    for preset in spec["presets"]:
        tree = preset["tree"]
        assert tree["total"] == preset["total"]
        for node, _ in walk(tree):
            own = sum(p["count"] for p in node["params"] if "tied_to" not in p)
            inside = sum(c["total"] * c["repeat"] for c in node["children"])
            assert node["total"] == own + inside, (preset["id"], node["module"])


def test_repeated_blocks_fold_into_one(spec):
    for preset in spec["presets"]:
        layers = preset["config"]["n_layers"]
        assert [n["repeat"] for n, _ in walk(preset["tree"]) if n.get("repeat", 1) > 1] == [layers]


def test_every_class_and_role_is_explained(spec):
    """A new family's modules show up here without prose until someone writes it."""
    classes = spec["classes"]
    assert not [key for key, info in classes.items() if "note" not in info]
    for info in classes.values():
        for target in info["note"].get("see", []):
            assert (explain.REPO / target.split("#")[0]).exists(), target

    roles = set()
    for preset in spec["presets"]:
        for node, parent in walk(preset["tree"]):
            if parent is not None:
                attr = node["module"].removeprefix(parent["module"]).lstrip(".")
                if attr := re.sub(r"(^|\.)\d+$", "", attr):
                    roles.add(f"{parent['class']}.{attr}")
    assert not set(spec["roles"]) - roles, "notes for modules that no longer exist"


def test_source_excerpts_keep_their_license_headers(spec):
    for key, info in spec["classes"].items():
        if "path" in info and info["path"].startswith("src/llm_training/model/"):
            assert any("Copyright" in line for line in info["header"]), key
