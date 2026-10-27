"""Each trainer's resume fingerprint covers every package module its trainer imports.

Follows module-level imports only. Imports inside functions are how one family's code stays
out of another family's runs (`model/registry.py`), so those paths are listed by hand in the
trainers and checked by their own tests.
"""

import ast
from pathlib import Path

import pytest

import llm_training
from llm_training.config import load_config
from llm_training.reasoning import trainer as grpo_trainer
from llm_training.training import trainer as lm_trainer
from llm_training.training.provenance import _NOT_TRAINING_CODE

PACKAGE = Path(llm_training.__file__).parent


def _module_file(base):
    for candidate in (base.with_suffix(".py"), base / "__init__.py"):
        if candidate.is_file():
            return candidate
    return None


def module_level_imports(path):
    """Package files imported by `path` at module level (not inside functions)."""
    found = set()
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.ImportFrom):
            if node.level:
                base = path.parent
                for _ in range(node.level - 1):
                    base = base.parent
            elif node.module and node.module.split(".")[0] == "llm_training":
                base = PACKAGE.parent
            else:
                continue
            target = base.joinpath(*node.module.split(".")) if node.module else base
            if module := _module_file(target):
                found.add(module)
            for alias in node.names:  # `from .model.qwen3 import pretrained`
                if module := _module_file(target / alias.name):
                    found.add(module)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] == "llm_training":
                    if module := _module_file(PACKAGE.parent.joinpath(*alias.name.split("."))):
                        found.add(module)
    return found


def closure(start):
    seen, stack = set(), [start]
    while stack:
        path = stack.pop()
        if path not in seen:
            seen.add(path)
            stack.extend(module_level_imports(path))
    return seen


def covered(code):
    roots = [PACKAGE / entry for entry in code]
    return {p for root in roots for p in ([root] if root.is_file() else root.rglob("*.py"))}


@pytest.mark.parametrize(
    "start, code",
    [
        (Path(grpo_trainer.__file__), grpo_trainer.GRPO_CODE),
        (
            Path(lm_trainer.__file__),
            lm_trainer.lm_training_code(load_config("configs/smoke.yaml")),
        ),
    ],
    ids=["grpo", "next-token"],
)
def test_fingerprint_covers_module_level_imports(start, code):
    missing = {
        str(path.relative_to(PACKAGE))
        for path in closure(start) - covered(code)
        if path.name not in _NOT_TRAINING_CODE
    }
    assert not missing, f"Imported but not fingerprinted: {sorted(missing)}"
