"""Run records shared by every training loop: atomic JSON, environment, git state, and the
implementation fingerprint that guards exact resume."""

import ast
import hashlib
import importlib.metadata
import json
import platform
import subprocess
from pathlib import Path

import numpy as np
import torch

# Files that cannot change training numerics (CLI plumbing, inspection tooling).
_NOT_TRAINING_CODE = {"__init__.py", "__main__.py", "cli.py", "prediction.py"}


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def collect_runtime_metadata():
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "torch": str(torch.__version__),
        "cuda_available": torch.cuda.is_available(),
        "mps_available": torch.backends.mps.is_available(),
        "numpy": np.__version__,
        "tiktoken": importlib.metadata.version("tiktoken"),
    }


def git_provenance():
    root = Path(__file__).resolve().parents[3]
    try:
        if not (root / "pyproject.toml").is_file():
            raise OSError("Installed wheel has no working-tree metadata")
        status = subprocess.check_output(["git", "status", "--short"], text=True, cwd=root)
        try:
            commit = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL, cwd=root
            ).strip()
        except subprocess.CalledProcessError:
            commit = None
        return {"commit": commit, "status": status}
    except OSError, subprocess.CalledProcessError:
        return {"commit": None, "note": "No working-tree metadata in installed wheel"}


def implementation_fingerprint(package: Path, code=None):
    """Hash behavior-relevant source: the normalized AST, so comments, docstrings, and
    formatting do not change it but any edit to executable code does.

    `code` lists the package-relative files and directories a training loop runs; `None`
    hashes the whole package. Scoping it keeps one model family's edits from marking
    another family's checkpoints as drifted.
    """
    roots = [package] if code is None else [package / entry for entry in code]
    if missing := [str(root) for root in roots if not root.exists()]:
        # A renamed module would otherwise silently drop out of the guard.
        raise ValueError(f"Fingerprinted code paths do not exist: {missing}")
    paths = {p for root in roots for p in ([root] if root.is_file() else root.rglob("*.py"))}
    digest = hashlib.sha256()
    for path in sorted(paths):
        if path.name in _NOT_TRAINING_CODE:
            continue
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
                body = node.body
                if (
                    body
                    and isinstance(body[0], ast.Expr)
                    and isinstance(body[0].value, ast.Constant)
                    and isinstance(body[0].value.value, str)
                ):
                    body[0] = ast.Pass()
        digest.update(str(path.relative_to(package)).encode())
        digest.update(ast.dump(tree).encode())
    return digest.hexdigest()
