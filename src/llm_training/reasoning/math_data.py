"""Pinned MATH problem sets: explicit acquisition, verified loading.

`train` is the 12,000 MATH problems that do not overlap MATH-500, the set *Build a Reasoning
Model (From Scratch)* trains GRPO on; `math500` is the MATH-500 test set its chapters evaluate
on. Both URLs name a commit, where upstream reads `main`, and both files are checked against
SHA-256 digests. Only `prepare_math_data` touches the network.
"""

import hashlib
import json
import urllib.request
from pathlib import Path

DATASETS = {
    "train": {
        "file": "math_full_minus_math500.json",
        "url": "https://raw.githubusercontent.com/rasbt/math_full_minus_math500/"
        "1636438eae81f1a7179de613f1c4aa52782b3a18/math_full_minus_math500.json",
        "sha256": "25677d1623060e59c23dde70ec570fe9f3f1cc9bdead787e1fa592018fb54a94",
        "count": 12000,
    },
    "math500": {
        "file": "math500_test.json",
        "url": "https://raw.githubusercontent.com/rasbt/reasoning-from-scratch/"
        "a788466dc85cfe8617b6c0b809ed4ab9084485de/ch03/01_main-chapter-code/math500_test.json",
        "sha256": "a9eccff7e25ecaf3952614e5b92fe081b4f0549c68ae6a387b96e131df9ddf41",
        "count": 500,
    },
}


def prepare_math_data(path):
    """Download any missing or mismatched pinned file into `path` (~11 MB)."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    for spec in DATASETS.values():
        target = path / spec["file"]
        if target.is_file() and _sha256(target.read_bytes()) == spec["sha256"]:
            continue
        with urllib.request.urlopen(spec["url"], timeout=60) as response:
            data = response.read()
        if _sha256(data) != spec["sha256"]:
            raise ValueError(f"Downloaded {spec['file']} SHA256 mismatch")
        partial = target.with_name(target.name + ".partial")
        partial.write_bytes(data)
        partial.replace(target)


def load_math_data(path, name):
    """Return the verified problem list: dicts with at least `problem` and `answer`."""
    spec = DATASETS[name]
    target = Path(path) / spec["file"]
    try:
        data = target.read_bytes()
    except OSError as exc:
        raise ValueError(f"{target} is missing; run `llm-training grpo-prepare` first") from exc
    if _sha256(data) != spec["sha256"]:
        raise ValueError(f"{target} differs from the pinned file; run `llm-training grpo-prepare`")
    problems = json.loads(data)
    if len(problems) != spec["count"]:
        raise ValueError(f"{target} has {len(problems)} problems, expected {spec['count']}")
    return problems


def dataset_identity():
    return {name: {"url": s["url"], "sha256": s["sha256"]} for name, s in DATASETS.items()}


def _sha256(data):
    return hashlib.sha256(data).hexdigest()
