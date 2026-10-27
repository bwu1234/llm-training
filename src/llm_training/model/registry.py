"""The model families the next-token trainer builds, and the released weights it starts from.

A family is an architecture (`model/<family>/model.py`) plus its tokenizer, described to the
rest of the package by one `Family` (`model/family.py`) in `model/<family>/family.py`. Both
sit behind small shared interfaces: a model maps token ids to logits, and a tokenizer offers
the subset of tiktoken's `Encoding` the trainer uses (`encode`, `encode_ordinary_batch`,
`decode`, `decode_single_token_bytes`, `eot_token`, `n_vocab`).

`init_from` names either a pinned release below or a project `model.pt`, which carries its
family, configuration and tokenizer identity, so one stage's output starts the next.

Adding a family:
1. Write `model/<name>/` with its model, tokenizer, and `family.py` defining `FAMILY`.
2. Add `<name>` to `FAMILIES`, and its pinned releases, if any, to `PRETRAINED`.
3. Add a recipe under `configs/`. `tests/unit/test_families.py` checks the declaration and
   `tests/unit/test_fingerprint_scope.py` checks that resume fingerprints its code.
"""

import importlib
import pickle

import torch

FAMILIES = ("gpt2", "qwen3")
PRETRAINED = {"gpt2-small-124m": "gpt2", "gpt2-medium-355m": "gpt2", "qwen3-0.6b-base": "qwen3"}
# Recipes may omit `model.arch`, and files written before Qwen3 support carry none.
DEFAULT_FAMILY = "gpt2"
MODEL_FILE_FORMAT = 1


def family(name):
    """The named family's `Family`, importing only that family's modules.

    A run imports no other family's code, so the resume fingerprint can leave it out.
    """
    if name not in FAMILIES:
        raise ValueError(f"model.arch must be one of {sorted(FAMILIES)}")
    return importlib.import_module(f"{__package__}.{name}.family").FAMILY


def family_name(model_config):
    """The family a model-config mapping (a recipe's or a saved file's) belongs to."""
    return model_config.get("arch", DEFAULT_FAMILY)


def build_model(config):
    """Construct the family's model from a model-config mapping."""
    return family(family_name(config)).build(config)


def check_model_file(saved, path):
    """Validate a loaded project `model.pt` and fill in its family; return it."""
    if (
        not isinstance(saved, dict)
        or saved.get("format_version") != MODEL_FILE_FORMAT
        or not isinstance(saved.get("model_config"), dict)
        or "model_state_dict" not in saved
    ):
        raise ValueError(f"{path} is not a project model.pt")
    saved["arch"] = saved.get("arch") or family_name(saved["model_config"])
    return saved


def read_model_file(path):
    """Load a project `model.pt`: weights plus the family and config needed to rebuild them."""
    try:
        saved = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    except (RuntimeError, EOFError, pickle.UnpicklingError) as exc:
        raise ValueError(f"{path} is not a project model.pt: {exc}") from exc
    return check_model_file(saved, path)
