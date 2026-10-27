"""What a model family provides to the next-token trainer and to inference.

Each family declares one `Family` in `model/<name>/family.py`; `model/registry.py` finds it
by name. Nothing outside a family's package branches on the family's name.
"""

from collections.abc import Callable
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Family:
    name: str
    # The recipe's `model` section: a frozen dataclass whose `arch` field defaults to `name`.
    config: type
    # Configuration keys that fix parameter shapes; the rest (dropout, attention kernel, tying)
    # may differ between the weights' source and the run that loads them.
    architecture_keys: tuple[str, ...]
    # Non-shape model fields a model.pt passes to the run that starts from it.
    inherited_build_keys: tuple[str, ...]
    # Where `prepare` puts the tokenizer when a recipe does not say otherwise.
    default_tokenizer_cache: str
    build: Callable  # (model-config mapping) -> nn.Module
    load_weights: Callable  # (model, state dict) -> model; strict
    prepare_tokenizer: Callable  # (path) -> None; the only tokenizer step that downloads
    load_tokenizer: Callable  # (path) -> (tokenizer, identity manifest), verified
    # (lm.config.Config) -> None; runs after the shared model checks and raises ValueError.
    validate: Callable
    # Pinned releases, named in `registry.PRETRAINED`; each takes the release name.
    release_architecture: Callable  # (name) -> {architecture key: value}
    prepare_release: Callable  # (name, path) -> None
    release_state: Callable  # (name, path) -> (state dict, {"url", "sha256"})
    # Recipe defaults a run starting from a release takes unless the recipe overrides them.
    release_defaults: dict = field(default_factory=dict)
    # Packages the family's code runs whose versions an exact resume must match, beyond the
    # environment every run records.
    resume_packages: tuple[str, ...] = ()


def positive_int(value, name):
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
