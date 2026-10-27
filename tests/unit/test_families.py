"""Every registered family declares a complete `Family`, and nothing else is accepted."""

from dataclasses import fields

import pytest

from llm_training.lm.config import config_from_dict, load_config
from llm_training.model.registry import FAMILIES, PRETRAINED, family


@pytest.mark.parametrize("name", FAMILIES)
def test_family_declaration_is_consistent(name):
    declared = family(name)
    assert declared.name == name
    config_fields = {f.name for f in fields(declared.config)}
    assert declared.config().arch == name
    assert set(declared.architecture_keys) <= config_fields
    assert set(declared.inherited_build_keys) <= config_fields
    assert not set(declared.architecture_keys) & set(declared.inherited_build_keys)
    assert set(declared.release_defaults) <= config_fields
    for release in (r for r, owner in PRETRAINED.items() if owner == name):
        assert set(declared.release_architecture(release)) == set(declared.architecture_keys)


def test_every_release_names_a_registered_family():
    assert set(PRETRAINED.values()) <= set(FAMILIES)


def test_unknown_family_is_refused_rather_than_falling_through():
    raw = load_config("configs/smoke.yaml").to_dict()
    raw["model"]["arch"] = "llama3"
    with pytest.raises(ValueError, match="model.arch must be one of"):
        config_from_dict(raw, ".")
    with pytest.raises(ValueError, match="model.arch must be one of"):
        family("llama3")
