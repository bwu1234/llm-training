"""docs/reference/model-config.md lists every model-config field, default and shape field."""

import re
from dataclasses import fields
from pathlib import Path

import pytest

from llm_training.model.registry import FAMILIES, family

PAGE = Path(__file__).resolve().parents[2] / "docs/reference/model-config.md"
SHARED_HEADING = "Fields every family has"


def sections():
    """Heading text -> the table rows under it, each row a list of cell strings."""
    found, current = {}, None
    for line in PAGE.read_text().splitlines():
        if line.startswith("## "):
            current = found.setdefault(line[3:].strip(), [])
        elif current is not None and line.startswith("| `"):
            current.append([cell.strip() for cell in line.strip().strip("|").split("|")])
    return found


def family_rows(name):
    matches = [rows for heading, rows in sections().items() if f"`arch: {name}`" in heading]
    assert len(matches) == 1, f"docs/reference/model-config.md needs one `arch: {name}` section"
    return {row[0].strip("`"): row for row in matches[0]}


def shared_rows():
    return {row[0].strip("`"): row for row in sections()[SHARED_HEADING]}


def shown(value):
    """A default as the page writes it: null, true, 1,024, 0.1, pytorch."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    return f"{value:,}" if isinstance(value, (int, float)) else str(value)


def defaults(name):
    return {f.name: f.default for f in fields(family(name).config)}


@pytest.mark.parametrize("name", FAMILIES)
def test_page_lists_exactly_the_families_fields(name):
    documented = set(family_rows(name)) | set(shared_rows())
    assert not set(family_rows(name)) & set(shared_rows())
    assert documented == set(defaults(name))


@pytest.mark.parametrize("name", FAMILIES)
def test_documented_defaults_match_the_dataclass(name):
    rows = family_rows(name) | shared_rows()
    for field, default in defaults(name).items():
        assert rows[field][1].strip("`") == shown(default), f"{name}: default of {field}"


@pytest.mark.parametrize("name", FAMILIES)
def test_fixed_by_weights_column_matches_architecture_keys(name):
    keys = set(family(name).architecture_keys)
    for field, row in family_rows(name).items():
        assert row[2] in ("yes", "no"), f"{name}: {field} needs yes or no"
        assert (row[2] == "yes") == (field in keys), f"{name}: {field} fixed-by-weights"


@pytest.mark.parametrize("name", FAMILIES)
def test_every_field_has_a_meaning(name):
    for field, row in (family_rows(name) | shared_rows()).items():
        assert len(re.sub(r"[`\s]", "", row[-1])) > 10, f"{name}: {field} has no meaning"
