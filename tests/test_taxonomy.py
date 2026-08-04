"""TAXONOMY.md, taxonomy.py and the check registry must agree.

Documentation that drifts from the code is worse than no documentation, because
a reader has no way to tell which one is lying.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hindsight.checks.base import registered_ids
from hindsight.models import Category
from hindsight.taxonomy import SPECS, TAXONOMY_VERSION, spec, static_check_ids

DOC = Path(__file__).resolve().parent.parent / "TAXONOMY.md"


@pytest.fixture(scope="module")
def document() -> str:
    """The spec, with inline code formatting stripped so names compare cleanly."""
    return DOC.read_text(encoding="utf-8").replace("`", "")


def test_document_pins_the_current_version(document: str) -> None:
    assert f"**Version {TAXONOMY_VERSION}**" in document


@pytest.mark.parametrize("check_id", sorted(SPECS))
def test_every_spec_is_documented(check_id: str, document: str) -> None:
    assert check_id in document, f"{check_id} is not mentioned in TAXONOMY.md"
    assert spec(check_id).name in document, f"{check_id}'s name is not in TAXONOMY.md"


def test_every_static_spec_has_an_implementation() -> None:
    assert set(static_check_ids()) == set(registered_ids())


def test_every_registered_check_has_a_spec() -> None:
    for check_id in registered_ids():
        assert spec(check_id).id == check_id


def test_every_category_has_at_least_one_check() -> None:
    covered = {spec(check_id).category for check_id in SPECS}
    assert covered == set(Category)


def test_specs_carry_a_mechanism_and_a_fix() -> None:
    for check_id, definition in SPECS.items():
        assert len(definition.mechanism) > 80, f"{check_id} needs a real mechanism explanation"
        assert len(definition.fix) > 40, f"{check_id} needs a real remedy"


def test_unknown_check_id_is_a_loud_error() -> None:
    with pytest.raises(KeyError):
        spec("ZZ999")
