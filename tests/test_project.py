"""Cross-file resolution: a check can see what a file imports.

The property that matters is the one at the bottom of this file: the same code
that fires three cost findings when read one file at a time is silent once its
imports are resolved. That was the single largest source of false positives in
the static layer.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from conftest import context

from hindsight import audit
from hindsight.checks.base import run_all
from hindsight.collector import collect
from hindsight.context import COST_TERMS
from hindsight.project import ProjectIndex, index_module, link, module_name


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("backtest.py", "backtest"),
        ("pkg/mod.py", "pkg.mod"),
        ("pkg/__init__.py", "pkg"),
        ("a/b/c.py", "a.b.c"),
    ],
)
def test_module_names(path: str, expected: str) -> None:
    assert module_name(path) == expected


def test_index_records_top_level_definitions() -> None:
    index = index_module(
        context(
            """
            COST_BPS = 2.0

            def apply(gross, weight):
                return gross - weight * COST_BPS
            """,
            path="costs.py",
        )
    )
    assert index.name == "costs"
    assert set(index.definitions) == {"COST_BPS", "apply"}
    assert "COST_BPS" in index.definitions["apply"]


def _index(**files: str) -> tuple[ProjectIndex, dict]:
    contexts = [context(source, path=path) for path, source in files.items()]
    return ProjectIndex.build(contexts), {c.path: c for c in contexts}


def test_from_import_pulls_in_the_named_definition() -> None:
    project, contexts = _index(
        **{
            "costs.py": "COMMISSION_BPS = 1.0\n",
            "backtest.py": "from costs import COMMISSION_BPS\n",
        }
    )
    vocabulary = project.imported_vocabulary(contexts["backtest.py"])
    assert "COMMISSION_BPS" in vocabulary


def test_an_unimported_definition_stays_out_of_scope() -> None:
    """Importing one name from a module must not drag in the whole module."""
    project, contexts = _index(
        **{
            "config.py": "FEE_BPS = 0.0\nLEVERAGE = 3.0\n",
            "backtest.py": "from config import LEVERAGE\n",
        }
    )
    vocabulary = project.imported_vocabulary(contexts["backtest.py"])
    assert "LEVERAGE" in vocabulary
    assert "FEE_BPS" not in vocabulary
    assert not any(COST_TERMS.search(token) for token in vocabulary)


def test_a_function_contributes_its_body_vocabulary() -> None:
    project, contexts = _index(
        **{
            "execution.py": (
                "SPREAD_BPS = 2.0\n\n"
                "def net(gross, weight):\n"
                "    commission = weight * SPREAD_BPS\n"
                "    return gross - commission\n"
            ),
            "backtest.py": "from execution import net\n",
        }
    )
    vocabulary = project.imported_vocabulary(contexts["backtest.py"])
    assert "commission" in vocabulary
    assert any(COST_TERMS.search(token) for token in vocabulary)


def test_plain_import_pulls_in_the_module() -> None:
    project, contexts = _index(
        **{"costs.py": "COMMISSION_BPS = 1.0\n", "backtest.py": "import costs\n"}
    )
    assert "COMMISSION_BPS" in project.imported_vocabulary(contexts["backtest.py"])


def test_package_relative_imports_resolve() -> None:
    project, contexts = _index(
        **{
            "pkg/costs.py": "BORROW_BPS = 30.0\n",
            "pkg/backtest.py": "from .costs import BORROW_BPS\n",
        }
    )
    assert "BORROW_BPS" in project.imported_vocabulary(contexts["pkg/backtest.py"])


def test_star_imports_pull_in_everything() -> None:
    project, contexts = _index(
        **{"costs.py": "COMMISSION_BPS = 1.0\n", "backtest.py": "from costs import *\n"}
    )
    assert "COMMISSION_BPS" in project.imported_vocabulary(contexts["backtest.py"])


def test_third_party_imports_resolve_to_nothing_but_their_own_name() -> None:
    project, contexts = _index(**{"backtest.py": "import pandas as pd\nfrom numpy import sign\n"})
    vocabulary = project.imported_vocabulary(contexts["backtest.py"])
    assert vocabulary == {"pd", "sign"}


def test_the_bound_name_survives_an_unresolvable_module() -> None:
    """`from costs import COST_BPS` says there is a cost model even if costs.py
    is outside the audited tree."""
    project, contexts = _index(**{"backtest.py": "from vendor.costs import COST_BPS\n"})
    assert "COST_BPS" in project.imported_vocabulary(contexts["backtest.py"])


def test_link_attaches_vocabulary_to_every_context(fixtures_dir: Path) -> None:
    contexts, _ = collect(fixtures_dir / "cross_file")
    link(contexts)
    backtest = next(c for c in contexts if c.path == "backtest.py")
    assert any(COST_TERMS.search(token) for token in backtest.external_vocabulary)


# ------------------------------------------------------------ the whole point


def test_reading_one_file_at_a_time_produces_three_false_positives(fixtures_dir: Path) -> None:
    contexts, _ = collect(fixtures_dir / "cross_file")
    unlinked = [f.check_id for ctx in contexts for f in run_all(ctx)]
    assert sorted(unlinked) == ["CF001", "CF002", "CF004"]


def test_resolving_imports_silences_all_three(fixtures_dir: Path) -> None:
    assert audit(fixtures_dir / "cross_file").findings == []
