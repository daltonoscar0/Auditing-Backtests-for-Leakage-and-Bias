"""Check protocol and registry.

Each static check is a class with an ``id`` matching a spec in
``hindsight.taxonomy`` and a ``run`` method that yields findings for one file.
Checks are stateless and see one file at a time; anything cross-file belongs in
the merge layer.
"""

from __future__ import annotations

import ast
from collections.abc import Iterable, Iterator
from typing import Any, ClassVar

from .. import astutils as A
from ..context import FileContext
from ..models import Confidence, Finding, Location, Severity, Source
from ..taxonomy import CheckSpec, spec


class Check:
    """Base class for a static detection rule."""

    id: ClassVar[str] = ""

    @property
    def spec(self) -> CheckSpec:
        return spec(self.id)

    def run(self, ctx: FileContext) -> Iterable[Finding]:  # pragma: no cover - interface
        raise NotImplementedError

    # ------------------------------------------------------------------ emit
    def finding(
        self,
        ctx: FileContext,
        node: ast.AST,
        *,
        confidence: Confidence = Confidence.HIGH,
        detail: str | None = None,
        fix: str | None = None,
        evidence: dict[str, Any] | None = None,
        severity: Severity | None = None,
        end_line: int | None = None,
    ) -> Finding:
        """Build a finding from this check's spec and a citation node.

        `detail` is appended to the spec's mechanism text and is where a check
        says what it actually saw in *this* file, as opposed to what the pattern
        means in general.
        """
        start, end = A.span(node)
        mechanism = self.spec.mechanism
        if detail:
            mechanism = f"{mechanism}\n\nHere: {detail}"
        return Finding(
            check_id=self.id,
            category=self.spec.category,
            title=self.spec.name,
            severity=severity or self.spec.severity,
            confidence=confidence,
            location=Location(ctx.path, start, end_line if end_line is not None else end),
            snippet=ctx.snippet(node),
            mechanism=mechanism,
            fix=fix,
            source=Source.STATIC,
            evidence=evidence or {},
        )


_REGISTRY: dict[str, Check] = {}


def register(cls: type[Check]) -> type[Check]:
    """Class decorator adding a check to the global registry."""
    if not cls.id:
        raise ValueError(f"{cls.__name__} must declare an id")
    if cls.id in _REGISTRY:
        raise ValueError(f"duplicate check id {cls.id}")
    spec(cls.id)  # fails loudly if the check has no taxonomy entry
    _REGISTRY[cls.id] = cls()
    return cls


def all_checks() -> list[Check]:
    return [_REGISTRY[key] for key in sorted(_REGISTRY)]


def get_check(check_id: str) -> Check:
    return _REGISTRY[check_id]


def registered_ids() -> list[str]:
    return sorted(_REGISTRY)


def run_all(ctx: FileContext, *, only: set[str] | None = None) -> Iterator[Finding]:
    for check in all_checks():
        if only is not None and check.id not in only:
            continue
        yield from check.run(ctx)
