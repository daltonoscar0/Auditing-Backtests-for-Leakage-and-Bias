"""Audit orchestration: collect, run both layers, merge, return."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from . import checks as _checks  # noqa: F401  - importing registers every check
from .checks.base import registered_ids, run_all
from .collector import collect
from .llm.provider import Provider, Usage
from .llm.reasoning import DEFAULT_WORKERS, LLMPass
from .merge import merge
from .models import AuditResult, Category, Finding
from .project import link


def audit(
    root: Path,
    *,
    only: Sequence[str] | None = None,
    provider: Provider | None = None,
    llm_categories: Sequence[Category] | None = None,
    llm_workers: int = DEFAULT_WORKERS,
    llm_gate: bool = True,
) -> AuditResult:
    """Audit the repository at `root`.

    The tree is never written to. `only` restricts the static layer to a set of
    check ids; `provider` enables the model pass, which is otherwise skipped
    entirely - no import, no network, no credentials.
    """
    root = root.resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"{root} is not a directory")

    selected = set(only) if only else None
    if selected is not None:
        unknown = selected - set(registered_ids())
        if unknown:
            raise ValueError(f"unknown check id(s): {', '.join(sorted(unknown))}")

    contexts, errors = collect(root)
    # Resolve imports before any check runs, so every file's notion of what is
    # in scope includes what it pulls in from elsewhere in the repository.
    link(contexts)

    static: list[Finding] = []
    for ctx in contexts:
        static.extend(run_all(ctx, only=selected))

    llm_findings: list[Finding] = []
    usage: dict | None = None
    if provider is not None:
        pass_ = LLMPass(
            provider=provider,
            categories=tuple(llm_categories) if llm_categories else tuple(Category),
            workers=llm_workers,
            gate=llm_gate,
        )
        llm_findings = pass_.run(contexts)
        errors.extend(pass_.errors)
        errors.extend(f"discarded model finding: {r.reason}" for r in pass_.rejected)
        usage = getattr(provider, "usage", Usage()).to_dict()
        usage["skipped_requests"] = len(pass_.skipped)

    return AuditResult(
        root=str(root),
        findings=merge(static, llm_findings),
        files_scanned=len(contexts),
        checks_run=sorted(selected) if selected else registered_ids(),
        llm_enabled=provider is not None,
        errors=errors,
        llm_usage=usage,
    )
