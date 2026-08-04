"""Command-line interface.

    hindsight audit <path> [--format md|json] [--fail-on <severity>] [--only <ids>]

Exit codes:
    0  audit completed; no finding at or above --fail-on (or --fail-on not given)
    1  audit completed; at least one finding at or above --fail-on
    2  the audit could not be run (bad path, unknown check id, provider failure)
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from . import checks as _checks  # noqa: F401  - registers checks for --list-checks
from .audit import audit
from .checks.base import registered_ids
from .llm.provider import ProviderError, build_provider
from .models import Severity
from .readonly import ReadOnlyViolation
from .report import json_report, markdown
from .taxonomy import TAXONOMY_VERSION, spec

EXIT_OK = 0
EXIT_FINDINGS = 1
EXIT_ERROR = 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hindsight",
        description=(
            "Audit a trading-research repository for lookahead, leakage and backtest bias. "
            "The target directory is only ever read."
        ),
    )
    parser.add_argument(
        "--version", action="version", version=f"hindsight (taxonomy {TAXONOMY_VERSION})"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    audit_parser = subparsers.add_parser("audit", help="audit a repository")
    audit_parser.add_argument("path", type=Path, help="path to the repository to audit")
    audit_parser.add_argument(
        "--format",
        choices=("md", "json"),
        default="md",
        help="report format: md for humans (default), json for CI",
    )
    audit_parser.add_argument(
        "--fail-on",
        choices=[s.value for s in Severity],
        default=None,
        help="exit 1 if any finding is at or above this severity (default: always exit 0)",
    )
    audit_parser.add_argument(
        "--only",
        default=None,
        help="comma-separated check ids to run, e.g. LA001,CV003 (default: all)",
    )
    audit_parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="write the report to this file instead of stdout",
    )
    audit_parser.add_argument(
        "--llm",
        action="store_true",
        help="enable the model reasoning pass (off by default; requires network access)",
    )
    audit_parser.add_argument(
        "--no-llm",
        action="store_true",
        help="explicitly disable the model pass; this is already the default",
    )
    audit_parser.add_argument(
        "--provider",
        choices=("anthropic", "mock"),
        default="anthropic",
        help="provider for the model pass (default: anthropic)",
    )

    checks_parser = subparsers.add_parser("checks", help="list the available checks")
    checks_parser.add_argument(
        "--format", choices=("text", "json"), default="text", help="output format"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "checks":
        return _list_checks(args.format)
    return _run_audit(args)


def _list_checks(fmt: str) -> int:
    specs = [spec(check_id) for check_id in registered_ids()]
    if fmt == "json":
        import json

        print(
            json.dumps(
                {
                    "taxonomy_version": TAXONOMY_VERSION,
                    "checks": [
                        {
                            "id": s.id,
                            "category": s.category.value,
                            "name": s.name,
                            "severity": s.severity.value,
                            "strategy": s.strategy,
                        }
                        for s in specs
                    ],
                },
                indent=2,
            )
        )
        return EXIT_OK

    width = max(len(s.id) for s in specs)
    print(f"hindsight static checks (taxonomy {TAXONOMY_VERSION})\n")
    for s in specs:
        print(f"  {s.id:<{width}}  {s.severity.value:<8}  {s.category.value:<13}  {s.name}")
    print(f"\n{len(specs)} static checks. See TAXONOMY.md for the model-pass checks.")
    return EXIT_OK


def _run_audit(args: argparse.Namespace) -> int:
    only = [part.strip().upper() for part in args.only.split(",")] if args.only else None

    provider = None
    if args.llm and not args.no_llm:
        try:
            provider = build_provider(args.provider)
        except ProviderError as exc:
            print(f"hindsight: {exc}", file=sys.stderr)
            return EXIT_ERROR

    try:
        result = audit(args.path, only=only, provider=provider)
    except (NotADirectoryError, FileNotFoundError, ValueError, ReadOnlyViolation) as exc:
        print(f"hindsight: {exc}", file=sys.stderr)
        return EXIT_ERROR

    report = markdown.render(result) if args.format == "md" else json_report.render(result) + "\n"

    if args.out:
        args.out.write_text(report, encoding="utf-8")
        print(f"wrote {args.out}", file=sys.stderr)
    else:
        sys.stdout.write(report)

    if args.fail_on:
        threshold = Severity(args.fail_on)
        worst = result.max_severity()
        if worst is not None and worst.rank >= threshold.rank:
            print(
                f"hindsight: {worst.value} finding(s) at or above --fail-on {threshold.value}",
                file=sys.stderr,
            )
            return EXIT_FINDINGS
    return EXIT_OK


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
