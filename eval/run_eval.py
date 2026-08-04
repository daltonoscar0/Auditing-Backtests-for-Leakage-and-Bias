#!/usr/bin/env python3
"""Score the auditor against the planted-bug fixtures.

Recall is per category, matched line-tolerantly (default +/-3 lines) because a
check may legitimately cite the statement rather than the exact sub-expression.
The clean fixture is scored separately and is pass/fail: any finding there is a
false positive, and false positives are what get an audit tool switched off.

    python eval/run_eval.py
    python eval/run_eval.py --tolerance 0 --json
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from hindsight import audit  # noqa: E402
from hindsight.models import Category, Finding  # noqa: E402
from hindsight.taxonomy import TAXONOMY_VERSION  # noqa: E402

FIXTURES_DIR = REPO_ROOT / "fixtures"
CLEAN_FIXTURE = "clean"
DEFAULT_TOLERANCE = 3
RECALL_TARGET = 0.90


@dataclass(frozen=True)
class Expectation:
    check_id: str
    category: str
    file: str
    line: int
    note: str = ""

    def matches(self, finding: Finding, tolerance: int) -> bool:
        """A finding satisfies an expectation if it is the same defect, near enough.

        Category rather than check id is the primary key: the expectation records
        what kind of mistake was planted, and a check id is an implementation
        detail that may be split or renamed without the fixture changing.
        """
        if finding.category.value != self.category:
            return False
        if finding.location.path != self.file:
            return False
        start, end = finding.location.span
        return start - tolerance <= self.line <= end + tolerance


@dataclass
class FixtureResult:
    name: str
    expectations: list[Expectation]
    findings: list[Finding]
    matched: list[tuple[Expectation, Finding]] = field(default_factory=list)
    missed: list[Expectation] = field(default_factory=list)
    unexpected: list[Finding] = field(default_factory=list)
    id_mismatches: list[tuple[Expectation, Finding]] = field(default_factory=list)

    @property
    def recall(self) -> float:
        if not self.expectations:
            return 1.0
        return len(self.matched) / len(self.expectations)


def load_expectations(path: Path) -> tuple[list[Expectation], str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    version = payload.get("taxonomy_version", "")
    if version and version != TAXONOMY_VERSION:
        raise SystemExit(
            f"{path.relative_to(REPO_ROOT)} pins taxonomy {version}, but the code is "
            f"{TAXONOMY_VERSION}. Re-verify the fixture before bumping it."
        )
    return [Expectation(**item) for item in payload["expected"]], payload.get("description", "")


def score_fixture(name: str, tolerance: int) -> FixtureResult:
    directory = FIXTURES_DIR / name
    expectations, _ = load_expectations(directory / "expected_findings.json")
    result = audit(directory)

    remaining = list(result.findings)
    outcome = FixtureResult(name=name, expectations=expectations, findings=result.findings)

    for expectation in expectations:
        hit = next((f for f in remaining if expectation.matches(f, tolerance)), None)
        if hit is None:
            outcome.missed.append(expectation)
            continue
        remaining.remove(hit)
        outcome.matched.append((expectation, hit))
        if hit.check_id != expectation.check_id:
            outcome.id_mismatches.append((expectation, hit))

    outcome.unexpected = remaining
    return outcome


def discover_fixtures() -> list[str]:
    return sorted(
        p.name for p in FIXTURES_DIR.iterdir() if (p / "expected_findings.json").is_file()
    )


def category_recall(results: list[FixtureResult]) -> dict[str, tuple[int, int]]:
    """(matched, expected) per category, across every fixture."""
    totals: dict[str, list[int]] = {c.value: [0, 0] for c in Category}
    for result in results:
        for expectation in result.expectations:
            totals[expectation.category][1] += 1
        for expectation, _ in result.matched:
            totals[expectation.category][0] += 1
    return {k: (v[0], v[1]) for k, v in totals.items()}


def _bar(fraction: float, width: int = 16) -> str:
    filled = round(fraction * width)
    return "#" * filled + "." * (width - filled)


def print_scoreboard(results: list[FixtureResult], tolerance: int) -> None:
    print(f"Hindsight eval — taxonomy {TAXONOMY_VERSION}, line tolerance +/-{tolerance}")
    print()

    print("Per fixture")
    print(f"  {'fixture':<16} {'found':>7} {'planted':>8} {'recall':>8}  {'extra':>5}")
    print(f"  {'-' * 16} {'-' * 7:>7} {'-' * 8:>8} {'-' * 8:>8}  {'-' * 5:>5}")
    for result in results:
        recall = "n/a" if not result.expectations else f"{result.recall:6.1%}"
        print(
            f"  {result.name:<16} {len(result.matched):>7} {len(result.expectations):>8} "
            f"{recall:>8}  {len(result.unexpected):>5}"
        )
    print()

    print("Per category (planted bugs across all fixtures)")
    print(f"  {'category':<14} {'recall':>8}  {'hits':>9}")
    print(f"  {'-' * 14} {'-' * 8:>8}  {'-' * 9:>9}")
    for category, (found, total) in category_recall(results).items():
        if total == 0:
            print(f"  {category:<14} {'n/a':>8}  {'0/0':>9}")
            continue
        fraction = found / total
        print(f"  {category:<14} {fraction:>7.1%}  {f'{found}/{total}':>9}  {_bar(fraction)}")
    print()


def print_detail(results: list[FixtureResult]) -> None:
    for result in results:
        if not (result.missed or result.unexpected or result.id_mismatches):
            continue
        print(f"{result.name}:")
        for expectation in result.missed:
            print(
                f"  MISSED     {expectation.category:<14} {expectation.file}:{expectation.line}"
                f"  ({expectation.check_id}) {expectation.note}"
            )
        for finding in result.unexpected:
            label = "FALSE POS " if result.name == CLEAN_FIXTURE else "UNEXPECTED"
            print(
                f"  {label} {finding.category.value:<14} {finding.location}"
                f"  ({finding.check_id}) {finding.title}"
            )
        for expectation, finding in result.id_mismatches:
            print(
                f"  CHECK-ID   {expectation.file}:{expectation.line} expected "
                f"{expectation.check_id}, got {finding.check_id}"
            )
        print()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Score Hindsight against its fixtures.")
    parser.add_argument("--tolerance", type=int, default=DEFAULT_TOLERANCE)
    parser.add_argument("--target", type=float, default=RECALL_TARGET)
    parser.add_argument("--json", action="store_true", help="emit machine-readable results")
    parser.add_argument("--fixture", action="append", help="score only these fixtures")
    args = parser.parse_args(argv)

    names = args.fixture or discover_fixtures()
    results = [score_fixture(name, args.tolerance) for name in names]

    planted = [r for r in results if r.expectations]
    total_expected = sum(len(r.expectations) for r in planted)
    total_matched = sum(len(r.matched) for r in planted)
    overall = total_matched / total_expected if total_expected else 1.0

    clean = next((r for r in results if r.name == CLEAN_FIXTURE), None)
    false_positives = len(clean.unexpected) if clean else 0

    if args.json:
        print(
            json.dumps(
                {
                    "taxonomy_version": TAXONOMY_VERSION,
                    "tolerance": args.tolerance,
                    "overall_recall": overall,
                    "false_positives_on_clean": false_positives,
                    "by_category": {
                        k: {"found": v[0], "expected": v[1]}
                        for k, v in category_recall(results).items()
                    },
                    "by_fixture": {
                        r.name: {
                            "expected": len(r.expectations),
                            "matched": len(r.matched),
                            "missed": len(r.missed),
                            "unexpected": len(r.unexpected),
                        }
                        for r in results
                    },
                },
                indent=2,
            )
        )
    else:
        print_scoreboard(results, args.tolerance)
        print_detail(results)
        print(f"Overall static recall  {overall:.1%}  ({total_matched}/{total_expected})")
        print(f"False positives on clean/  {false_positives}")
        print()

    failures: list[str] = []
    if overall < args.target:
        failures.append(f"recall {overall:.1%} is below the {args.target:.0%} target")
    if false_positives:
        failures.append(f"{false_positives} finding(s) on the clean fixture")

    if failures:
        if not args.json:
            for failure in failures:
                print(f"FAIL: {failure}")
        return 1
    if not args.json:
        print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
