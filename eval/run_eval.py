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
from hindsight.llm.cassette import (  # noqa: E402
    MODEL_GENERATED,
    Cassette,
    CassetteProvider,
)
from hindsight.models import Category, Finding  # noqa: E402
from hindsight.taxonomy import TAXONOMY_VERSION  # noqa: E402

FIXTURES_DIR = REPO_ROOT / "fixtures"
CLEAN_FIXTURE = "clean"
CASSETTE_NAME = "cassette.json"
DEFAULT_TOLERANCE = 3
RECALL_TARGET = 0.90


@dataclass(frozen=True)
class Expectation:
    check_id: str
    category: str
    file: str
    line: int
    cell: int | None = None
    """Notebook cell, when the fixture is a ``.ipynb``. `line` is then the line
    within that cell, and cell 4 line 2 must not match cell 6 line 2."""
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
        if finding.location.cell != self.cell:
            return False
        start, end = finding.location.span
        return start - tolerance <= self.line <= end + tolerance


@dataclass
class FixtureResult:
    name: str
    expectations: list[Expectation]
    findings: list[Finding]
    layer: str = "static"
    """`static` fixtures are scored by the AST checks alone. `llm` fixtures carry a
    cassette and are scored through the model pass."""
    provenance: str = ""
    """Cassette provenance, when one was used. `live` means recorded from the API;
    `hand-authored` means the run proves the pipeline works, not that the model
    can find the defect."""
    matched: list[tuple[Expectation, Finding]] = field(default_factory=list)
    missed: list[Expectation] = field(default_factory=list)
    unexpected: list[Finding] = field(default_factory=list)
    id_mismatches: list[tuple[Expectation, Finding]] = field(default_factory=list)
    anchor_mismatches: list[tuple[Expectation, Finding]] = field(default_factory=list)
    """Expectations with no match inside the line tolerance, but where the same
    check did fire elsewhere in the same file. A defect that spans several
    statements has more than one defensible citation, and 'found it, anchored it
    somewhere else' is a different result from 'did not find it'. These still
    count as misses in the recall figure - the anchor is part of what makes a
    finding actionable - but the distinction is reported rather than hidden."""

    @property
    def recall(self) -> float:
        if not self.expectations:
            return 1.0
        return len(self.matched) / len(self.expectations)


def load_expectations(path: Path) -> tuple[list[Expectation], str, str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    version = payload.get("taxonomy_version", "")
    if version and version != TAXONOMY_VERSION:
        raise SystemExit(
            f"{path.relative_to(REPO_ROOT)} pins taxonomy {version}, but the code is "
            f"{TAXONOMY_VERSION}. Re-verify the fixture before bumping it."
        )
    return (
        [Expectation(**item) for item in payload["expected"]],
        payload.get("description", ""),
        payload.get("layer", "static"),
    )


def score_fixture(name: str, tolerance: int) -> FixtureResult:
    directory = FIXTURES_DIR / name
    expectations, _, layer = load_expectations(directory / "expected_findings.json")

    cassette_path = directory / CASSETTE_NAME
    provider = None
    provenance = ""
    if cassette_path.is_file():
        provider = CassetteProvider.from_path(cassette_path)
        provenance = Cassette.load(cassette_path).provenance

    # Workers are pinned to 1 so the eval is reproducible and so a cassette miss
    # surfaces as one clear error rather than several interleaved ones.
    result = audit(directory, provider=provider, llm_workers=1)

    remaining = list(result.findings)
    outcome = FixtureResult(
        name=name,
        expectations=expectations,
        findings=result.findings,
        layer=layer,
        provenance=provenance,
    )

    for expectation in expectations:
        # Category is the scoring key, but when several findings of one category
        # land on one statement, category alone will happily consume the wrong
        # one and leave the right one looking unexpected. Prefer an exact check
        # before falling back.
        candidates = [f for f in remaining if expectation.matches(f, tolerance)]
        hit = next(
            (f for f in candidates if f.check_id == expectation.check_id),
            candidates[0] if candidates else None,
        )
        if hit is None:
            outcome.missed.append(expectation)
            elsewhere = next(
                (
                    f
                    for f in remaining
                    if f.location.path == expectation.file and f.check_id == expectation.check_id
                ),
                None,
            )
            if elsewhere is not None:
                outcome.anchor_mismatches.append((expectation, elsewhere))
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

    static = [r for r in results if r.layer == "static"]
    llm = [r for r in results if r.layer == "llm"]

    print("Layer 1 — static checks")
    print(f"  {'fixture':<16} {'found':>7} {'planted':>8} {'recall':>8}  {'extra':>5}")
    print(f"  {'-' * 16} {'-' * 7:>7} {'-' * 8:>8} {'-' * 8:>8}  {'-' * 5:>5}")
    for result in static:
        recall = "n/a" if not result.expectations else f"{result.recall:6.1%}"
        print(
            f"  {result.name:<16} {len(result.matched):>7} {len(result.expectations):>8} "
            f"{recall:>8}  {len(result.unexpected):>5}"
        )
    print()

    print("Per category (planted bugs across the static fixtures)")
    print(f"  {'category':<14} {'recall':>8}  {'hits':>9}")
    print(f"  {'-' * 14} {'-' * 8:>8}  {'-' * 9:>9}")
    for category, (found, total) in category_recall(static).items():
        if total == 0:
            print(f"  {category:<14} {'n/a':>8}  {'0/0':>9}")
            continue
        fraction = found / total
        print(f"  {category:<14} {fraction:>7.1%}  {f'{found}/{total}':>9}  {_bar(fraction)}")
    print()

    if llm:
        print("Layer 2 — model pass (replayed from cassette)")
        print(f"  {'fixture':<16} {'found':>7} {'planted':>8} {'recall':>8}  {'provenance':<14}")
        print(f"  {'-' * 16} {'-' * 7:>7} {'-' * 8:>8} {'-' * 8:>8}  {'-' * 14}")
        for result in llm:
            recall = "n/a" if not result.expectations else f"{result.recall:6.1%}"
            print(
                f"  {result.name:<16} {len(result.matched):>7} {len(result.expectations):>8} "
                f"{recall:>8}  {result.provenance:<14}"
            )
        print()
        if all(r.provenance == "live" for r in llm):
            pass
        elif all(r.provenance in MODEL_GENERATED for r in llm):
            print(
                "  NOTE: responses are model-generated but were not recorded through\n"
                "  AnthropicProvider, so the API wire format remains verified only by\n"
                "  the fake-client tests. Re-record through the API with: make record"
            )
        else:
            print(
                "  NOTE: no model recording. These rows prove the Layer 2 pipeline works\n"
                "  end to end and that the fixture is statically clean. They say nothing\n"
                "  about whether the model finds these defects. Re-record with: make record"
            )
        print()


def print_detail(results: list[FixtureResult]) -> None:
    for result in results:
        if not (result.missed or result.unexpected or result.id_mismatches):
            continue
        print(f"{result.name}:")
        anchored = {id(e) for e, _ in result.anchor_mismatches}
        for expectation in result.missed:
            if id(expectation) in anchored:
                continue
            print(
                f"  MISSED     {expectation.category:<14} {expectation.file}:{expectation.line}"
                f"  ({expectation.check_id}) {expectation.note}"
            )
        for expectation, finding in result.anchor_mismatches:
            print(
                f"  ANCHOR     {expectation.category:<14} {expectation.file}:"
                f"{expectation.line} expected, {finding.check_id} reported at "
                f"{finding.location} instead"
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

    static = [r for r in results if r.layer == "static" and r.expectations]
    total_expected = sum(len(r.expectations) for r in static)
    total_matched = sum(len(r.matched) for r in static)
    overall = total_matched / total_expected if total_expected else 1.0

    llm = [r for r in results if r.layer == "llm"]
    llm_expected = sum(len(r.expectations) for r in llm)
    llm_matched = sum(len(r.matched) for r in llm)
    llm_recall = llm_matched / llm_expected if llm_expected else 1.0
    llm_live = bool(llm) and all(r.provenance == "live" for r in llm)
    llm_model_generated = bool(llm) and all(r.provenance in MODEL_GENERATED for r in llm)
    llm_anchor = sum(len(r.anchor_mismatches) for r in llm)

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
                    "llm_pipeline_recall": llm_recall,
                    "llm_recording_is_live": llm_live,
                    "llm_recording_is_model_generated": llm_model_generated,
                    "llm_anchor_mismatches": llm_anchor,
                    "by_category": {
                        k: {"found": v[0], "expected": v[1]}
                        for k, v in category_recall(static).items()
                    },
                    "by_fixture": {
                        r.name: {
                            "layer": r.layer,
                            "provenance": r.provenance,
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
        if llm:
            print(
                f"Layer 2 recall         {llm_recall:.1%}  ({llm_matched}/{llm_expected})"
                + (f", {llm_anchor} anchored elsewhere" if llm_anchor else "")
                + f"  [{llm[0].provenance}]"
            )
        print()

    failures: list[str] = []
    if overall < args.target:
        failures.append(f"recall {overall:.1%} is below the {args.target:.0%} target")
    if false_positives:
        failures.append(f"{false_positives} finding(s) on the clean fixture")
    # A Layer 2 shortfall against a model-generated cassette is a measurement, not
    # a regression, so it is reported rather than failing the build. Against a
    # hand-authored one it means the pipeline dropped a recorded finding, which is.
    if llm_expected and llm_recall < 1.0 and not llm_model_generated:
        failures.append(
            f"the Layer 2 pipeline dropped {llm_expected - llm_matched} recorded finding(s)"
        )

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
