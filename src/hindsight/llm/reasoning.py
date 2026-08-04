"""The Layer 2 model pass: structured findings, validated against the source.

The validation in :func:`validate` is the load-bearing part. A model asked to
audit code will occasionally produce a finding that reads well and points at
nothing. Every such finding is dropped here rather than shown to a reader, on
the principle that an audit tool's credibility is spent one false positive at a
time.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from ..context import FileContext
from ..models import Category, Confidence, Finding, Location, Severity, Source
from ..taxonomy import SPECS
from .prompts import SYSTEM, build_user_prompt
from .provider import Provider, ProviderError
from .schema import FINDINGS_SCHEMA

DEFAULT_WORKERS = 4

# A category is worth asking about only if the file contains something it could
# possibly apply to. Six requests per file is the naive cost; in practice most
# files are relevant to one or two categories, and skipping the rest is the
# difference between a CI step and something nobody runs.
#
# Every pattern here is deliberately loose. A skipped category is a guaranteed
# miss, so the bar for skipping is "there is nothing in this file the category
# could even attach to", not "this file looks fine".
RELEVANCE: dict[Category, re.Pattern[str]] = {
    Category.LOOKAHEAD: re.compile(
        r"shift|rolling|expanding|ewm|resample|merge|join|fillna|ffill|bfill|asof|"
        r"pct_change|diff|lag|lead|align|reindex|feature|signal|factor",
        re.IGNORECASE,
    ),
    Category.LABEL_LEAK: re.compile(
        r"fit|train|target|label|^y$|predict|feature|X_train|design|model", re.IGNORECASE
    ),
    Category.SURVIVORSHIP: re.compile(
        r"universe|constituent|ticker|symbol|member|listing|delist|panel|screen|"
        r"index|read_csv|read_parquet|load|merge|join|isin",
        re.IGNORECASE,
    ),
    Category.UNPURGED_CV: re.compile(
        r"split|fold|cv|cross_val|validate|validation|train|test|holdout|oos|"
        r"purge|embargo|backtest|walk_forward|fit",
        re.IGNORECASE,
    ),
    Category.COST_FANTASY: re.compile(
        r"position|weight|trade|order|fill|execut|pnl|profit|return|ret|price|"
        r"slippage|cost|fee|commission|borrow|turnover|notional|volume|liquidity|size",
        re.IGNORECASE,
    ),
    # Widened after this gate silently skipped the one category that mattered on
    # fixtures/llm_only/study.py, whose defect is a selection log with no sweep
    # vocabulary in it at all. Iterative refinement against a fixed holdout is
    # the hardest SWEEP-OVERFIT case to spot precisely because it does not look
    # like a search.
    Category.SWEEP_OVERFIT: re.compile(
        r"grid|param|search|sweep|tune|tuning|best|optim|score|sharpe|trial|"
        r"candidate|threshold|product|itertools|GridSearch|Randomized|"
        r"holdout|oos|round|select|evaluat|experiment|iterat|variant|attempt|"
        r"compare|result|version|baseline",
        re.IGNORECASE,
    ),
}


def relevant_categories(ctx: FileContext, categories: Sequence[Category]) -> list[Category]:
    """Categories worth spending a request on for this file."""
    return [c for c in categories if any(RELEVANCE[c].search(t) for t in ctx.vocabulary)]


@dataclass
class RejectedFinding:
    """A model finding that failed validation, kept for diagnostics."""

    reason: str
    payload: dict


@dataclass
class LLMPass:
    """Runs one structured request per relevant (file, category) and validates the results."""

    provider: Provider
    categories: Sequence[Category] = tuple(Category)
    max_lines: int = 400
    workers: int = DEFAULT_WORKERS
    gate: bool = True
    """Whether to skip categories with no lexical foothold in the file. Turn off
    to spend the full six requests per file."""
    rejected: list[RejectedFinding] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)

    def plan(self, contexts: Iterable[FileContext]) -> list[tuple[FileContext, Category]]:
        """The (file, category) requests this pass will make."""
        requests: list[tuple[FileContext, Category]] = []
        for ctx in contexts:
            wanted = (
                relevant_categories(ctx, self.categories) if self.gate else list(self.categories)
            )
            for category in self.categories:
                if category in wanted:
                    requests.append((ctx, category))
                else:
                    self.skipped.append(f"{ctx.path} [{category.value}]")
        return requests

    def run(self, contexts: Iterable[FileContext]) -> list[Finding]:
        requests = self.plan(contexts)
        if not requests:
            return []
        if self.workers <= 1:
            return [f for ctx, cat in requests for f in self.run_one(ctx, cat)]

        # Requests are independent and dominated by round-trip latency, so a small
        # thread pool turns a serial wait into a parallel one. Results are
        # collected in request order to keep runs reproducible.
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            batches = list(pool.map(lambda item: self.run_one(*item), requests))
        return [finding for batch in batches for finding in batch]

    def run_one(self, ctx: FileContext, category: Category) -> list[Finding]:
        prompt = build_user_prompt(ctx, category, max_lines=self.max_lines)
        try:
            raw = self.provider.complete(system=SYSTEM, user=prompt, schema=FINDINGS_SCHEMA)
        except ProviderError as exc:
            self.errors.append(f"{ctx.path} [{category.value}]: {exc}")
            return []

        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            self.errors.append(f"{ctx.path} [{category.value}]: response was not JSON ({exc})")
            return []

        results: list[Finding] = []
        for item in payload.get("findings", []):
            finding, rejection = validate(item, ctx, category)
            if finding is not None:
                results.append(finding)
            elif rejection is not None:
                self.rejected.append(rejection)
        return results


def validate(
    item: dict, ctx: FileContext, category: Category
) -> tuple[Finding | None, RejectedFinding | None]:
    """Turn one model finding into a :class:`Finding`, or reject it with a reason."""
    check_id = str(item.get("check_id", "")).strip().upper()
    if check_id not in SPECS:
        return None, RejectedFinding(f"unknown check id {check_id!r}", item)
    spec = SPECS[check_id]

    if spec.category is not category:
        return None, RejectedFinding(
            f"{check_id} belongs to {spec.category.value}, not the requested {category.value}",
            item,
        )
    # A model finding against a purely-static check id is not rejected here. It may
    # be a genuine instance the AST pass could not resolve, and if it duplicates a
    # static hit the merge layer folds the two together rather than reporting both.

    path = str(item.get("file", "")).strip()
    if path != ctx.path:
        return None, RejectedFinding(f"cited {path!r}, which is not the file under audit", item)

    try:
        line = int(item["line"])
        end_line = int(item.get("end_line", line))
    except (KeyError, TypeError, ValueError):
        return None, RejectedFinding("line numbers missing or non-integer", item)

    if not 1 <= line <= len(ctx.lines):
        return None, RejectedFinding(f"line {line} is outside the file", item)
    end_line = max(line, min(end_line, len(ctx.lines)))

    quoted = str(item.get("quoted_line", ""))
    actual = ctx.line_text(line)
    if quoted.strip() and quoted.strip() != actual.strip():
        return None, RejectedFinding(
            f"quoted text does not match line {line}: quoted {quoted.strip()!r}, "
            f"file has {actual.strip()!r}",
            item,
        )
    if not actual.strip():
        return None, RejectedFinding(f"line {line} is blank", item)

    mechanism = str(item.get("mechanism", "")).strip()
    if not mechanism:
        return None, RejectedFinding("no mechanism given", item)

    return (
        Finding(
            check_id=check_id,
            category=spec.category,
            title=str(item.get("title") or spec.name),
            severity=_enum(Severity, item.get("severity"), spec.severity),
            confidence=_enum(Confidence, item.get("confidence"), Confidence.MEDIUM),
            location=Location(ctx.path, line, end_line),
            snippet="\n".join(ctx.lines[line - 1 : min(end_line, line + 2)]).strip(),
            mechanism=mechanism,
            fix=None,
            source=Source.LLM,
            evidence={"reported_by": "llm"},
        ),
        None,
    )


def _enum(enum_cls, value, default):  # type: ignore[no-untyped-def]
    try:
        return enum_cls(value)
    except ValueError:
        return default
