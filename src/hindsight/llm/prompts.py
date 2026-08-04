"""Prompt construction for the model pass.

One request per (file, category). Keeping the categories separate costs more
calls but produces sharper findings: a prompt that asks for six unrelated defect
classes at once reliably returns the easy ones and misses the rest.
"""

from __future__ import annotations

from ..context import FileContext
from ..models import Category
from ..taxonomy import TAXONOMY_VERSION, specs_for

SYSTEM = f"""\
You are a skeptical senior quantitative researcher reviewing someone else's backtest \
before it goes in front of an investment committee. Your job is to find the reasons the \
reported performance will not survive live trading.

You are auditing against Hindsight taxonomy version {TAXONOMY_VERSION}. Only report defects \
that instantiate one of the check ids given in the user message. If a file is clean with \
respect to those checks, return an empty findings list - that is a useful and common answer, \
and a false positive costs more than a miss because it teaches the reader to ignore the tool.

Rules you must follow:

1. Cite a real line. Every finding must give the 1-based line number of the single most \
relevant line, and must reproduce that line verbatim in `quoted_line`. Findings whose quoted \
text does not match the file at that line are discarded before the reader ever sees them.
2. Explain the mechanism for THIS code, not the pattern in general. Say what the value is, \
where it comes from, and why it could not have been known at the time it is used.
3. Do not report a defect you cannot point at. Suspicion about code you were not shown is \
not a finding.
4. Do not repeat what a mechanical pattern-matcher would already catch: a literal \
`shift(-1)`, a literal `center=True`, a literal `KFold`. You are here for the cases that need \
reading comprehension - alignment that depends on join semantics, costs that depend on the \
execution model, validation schemes whose correctness depends on how the labels were built.
5. Judge the code as written. Do not assume a helper you were not shown fixes the problem, \
and do not assume it causes one.
"""

CATEGORY_FRAMING: dict[Category, str] = {
    Category.LOOKAHEAD: (
        "Ask of every feature: could a person standing at time t, with only what had been "
        "published by then, have computed this number? Pay particular attention to join keys "
        "that carry an effective date rather than an availability date, to vendor data with a "
        "publication lag, and to resampling whose bucket boundary sits after the bar it labels."
    ),
    Category.LABEL_LEAK: (
        "Trace what the target is built from, then trace what the feature matrix is built "
        "from, and look for any path connecting them. Include indirect paths: a column derived "
        "from the target, a transform fitted on a frame that contains it, an index or key that "
        "encodes the outcome."
    ),
    Category.SURVIVORSHIP: (
        "Ask how the set of names being tested was assembled, and whether that assembly used "
        "information from after the start of the sample. Universe tables with no date "
        "dimension, joins against a present-day metadata table, and filters on attributes that "
        "change over a company's life are the usual routes."
    ),
    Category.UNPURGED_CV: (
        "The reference standard is purged, embargoed walk-forward validation (Lopez de Prado, "
        "Advances in Financial Machine Learning, ch. 7). Check both requirements: training "
        "observations whose label horizon overlaps the test window must be purged, and a "
        "further embargo must follow the test window. Consider whether the split respects both "
        "the time axis and any grouping axis."
    ),
    Category.COST_FANTASY: (
        "Compare what the backtest charges against what trading would cost. Consider the "
        "execution assumption implied by the fill price, the turnover implied by the "
        "rebalancing rule, the borrow cost implied by any short exposure, and whether the "
        "position sizes are attainable against the traded volume."
    ),
    Category.SWEEP_OVERFIT: (
        "Count the trials, then ask what was reported. A number selected as the best of many "
        "is biased upward by construction. Look for the test set being consulted during model "
        "selection, and for a headline statistic presented without any correction for the "
        "number of times the data was queried."
    ),
}


def prompt_key(user: str) -> str:
    """Stable identity of a request, read back off the prompt itself.

    ``<path>::<CATEGORY>``. Used to key cassettes and the mock provider, so a
    recorded response can never be replayed against a different file or a
    different category than the one it was recorded for.
    """
    path = ""
    category = ""
    for line in user.splitlines():
        if line.startswith("FILE: "):
            path = line[len("FILE: ") :].strip()
        elif line.startswith("CATEGORY: "):
            category = line[len("CATEGORY: ") :].strip()
        if path and category:
            break
    return f"{path}::{category}" if category else path


def checks_block(category: Category) -> str:
    """Render the taxonomy entries the model is allowed to report for a category."""
    lines: list[str] = []
    for spec in specs_for(category):
        note = (
            ""
            if spec.strategy == "llm"
            else " (a static rule already covers the mechanical form of this; report only "
            "instances that rule would miss)"
        )
        lines.append(f"- {spec.id} — {spec.name} [default severity: {spec.severity.value}]{note}")
        lines.append(f"    mechanism: {spec.mechanism}")
    return "\n".join(lines)


def numbered_source(ctx: FileContext, *, max_lines: int = 400) -> tuple[str, int]:
    """Render the file with line numbers, truncated to `max_lines`.

    Returns the text and the number of lines actually included, so the caller can
    tell the reader when an audit saw only part of a file rather than silently
    reporting on a truncated view.
    """
    shown = ctx.lines[:max_lines]
    width = len(str(len(shown)))
    body = "\n".join(f"{i:>{width}} | {line}" for i, line in enumerate(shown, start=1))
    return body, len(shown)


def build_user_prompt(ctx: FileContext, category: Category, *, max_lines: int = 400) -> str:
    body, shown = numbered_source(ctx, max_lines=max_lines)
    truncated = ""
    if shown < len(ctx.lines):
        truncated = (
            f"\n\n(The file continues past line {shown}; you are seeing the first {shown} "
            f"of {len(ctx.lines)} lines. Do not report on code you cannot see.)"
        )

    facts = [
        f"time-ordered data: {'yes' if ctx.is_temporal else 'not established'}",
        f"target column(s): {sorted(ctx.target_columns) or 'not identified'}",
        f"forward label horizon: {ctx.forward_horizon or 'not identified'}",
    ]

    return f"""\
FILE: {ctx.path}
CATEGORY: {category.value}

Audit this file for {category.value} defects only.

{CATEGORY_FRAMING[category]}

Report only these checks:
{checks_block(category)}

What the static pass established about this file (may be incomplete, and you should
correct it if the source says otherwise):
{chr(10).join(f"- {fact}" for fact in facts)}

Source, with 1-based line numbers. The numbers are not part of the file; cite them, do
not quote them in `quoted_line`.

{body}{truncated}
"""
