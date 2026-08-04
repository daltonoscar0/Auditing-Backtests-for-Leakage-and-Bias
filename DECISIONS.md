# Decisions

Every judgement call made while building this, and the reasoning behind it.
Written so that a future maintainer can tell which choices are load-bearing and
which were arbitrary.

---

## Scope and philosophy

### 1. False positives are the failure mode that matters

An audit tool gets used exactly as long as its findings are worth reading. One
confident report of a bug that isn't there teaches the reader to skim, and a
skimmed report is worth nothing. Every design tension in this repo was resolved
in favour of precision:

- Checks are suppressed by cheap evidence that the author already handled the
  issue (a cost term in scope, a purge gap, an embargo, a `Pipeline` wrapper),
  even though that suppression will occasionally hide a real defect.
- Findings whose only evidence is a name match are emitted at `medium`
  confidence and say so.
- The `clean/` fixture exists specifically to fail the build the first time a
  new rule over-fires. It is the most valuable fixture in the repo.

The cost of this posture is recall on messy real-world code. That is the right
trade for a tool whose output a human has to act on.

### 2. The taxonomy is versioned and the tool states its limits

`TAXONOMY.md` ends with an explicit list of what Hindsight does *not* check —
data errors, regime dependence, capacity, execution microstructure, statistical
inference beyond multiple testing. A clean report says "none of these 29 patterns
were found", not "this research is correct", and the report renderer says so in
those words. An audit tool that implies completeness is worse than none.

### 3. Read-only is enforced, not merely intended

`readonly.read_text` rejects any file mode other than a plain read, and
`tests/test_readonly.py` hashes every fixture tree before and after a full audit
(with and without the model pass) and asserts byte-identity. Someone will point
this at a working tree they have not committed.

---

## Architecture

### 4. Two layers, and the static one has to stand alone

Layer 1 is pure `ast`, zero third-party imports, no network. Layer 2 is
model-backed and **off by default**. An auditor that cannot run in CI without
credentials and egress is an auditor that does not run in CI. `--llm` opts in;
`--no-llm` is documented but redundant.

### 5. Per-file context, computed once

Most checks need the same handful of facts — is this time-ordered data, which
column is the target, how far forward does the label look, is there a cost term
in scope. `FileContext` derives each once and memoises it, so every check agrees
on the answer. The alternative, each check re-deriving what it needs, produces
findings that contradict each other in the same report.

### 6. Checks see one file at a time

No cross-file analysis, no import resolution, no call graph. This is a real
limitation: a cost model defined in `costs.py` and applied in `backtest.py` is
invisible to `CF001`, which is why the `cost_fantasy` fixture splits `config.py`
from `backtest.py` and the tool reports the uncosted PnL. Cross-file analysis is
the single largest available accuracy gain and is deferred to Milestone 3.

### 7. Module-level integer constants are resolved

`shift(-HORIZON)` and `TimeSeriesSplit(gap=HORIZON)` are how people actually
write this code. `FileContext.resolve_int` follows one level of module-constant
indirection. Without it, `CV003` would fire on a correctly-gapped splitter — a
false positive on exactly the code that got it right.

### 8. Windowed views bound to a name are tracked

`history = series.expanding(); (x - history.mean()) / history.std()` is a correct
trailing z-score whose windowing is invisible at the point of use.
`FileContext.windowed_names` records variables bound to a
rolling/expanding/groupby/resample view so `LA005` does not report it. This case
appeared while writing the `clean/` fixture, which is the fixture doing its job.

---

## Taxonomy design

### 9. Check id is the unit of detection; category is the unit of reporting

Fixtures assert against categories, so a check can be split, merged or renamed
without rewriting every `expected_findings.json`. The eval still reports check-id
mismatches, and `tests/test_eval.py` fails on them, so a rename is a deliberate
act rather than a silent drift.

### 10. `CV003` and `CV004` are mutually exclusive by construction

A splitter with no gap raises `CV003`; a splitter with a gap smaller than the
label horizon raises `CV004`. Overlapping them would double-report one mistake.
`TAXONOMY.md` states this explicitly because it is not obvious from the names.

### 11. A negative shift that builds the target is not a lookahead

`df["fwd_ret"] = df["close"].shift(-5)` is how forward-return labels are made.
`LA001` suppresses it when the assignment target is the identified target column.
Whether that label is purged out of the training folds is `CV004`'s problem.

### 12. Target-root matching is length-gated

A target literally named `y` would otherwise claim `yield_spread` as a
derivative. `is_target_like` requires exact equality for short roots, a `_`
separator for roots of three or more characters, and only allows bare prefix
matching at five or more.

### 13. `LA005` requires the subtraction

`ret.mean() / ret.std()` is a Sharpe ratio, not a normalisation. Requiring an
explicit `x - x.mean()` numerator is what separates the two. This is checked
directly in `tests/test_static_checks.py`.

### 14. The whole UNPURGED-CV category is gated on temporality

Shuffled cross-validation on genuinely cross-sectional data is correct practice.
`FileContext.is_temporal` needs one strong signal (a datetime conversion, a
`resample`, a `parse_dates`, a `TimeSeriesSplit`) or two weak ones. Without this
gate the tool would be unusable in any repository that also contains
non-time-series work.

### 15. `range()` is not a parameter grid

An earlier version of `SO001` treated `for n in range(...)` as a sweep. Loop
counters are far more common than sweeps written that way, and the rule fired on
the `clean/` fixture's purged-walk-forward implementation. Removed.

### 16. Cost checks are suppressed by any cost term in scope

`CF001`, `CF002` and `CF004` go quiet if any identifier or string in the
enclosing scope (plus module level) matches `cost|fee|commission|slippage|
spread|bps|borrow|impact|financing`. Hindsight can tell whether a cost model
exists; it cannot statically tell whether the number in it is defensible.
Flagging research that forgot costs is a claim the tool can support. Grading
someone's slippage assumption is not.

### 17. The `SO004` trial threshold is 25

Deliberately low. The upward bias in the maximum of many noisy estimates is
already material at a few dozen trials. The finding is `medium` severity
precisely because the defensible threshold depends on how correlated the trials
are, which is not visible in the code.

### 18. `SV001` reports once per offending name

A helper called `current_constituents` matches at its definition and again at
every call site. Three citations of one mistake is two too many, so the check
deduplicates on the matched token.

---

## Fixtures and eval

### 19. Fixtures are realistic and un-annotated

No `# BUG HERE` comments. The planted bugs are recorded only in
`expected_findings.json`, so the fixture reads as code someone might actually
have written and the eval cannot be gamed by matching on a marker.

### 20. Fixtures are not executed

They are syntactically valid and semantically coherent, and they import pandas,
numpy and scikit-learn the way real research code does — but the auditor parses
them with `ast` and never imports them. Requiring the eval to *run* them would
mean requiring pandas and scikit-learn to run the eval, which contradicts the
zero-heavy-dependency constraint on the core. They are installable for hand
inspection via the `fixtures` extra.

### 21. Recall is matched line-tolerantly at ±3, but tested at 0

The eval's default tolerance is ±3 lines because a check may legitimately cite an
enclosing statement rather than an exact sub-expression.
`tests/test_eval.py::test_recall_holds_at_zero_tolerance` then asserts that in
practice every check cites the exact line anyway. The tolerance is headroom for
future checks, not slack the current ones are using.

### 22. Unexpected findings are reported per fixture but only fail on `clean/`

A category fixture may legitimately contain incidental findings from other
categories. In practice each was written to contain exactly its planted set, and
the eval prints any extras as `UNEXPECTED` so a regression is visible without
being fatal. On `clean/` an extra is a false positive and fails the build.

### 23. Some cross-category interactions are avoided by fixture design

`LA004` needs a split call in the file, so the `lookahead` fixture uses a
correctly-gapped `TimeSeriesSplit` that no CV check fires on. `CF003` needs a
zero cost constant, whose presence would suppress `CF001` in the same scope, so
the `cost_fantasy` fixture puts it in a separate `config.py` that `backtest.py`
never imports — which is also the most realistic version of that mistake.

---

## Model pass (Milestone 2)

### 24. One request per (file, category)

More calls, sharper findings. A prompt that asks for six unrelated defect classes
at once reliably returns the easy ones and misses the rest.

### 25. Uncited findings are discarded, not downgraded

The model must return the 1-based line number *and* reproduce that line verbatim
in `quoted_line`. Findings whose quoted text does not match the file at that
line, or that cite a blank line, a line outside the file, or a different file,
are dropped before a reader ever sees them, with the reason recorded in
`AuditResult.errors`.

### 26. Uncorroborated model findings are capped at medium confidence

A static hit confirmed by the model pass is promoted to `high` and marked
`static+llm`. A model finding with no static counterpart is kept — it is the
whole point of Layer 2 — but capped at `medium`, because one pass asserting
something is weaker evidence than two and the report should say so.

### 27. `MockProvider` returns nothing by default

Unmapped prompts yield an empty finding list. A mocked model pass must not be
able to inflate the eval's recall.

### 28. Model choice and API shape

`claude-opus-5` via the Messages API with structured outputs
(`output_config.format` with a JSON schema) and `effort: high`. The `anthropic`
package is imported lazily inside `AnthropicProvider.client()`, so installing
without the `llm` extra works and a static-only audit never pays the import.
`stop_reason == "refusal"` is checked before reading `content`, and surfaces as a
`ProviderError` suggesting `--no-llm`.

---

## Engineering

### 29. Only `.py` files are audited

Notebooks are extremely common in trading research and are the obvious next
target, but a citation into a `.ipynb` needs a cell/line mapping that means
something to a reader. Half-doing it would produce line numbers that point
nowhere. Deferred, and listed in the README as a known gap.

### 30. A file that does not parse is reported, not fatal

Audited repos are other people's code and may target a different Python version.
Skipping one module and saying so in the report's Notes section beats refusing to
audit the repository.

### 31. `--fail-on` defaults to off

`hindsight audit` exits 0 by default even with findings, so it can be used
exploratorily without wrapping it in `|| true`. CI opts into gating explicitly.
Exit 2 is reserved for "the audit could not run", so a broken invocation is never
mistaken for a clean repository.

### 32. Fixes are diffs, never applied

Hindsight is read-only, so `--fix` would contradict its central guarantee. Checks
whose remedy is unambiguous and local emit a unified diff for `git apply`.
`fixes.replace_on_line` returns `None` when its pattern does not match rather
than guessing: a missing patch is fine, a wrong one is not. `CF003`'s patch
inserts a placeholder value and the finding text says explicitly that it is a
placeholder.

### 33. `FileContext` is not slotted

Every derived fact on it is a `cached_property`, which needs an instance
`__dict__`. `Finding`, `Location` and `AuditResult` are slotted and frozen.

### 34. Python 3.11+

For `enum.StrEnum`, PEP 604 unions in `isinstance`, and current `ast` node
shapes. The audited repository can be any Python version that `ast.parse`
accepts.

---

## Model pass, continued (Milestone 2)

### 35. Requests are keyed `<path>::<CATEGORY>`, read back off the prompt

The prompt carries a `CATEGORY:` header alongside `FILE:`, and `prompt_key`
parses both back out. This is what lets a cassette be keyed by request identity,
and it means a recorded response can never be replayed against a different file
or a different category than the one it was recorded for.

### 36. Cassettes, with provenance recorded in the file

The model pass is non-deterministic, costs money and needs network access. None
of those belong in `pytest` or `make eval`. A cassette is a JSON file of recorded
responses; `--llm-record` captures one against the live API and `--llm-replay`
replays it forever afterwards.

The `provenance` field distinguishes `live` from `hand-authored`, and the eval
prints it, because the two support very different claims. A hand-authored
cassette proves the pipeline works end to end — prompt keying, schema, citation
validation, merge, render — and proves the fixture is statically clean. Only a
live one says anything about whether the model can find the defect. Conflating
them would be the single most misleading thing this repo could do, so the
distinction is in the file format, in the scoreboard and in the README.

### 37. A cassette miss is loud

`CassetteProvider` is strict by default: a prompt with no recorded response
raises rather than returning an empty finding list. A cassette that has drifted
out of sync with the prompt builder should fail the build, not quietly score
zero and look like a model that found nothing.

### 38. The relevance gate is a cost lever with a real risk

Six requests per file is the naive cost. Most files can only possibly trip one or
two categories, so `relevant_categories` skips the rest — about 45% of requests
on `fixtures/mixed`, 30% on `clean/`.

The risk is not hypothetical. The first version silently skipped SWEEP-OVERFIT on
`fixtures/llm_only/study.py`, whose entire defect is a selection log that
contains no sweep vocabulary at all — iterative refinement against a fixed
holdout is the hardest case in that category precisely because it does not look
like a search. The patterns were widened, and
`test_the_gate_never_drops_a_category_a_fixture_expects` now fails the build if
the gate would skip a category any fixture plants a bug in.

Every pattern is deliberately loose: a skipped category is a guaranteed miss, so
the bar for skipping is "there is nothing here the category could attach to", not
"this looks fine". `--llm-all-categories` turns the gate off entirely.

### 39. Requests run in a small thread pool

They are independent and dominated by round-trip latency. Results are collected
in request order so runs stay reproducible, and the eval pins workers to 1 so a
cassette miss surfaces as one clear error rather than several interleaved ones.

### 40. The system prompt gets the cache breakpoint

It is byte-identical across every request in a run and it is the longest fixed
prefix, so it is the one thing worth caching. The per-file cost then reduces to
roughly the file itself.

### 41. Token usage and an estimated cost are reported

`AuditResult.llm_usage` carries requests, tokens, cached tokens and a dollar
estimate at the published rate, and both renderers print it. A tool that quietly
spends money is a tool that gets switched off after the first invoice. The
estimate is explicitly an estimate; the invoice is authoritative.

### 42. The live provider is tested against a fake client

`tests/test_provider.py` injects a fake client and asserts the request shape:
`claude-opus-5`, `output_config.format` carrying the JSON schema, the system
prompt sent as a cached block, no `temperature`/`top_p`/`top_k`/`thinking`
(all rejected on this model), refusal checked before `content` is read, and usage
accumulated across calls. That pins everything the wire format depends on without
needing credentials, which is what makes it runnable in CI.

### 43. `fixtures/llm_only` must be invisible to the static layer

If the AST checks can see its defects, it is not testing Layer 2.
`test_the_llm_only_fixture_is_invisible_to_the_static_layer` asserts the static
audit returns nothing. The six planted defects are all semantic: a join on
`period_end` rather than `filing_date`, a label sharing a bar with its features, a
date-less reference table applied across history, `GroupKFold` with no time
constraint, order sizing with no volume check, and a holdout queried five times.

### 44. The eval reports two layers separately

The headline static recall number is computed over static fixtures only, and the
Layer 2 row is printed separately with its provenance attached. Folding
hand-authored cassette hits into a single recall figure would inflate a number
people will quote.
