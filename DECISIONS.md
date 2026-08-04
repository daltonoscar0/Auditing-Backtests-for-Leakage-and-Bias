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

### 45. The model pass was recorded by subagents, not by the API

No API key was available, so the cassette was recorded by giving Claude Opus 5
subagents Hindsight's exact system and user prompts — one per (file, category),
each with a clean context and an explicit instruction not to read anything else
in the repository, since the fixture directory contains the answer key.

This is marked `subagent` rather than `live` because it settles only one of the
two questions. It shows the model finds these defects; it says nothing about
whether `AnthropicProvider` speaks the wire format correctly, which remains
pinned by fake-client tests alone. Conflating the two would have been the easy
and dishonest move.

### 46. The eval distinguishes a wrong anchor from a miss

The model found all six planted defects but cited two of them on a different
line from the one the fixture names — the label leak on the leaking feature
rather than on the label, and the holdout reuse on the docstring claiming the
headline rather than on the table of rounds. Both are defensible citations of the
same defect.

They still count as misses, because where a finding points is part of what makes
it actionable, and moving the expectation to match the model after seeing the
result would be scoring the exam after reading the answers. But the eval now
reports them as `ANCHOR` rather than `MISSED`, because "found it, cited it
elsewhere" and "did not find it" are different results and folding them together
loses information.

`test_static_recall_holds_at_zero_tolerance` is scoped to the static layer for
the same reason: holding a model to a line a fixture author chose measures
agreement about citation style, not detection.

### 47. Unplanted findings are reported, not absorbed

The recorded run produced seven findings nobody planted. Several are real — a
rolling window whose right edge includes the label bar, an exact algebraic
reconstruction of the target from two features, and a frame-wide `ffill` that
leaks one ticker's fundamentals into another's early history, which is a genuine
bug in the fixture that its author did not intend.

They are left as `UNEXPECTED` in the eval output and the fixture's
`expected_findings.json` is unchanged. Adding them to the answer key after the
fact would make the fixture's expectations derived from the model's output,
which destroys the only thing an independent answer key is for.

The `ffill` bug is the exception, because it is a defect rather than an
undocumented instance of a planted one: it was an accidental correctness error,
not one of the six semantic traps the fixture exists to hold. It has been fixed,
which shifted every line below it and invalidated the pipeline.py half of the
cassette, so those five prompts were re-recorded from scratch. The study.py
entries are from the original run, that file being untouched. The re-recorded run
scored the same 4/6 with the same two anchors, and no longer reports the `ffill`.

The remaining unplanted findings on `pipeline.py` are not bugs to fix. The
realised-vol window and the algebraic reconstruction of the target are both
instances of the planted `LL004` - features and label drawn from the same bar -
manifesting at sites the fixture author did not enumerate. The fixture is
working; the note in its answer key was narrower than the defect it describes.

### 49. The eval matched expectations category-first, and mis-scored

`Expectation.matches` keys on category, which is right for the reason given in
decision 9. But when two findings of one category land on one statement, the
first one encountered satisfied the expectation and the genuinely-expected
finding fell through to `UNEXPECTED`. A `CF002` consumed the `CF005`
expectation, and the real `CF005` was reported as unexpected on the line
directly above.

Matching now prefers a finding whose check id equals the expectation's, and only
falls back to category when none does. Found by reading the eval's own
`CHECK-ID` diagnostic output on the recorded run.

### 48. A Layer 2 shortfall reports rather than fails

Against a hand-authored cassette, a missing finding means the pipeline dropped
something and the build should break. Against a model-generated one it is a
measurement of the model, and a build that fails when a model changes its mind
is a build nobody can keep green. The eval gates on the static layer and the
clean fixture, and reports Layer 2.

---

## Cross-file, notebooks and adoption (Milestone 3)

### 50. Imports are resolved one level deep, not transitively

`ProjectIndex` maps every module in the audited tree to what it defines, then
works out which of those definitions each file imports. That vocabulary joins the
file's own when a check asks whether there is a cost term in scope. The
`cross_file` fixture is the proof: read one file at a time it raises `CF001`,
`CF002` and `CF004`; with imports resolved it is silent, and a test asserts both
halves of that.

One level, deliberately. Transitive resolution would let a single mention of the
word `fee` anywhere in a repository suppress every cost finding in it, trading a
false-positive problem for a worse and much less visible false-negative one.

An import of a name that cannot be resolved still contributes the bound name:
`from vendor.costs import COST_BPS` tells a reader there is a cost model whether
or not `vendor/` is inside the audited tree.

### 51. Notebook cells are analysed together and cited separately

Concatenating cells into one synthetic module is what makes cross-cell analysis
work, and cross-cell is the normal case — a target defined in cell 2 and used in
cell 7. Treating each cell as its own file would miss all of it.

But a line number into that concatenation is not a coordinate anyone can act on,
so `Location` carries a cell and every citation reads `cell 4, line 2`. Cells
that are not Python — IPython magics, shell escapes, half-written drafts — are
blanked rather than dropped, which keeps every later cell's numbering honest.

No unified diff is offered for a notebook: the lines on disk are JSON string
fragments, so a patch against the flattened source would not apply to anything.

### 52. Baselines fingerprint the code, not the line

A line-numbered suppression file goes stale the first time somebody adds an
import, and a stale one is worse than none: it stops suppressing the finding it
named and starts suppressing whatever moved into that line. Entries are keyed on
a hash of check id, path, cell and the normalised text of the cited code, and a
test asserts a baseline survives an edit above the finding it covers.

### 53. Hindsight audits itself in CI, and that found a false positive

`SV001` fired on `checks/survivorship.py`, because the regex that looks for
present-day universe names contains the string `current_constituents`, and the
constant holding it is called `CURRENT_UNIVERSE`. Both the pattern and the name
bound to it are now skipped when the assignment is a compiled regex.

This generalises: any repository that holds a vocabulary of bad names — a linter,
a data dictionary, a validation schema — would have hit the same thing. The
self-audit step in CI exists to keep catching this class of problem.
