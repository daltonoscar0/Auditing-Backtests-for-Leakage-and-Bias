# Hindsight

Audit a trading-research repository for the reasons its backtest will not survive
live trading.

Hindsight reads a repo the way a skeptical senior quant reads someone else's
notebook: looking for future information in the features, the answer in the
feature matrix, a universe chosen with knowledge of who survived, a validation
scheme that leaks across the fold boundary, a cost model that charges nothing,
and a headline number that is the maximum of a hundred trials.

It is **strictly read-only** with respect to the audited tree, and the static
layer runs offline with no dependencies outside the standard library.

```
$ hindsight audit ./strategy-repo

# Hindsight audit

**Target:** `/Users/oscar/strategy-repo`
**Scanned:** 6 Python files · 23 checks · taxonomy 1.0.0 · model pass off

## Summary

**9 findings.**

| Severity | Count |
| --- | ---: |
| critical | 6 |
| high | 2 |
| medium | 1 |

### 1. CF001 — PnL arithmetic with no cost term

**critical** · COST-FANTASY · confidence high · detected by static

`backtest.py:17`

```python
pnl = position.shift(1) * panel["ret_1"]
```

**Why this matters**

Returns are computed as position times market return with nothing subtracted.
Every rebalance is free. For anything above weekly turnover the omitted cost is
usually larger than the entire measured edge, so a strategy that looks profitable
gross is loss-making net.

Here: `pnl` is `position` times `ret_1` with no cost term anywhere in scope.
```

## Install

Requires Python 3.11 or newer.

```sh
git clone <this repo> && cd hindsight
make install          # creates .venv and installs with dev extras
make test             # 180 tests
make eval             # score against the planted-bug fixtures
```

Or without the Makefile:

```sh
pip install -e ".[dev]"
```

Optional extras: `.[llm]` adds the `anthropic` client for the model pass;
`.[fixtures]` adds pandas/numpy/scikit-learn if you want to execute the fixture
repos by hand (the auditor itself never imports them).

## Usage

```
hindsight audit <path> [--format md|json] [--fail-on <severity>] [--only <ids>]
                       [--out <file>] [--llm] [--provider anthropic|mock]
hindsight checks [--format text|json]
```

| Flag | Meaning |
| --- | --- |
| `--format md` | Markdown report for humans (default) |
| `--format json` | Machine-readable report for CI, pinned to a taxonomy version |
| `--fail-on <sev>` | Exit 1 if any finding is at or above this severity. Default: never fail |
| `--only <ids>` | Comma-separated check ids, e.g. `--only LA001,CV003` |
| `--out <file>` | Write the report to a file instead of stdout |
| `--llm` | Enable the model reasoning pass (off by default; needs network) |
| `--llm-record FILE` | Record every model response to a cassette for offline replay |
| `--llm-replay FILE` | Replay a recorded cassette instead of calling the API |
| `--llm-workers N` | Parallel model requests (default 4) |
| `--llm-all-categories` | Ask every category about every file instead of skipping irrelevant ones |

Exit codes: `0` audit ran and nothing tripped `--fail-on`; `1` findings at or
above the threshold; `2` the audit could not run (bad path, unknown check id,
provider failure). Exit 2 is deliberately distinct so a broken invocation is
never mistaken for a clean repository.

In CI:

```yaml
- run: hindsight audit . --format json --fail-on critical --out hindsight.json
```

As a library:

```python
from pathlib import Path
from hindsight import audit

result = audit(Path("./strategy-repo"))
for finding in result.findings:
    print(finding.check_id, finding.location, finding.title)
```

## What it looks for

Six categories, 29 checks — 23 static, 6 reserved for the model pass. The
normative specification is [`TAXONOMY.md`](TAXONOMY.md); this is the summary.

| Category | The question it asks | Static checks |
| --- | --- | --- |
| **LOOKAHEAD** | Could a person standing at time `t` have computed this number? | Negative shifts, centred rolling windows, backward fill, transformers fitted before the split, full-sample normalisation |
| **LABEL-LEAK** | Is the answer reachable from the feature matrix? | Target in the design matrix (both list and drop construction), features engineered from the target, transformers fitted on a frame holding it |
| **SURVIVORSHIP** | Was the sample chosen knowing what happened to it? | Present-day constituent lists, listing-status filters, universes selected on whole-sample outcomes |
| **UNPURGED-CV** | Can information cross the train/test boundary? | Shuffled K-fold on time-ordered data, `train_test_split` defaults, walk-forward with no purge gap, gaps shorter than the label horizon |
| **COST-FANTASY** | Does the backtest charge what trading costs? | Uncosted PnL, fills at close or mid, cost parameters pinned to zero, shorts with no borrow |
| **SWEEP-OVERFIT** | Is the reported number the maximum of many trials? | Grids maximised on the test set, `best_score_` as out-of-sample, test-set early stopping, wide sweeps with no multiple-testing correction |

Reference standard for UNPURGED-CV is purged and embargoed walk-forward
cross-validation as specified in López de Prado, *Advances in Financial Machine
Learning*, ch. 7 — both requirements are checked: purge the training
observations whose label horizon overlaps the test window, and embargo a further
stretch after it.

Run `hindsight checks` for the full list with severities.

### What it does not look for

Stated explicitly, because a tool that implies completeness is worse than none:
data errors (bad splits, stale ticks, wrong corporate actions), regime
dependence, capacity, execution microstructure (queue position, partial fills,
adverse selection), and statistical inference beyond the multiple-testing check.
A clean report means none of the 29 patterns were found. It does not mean the
research is correct.

## How it works

```
  target repo
      │
      ▼
  collector ──► FileContext (per file: is this temporal? which column is the
      │          target? how far forward does the label look? is there a cost
      │          term in scope?)
      │
      ├──► Layer 1: 23 AST checks           ── always on, stdlib only, offline
      │
      └──► Layer 2: model reasoning pass    ── opt-in via --llm
               one structured request per RELEVANT (file, category);
               findings must cite a line and quote it verbatim,
               and are discarded if the quote does not match
      │
      ▼
  merge ──► dedupe, promote static+model agreement to high confidence,
      │     cap uncorroborated model findings at medium
      ▼
  report ──► markdown (humans) | json (CI)
```

Every finding carries: check id and category, severity, a `file:line` citation,
a plain-English explanation of *why* this biases results, the canonical remedy,
a confidence level, and — where the fix is mechanical — a unified diff you can
feed to `git apply`. Hindsight never applies a patch itself.

## Eval

`make eval` scores the auditor against fixture repos with deliberately planted
bugs, recorded in machine-readable `expected_findings.json` files. Recall is
matched line-tolerantly (±3 lines) per category; the `clean/` fixture is
pass/fail and any finding there is a false positive.

```
Hindsight eval — taxonomy 1.0.0, line tolerance +/-3

Layer 1 — static checks
  fixture            found  planted   recall  extra
  ---------------- ------- -------- --------  -----
  clean                  0        0      n/a      0
  cost_fantasy           5        5   100.0%      0
  label_leak             4        4   100.0%      0
  lookahead              6        6   100.0%      0
  mixed                  9        9   100.0%      0
  survivorship           4        4   100.0%      0
  sweep_overfit          5        5   100.0%      0
  unpurged_cv            4        4   100.0%      0

Per category (planted bugs across the static fixtures)
  category         recall       hits
  -------------- --------  ---------
  LOOKAHEAD       100.0%        8/8  ################
  LABEL-LEAK      100.0%        5/5  ################
  SURVIVORSHIP    100.0%        5/5  ################
  UNPURGED-CV     100.0%        5/5  ################
  COST-FANTASY    100.0%        7/7  ################
  SWEEP-OVERFIT   100.0%        7/7  ################

Layer 2 — model pass (replayed from cassette)
  fixture            found  planted   recall  provenance    
  ---------------- ------- -------- --------  --------------
  llm_only               4        6    66.7%  subagent      

  NOTE: responses are model-generated but were not recorded through
  AnthropicProvider, so the API wire format remains verified only by
  the fake-client tests. Re-record through the API with: make record

llm_only:
  ANCHOR     LABEL-LEAK     pipeline.py:47 expected, LL004 reported at pipeline.py:54 instead
  ANCHOR     SWEEP-OVERFIT  study.py:13 expected, SO005 reported at study.py:28-32 instead
  UNEXPECTED LABEL-LEAK     pipeline.py:54  (LL004) Same-bar close return makes the intraday label algebraically recoverable
  UNEXPECTED SURVIVORSHIP   study.py:19  (SV003) Round 5 universe excludes the two worst-performing sectors, a whole-sample outcome
  UNEXPECTED LOOKAHEAD      pipeline.py:37  (LA006) Frame-wide ffill leaks one ticker's later fundamentals into another ticker's earlier rows
  UNEXPECTED LOOKAHEAD      pipeline.py:54  (LA006) close_to_close feature contains the same session's close, which is the label
  UNEXPECTED LOOKAHEAD      pipeline.py:42  (LA006) Sector reference joined on ticker alone carries today's classification back over all history
  UNEXPECTED COST-FANTASY   pipeline.py:66-67  (CF005) Order sizing scales with capital and never checks against traded volume
  UNEXPECTED SWEEP-OVERFIT  study.py:28-32  (SO005) Headline Sharpe is the fifth revision scored on the same holdout used to choose all five
  CHECK-ID   pipeline.py:67 expected CF005, got CF002

Overall static recall  100.0%  (37/37)
False positives on clean/  0
Layer 2 recall         66.7%  (4/6), 2 anchored elsewhere  [subagent]

PASS
```

Target is ≥90% recall and zero findings on `clean/`; the harness exits non-zero
below either. It is wired into `pytest` (`tests/test_eval.py`), which
additionally asserts that recall holds at **zero** line tolerance — every check
cites the exact line, not merely the right neighbourhood.

The `clean/` fixture is the most useful thing in the repo. It is a genuinely
careful piece of research — point-in-time universe including delistings,
expanding-window normalisation, purged and embargoed walk-forward validation,
commission plus half-spread plus borrow costs, a deflated Sharpe — and it must
produce nothing. Two of the checks were softened during development because
they fired on it, which is exactly what it is there for.

## Testing

```sh
make test    # 180 tests
make lint    # ruff check + format --check
```

Every static check has both a positive case and a lookalike that must *not* fire:
`shift(1)`, `rolling(center=False)`, a Sharpe ratio rather than a z-score, an
expanding-window normalisation, shuffled CV on genuinely cross-sectional data, a
`train_test_split` with `shuffle=False`, a costed PnL, a short book that pays
borrow, a grid scored on a validation fold. False positives are what get an audit
tool switched off, so they are tested first. A meta-test fails the build if any
registered check lacks either kind of case.

`tests/test_readonly.py` hashes every fixture tree before and after a full audit
and asserts byte-identity.

## Repository layout

```
src/hindsight/
  models.py        Finding, Location, Severity, Confidence, AuditResult
  taxonomy.py      machine-readable mirror of TAXONOMY.md
  readonly.py      read-only file access and tree hashing
  collector.py     file discovery
  context.py       per-file derived facts every check shares
  astutils.py      AST helpers
  fixes.py         unified-diff generation
  merge.py         dedupe and confidence scoring
  checks/          one module per taxonomy category
  llm/             provider abstraction, prompts, schema, validation
  report/          markdown and json renderers
  cli.py           the hindsight command
fixtures/          one repo per category + clean/ + mixed/
eval/run_eval.py   scoreboard
tests/
TAXONOMY.md        the normative specification
DECISIONS.md       every judgement call and why
```

## Status and known gaps

Milestone 1 (static layer, fixtures, eval) is complete. Milestone 2 (model pass)
is complete apart from a live API recording.

**The model pass has been measured.** `fixtures/llm_only/cassette.json` holds
responses from Claude Opus 5, given Hindsight's exact prompts one per
(file, category), each with a clean context and no access to the fixture's
answer key. The result:

- **All six planted defects were identified.** Four were cited within the eval's
  ±3-line tolerance; two were anchored on a different but defensible line — the
  label-leak was cited on the leaking *feature* rather than the label, and the
  holdout-reuse on the docstring claiming the headline rather than the round
  table. The eval reports these as `ANCHOR`, and they still count as misses,
  because where a finding points is part of what makes it actionable.
- **Seven findings nobody planted**, most of them real: a rolling window whose
  right edge includes the label bar, an exact algebraic reconstruction of the
  target from two features, and a frame-wide `ffill` that leaks one ticker's
  fundamentals into another's history — that last one a genuine bug in the
  fixture its author did not intend.
- **One bug in Hindsight**, found by the exercise: the merge layer was collapsing
  two model findings that shared a category and sat close together, silently
  dropping a planted `CF005` because a `CF002` landed on the same statement.

The recording is marked `subagent`, not `live`: it evidences that the model finds
these defects, not that `AnthropicProvider` speaks the wire format. That last
step needs an API key (~$0.17):

```sh
export ANTHROPIC_API_KEY=...        # Console account; separate from a Claude subscription
make record                          # re-records through AnthropicProvider
make eval                            # provenance flips to `live`
```

Known gaps, in rough order of value:

1. **No cross-file analysis.** A cost model defined in one module and applied in
   another is invisible. This is the largest available accuracy gain.
2. **Never run on a repo I did not write.** The static fixtures and the static
   checks were co-designed, so that 100% is partly a measure of the fixtures.
   The Layer 2 number is more meaningful — the model had no answer key — but the
   fixture author and the tool author are still the same person.
3. **No live API recording.** Everything else in Layer 2 is verified; the wire
   format is pinned by fake-client tests.
4. **No notebook support.** `.ipynb` is where a lot of this research lives.
   Skipped rather than half-done, because a citation needs a line number that
   means something to the reader.
5. **Name-based heuristics.** `SURVIVORSHIP` in particular leans on identifier
   names and reports at `medium` confidence to say so.

## Licence

MIT.
