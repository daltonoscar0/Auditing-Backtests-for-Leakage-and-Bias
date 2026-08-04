# Hindsight finding taxonomy

**Version 1.0.0**

This is the normative specification. `src/hindsight/taxonomy.py` is the machine-readable
mirror of it, and `tests/test_taxonomy.py` fails the build if the two drift apart.

Six categories, each with a set of numbered checks. A check is the unit of detection; a
category is the unit of reporting. Fixture `expected_findings.json` files and stored JSON
reports pin against the version number above, so any change to what an existing id *means*
requires a version bump rather than an edit in place.

## Fields

Every check carries:

| Field | Meaning |
| --- | --- |
| `id` | Stable identifier, e.g. `LA001`. Never reused. |
| `name` | Short human label. |
| `strategy` | `static` (AST pass), `llm` (model reasoning pass), or `both`. |
| `severity` | Default severity. Individual findings may not deviate from it. |
| `mechanism` | Plain-English explanation of *why* the pattern biases results. Rendered into every report. |
| `fix` | The canonical remedy. Checks marked ✎ below also emit a unified diff. |

## Severity

| Severity | Meaning |
| --- | --- |
| `critical` | The result is invalid. The reported performance cannot be attained, at any cost level, by any implementation of the described strategy. |
| `high` | The result is materially overstated and the correction is likely to be large relative to the claimed edge. |
| `medium` | The result is biased in a known direction; magnitude depends on parameters not visible in the code. |
| `low` | Questionable practice with a bounded effect, or a latent hazard as the code grows. |

## Confidence

| Confidence | Assigned when |
| --- | --- |
| `high` | A static rule fired *and* the model pass independently confirmed it, or a static rule matched an unambiguous mechanical signature (`shift(-1)`, `center=True`). |
| `medium` | A single pass fired on a signature that admits benign readings. |
| `low` | Heuristic name matching, or a model finding with a citation but no static corroboration. |

---

## 1. LOOKAHEAD

*Future information reaching a feature that is indexed at time `t`.*

The unifying test: **could a person standing at time `t`, with only what had been published
by then, have computed this number?** If not, the feature is a lookahead regardless of how
it was produced.

| ID | Name | Strategy | Severity | |
| --- | --- | --- | --- | --- |
| `LA001` | Negative shift on a feature column | static | critical | ✎ |
| `LA002` | Centered rolling window | static | critical | ✎ |
| `LA003` | Backward fill on a feature column | static | high | ✎ |
| `LA004` | Transformer fitted before the train/test split | static | high | |
| `LA005` | Full-sample normalisation statistics | static | high | |
| `LA006` | Feature timestamped after the bar it indexes | llm | high | |

**`LA001`** — `.shift(-n)` moves data backwards in time, so the value on row `t` was observed
at `t+n`. Detected on any negative literal or unary-negated argument, positional or `periods=`.
Not raised for a negative shift whose result is assigned to a column identified as the target,
which is how forward-return labels are legitimately built — that construction is instead the
concern of `CV004`.

**`LA002`** — `rolling(n, center=True)` centres the window, so roughly `n/2` of the bars
contributing to the statistic at `t` occur after `t`. Detected on an explicit `center=True`
keyword. `center=False` and the omitted default are trailing and must not fire.

**`LA003`** — `bfill()`, `fillna(method="bfill")` and `fillna(method="backfill")` propagate a
later observation into earlier missing rows. Fires when the fill result is assigned to a
column that is not the target. `ffill` must never fire.

**`LA004`** — `.fit()` or `.fit_transform()` on a transformer at a source line preceding the
first split call in the same scope. The transform learns the test period's location and scale.
Suppressed when the transformer is constructed inside a `Pipeline`, which sklearn refits per
fold.

**`LA005`** — normalisation of the form `(df[c] - df[c].mean()) / df[c].std()`, where the
statistics are taken over the whole column. Also fires on whole-column `min()`/`max()` range
scaling. Expanding and rolling statistics must not fire.

**`LA006`** — model pass. Covers alignment that static analysis cannot resolve: publication
lag on fundamentals, resampling boundaries, joins whose keys carry an effective date rather
than an availability date.

## 2. LABEL-LEAK

*The target, or something derived from it, is reachable from the feature matrix.*

Distinguished from LOOKAHEAD by direction: lookahead is future market data entering a
feature; label leak is the answer itself entering a feature. They frequently co-occur.

| ID | Name | Strategy | Severity | |
| --- | --- | --- | --- | --- |
| `LL001` | Target column reachable from the feature matrix | static | critical | ✎ |
| `LL002` | Feature engineered from the target column | static | critical | |
| `LL003` | Transformer fitted on a frame that still contains the target | static | high | |
| `LL004` | No gap between the feature bar and the label horizon | llm | high | |

The target column is resolved from, in order of preference: the second positional argument to
`.fit(X, y)`; assignment to a variable named `y`, `target` or `label`; assignment to a column
whose name matches `^(y|target|label|fwd_|forward_|future_)`. Derived names are matched by
shared root, so a target `fwd_ret_5d` also protects `fwd_ret_5d_ma`.

**`LL001`** — the target's name, or a name sharing its root, survives into the feature matrix.
Detected on both construction styles: an explicit column list (`X = df[[...]]`) and
subtractive construction (`X = df.drop(columns=[...])`).

**`LL002`** — an assignment to a feature column whose right-hand side references the target
column. Lagging or smoothing does not clear this: the transformed series still carries the
label's signal.

**`LL003`** — `scaler.fit(df)` where `df` is known to contain the target column. Label
information reaches every feature through the shared covariance structure even if the column
is dropped afterwards.

**`LL004`** — model pass. Whether features and label need a gap between them depends on how
each bar is assembled and when it becomes observable.

## 3. SURVIVORSHIP

*The sample was selected using knowledge of what happened to its members.*

| ID | Name | Strategy | Severity | |
| --- | --- | --- | --- | --- |
| `SV001` | Universe drawn from current constituents | static | critical | |
| `SV002` | Delisted or inactive names filtered out of history | static | high | |
| `SV003` | Universe selected on a whole-sample outcome | static | critical | |
| `SV004` | Static universe table joined across history | llm | medium | |

**`SV001`** — a universe sourced from a name matching `current|latest|today|as_of_today|
_now$|active_constituents`, or a hard-coded ticker list applied to a historical panel. Name
matching is a heuristic and these findings are emitted at `medium` confidence unless the model
pass confirms.

**`SV002`** — a boolean mask on a status column (`is_active`, `listed`, `delisted`,
`still_trading`) applied to the historical panel. Cutting exactly the names with the worst
outcomes truncates the left tail of the return distribution.

**`SV003`** — a mask built from a statistic computed over a name's entire history — a total
return, a final price, a cumulative flag — used to select the universe. Detected on
`groupby(...).transform(...)` and whole-series aggregates appearing inside a subscript mask,
and on comparisons against columns whose names match `total_|cum_|final_|max_|lifetime_`.

**`SV004`** — model pass, for universe or metadata tables with no date dimension.

## 4. UNPURGED-CV

*The validation scheme lets information cross the train/test boundary.*

Reference standard: purged and embargoed walk-forward cross-validation as specified in
López de Prado, *Advances in Financial Machine Learning*, ch. 7. Two requirements, both
checked: training observations whose label horizon overlaps the test window must be **purged**,
and a further **embargo** must be applied after the test window because serial correlation
runs in both directions.

| ID | Name | Strategy | Severity | |
| --- | --- | --- | --- | --- |
| `CV001` | Shuffled K-fold on time-ordered data | static | critical | ✎ |
| `CV002` | `train_test_split` with shuffling on a time series | static | critical | ✎ |
| `CV003` | Walk-forward split with no purge gap or embargo | static | high | ✎ |
| `CV004` | Label horizon exceeds the purge gap | static | high | |
| `CV005` | Grouped split that ignores the time axis | llm | medium | |

All checks in this category are gated on the file being **time-ordered**, established by any
of: a datetime conversion or index, a date-like column literal, a `resample`, a `freq=`
argument, or a `sort_values` on a date-like key. This gate is what keeps shuffled CV on
genuinely cross-sectional data from firing.

**`CV001`** — `KFold`, `StratifiedKFold`, `ShuffleSplit`, `StratifiedShuffleSplit`,
`GroupShuffleSplit`, or an integer `cv=` passed to `cross_val_score` / `cross_validate` /
`GridSearchCV`, in a time-ordered file. `KFold(shuffle=False)` still fires: unshuffled folds
are contiguous but every fold except the last trains on data after its test window.

**`CV002`** — `train_test_split` on a time-ordered frame. Fires unless `shuffle=False` is
passed explicitly, because the parameter defaults to `True`.

**`CV003`** — `TimeSeriesSplit` constructed with no `gap=` argument, or `gap=0`, and no purge
or embargo implemented elsewhere in the file (established by an identifier or string matching
`purge|embargo`).

**`CV004`** — the file builds a label with a forward horizon of `h` bars (from the largest
negative shift, or a `horizon`/`lookahead` constant) and passes a positive `gap` smaller than
`h`. The `h - gap` overlapping bars carry outcomes realised inside the test window. The case of
no gap at all belongs to `CV003`; the two are mutually exclusive by construction, so a single
splitter never raises both.

**`CV005`** — model pass, for entity-grouped splits that leave the calendar axis unconstrained.

## 5. COST-FANTASY

*The backtest charges less than trading would.*

| ID | Name | Strategy | Severity | |
| --- | --- | --- | --- | --- |
| `CF001` | PnL arithmetic with no cost term | static | critical | |
| `CF002` | Fills at close or mid with no spread or slippage | static | high | |
| `CF003` | Cost parameter pinned to zero | static | high | ✎ |
| `CF004` | Short positions with no borrow cost | static | medium | |
| `CF005` | Unlimited liquidity assumed | llm | medium | |

A **cost term** is any identifier, column name or string literal matching
`cost|fee|commission|slippage|spread|bps|borrow|impact|turnover_cost`. Its presence anywhere
in the enclosing scope suppresses `CF001`, `CF002` and `CF004` — the tool's job is to catch
research that forgot costs, not to grade a cost model it cannot evaluate statically.

**`CF001`** — an assignment to a PnL-like target (`pnl`, `ret_net`, `returns`, `profit`,
`equity`, `strategy_ret`) whose right-hand side multiplies a position-like term (`position`,
`signal`, `weight`, `holding`, `exposure`) by a return, with no cost term anywhere in scope.

**`CF002`** — an assignment to a fill-price-like target (`fill`, `fill_price`, `exec_price`,
`entry_price`, `trade_price`) from a close or mid reference with no spread or slippage
adjustment in scope.

**`CF003`** — a cost-like parameter assigned a literal zero. Distinct from `CF001` in that the
accounting is present and correct; only the number is a placeholder, which is why the fix is a
one-line mechanical patch.

**`CF004`** — short exposure created explicitly (a negative position literal, `.clip(upper=0)`,
`np.sign`-based long/short construction, or an identifier matching `short`) with no borrow or
loan cost term in scope.

**`CF005`** — model pass, for participation and market impact.

## 6. SWEEP-OVERFIT

*The reported number is the maximum of many trials, presented as if it were one.*

| ID | Name | Strategy | Severity | |
| --- | --- | --- | --- | --- |
| `SO001` | Parameter grid scored on the test set and maximised | static | critical | |
| `SO002` | Search `best_score_` reported as out-of-sample | static | high | |
| `SO003` | Test set used for early stopping | static | critical | |
| `SO004` | Large sweep with no multiple-testing correction | static | medium | |
| `SO005` | Iterative refinement against a single holdout | llm | medium | |

A **correction term** is an identifier or string matching `deflated|dsr|bonferroni|
holm|multiple_test|family_wise|fdr|white_reality|spa_test`.

**`SO001`** — a loop over a parameter grid (`itertools.product`, nested `for` over literal
sequences, or iteration over a grid variable) whose body evaluates a metric against a
test-named variable, with a `max`, `sorted`, `idxmax` or `argmax` applied to the collected
results afterwards.

**`SO002`** — `best_score_` or `best_estimator_.score(X, y)` read on data that also fed the
search, with no separately-held evaluation set in the file.

**`SO003`** — the test set appears in `eval_set=`, `validation_data=`, or alongside
`early_stopping_rounds` / `callbacks` in a `.fit()` call.

**`SO004`** — the product of literal grid lengths exceeds **25** trials with no correction term
in the file. The threshold is deliberately low: the bias is already material at that size, and
the finding is `medium` severity precisely because the correct threshold depends on how
correlated the trials are.

**`SO005`** — model pass. Not visible in a single snapshot of the code; requires reading
history or commentary.

---

## What this taxonomy does not cover

Stated explicitly, because an audit tool that implies completeness is worse than none:

- **Data errors** — bad splits, stale ticks, wrong corporate actions, misaligned currencies.
- **Regime dependence** — a strategy that worked in one regime and is being tested in it.
- **Capacity** — beyond the liquidity heuristic in `CF005`.
- **Execution modelling** — queue position, partial fills, adverse selection at the touch.
- **Statistical inference** — beyond the multiple-testing check in `SO004`.

A clean Hindsight report means none of the 29 patterns above were found. It does not mean the
research is correct.
