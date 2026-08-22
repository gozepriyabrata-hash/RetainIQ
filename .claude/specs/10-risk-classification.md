# Spec + Plan: Risk Classification — Critical/High/Medium/Low Churn Risk Tiers

> Location note: this file lives at `.claude/specs/` per CLAUDE.md §4's repo
> tree, consistent with `01`–`09`. Spec and plan are combined in one file for
> the same reason. Numbered `10` (next after `09-probability-scoring.md`).
>
> Scope note: this is the **risk-tier half of CLAUDE.md §14 Phase 4** ("Risk
> tiers + Next-Best-Action engine, optional LLM"), not the whole phase — in
> the same spirit as how `02`–`07` extended Phase 1 and `09` bridged Phase 2
> to Phase 5 without claiming a full phase row. This spec turns the
> calibrated churn probability `09` already produces into the four named
> tiers CLAUDE.md §7 defines (Critical/High/Medium/Low) and stops there. It
> is deliberately **not** the Next-Best-Action engine (recommending what to
> do about a tier — a separate, larger spec), **not** an LLM insight
> generator, and **not** Phase 5 (no FastAPI route, no Streamlit view — same
> "added but wired to nothing" precedent `08` and `09` both set).
>
> Interpretation note — boundary inclusivity: CLAUDE.md §7 states "Critical
> > 70%, High 50–70%, Medium 30–50%, Low < 30%." Read literally, the 70%
> and 50% endpoints appear in two adjacent ranges each, which would double-
> count a customer scored at exactly 70% or 50% if taken naively. Resolved
> here, uniquely determined by the two ranges CLAUDE.md gives a strict
> inequality (`> 70%`, `< 30%`): a customer at exactly 70% cannot be
> Critical (`> 70%` excludes it) so falls in High; a customer at exactly
> 30% cannot be Low (`< 30%` excludes it) so falls in Medium. This forces
> **Low = [0, 30%), Medium = [30%, 50%), High = [50%, 70%], Critical =
> (70%, 100%]** — four disjoint, gap-free ranges with no ambiguous
> probability. Flagged per the create-spec workflow's instruction to state
> an assumption rather than silently pick one.
>
> Interpretation note — probability scale: `src/models/scoring.py`'s
> `score_customers` already returns both `churn_probability` (0–1 float)
> and `churn_probability_pct` (0–100 float). This spec's threshold
> constants are expressed on the **0–1 scale** (`0.30`/`0.50`/`0.70`),
> matching `src/models/evaluation.py`'s existing `DEFAULT_DECISION_THRESHOLD
> = 0.5` convention, and `assign_risk_tiers`/`classify_scored_customers`
> default to reading the `churn_probability` column (not the `_pct` one).
>
> Research note: verified against the real, currently-persisted calibrated
> model (`models/churn_model_calibrated.pkl`) during spec research by
> scoring all 7,043 raw customers with `scoring.score_customers` and
> tallying the resulting tier distribution using this spec's exact
> boundary rule (`np.select` on the 0–1 `churn_probability`, matching
> Requirement 2/3 below) — not estimated. Distribution: **Low 4,446
> (63.1%), Medium 1,063 (15.1%), High 1,055 (15.0%), Critical 479 (6.8%)**.
> No customer landed exactly on a boundary (`0.30`/`0.50`/`0.70`) in this
> dataset, so the boundary-inclusivity rule doesn't move any real customer
> between tiers today — it's still specified precisely for the (untested-
> on-this-data) case where one does. Observed actual churn rate by tier
> (sanity check against CLAUDE.md §9's "if this doesn't show up, something
> is wrong" spirit, applied to tiers instead of SHAP): Low 9.1%, Medium
> 39.6%, High 60.9%, Critical 83.5% — strictly increasing with tier,
> confirming the calibrated probability the tiers are built on is behaving
> as a real risk ladder, not an arbitrary cut.

---

## PART 1 — SPEC

### Feature

A risk-tier classification layer on top of the existing calibrated churn
probability (`09-probability-scoring.md`): given a churn probability,
assign one of four named risk tiers (Critical/High/Medium/Low, thresholds
fixed by CLAUDE.md §7), with a batch DataFrame version, a summary table,
a distribution chart, and a convenience function that goes straight from
raw customer attributes to `(churn_probability, risk_tier)` — the shape
CLAUDE.md §10's future `POST /predict` response needs.

### Problem / motivation

`09` gave RetainIQ a trustworthy probability number, but a retention
manager triaging hundreds of customers can't act on a raw float queue —
CLAUDE.md §7 already commits to four named tiers (Critical/High/Medium/Low)
specifically so risk can be triaged and prioritized at a glance, and §10's
planned `/predict` response is "churn probability **+ risk tier**," not
probability alone. Nothing in the repo today turns a probability into a
tier; `src/recommend/` is still an empty package. This spec is that
missing classification step — the prerequisite both the future
Next-Best-Action engine (which will branch its recommendation by tier) and
the future dashboard/API (which will color-code and filter by tier) need
before either can be built.

### Goals / non-goals

**Goals**
- Add `src/recommend/risk_tiers.py`: named threshold constants, a scalar
  classifier, a batch DataFrame classifier, a raw-customer-to-tier
  convenience wrapper around `09`'s scoring functions, a tier summary
  table, and a distribution chart — mirroring the
  constant/function/summary/plot shape `src/data/kpi.py` and
  `src/models/segmentation.py` already establish.
- Add `notebooks/09_risk_classification.ipynb` following `01`–`08`'s
  bootstrap-cell pattern, including the observed-churn-rate-by-tier sanity
  check from the Research note above.
- Add `tests/test_risk_tiers.py`.

**Non-goals**
- No Next-Best-Action engine, no recommended actions per tier, no LLM
  insight generation — a larger, separate spec. `src/recommend/` gains
  only `risk_tiers.py` here.
- No FastAPI endpoint, no Streamlit view, no model loaded at app startup —
  Phase 5, not this spec. `src/api/` and `app/` are untouched.
  `classify_scored_customers` is added as the function a future `/predict`
  route will call, exactly as `09`'s `score_customers` was added but wired
  to nothing.
- No change to the threshold values themselves, no configurability beyond
  the named constants (e.g. no per-deployment YAML override) — CLAUDE.md
  §7's four numbers are fixed, not user-tunable, in this spec.
- No change to `src/models/scoring.py`, `src/models/calibration.py`, or
  any `09` behavior — `classify_scored_customers` calls `09`'s functions
  as-is, never modifies them.
- No new third-party dependency.

### User stories

- As a **retention manager**, I want every customer sorted into
  Critical/High/Medium/Low at a glance instead of reading raw
  probabilities, so I can triage which accounts need attention first.
- As the **engineer (Priyabrata)**, I want one function
  (`classify_scored_customers`) that goes from raw customer attributes
  straight to `(churn_probability, risk_tier)`, so Phase 5's `/predict`
  endpoint is a thin wrapper around already-tested logic (CLAUDE.md §4:
  "Keep the API... thin"), not a place that reimplements tier math inline.
- As the **engineer**, I want the tier thresholds defined once as named
  constants (CLAUDE.md §8: "no magic numbers"), so `risk_tiers.py`,
  the future Next-Best-Action engine, and any dashboard filter all agree
  on exactly the same boundaries.
- As a **recruiter/reviewer**, I want the tier distribution and its
  observed-churn-rate-by-tier sanity check reported with real numbers (not
  asserted), so the classification reads as validated, not just declared.

### Functional requirements

1. `src/recommend/risk_tiers.py` MUST define `MEDIUM_THRESHOLD = 0.30`,
   `HIGH_THRESHOLD = 0.50`, `CRITICAL_THRESHOLD = 0.70` (0–1 scale, per the
   Interpretation notes above) and `RISK_TIER_LABELS = ["Low", "Medium",
   "High", "Critical"]` (ascending-severity order, used as the category
   order for the ordered `pandas.Categorical` results below) — no
   threshold value duplicated or re-derived elsewhere in this module.
2. MUST gain `classify_risk_tier(probability: float) -> str`: returns
   `"Low"` if `probability < MEDIUM_THRESHOLD`; `"Medium"` if
   `MEDIUM_THRESHOLD <= probability < HIGH_THRESHOLD`; `"High"` if
   `HIGH_THRESHOLD <= probability <= CRITICAL_THRESHOLD`; `"Critical"` if
   `probability > CRITICAL_THRESHOLD`. Raises `ValueError` (naming the bad
   value) if `probability` is `NaN` or outside `[0, 1]` — never silently
   clips or coerces out-of-range input.
3. MUST gain `assign_risk_tiers(df: pd.DataFrame, probability_column: str
   = "churn_probability") -> pd.DataFrame`: returns a copy of `df` with a
   new `"risk_tier"` column, dtype an **ordered** `pandas.Categorical`
   over `RISK_TIER_LABELS`. Raises `ValueError` naming
   `probability_column` if it is absent from `df`. Vectorized (e.g.
   `np.select`) using the identical boundary logic as
   `classify_risk_tier` — directly tested for row-by-row agreement between
   the two, so the batch and scalar paths can never silently diverge.
   Empty `df` in → empty `df` out (0 rows, `"risk_tier"` column present),
   not an error.
4. MUST gain `classify_scored_customers(raw_df: pd.DataFrame, pipeline=None,
   use_calibrated: bool = True) -> pd.DataFrame`: `scoring.score_customers(
   raw_df, pipeline, use_calibrated)` (Requirement 4 reuses `09`'s function
   unmodified) piped straight into `assign_risk_tiers` — the full
   raw-customer-in, `(customerID, churn_probability, churn_probability_pct,
   risk_tier)`-out convenience path a future `/predict` endpoint calls
   directly. Every `09` edge case (empty input, missing required column,
   missing calibrated artifact) propagates unchanged since this is a thin
   composition, not a reimplementation.
5. MUST gain `risk_tier_summary(df: pd.DataFrame, probability_column: str
   = "churn_probability") -> pd.DataFrame`: one row per tier in
   `["Critical", "High", "Medium", "Low"]` order (most-severe first,
   matching a triage worklist's natural reading order — the reverse of
   `RISK_TIER_LABELS`'s ascending storage order), columns `risk_tier`,
   `count`, `pct` (share of `len(df)`, 2dp, `0.0` not `NaN` on an empty
   `df`). Calls `assign_risk_tiers` internally rather than re-deriving
   tier assignment.
6. MUST gain `RISK_TIER_PALETTE = {"Low": <green>, "Medium": <yellow>,
   "High": <orange>, "Critical": <red>}` and
   `plot_risk_tier_distribution(df: pd.DataFrame, probability_column: str
   = "churn_probability") -> go.Figure`: a bar chart of `risk_tier_summary`'s
   counts in Critical→Low order, colored by `RISK_TIER_PALETTE`, following
   `kpi.py`'s `plot_mrr_breakdown` shape (a live-widget figure, not saved
   to `reports/figures/`).
7. `notebooks/09_risk_classification.ipynb` MUST follow `01`–`08`'s
   bootstrap-cell pattern. Sections, in order: problem framing (CLAUDE.md
   §7's four tiers, cite the boundary-inclusivity interpretation) → score
   every customer via `scoring.score_customers` → `assign_risk_tiers` +
   `risk_tier_summary` table → `plot_risk_tier_distribution` → sanity
   check: observed actual `Churn` rate per tier, must be strictly
   increasing Low→Critical (cites the verified Research-note numbers) →
   a handful of example customers with their tier via
   `classify_scored_customers` → key findings closing cell.
8. `tests/test_risk_tiers.py` MUST cover the Plan's "Tests to write"
   section in full.
9. None of the above may change `src/models/scoring.py`,
   `src/models/calibration.py`, `src/models/train.py`,
   `src/data/load_data.py`, `src/data/eda.py`, `src/data/cohorts.py`,
   `src/models/segmentation.py`, `src/data/lifecycle.py`,
   `src/data/kpi.py`, `src/explain/driver_analysis.py`, `src/api/`,
   `app/`, or any existing test/figure/notebook — all current tests must
   keep passing unmodified.

### Data & model impact

No new model, no new training feature, no schema change to any existing
artifact. `risk_tier` is a **derived, post-hoc label computed purely from
`churn_probability`** — a model *output*, not an input. Nothing in this
spec writes `risk_tier` (or `churn_probability`) back into
`load_clean_data()`'s output or any training DataFrame, and no function
here is reachable from `src/features/preprocessing.py` or `src/models/train.py`.

### ML guardrails (mandatory check)

- **No target/probability leakage:** `risk_tier` is derived exclusively
  from `churn_probability`, itself already a model output (never the raw
  `Churn` label). Golden rule 1 ("never let churn probability, or anything
  derived from it, become a model input") is upheld structurally — none of
  `risk_tiers.py`'s functions write into any DataFrame that
  `src/features/preprocessing.py` or `src/models/train.py` consumes; this
  spec's outputs flow one-way, toward reporting/serving, never back toward
  training.
- **Honest-AUC guard is unaffected:** this spec adds no new model and
  calls no training/evaluation code path — N/A beyond the check above.
- **Fitting/splitting/SMOTE:** N/A — no model is fit, split, or resampled
  anywhere in this spec; `risk_tiers.py` has zero scikit-learn/imbalanced-
  learn dependency, matching `evaluation.py`'s "dependency-light" style.
- **Imbalance/metric reporting:** N/A for classification metrics (this
  spec adds no classifier), but the tier distribution itself is reported
  as **counts and percentages** (Requirement 5), never a single aggregate
  number, and the notebook's sanity check reports the **observed churn
  rate per tier** (never accuracy) as the validation signal.
- **Reproducibility:** N/A — no randomness anywhere in this spec (pure
  threshold comparisons); `classify_risk_tier`/`assign_risk_tiers` are
  deterministic functions of their input by construction, needing no
  `random_state`.

### API / UI surface

None shipped. `classify_scored_customers` (Requirement 4) is added as the
exact function a future Phase 5 `/predict` route will call — `raw_df` →
`DataFrame` with `churn_probability` + `risk_tier` out, now the complete
shape CLAUDE.md §10's `POST /predict` response needs. No FastAPI route or
Streamlit view is wired up; `src/api/` and `app/` are untouched.

### Edge cases & failure states

- **Probability exactly on a boundary** (`0.30`, `0.50`, `0.70`): resolved
  deterministically per Requirement 2's inclusive/exclusive rule — directly
  tested at all three boundaries plus `0.0` and `1.0`.
- **Probability out of `[0, 1]` or `NaN`:** `classify_risk_tier` raises
  `ValueError` — directly tested. Unreachable via `classify_scored_customers`
  in normal operation since `09`'s `score_customers` already clips/rounds
  into range, but `classify_risk_tier`/`assign_risk_tiers` are still
  defensive on their own for any direct caller.
- **`probability_column` missing from `df`:** `assign_risk_tiers` raises
  `ValueError` naming the column — directly tested.
- **Empty `df` (0 rows):** `assign_risk_tiers` returns an empty `DataFrame`
  with the `risk_tier` column present; `risk_tier_summary` returns all four
  tiers with `count=0`, `pct=0.0` (not `NaN`) — directly tested.
- **`classify_scored_customers` inherits every `09` edge case** (empty
  `raw_df`, missing required feature column, missing calibrated model
  artifact) unchanged, since it is a thin composition over `09`'s
  functions — directly tested that these propagate rather than being
  swallowed or altered.

### Security notes

- **No new dependency**, no new untrusted-input surface beyond what `09`
  already accepts and validates: `assign_risk_tiers`/`risk_tier_summary`/
  `plot_risk_tier_distribution` operate on an already-numeric probability
  column (typically `09`'s own validated output); `classify_scored_customers`'s
  `raw_df` is the same untrusted-customer-row surface `09`'s
  `score_customers` already documents and mitigates (missing-column
  validation, safe numeric coercion, no dynamic code execution) —
  unchanged here, just reused.
- No secrets, no new environment variables, no network call, no file I/O
  (this module never reads/writes `models/` or `reports/` — figures are
  live dashboard-style objects returned in memory, matching `kpi.py`'s
  `plot_mrr_breakdown`).

### Success criteria

- `pytest -q` passes: all existing tests + `test_risk_tiers.py`, all
  green.
- `classify_risk_tier`/`assign_risk_tiers` correctly classify a hand-built
  set of probabilities spanning every tier and all three boundaries.
- `risk_tier_summary` on the full scored dataset reproduces the verified
  Research-note distribution (Low 4,446 / Medium 1,063 / High 1,055 /
  Critical 479 out of 7,043).
- The notebook's observed-churn-rate-by-tier sanity check is strictly
  increasing Low→Critical on real data (verified: 9.1% → 39.6% → 60.9% →
  83.5%).
- `notebooks/09_risk_classification.ipynb` runs top-to-bottom without
  error.
- `quality-reviewer` and `security-reviewer` report no unresolved
  findings on the diff.

### Out of scope

- Next-Best-Action engine, per-tier recommended actions, optional LLM
  insight generation — the rest of CLAUDE.md §14 Phase 4, a separate spec.
- Phase 5 (FastAPI `/predict`, Streamlit dashboard tier filter/coloring),
  Phase 6 (Evidently drift, Prefect retraining, Docker).
- Configurable/tunable thresholds (env var, config file, per-deployment
  override) — CLAUDE.md §7's four numbers are fixed constants here.
- Any change to `src/models/scoring.py`, `src/models/calibration.py`, or
  any other existing module's public behavior, output, or test.

---

## PART 2 — PLAN

### Approach

One new module (`src/recommend/risk_tiers.py`) with a scalar classifier as
the single source of truth for the boundary rule, a vectorized batch
version tested to agree with it row-for-row, and a composition function
that chains `09`'s scoring output straight into tier assignment —
mirroring `kpi.py`/`segmentation.py`'s established
constant/function/summary/plot shape rather than inventing a new module
pattern.

**Alternative rejected:** implement tier assignment with a single
`pd.cut(probability, bins=[...], labels=RISK_TIER_LABELS)` call instead of
`np.select`. Rejected because `pd.cut` is single-direction inclusive (all
bins either all-left-closed or all-right-closed), which cannot reproduce
CLAUDE.md §7's actual boundary rule (Critical strictly `>`, Low strictly
`<`, with High/Medium filling the inclusive-lower gaps in between) in one
call without post-hoc boundary patching — `np.select` expresses the exact
per-tier condition directly and stays trivially readable against
`classify_risk_tier`'s scalar version.

### Task breakdown

- [ ] **1. Create `src/recommend/risk_tiers.py`** — threshold constants,
      `RISK_TIER_LABELS`, `RISK_TIER_PALETTE`, `classify_risk_tier`,
      `assign_risk_tiers`, `classify_scored_customers`,
      `risk_tier_summary`, `plot_risk_tier_distribution` (Requirements
      1–6).
- [ ] **2. Create `notebooks/09_risk_classification.ipynb`** — bootstrap
      cell copied from `08_probability_calibration.ipynb`; sections per
      Functional Requirement 7.
- [ ] **3. Add `tests/test_risk_tiers.py`** — see Tests to write below.
- [ ] **4. Run the full suite** — `pytest -q`, confirm all existing + new
      tests pass.
- [ ] **5. Commit** — `src/recommend/risk_tiers.py`,
      `notebooks/09_risk_classification.ipynb`,
      `tests/test_risk_tiers.py`, commit message `phase 4: risk
      classification — Critical/High/Medium/Low tiers`. No files under
      `data/`, `models/`, or `mlruns/` are touched by this spec.

### Tests to write (hand to test-writer)

- `tests/test_risk_tiers.py::test_classify_risk_tier_interior_values` —
  one representative probability per tier (e.g. `0.10`, `0.40`, `0.60`,
  `0.90`) → `"Low"`/`"Medium"`/`"High"`/`"Critical"` respectively.
- `tests/test_risk_tiers.py::test_classify_risk_tier_boundaries` — `0.30`
  → `"Medium"`; `0.50` → `"High"`; `0.70` → `"High"`
  (not `"Critical"`); values just above/below each boundary
  (`0.2999`/`0.3001`, `0.4999`/`0.5001`, `0.6999`/`0.7001`) land on the
  expected side.
- `tests/test_risk_tiers.py::test_classify_risk_tier_extremes` — `0.0` →
  `"Low"`; `1.0` → `"Critical"`.
- `tests/test_risk_tiers.py::test_classify_risk_tier_raises_out_of_range` —
  `-0.01` and `1.01` both raise `ValueError`.
- `tests/test_risk_tiers.py::test_classify_risk_tier_raises_on_nan` —
  `float("nan")` raises `ValueError`.
- `tests/test_risk_tiers.py::test_assign_risk_tiers_matches_scalar_classifier` —
  a synthetic `DataFrame` of ~50 random probabilities (`np.random.default_rng(42)`)
  → every row's `risk_tier` equals `classify_risk_tier` applied to that
  row's probability individually (the batch/scalar agreement guard named
  in Requirement 3).
- `tests/test_risk_tiers.py::test_assign_risk_tiers_is_ordered_categorical` —
  result `df["risk_tier"].dtype` is an ordered `CategoricalDtype` with
  categories `RISK_TIER_LABELS`.
- `tests/test_risk_tiers.py::test_assign_risk_tiers_raises_on_missing_column` —
  a `df` without `churn_probability` (and no override) → `ValueError`
  naming the column.
- `tests/test_risk_tiers.py::test_assign_risk_tiers_empty_input_returns_empty_output` —
  0-row `df` in → 0-row `df` out with `risk_tier` column present.
- `tests/test_risk_tiers.py::test_risk_tier_summary_counts_and_percentages` —
  a hand-built `df` with a known tier split → `risk_tier_summary`'s
  `count`/`pct` match by hand computation, rows in Critical→Low order.
- `tests/test_risk_tiers.py::test_risk_tier_summary_empty_input_all_zero` —
  0-row `df` in → all four tiers present with `count=0`, `pct=0.0`.
- `tests/test_risk_tiers.py::test_risk_tier_summary_matches_verified_real_distribution` —
  `risk_tier_summary(scoring.score_customers(load_raw_data().drop(columns=[TARGET_COLUMN])))`
  on the real dataset reproduces the verified Research-note counts (Low
  4,446 / Medium 1,063 / High 1,055 / Critical 479) — intentionally
  brittle, matching `08`/`09`'s precedent for real-data regression guards.
- `tests/test_risk_tiers.py::test_classify_scored_customers_returns_probability_and_tier` —
  a few real raw customer rows → result has `churn_probability` and
  `risk_tier` columns, each row's tier matches
  `classify_risk_tier(row["churn_probability"])`.
- `tests/test_risk_tiers.py::test_classify_scored_customers_propagates_scoring_errors` —
  a `raw_df` missing a required feature column → the same `ValueError`
  `scoring.score_customers` would raise, unaltered.
- `tests/test_risk_tiers.py::test_plot_risk_tier_distribution_returns_figure` —
  a hand-built `df` → returns a `plotly.graph_objects.Figure` with 4 bars.

### Quality gates

1. **test-runner** — run `pytest -q`, confirm all tests (existing + new)
   pass, fix any straightforward regression.
2. **quality-reviewer** — review the boundary-inclusivity logic for exact
   agreement with the Interpretation note, the `np.select`/scalar
   agreement test, `risk_tier`'s one-way (output-only, never
   feature-bound) data flow, and CLAUDE.md §8 adherence (named constants,
   type hints, docstrings).
3. **security-reviewer** — confirm no new dependency, confirm
   `classify_scored_customers` introduces no new untrusted-input handling
   beyond what `09`'s `score_customers` already validates, confirm no
   file/network I/O was added.
4. Commit only after both reviews are clean.

### Risks / rollback

- **Risk:** a future change to `09`'s calibrated model shifts the
  probability distribution enough that
  `test_risk_tier_summary_matches_verified_real_distribution` no longer
  holds. **Mitigation:** intentional brittleness, matching `08`/`09`'s
  philosophy — this test should fail loudly on real distributional shift
  (e.g. after a retrain moves the winning model or a Kaggle re-download
  changes the data), prompting a deliberate re-verification and constant
  update rather than silently trusting stale numbers.
- **Risk:** a reviewer assumes `risk_tier` is safe to feed back into
  `src/features/preprocessing.py` as a feature (it is derived from
  probability, so doing so would violate CLAUDE.md's golden rule 1).
  **Mitigation:** stated explicitly in ML guardrails and Data & model
  impact above, and no function in this spec is reachable from any
  training code path.
- **Rollback:** single commit (Task 5) covering only new, additive files
  (`src/recommend/risk_tiers.py`, one notebook, one test file) — no
  existing file is modified. `git revert` is clean; nothing under
  `models/`/`mlruns/`/`data/` is touched.

### Definition of done

- All 5 tasks checked off.
- `pytest -q` green (all existing tests + `test_risk_tiers.py`).
- `notebooks/09_risk_classification.ipynb` executes top-to-bottom without
  error, including the strictly-increasing observed-churn-rate-by-tier
  sanity check.
- `quality-reviewer` and `security-reviewer` report no unresolved
  findings.
- All Success Criteria in Part 1 are met.
