# Spec + Plan: Tenure-Based Cohort Analysis — Retention Curves & Churn Trend

> Location note: this file lives at `.claude/specs/` per CLAUDE.md §4's repo
> tree, which marks `.claude` as "claude.md and also Specs folder" — consistent
> with `01-clean-dataset-eda-notebook-kpi-charts.md` and
> `02-churn-patterns.md`. Spec and plan are combined in one file for the same
> reason as those two.
>
> Scope note: like `02`, this is **not** a new row in CLAUDE.md §14's phase
> tracker. It is a further Phase 1 EDA deep-dive — cohorting the already-clean
> data by `tenure` and visualizing retention/churn trend — and touches no
> model, pipeline, or API code.
>
> Methodology note: the user was asked how to define "retention curve" given
> this dataset is a single cross-sectional snapshot (one `tenure` + one
> `Churn` value per customer), not repeated time-series observations. They
> chose the **empirical retention curve** (`% of customers with tenure ≥ m`,
> i.e. 1 − ECDF of tenure) over a true Kaplan–Meier survival estimate, to
> avoid adding `lifelines` (not in CLAUDE.md §3's approved stack) for a
> Phase-1-level EDA feature. This spec documents that curve's known bias
> (no censoring correction) explicitly rather than presenting it as
> statistically rigorous.

---

## PART 1 — SPEC

### Feature

A new `src/data/cohorts.py` module that bins customers into the four
requested tenure cohorts (0–12, 12–24, 24–48, 48–72 months), reports churn
rate / retention rate per cohort, computes an empirical retention curve
(1 − ECDF of tenure) at monthly granularity, and renders both as charts —
consumed by a new `notebooks/03_cohort_analysis.ipynb` that narrates the
findings.

### Problem / motivation

CLAUDE.md §6's signal table already asserts "Churn concentrated in first 12
months" as a pattern the model must recover, but nothing in the repo
currently **quantifies or visualizes** that by tenure band. `src/data/eda.py`
has `churn_rate_by_segment()`, but it only operates on existing categorical
columns (`Contract`, `InternetService`, etc.) — `tenure` is numeric and has no
binning logic anywhere. Verified against the current cleaned data
(`load_clean_data()`, 7,043 rows):

1. **The 4 requested tenure bands partition the data with no gaps and no
   overflow.** `tenure` ranges from 0 to exactly 72 (no customer exceeds 72
   months), so bins `(0,12], (12,24], (24,48], (48,72]` (with 0 folded into
   the first band) assign every one of the 7,043 rows to exactly one cohort:
   0–12 → 2,186 customers, 12–24 → 1,024, 24–48 → 1,594, 48–72 → 2,239.
2. **Churn rate drops monotonically and steeply across the four bands**:
   47.44% (0–12) → 28.71% (12–24) → 20.39% (24–48) → 9.51% (48–72). This is
   the concrete, checkable version of the CLAUDE.md §6 tenure signal — new
   customers are ~5x more likely to churn than customers past 4 years.
3. **No retention curve of any kind exists in the codebase.** The closest
   existing artifact, `tenure_distribution.png`, is a churn-split histogram,
   not a cumulative retention view — it can't answer "what fraction of
   customers make it to month 24?" directly.

### Goals / non-goals

**Goals**
- Add `src/data/cohorts.py` with tenure-cohort assignment, per-cohort
  churn/retention summary, and an empirical retention curve — all as pure,
  testable functions following `src/data/eda.py`'s existing pattern (analysis
  function returns a DataFrame; a parallel `plot_*` function saves a figure
  via the same `_save_fig` helper).
- Add two new charts: churn rate **and** retention rate by tenure cohort (in
  cohort order, not sorted descending — the trend across bands is the point),
  and the empirical retention curve (retention % vs. tenure month, 0–72).
- Create `notebooks/03_cohort_analysis.ipynb` narrating: cohort assignment →
  per-cohort table → churn-trend chart → retention-curve chart → key findings,
  following `01_eda.ipynb`/`02_churn_patterns.ipynb`'s bootstrap-cell pattern.
- Add `python -m src.data.cohorts` as a standalone runnable entry point
  (mirrors `python -m src.data.eda`), documented in CLAUDE.md §5.
- Add pytest coverage in `tests/test_cohorts.py`.

**Non-goals**
- No true Kaplan–Meier / censoring-corrected survival curve — explicitly
  deferred per the user's methodology decision above; would require adding
  `lifelines`, which is out of CLAUDE.md §3's stack.
- No feature-engineering decision (e.g., adding `TenureCohort` as a model
  input) — that's Phase 2's call in `src/features/`, and CLAUDE.md's leakage
  guardrail applies once it's made there, not here.
- No cross-tab of cohort × other segments (e.g., cohort × Contract) — single-
  dimension cohorting only, matching the literal request.
- No change to `src/data/eda.py`, `generate_all_figures()`, `clean_data()`,
  or any existing test/figure.
- No new phase-tracker row in CLAUDE.md §14.

### User stories

- As the **engineer (Priyabrata)**, I want tenure cohorts and their churn
  rates computed and charted with reusable functions (not one-off notebook
  math) so the same logic can back a future dashboard "retention by cohort"
  view in Phase 5 without rewriting it.
- As a **retention manager** reviewing the notebook, I want a single chart
  showing churn risk falling from ~47% to ~9.5% across tenure bands, so I can
  see exactly where retention investment (onboarding, first-year offers)
  matters most.
- As a **recruiter/reviewer**, I want the retention curve's limitation (no
  censoring correction) stated plainly next to the chart, so the analysis
  reads as rigorous rather than overclaiming statistical precision it doesn't
  have.

### Functional requirements

1. `src/data/cohorts.py` MUST define `TENURE_BIN_EDGES = [0, 12, 24, 48, 72]`
   and `TENURE_COHORT_LABELS = ["0-12", "12-24", "24-48", "48-72"]` as named
   constants (no magic numbers inlined elsewhere, per CLAUDE.md §8).
2. MUST gain `assign_tenure_cohort(df: pd.DataFrame) -> pd.Series` returning
   an ordered `pd.Categorical` (categories in `TENURE_COHORT_LABELS` order)
   built via `pd.cut(df["tenure"], bins=[-1, 12, 24, 48, 72], labels=TENURE_COHORT_LABELS)`
   — the `-1` lower bound so `tenure == 0` falls in `"0-12"` rather than
   producing `NaN`. Every row MUST receive a non-null cohort for `tenure` in
   `[0, 72]`; a row with `tenure > 72` (not present today, but a future
   re-download could have one) falls outside all bins and MUST come back as
   `NaN`, not silently clipped into `"48-72"` — documented as a known edge
   case (see below), not silently handled.
3. MUST gain `cohort_summary(df: pd.DataFrame) -> pd.DataFrame` with one row
   per cohort **in `TENURE_COHORT_LABELS` order** (not sorted by rate), columns
   `cohort`, `customers`, `churn_rate` (%, 2dp), `retention_rate` (%, 2dp,
   `= 100 - churn_rate`). Must not drop or reorder cohorts even if one is
   empty (`observed=False` semantics — zero-count cohort still gets a row).
4. MUST gain
   `empirical_retention_curve(df: pd.DataFrame, max_month: int = 72) -> pd.DataFrame`
   with one row per integer month `0..max_month`, columns `month`,
   `retention_pct` (`= (tenure >= month).mean() * 100`), `n_at_risk`
   (count of customers with `tenure >= month`). MUST be monotonically
   non-increasing in `retention_pct` by construction (verified on current
   data: 100.0 at month 0 down to 5.14 at month 72) and MUST return `0.0`
   (not `NaN`/error) for any `month` beyond the data's actual max tenure.
5. MUST gain `plot_cohort_churn_trend(df, out_dir=FIGURES_DIR) -> Path` — a
   grouped/paired bar chart (churn_rate and retention_rate per cohort, cohort
   order preserved left-to-right) saved as
   `reports/figures/cohort_churn_trend.png`.
6. MUST gain `plot_retention_curve(df, out_dir=FIGURES_DIR) -> Path` — a line
   chart of `retention_pct` vs. `month` (0–72), with vertical reference lines
   at the three interior cohort boundaries (12, 24, 48), saved as
   `reports/figures/retention_curve.png`. The chart or its notebook cell MUST
   carry a one-line caption noting it is not censoring-corrected (see
   methodology note above).
7. MUST gain `generate_cohort_figures(df, out_dir=FIGURES_DIR) -> list[Path]`
   and a `main()` calling it via `load_clean_data()`, runnable as
   `python -m src.data.cohorts` — mirrors `src/data/eda.py`'s
   `generate_all_figures()`/`main()` pattern but stays a separate entry point
   (this module is not wired into `eda.generate_all_figures()`, per
   Non-goals — no cross-module coupling for a reporting-only addition).
8. `notebooks/03_cohort_analysis.ipynb` MUST be created, following
   `01_eda.ipynb`/`02_churn_patterns.ipynb`'s bootstrap-cell pattern (sys.path
   setup, imports from `src.data.cohorts`/`src.data.load_data`, `IPython.display.Image`
   for saved PNGs — no inline analysis logic). Sections, in order: cohort
   assignment + `cohort_summary()` table; churn-trend chart + narration of the
   47.44%→9.51% drop; retention-curve chart + the censoring-bias caveat; key
   findings summary matching the closing-cell style of `01`/`02`.
9. `tests/test_cohorts.py` MUST cover: `assign_tenure_cohort` produces the
   verified counts (2186/1024/1594/2239) on `clean_df` with zero nulls;
   `cohort_summary` rows are in cohort order with `retention_rate == 100 -
   churn_rate` and `customers` summing to `len(clean_df)`; churn rate is
   strictly decreasing across the 4 cohorts on current data (locks in the
   CLAUDE.md §6 tenure signal); `empirical_retention_curve` starts at 100.0,
   is monotonically non-increasing, and returns `0.0` for `month` beyond the
   observed max tenure.
10. None of the above may change `clean_data()`'s output, `src/data/eda.py`,
    or any existing test/figure — all current tests must keep passing
    unmodified.

### Data & model impact

None to the model or preprocessing pipeline. `TenureCohort` is not written
back into the cleaned frame returned by `load_clean_data()` — `cohorts.py`
computes it internally per function call, so nothing in `src/features/`
(Phase 2, not yet built) is affected. If a future phase wants
`TenureCohort` as a model feature, that is a separate, explicit Phase 2
decision subject to CLAUDE.md's leakage guardrail (bin edges are fixed
constants derived from `tenure` itself, not from `Churn`, so using them as a
feature would not leak the label — noted for whoever makes that call later,
not acted on here).

### ML guardrails (mandatory check)

N/A for leakage/splitting/resampling — no model path is touched; this is a
reporting/EDA module operating on the already-cleaned frame. Confirmed no
guardrail regression: `assign_tenure_cohort` bins on `tenure` only (never on
`Churn` or a churn probability); `cohort_summary`/`empirical_retention_curve`
read `Churn` only to report descriptive rates, never write it back as a
transformed feature. No train/test split occurs in this feature (not yet
applicable — same as `01`/`02`'s specs noted).

### API / UI surface

None — no FastAPI endpoint or Streamlit view. `src/data/cohorts.py` stays
notebook/report-facing, consistent with `eda.py`.

### Edge cases & failure states

- **A future Kaggle re-download has a customer with `tenure > 72`**:
  `assign_tenure_cohort` returns `NaN` for that row (Functional Requirement
  2) rather than silently folding it into `"48-72"` — a `NaN` cohort is a
  visible signal (shows as a count in `cohort_summary` totals not matching
  `len(df)`) that the bin edges need revisiting, not a silent misclassification.
- **`empirical_retention_curve` at `month=0`**: `retention_pct == 100.0` by
  definition (`tenure >= 0` is always true); this is asserted directly in the
  test suite as the curve's fixed starting point.
- **`month` beyond the observed max tenure** (e.g. `max_month=100` when data
  tops out at 72): returns `0.0` for months 73–100, not `NaN` or a raised
  error — `(tenure >= month).mean()` is well-defined and correctly evaluates
  to 0 when no row satisfies it.
- **Empty input DataFrame**: `cohort_summary` returns 4 rows (one per cohort
  label) with `customers == 0` and `churn_rate`/`retention_rate` as `NaN`
  (mean of an empty group) rather than raising — acceptable since no current
  caller passes an empty frame; not asserted by a dedicated test but the
  function must not crash the notebook if it's ever invoked mid-filter.
- **Retention curve is misread as a true survival estimate**: mitigated by
  the mandatory caption/narration in Functional Requirement 6 and 8 — the
  curve mixes still-active and already-churned customers at each month
  without a censoring correction, so e.g. the drop to 5.14% at month 72
  reflects "few customers have been on the books that long yet," not "95% of
  customers churn by month 72."

### Security notes

None — no new untrusted input, secret, network call, or dependency. All new
functions operate on the already-loaded, already-cleaned local DataFrame
using pandas/matplotlib/seaborn, all already in `requirements.txt`. No
`lifelines` or other new package added, per the user's methodology decision.

### Success criteria

- `pytest -q` passes: all existing tests (12, per `02`'s spec) + the new
  `tests/test_cohorts.py` module, all green.
- `notebooks/03_cohort_analysis.ipynb` runs top-to-bottom without error and
  renders every section named in Functional Requirement 8.
- `reports/figures/` contains `cohort_churn_trend.png` and
  `retention_curve.png` alongside the 13 existing Phase 1 figures (15 total),
  with no existing filename changed.
- `python -m src.data.cohorts` is a working, idempotent entry point producing
  both new figures, documented in CLAUDE.md §5.
- `quality-reviewer` and `security-reviewer` report no unresolved findings on
  the diff.

### Out of scope

- True Kaplan–Meier survival curve / `lifelines` dependency — explicitly
  deferred per the user's decision; revisit only if the user later wants
  censoring-corrected retention analysis.
- Wiring `TenureCohort` into `src/features/` as a model input — a Phase 2
  decision, not made here.
- Cross-tabbing cohort with other segment columns (Contract, InternetService,
  etc.) — single-dimension cohorting only.
- A new row in CLAUDE.md §14's phase tracker.
- Any change to `01_eda.ipynb` or `02_churn_patterns.ipynb`.

---

## PART 2 — PLAN

### Approach

Add `src/data/cohorts.py` as a self-contained sibling to `src/data/eda.py`,
reusing its exact function-pair pattern (`compute_* -> DataFrame`,
`plot_* -> Path` via the same `_save_fig` helper imported from `eda.py`) so
the new module reads as idiomatic to anyone who's already read `eda.py`. Keep
it a separate module (not folded into `eda.py`) since cohorting is a distinct
analytical concern with its own constants and its own `main()`, matching
CLAUDE.md §5's "expose runnable scripts as `python -m src.<module>`"
convention independently of the existing EDA entry point.

**Alternative rejected:** adding cohort functions directly into
`src/data/eda.py` and wiring them into `generate_all_figures()`. Rejected
because it would grow an already-sizable module with a conceptually distinct
concern (cohort binning + retention curve vs. general distribution/segment
EDA) and couple two entry points that should be able to run independently;
a separate module with its own `main()` keeps both small and keeps this
feature's diff isolated to new files plus two documentation edits.

### Task breakdown

- [ ] **1. Create `src/data/cohorts.py`** — constants
      (`TENURE_BIN_EDGES`, `TENURE_COHORT_LABELS`), `assign_tenure_cohort`,
      `cohort_summary`, `empirical_retention_curve`, `plot_cohort_churn_trend`,
      `plot_retention_curve`, `generate_cohort_figures`, `main()`. Import
      `_save_fig`, `FIGURES_DIR`, `CHURN_PALETTE` from `src.data.eda` to avoid
      duplicating the figure-saving helper.
- [ ] **2. Run `python -m src.data.cohorts`** to generate
      `reports/figures/cohort_churn_trend.png` and `retention_curve.png`;
      confirm via `git status`/`git diff --stat` that no existing figure file
      changed.
- [ ] **3. Create `notebooks/03_cohort_analysis.ipynb`** — bootstrap cell
      copied from `02_churn_patterns.ipynb`; sections per Functional
      Requirement 8.
- [ ] **4. Add `tests/test_cohorts.py`** — cover Functional Requirement 9
      (verified cohort counts, cohort-order summary invariants, monotonic
      churn-rate decrease, retention-curve boundary/monotonicity checks).
- [ ] **5. Document the new entry point** — add `python -m src.data.cohorts`
      to CLAUDE.md §5's command list, next to the existing
      `python -m src.data.eda` line.
- [ ] **6. Run the full suite** — `pytest -q`, confirm all existing + new
      tests pass.
- [ ] **7. Commit** — `src/data/cohorts.py`,
      `notebooks/03_cohort_analysis.ipynb`, `tests/test_cohorts.py`,
      `reports/figures/cohort_churn_trend.png`, `reports/figures/retention_curve.png`,
      `.claude/CLAUDE.md`, commit message
      `phase 1: tenure cohort analysis, retention curve, churn trend by band`.

### Tests to write (hand to test-writer)

- `tests/test_cohorts.py::test_assign_tenure_cohort_matches_verified_counts` —
  `assign_tenure_cohort(clean_df).value_counts()` equals
  `{"0-12": 2186, "12-24": 1024, "24-48": 1594, "48-72": 2239}`, and no nulls.
- `tests/test_cohorts.py::test_cohort_summary_order_and_retention_identity` —
  `cohort_summary(clean_df)["cohort"].tolist() == TENURE_COHORT_LABELS`
  (order preserved, not sorted); `retention_rate == 100 - churn_rate` for
  every row (within floating-point tolerance); `customers.sum() ==
  len(clean_df)`.
- `tests/test_cohorts.py::test_cohort_churn_rate_strictly_decreasing` —
  `cohort_summary(clean_df)["churn_rate"]` is strictly decreasing across the
  4 rows on current data (locks in the 47.44/28.71/20.39/9.51 pattern).
- `tests/test_cohorts.py::test_empirical_retention_curve_starts_at_100` —
  `empirical_retention_curve(clean_df).iloc[0]` has `month == 0` and
  `retention_pct == 100.0`.
- `tests/test_cohorts.py::test_empirical_retention_curve_monotonic_nonincreasing` —
  `retention_pct` never increases as `month` increases, over `0..72`.
- `tests/test_cohorts.py::test_empirical_retention_curve_beyond_max_tenure_is_zero` —
  `empirical_retention_curve(clean_df, max_month=80)` returns `0.0` for
  months `73..80`, no error.

### Quality gates

1. **test-runner** — run `pytest -q`, confirm all tests (existing + new)
   pass, fix any straightforward regression.
2. **quality-reviewer** — review `cohorts.py`'s binning logic (the `-1`
   lower-bound trick, `NaN` handling above 72), the retention-curve formula,
   the notebook's reuse-not-duplicate structure, and CLAUDE.md §8 adherence
   (named bin-edge constants, type hints, docstrings).
3. **security-reviewer** — confirm no new untrusted input path or dependency
   is introduced (there isn't one).
4. Commit only after both reviews are clean.

### Risks / rollback

- **Risk:** the retention curve's censoring bias is misread by a reviewer as
  a modeling error rather than a documented, intentional simplification.
  **Mitigation:** Functional Requirement 6/8's mandatory caption plus this
  spec's methodology note make the limitation explicit and traceable to an
  explicit user decision, not an oversight.
- **Risk:** a future Kaggle re-download shifts the tenure distribution enough
  that the strictly-decreasing churn-rate test (locked-in pattern) fails.
  **Mitigation:** intentional, matching `02`'s brittleness philosophy — it
  should fail loudly so the change gets reviewed, not silently pass on stale
  assumptions.
- **Rollback:** single commit (Task 7) covering only additive files (new
  module, new notebook, new tests, new PNGs, one CLAUDE.md doc line) — `git
  revert` is clean since nothing existing is modified in place.

### Definition of done

- All 7 tasks checked off.
- `pytest -q` green (all existing tests + 6 new `test_cohorts.py` tests).
- `notebooks/03_cohort_analysis.ipynb` executes top-to-bottom without error.
- `reports/figures/` has 15 PNGs (13 existing + 2 new), no existing one
  altered.
- CLAUDE.md §5 documents `python -m src.data.cohorts`.
- `quality-reviewer` and `security-reviewer` report no unresolved findings.
- All Success Criteria in Part 1 are met.
