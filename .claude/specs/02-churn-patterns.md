# Spec + Plan: Churn Patterns — Distributions, Outliers & Segment Deep-Dive

> Location note: this file lives at `.claude/specs/` per CLAUDE.md §4's repo
> tree, which marks `.claude` as "claude.md and also Specs folder" — consistent
> with `01-clean-dataset-eda-notebook-kpi-charts.md`. Spec and plan are combined
> in one file for the same reason as `01`.
>
> Scope note: this is **not** a new row in CLAUDE.md §14's phase tracker.
> Phase 1 ("Data loading, cleaning, EDA, first tests") is already marked ☑, and
> most of what this feature asks for — the 11 missing `TotalCharges` values,
> churn rate by Contract/InternetService/PaymentMethod — already exists and is
> tested in `src/data/`. This feature is a **deep-dive extension of the Phase 1
> EDA**, adding the one real analytical gap (outlier detection) and packaging
> the segment/distribution findings into a dedicated, presentation-ready
> notebook. It does not touch the model, pipeline, or API.

---

## PART 1 — SPEC

### Feature

A dedicated `notebooks/02_churn_patterns.ipynb` that deepens Phase 1's EDA:
full distribution analysis (histograms + boxplots + skew/kurtosis) for
`tenure`, `MonthlyCharges`, `TotalCharges`; a documented investigation of the
11 missing `TotalCharges` rows; new IQR-based outlier detection for the three
numeric columns; and a consolidated churn-rate-by-segment section (Contract,
InternetService, PaymentMethod) that reuses and narrates the existing Phase 1
charts rather than re-deriving them.

### Problem / motivation

Phase 1 already computes and charts nearly everything this request names —
`churn_rate_by_segment()` and `plot_churn_rate_by_segment()` cover Contract,
InternetService, and PaymentMethod; `clean_data()` already resolves the 11
blank `TotalCharges` rows; `plot_charges_distribution()` and
`plot_tenure_distribution()` already chart the three numeric columns. What's
missing, verified by direct execution against the current cleaned data:

1. **No outlier analysis exists anywhere in the codebase.** `src/data/eda.py`
   has no boxplots and no IQR/z-score outlier flagging. Direct computation
   (IQR method, `1.5×IQR` fences) on the current cleaned data finds **zero
   outliers** in `tenure`, `MonthlyCharges`, or `TotalCharges` — every value
   sits inside its `[Q1-1.5×IQR, Q3+1.5×IQR]` fence. That is itself a finding
   worth stating explicitly (bounded billing/tenure fields, no data-entry
   anomalies to clean), not something to leave unverified.
2. **No skew/kurtosis statistics are reported.** `TotalCharges` is
   right-skewed (skew ≈ 0.96, roughly the threshold analysts treat as
   "moderately skewed"); `tenure` (skew ≈ 0.24) and `MonthlyCharges`
   (skew ≈ -0.22) are close to symmetric. This is relevant to Phase 2 (a
   tree-based model doesn't need a transform, but a log-transform of
   `TotalCharges` would help logistic regression's baseline) and is presently
   undocumented anywhere in the repo.
3. **The 11 missing `TotalCharges` rows are cleaned but not investigated.**
   `clean_data()` imputes them to 0 and one test
   (`test_tenure_zero_customers_have_zero_total_charges`) confirms they all
   have `tenure == 0`, but nobody has looked at *why* — direct inspection
   shows all 11 are brand-new signups on **One-year or Two-year contracts**
   (10 Two-year, 1 One-year — none month-to-month) and **all 11 have
   `Churn == No`**. That's a real, checkable pattern (new customers on longer
   contracts, captured before their first bill, unsurprisingly still retained)
   worth documenting so a reviewer doesn't have to re-derive it.
4. **The existing segment/distribution charts are scattered across Phase 1's
   generic notebook sections**, interleaved with shape/dtype/missing-value
   checks. There is no single notebook a reviewer can open to see "churn
   patterns" as a coherent narrative.

### Goals / non-goals

**Goals**
- Add IQR-based outlier detection to `src/data/eda.py` for `tenure`,
  `MonthlyCharges`, `TotalCharges`: a reusable function returning outlier
  counts/bounds per column, plus boxplot charts by churn status.
- Add skewness/kurtosis reporting for the same three columns (reuse
  pandas' `.skew()`/`.kurt()`; no new dependency).
- Add a documented investigation of the 11 missing `TotalCharges` rows:
  tenure, Contract, and Churn breakdown, as both a notebook cell and a new
  pytest assertion.
- Create `notebooks/02_churn_patterns.ipynb` that imports the same
  `src/data/eda.py` / `src/data/load_data.py` helpers (no logic duplicated in
  the notebook, per CLAUDE.md §8/§4) and presents: distributions (with
  outlier boxplots), the missing-`TotalCharges` investigation, and
  churn-rate-by-segment for Contract / InternetService / PaymentMethod
  (reusing `churn_rate_by_segment()` and the existing PNGs — not
  regenerating charts that already exist and are correct).
- Add pytest coverage for the new outlier/skew functions.

**Non-goals**
- No outlier *treatment* (capping, winsorizing, removal) — this feature only
  flags and reports. Per user decision, treatment is deferred to whichever
  Phase 2 model/preprocessing step needs it, since tree-based models
  (XGBoost/LightGBM, CLAUDE.md §3's primary candidates) don't require it and a
  premature transform could hurt interpretability.
- No changes to `clean_data()`'s behavior, `TARGET_COLUMN` handling, or any
  existing Phase 1 test.
- No re-derivation of `churn_rate_by_segment()`/`plot_churn_rate_by_segment()`
  — these already exist, are correct, and are tested indirectly via
  `generate_all_figures()`; this feature narrates and reuses them.
- No new top-level phase in CLAUDE.md §14 — this stays inside Phase 1's
  existing ☑ scope as a documented extension, not a phase-tracker change.
- No feature-engineering decisions (binning, encoding) — that's Phase 2
  (`src/features/`).

### User stories

- As the **engineer (Priyabrata)**, I want a single notebook that tells the
  "churn patterns" story end to end (distributions → outliers → missing-data
  investigation → segment churn rates) so I don't have to reconstruct the
  narrative from Phase 1's more mechanical, checklist-style notebook.
- As a **recruiter/reviewer**, I want to see that outlier analysis was done
  and its result stated plainly (zero IQR outliers found, and why that's
  plausible for this dataset) rather than silently skipped, so I can trust the
  EDA is complete rather than incomplete-but-unstated.
- As the **engineer**, I want the 11 missing-`TotalCharges` rows explained
  (who they are, why they're blank, why imputing to 0 is defensible) with a
  test that locks the finding in place, so a future re-download of the Kaggle
  CSV that changes this pattern fails loudly instead of silently.
- As the **engineer preparing for Phase 2**, I want `TotalCharges`'s skew
  documented now so the feature-pipeline decision (log-transform for linear
  baselines, none needed for trees) has evidence behind it instead of being
  guessed at during Phase 2.

### Functional requirements

1. `src/data/eda.py` MUST gain a function
   `detect_outliers_iqr(df: pd.DataFrame, column: str, k: float = 1.5) -> pd.DataFrame`
   that computes `Q1`, `Q3`, `IQR`, the `[Q1-k·IQR, Q3+k·IQR]` fence, and
   returns the subset of rows outside the fence (empty frame if none). `k`
   defaults to the standard `1.5` Tukey multiplier; no magic number inlined
   elsewhere (CLAUDE.md §8).
2. `src/data/eda.py` MUST gain
   `summarize_outliers(df: pd.DataFrame, columns: list[str] = NUMERIC_COLUMNS) -> pd.DataFrame`
   returning one row per column with `q1`, `q3`, `iqr`, `lower_bound`,
   `upper_bound`, `n_outliers`, `pct_outliers`.
3. `src/data/eda.py` MUST gain
   `distribution_stats(df: pd.DataFrame, columns: list[str] = NUMERIC_COLUMNS) -> pd.DataFrame`
   returning `mean`, `median`, `std`, `skew`, `kurtosis`, `min`, `max` per
   column.
4. `src/data/eda.py` MUST gain `plot_outlier_boxplot(df, column, out_dir=FIGURES_DIR) -> Path`,
   a boxplot of `column` split by `Churn` (0/"No" vs 1/"Yes", consistent
   palette with the existing charts), saved as
   `reports/figures/{column.lower()}_boxplot.png`. Add one for each of
   `tenure`, `MonthlyCharges`, `TotalCharges` to `generate_all_figures()`.
5. A new notebook `notebooks/02_churn_patterns.ipynb` MUST be created,
   following `01_eda.ipynb`'s existing pattern (sys.path bootstrap cell,
   `from src.data.eda import ...` / `from src.data.load_data import ...`,
   `IPython.display.Image` to render saved PNGs — no inline logic). Sections,
   in order:
   - Distribution analysis: `distribution_stats()` table + histogram/boxplot
     for each of the 3 numeric columns.
   - The 11 missing `TotalCharges` values: reproduce the tenure/Contract/Churn
     breakdown from raw data (before imputation) and state the finding.
   - Outlier analysis: `summarize_outliers()` table + boxplots; state the
     zero-outliers finding and why it's plausible.
   - Churn rate by Contract / InternetService / PaymentMethod: reuse
     `churn_rate_by_segment()` + existing PNGs (no new segment charts).
   - Key findings summary mirroring `01_eda.ipynb`'s closing markdown cell's
     style.
6. A new test module `tests/test_eda.py` MUST cover: `detect_outliers_iqr`
   returns 0 rows for `tenure`/`MonthlyCharges`/`TotalCharges` on the current
   cleaned data (locks in today's verified finding); `summarize_outliers`
   returns one row per requested column with non-negative `n_outliers`;
   `distribution_stats` returns finite values (no NaN/inf) for all three
   columns.
7. `tests/test_data.py` MUST gain a test asserting the missing-`TotalCharges`
   investigation's finding: on raw (pre-clean) data, the 11 rows with blank
   `TotalCharges` all have `tenure == 0`, `Churn == "No"`, and
   `Contract in {"One year", "Two year"}` — locking in the pattern documented
   above so a future dataset change surfaces loudly instead of silently.
8. `generate_all_figures()` MUST remain idempotent and MUST NOT change any
   existing figure's filename or content — only append the 3 new boxplot
   files.
9. None of the above may change `clean_data()`'s output or any existing
   Phase 1 test's result — all 8 current tests in `tests/test_data.py` plus
   the new tests must pass.

### Data & model impact

None to the model or preprocessing pipeline (`src/features/` doesn't exist
yet — Phase 2). No feature engineering happens here; outlier detection and
skew stats are reporting-only, per the non-goals. The skew finding on
`TotalCharges` (≈0.96) is informational for whoever designs Phase 2's
`ColumnTransformer`, not something this feature acts on.

### ML guardrails (mandatory check)

N/A for leakage/splitting/resampling — no model path is touched. Relevant
guardrails already respected: `detect_outliers_iqr`/`summarize_outliers`/
`distribution_stats` operate on the already-cleaned frame (post `Churn`
binary-mapping) purely for reporting; they don't feed anything back into a
feature or a model. No split occurs in this feature (`random_state=42` is not
yet applicable — same as `01`'s spec noted). If Phase 2 later acts on the skew
finding (e.g. `np.log1p(TotalCharges)`), that transform MUST be fit inside the
train-only branch of the Phase 2 pipeline, not here.

### API / UI surface

None — no FastAPI endpoint or Streamlit view. `src/data/eda.py` stays
notebook/report-facing, consistent with Phase 1.

### Edge cases & failure states

- **`detect_outliers_iqr` on a column with zero variance** (`IQR == 0`, e.g.
  a hypothetical constant column): fences collapse to the constant value
  itself; every non-equal value would flag as an outlier. Not an issue for
  the three columns in scope (all have real spread), but the function should
  not divide by `IQR` (it doesn't — Tukey fences are additive), so this
  degrades gracefully rather than raising.
- **A future Kaggle re-download changes the missing-`TotalCharges` pattern**
  (e.g. some blank-charge customers churn, or appear on month-to-month
  contracts): Functional Requirement 7's new test is intentionally brittle
  and MUST fail loudly in that case, consistent with `01`'s existing
  `test_tenure_zero_customers_have_zero_total_charges` test's philosophy.
- **`summarize_outliers`/`distribution_stats` called with an empty
  `columns` list**: returns an empty frame with the right column headers,
  not an error — acceptable, no caller in this feature does this today.
- **Boxplot figures silently overwrite an existing file of the same name**:
  not a concern here since the three new filenames
  (`tenure_boxplot.png`, `monthlycharges_boxplot.png`,
  `totalcharges_boxplot.png`) don't collide with any existing Phase 1 figure
  name.

### Security notes

None — no new untrusted input, secret, network call, or dependency. All new
functions operate on the already-loaded, already-cleaned local DataFrame
using pandas/matplotlib/seaborn, all already in `requirements.txt`.

### Success criteria

- `pytest -q` passes: 8 existing `tests/test_data.py` tests + 1 new one there
  + the new `tests/test_eda.py` module, all green.
- `notebooks/02_churn_patterns.ipynb` runs top-to-bottom without error and
  renders every section named in Functional Requirement 5.
- `reports/figures/` contains the 3 new boxplot PNGs alongside the 10
  existing Phase 1 figures (13 total), with no existing filename changed.
- `python -m src.data.eda` (i.e. `generate_all_figures()`) remains a single
  idempotent entry point producing all 13 figures.
- `quality-reviewer` and `security-reviewer` report no unresolved findings on
  the diff.

### Out of scope

- Any outlier *treatment* (capping/winsorizing/removal) — flagged as a
  deferred decision for Phase 2, per the user's explicit choice.
- A new row in CLAUDE.md §14's phase tracker — this stays inside Phase 1.
- Statistical significance testing (chi-square, etc.) on the segment churn
  rates — not requested; the existing descriptive churn-rate-by-segment
  output is sufficient for this feature's scope.
- Any change to `01_eda.ipynb` — it stays as-is; `02_churn_patterns.ipynb` is
  additive, not a replacement.

---

## PART 2 — PLAN

### Approach

Extend `src/data/eda.py` with pure, testable functions
(`detect_outliers_iqr`, `summarize_outliers`, `distribution_stats`,
`plot_outlier_boxplot`) following the exact pattern already used by
`churn_rate_by_segment`/`plot_churn_rate_by_segment` (analysis function
returns a DataFrame, a parallel `plot_*` function saves a figure via the
existing `_save_fig` helper). Then build `02_churn_patterns.ipynb` as a thin
notebook that imports and narrates those functions plus the existing
segment-chart helpers — mirroring `01_eda.ipynb`'s own bootstrap-cell pattern
so both notebooks stay consistent.

**Alternative rejected:** doing the outlier/skew analysis directly in notebook
cells (pandas one-liners, no `src/data/eda.py` functions). Rejected because
CLAUDE.md §8 requires notebooks to import from `src/`, not embed logic
inline, and because un-encapsulated notebook logic can't be unit-tested
(Functional Requirement 6 needs `pytest`-callable functions).

### Task breakdown

- [ ] **1. Add outlier/distribution functions to `src/data/eda.py`** —
      `detect_outliers_iqr`, `summarize_outliers`, `distribution_stats`,
      `plot_outlier_boxplot`; wire the 3 boxplots into `generate_all_figures()`.
- [ ] **2. Run `python -m src.data.eda`** to regenerate `reports/figures/`
      and confirm the 3 new boxplot PNGs appear alongside the 10 existing
      ones, with no existing file changed (`git status` / `git diff --stat`
      on `reports/figures/` should show only additions).
- [ ] **3. Create `notebooks/02_churn_patterns.ipynb`** — bootstrap cell
      copied from `01_eda.ipynb`; sections per Functional Requirement 5
      (distributions, missing-`TotalCharges` investigation, outliers,
      segment churn rates, key findings).
- [ ] **4. Add `tests/test_eda.py`** — cover Functional Requirement 6 (zero
      outliers on current data for the 3 numeric columns; shape/finiteness
      checks on `summarize_outliers`/`distribution_stats`).
- [ ] **5. Add missing-`TotalCharges` pattern test to `tests/test_data.py`**
      — Functional Requirement 7 (tenure/Contract/Churn breakdown of the 11
      blank rows, on raw pre-clean data).
- [ ] **6. Run the full suite** — `pytest -q`, confirm all existing + new
      tests pass.
- [ ] **7. Commit** — `src/data/eda.py`, `notebooks/02_churn_patterns.ipynb`,
      `tests/test_eda.py`, `tests/test_data.py`, `reports/figures/*_boxplot.png`,
      commit message `phase 1: outlier analysis and churn-patterns notebook`.

### Tests to write (hand to test-writer)

- `tests/test_eda.py::test_detect_outliers_iqr_finds_none_in_current_data` —
  parametrized over `tenure`, `MonthlyCharges`, `TotalCharges`; asserts
  `len(detect_outliers_iqr(clean_df, col)) == 0` (locks in today's verified
  zero-outlier finding).
- `tests/test_eda.py::test_summarize_outliers_shape` — one row per input
  column, `n_outliers >= 0`, `pct_outliers` between 0 and 100.
- `tests/test_eda.py::test_distribution_stats_no_nan_or_inf` — every value in
  the returned frame is finite.
- `tests/test_data.py::test_missing_total_charges_pattern` — on raw data, the
  11 blank-`TotalCharges` rows have `tenure == 0` for all 11, `Churn == "No"`
  for all 11, and `Contract` in `{"One year", "Two year"}` for all 11 (10 Two
  year, 1 One year — assert the exact split so a shift is caught, not just
  set membership).

### Quality gates

1. **test-runner** — run `pytest -q`, confirm all tests (8 existing + 4 new)
   pass, fix any straightforward regression.
2. **quality-reviewer** — review the 4 new `eda.py` functions for
   correctness (Tukey fence math, skew/kurtosis calls), the notebook's
   reuse-not-duplicate structure, and adherence to CLAUDE.md §8 (no magic
   numbers, type hints, docstrings).
3. **security-reviewer** — spot-check no new untrusted input path is
   introduced (there isn't one — everything reads the already-local cleaned
   DataFrame); confirm no dependency was added outside `requirements.txt`.
4. Commit only after both reviews are clean.

### Risks / rollback

- **Risk:** the zero-outliers finding is dataset-order-sensitive or
  `k=1.5`-sensitive in a way that looks like a bug rather than a real
  pattern. **Mitigation:** Task 2's manual regeneration plus the locked-in
  test (Functional Requirement 6) make the finding explicit and re-verifiable
  on every future test run, not just asserted once in this session's
  research.
- **Risk:** notebook execution order drift (a cell run out of order
  produces a different result than top-to-bottom). **Mitigation:** Success
  Criteria requires the notebook to run top-to-bottom clean before this is
  considered done — verify with "Restart Kernel and Run All" or
  `jupyter nbconvert --execute` before committing.
- **Rollback:** single commit (Task 7) covering only additive files (new
  functions in `eda.py`, new notebook, new tests, new PNGs) — `git revert` is
  clean since nothing existing is modified in place.

### Definition of done

- All 7 tasks checked off.
- `pytest -q` green (8 existing `test_data.py` + 1 new there + 3 new
  `test_eda.py` = 12 total).
- `notebooks/02_churn_patterns.ipynb` executes top-to-bottom without error.
- `reports/figures/` has 13 PNGs (10 existing + 3 new boxplots), no existing
  one altered.
- `quality-reviewer` and `security-reviewer` report no unresolved findings.
- All Success Criteria in Part 1 are met.
