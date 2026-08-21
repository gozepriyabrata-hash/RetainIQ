# Spec + Plan: Retention Funnel — Rule-Based Customer Lifecycle Stages

> Location note: this file lives at `.claude/specs/` per CLAUDE.md §4's repo
> tree, which marks `.claude` as "claude.md and also Specs folder" — consistent
> with `01`–`04`. Spec and plan are combined in one file for the same reason.
>
> Scope note: like `04-customer-segmentation.md`, this is genuinely **new
> capability**, not a Phase 1 EDA deep-dive — it's a rule-based, multi-signal
> classifier (not pure `pd.cut` binning like `cohorts.py`, and not a fitted
> K-Means model like `segmentation.py`). It composes on both: it reuses
> `tenure` banding logic in spirit from `cohorts.py` and imports
> `compute_service_count` directly from `segmentation.py` rather than
> redefining it. It is **not** CLAUDE.md §14's Phase 2 — no supervised model,
> no train/test split, no MLflow. No phase-tracker row changes.
>
> Methodology note: the request names four stages — "new (low tenure) →
> engaged → loyal (long-tenure, multi-service) → at-risk" — as an arrow chain,
> but "At-Risk" is not a tenure band the way New/Engaged/Loyal are: a
> brand-new customer and a 5-year customer can both show the same churn-risk
> behavior. This spec treats **At-Risk as an override that takes priority
> over the tenure/service staging**, not a fourth tenure band — a customer is
> classified New/Engaged/Loyal by tenure and service depth *only if* they
> don't already show enough risk behavior. This is stated explicitly because
> it's a reinterpretation of the literal arrow-chain wording, same as `04`
> stated its RFM substitution explicitly rather than leaving it implicit.
> Every design constant below (`AT_RISK_SIGNAL_THRESHOLD`, `NEW_TENURE_MAX`,
> `LOYAL_TENURE_MIN`, `LOYAL_SERVICE_MIN`) was chosen by inspecting churn-rate
> separation across candidate values on the current cleaned dataset (verified
> below), not guessed — see ML guardrails for why this is disclosed as a
> one-time offline design choice, not a leakage risk in the shipped code.

---

## PART 1 — SPEC

### Feature

A new `src/data/lifecycle.py` module that classifies every customer into
exactly one of four mutually-exclusive lifecycle stages — `New`, `Engaged`,
`Loyal`, `At-Risk` — using only `tenure`, a reused `ServiceCount`, and four
CLAUDE.md §6-documented risk indicators (contract type, payment method, tech
support, internet type). Reports a per-stage summary and a risk-signal
disclosure table, and renders two charts — consumed by a new
`notebooks/05_retention_funnel.ipynb`.

### Problem / motivation

Nothing in the repo today assigns customers to a lifecycle/relationship
stage. `cohorts.py` bins by `tenure` alone; `segmentation.py` clusters by
value (tenure, spend, service depth) but has no notion of "at risk." Neither
answers "which customers need retention attention right now, independent of
how long they've been a customer." Verified directly against
`load_clean_data()`'s current 7,043 rows with the exact logic this feature
implements:

1. **A composite risk-signal count built from CLAUDE.md §6's own signal
   table is strongly and monotonically predictive of churn**, confirming the
   signals are worth combining rather than used one at a time. Counting how
   many of {`Contract == "Month-to-month"`, `PaymentMethod == "Electronic
   check"`, `TechSupport == "No"`, `InternetService == "Fiber optic"`} apply
   per customer (0–4; `Tenure` deliberately excluded from this count since
   tenure already drives the New/Engaged/Loyal split separately — see
   Functional Requirement 2) gives a churn rate that climbs from **2.60%** at
   0 signals to **13.02%** (1), **24.14%** (2), **43.52%** (3), **62.92%**
   (4) — a >24x spread across the range, on population sizes 1,691 / 1,513 /
   1,359 / 1,342 / 1,138 respectively.
2. **At `AT_RISK_SIGNAL_THRESHOLD = 3` (≥3 of 4 signals), thresholding At-Risk
   first and then splitting the remainder by tenure/service depth produces a
   well-separated 4-stage population**, computed with `NEW_TENURE_MAX = 12`,
   `LOYAL_TENURE_MIN = 24`, `LOYAL_SERVICE_MIN = 5`:

   | Stage | Customers | Avg tenure | Avg ServiceCount | Churn rate |
   |---|---|---|---|---|
   | New | 1,079 | 4.95 mo | 2.36 | 27.71% |
   | Engaged | 1,720 | 39.35 mo | 2.37 | 6.28% |
   | Loyal | 1,764 | 57.08 mo | 6.80 | 9.18% |
   | At-Risk | 2,480 | 21.89 mo | 4.26 | 52.42% |

   Sums to 7,043 with no unassigned rows. At-Risk is by far the
   highest-churn stage (52.42%, ~2x the next-highest) despite a *moderate*
   average tenure (21.89 mo) — behavioral signals separate churn risk better
   than tenure alone.
3. **The override matters, not just the tenure band**: without the At-Risk
   override, customers with `tenure ≤ 12` churn at 47.44% (the `cohorts.py`
   `"0-12"` figure). After carving the high-risk fraction out into At-Risk,
   the *remaining* New stage churns at 27.71% — the tenure effect and the
   behavioral-risk effect are both real, but conflating them overstates how
   much "just being new" explains, since among the 1,079 New-and-otherwise
   customers, most of the churn is concentrated where risk signals are also
   present.
4. **Loyal correctly requires both long tenure and service depth, not
   either alone**: a customer with `tenure ≥ 24` but `ServiceCount < 5` (long
   tenure, only 1–4 services) falls to `Engaged`, not `Loyal` — this
   distinguishes a long-tenured but lightly-subscribed customer from one
   who's both stuck around *and* bought into multiple services, matching the
   literal "long-tenure, multi-service" wording.

### Goals / non-goals

**Goals**
- Add `src/data/lifecycle.py` with a leakage-safe risk-signal counter, a
  stage-assignment function producing an ordered `Low`→... no — ordered
  `New`/`Engaged`/`Loyal`/`At-Risk` categorical, a per-stage summary, and a
  risk-signal disclosure table, following `cohorts.py`'s
  `compute_*`/`plot_*` function-pair pattern.
- Reuse `compute_service_count` from `src.models.segmentation` — do not
  redefine `ServiceCount` logic in a second place.
- Add two charts: lifecycle-stage customer-count distribution (stage order,
  explicitly **not** framed as a monotonically-decreasing conversion
  funnel — see Edge cases), and churn rate by stage.
- Create `notebooks/05_retention_funnel.ipynb` narrating: risk-signal
  construction and its disclosure table → stage-assignment logic (At-Risk
  priority, then tenure/service split) → stage summary table → both charts
  and the At-Risk / New findings above → key findings, following
  `01`–`04`'s bootstrap-cell notebook pattern.
- Add `python -m src.data.lifecycle` as a standalone runnable entry point,
  documented in CLAUDE.md §5.
- Add pytest coverage in `tests/test_lifecycle.py`, including a leakage guard
  (`Churn` never read by the risk-signal or stage-assignment functions) and
  a hand-computed fixture exercising the At-Risk override explicitly.

**Non-goals**
- No wiring of `LifecycleStage` into Phase 2's supervised churn pipeline
  (`src/features/`) or the FastAPI contract — not one of CLAUDE.md §10's four
  endpoints. If a future phase wants it as a model feature, see the ML
  guardrails note on re-tuning constants outside any CV loop.
- No true behavioral/time-series lifecycle transition tracking (e.g., "this
  customer moved from Engaged to At-Risk last quarter") — this dataset is a
  single cross-sectional snapshot with no repeated observations per
  customer, so only a *current* stage can be assigned, never a transition.
- No statistical optimization (grid search / decision-tree fit) of the four
  threshold constants — they are fixed, disclosed business-rule constants
  chosen by inspecting churn-rate separation (Problem/motivation, finding 1),
  not fit by any shipped function.
- No model persistence — matches `cohorts.py`/`segmentation.py`'s
  reporting-only precedent.
- No new phase-tracker row in CLAUDE.md §14, and no change to
  `src/data/eda.py`, `src/data/cohorts.py`, `src/models/segmentation.py`, or
  any existing test/figure.

### User stories

- As the **engineer (Priyabrata)**, I want a reusable, tested lifecycle-stage
  assignment (not one-off notebook logic) so the same function can back a
  future "stage" filter on the Phase 5 dashboard without rewriting it.
- As a **retention manager**, I want to see that At-Risk customers churn at
  more than 2x the rate of any other stage, *independent of how long
  they've been a customer*, so retention outreach can be targeted by
  behavior signals rather than assuming only new customers are at risk.
- As a **recruiter/reviewer**, I want the At-Risk-overrides-tenure design
  choice and the four threshold constants' derivation stated plainly, so the
  analysis reads as a deliberate, disclosed design rather than an
  arbitrary or overfit rule.

### Functional requirements

1. `src/data/lifecycle.py` MUST define named constants: `CONTRACT_RISK_VALUE
   = "Month-to-month"`, `PAYMENT_RISK_VALUE = "Electronic check"`,
   `TECHSUPPORT_RISK_VALUE = "No"`, `INTERNET_RISK_VALUE = "Fiber optic"`,
   `AT_RISK_SIGNAL_THRESHOLD = 3`, `NEW_TENURE_MAX = 12`, `LOYAL_TENURE_MIN =
   24`, `LOYAL_SERVICE_MIN = 5`, `LIFECYCLE_STAGE_LABELS = ["New", "Engaged",
   "Loyal", "At-Risk"]` — no magic numbers inlined elsewhere (CLAUDE.md §8).
2. MUST gain `compute_risk_signal_count(df: pd.DataFrame) -> pd.Series`
   returning an integer count (0–4) of how many of the four risk conditions
   hold per row (`Contract == CONTRACT_RISK_VALUE`, `PaymentMethod ==
   PAYMENT_RISK_VALUE`, `TechSupport == TECHSUPPORT_RISK_VALUE`,
   `InternetService == INTERNET_RISK_VALUE`). MUST NOT read `Churn` or
   `tenure`. Tenure is deliberately excluded from the risk-signal count
   since it drives the base New/Engaged/Loyal split separately (see
   Requirement 3) — including it here would double-count the same signal.
3. MUST gain `assign_lifecycle_stage(df: pd.DataFrame) -> pd.Series`
   returning an ordered `pd.Categorical` (categories in
   `LIFECYCLE_STAGE_LABELS` order) aligned to `df`'s index, assigned by this
   priority: (a) `At-Risk` if `compute_risk_signal_count(df) >=
   AT_RISK_SIGNAL_THRESHOLD`; else (b) `New` if `tenure <= NEW_TENURE_MAX`;
   else (c) `Loyal` if `tenure >= LOYAL_TENURE_MIN` AND
   `compute_service_count(df) >= LOYAL_SERVICE_MIN`; else (d) `Engaged`
   (catch-all). Every row MUST receive exactly one non-null stage — no `NaN`
   path exists since (d) is a catch-all, unlike `cohorts.py`'s bin edges.
   MUST import `compute_service_count` from `src.models.segmentation` rather
   than redefining it.
4. MUST gain `lifecycle_summary(df: pd.DataFrame) -> pd.DataFrame` with one
   row per stage **in `LIFECYCLE_STAGE_LABELS` order**, columns `stage`,
   `customers`, `avg_tenure`, `avg_service_count` (both 2dp), `churn_rate`
   (%, 2dp). Must not drop or reorder stages even if one is empty
   (`observed=False` groupby semantics, matching `cohorts.py`/
   `segmentation.py`'s precedent).
5. MUST gain `risk_signal_diagnostic(df: pd.DataFrame) -> pd.DataFrame` with
   one row per raw `risk_count` value `0..4`, columns `risk_count`,
   `customers`, `churn_rate` (%, 2dp) — a disclosure table showing the
   monotonic relationship justifying `AT_RISK_SIGNAL_THRESHOLD`, not used to
   override it at call time.
6. MUST gain `plot_lifecycle_stage_distribution(df, out_dir=FIGURES_DIR) ->
   Path` — a bar chart of `customers` per stage in `LIFECYCLE_STAGE_LABELS`
   order, saved as `reports/figures/lifecycle_stage_distribution.png`. The
   chart title or notebook cell MUST note this is a stage distribution over
   a single snapshot, not a sequential conversion funnel with
   monotonically-decreasing counts (see Edge cases).
7. MUST gain `plot_lifecycle_churn_rate(df, out_dir=FIGURES_DIR) -> Path` —
   a bar chart of `churn_rate` per stage in `LIFECYCLE_STAGE_LABELS` order,
   saved as `reports/figures/lifecycle_churn_rate.png`.
8. MUST gain `generate_lifecycle_figures(df, out_dir=FIGURES_DIR) ->
   list[Path]` and a `main()` calling it via `load_clean_data()`, runnable
   as `python -m src.data.lifecycle`.
9. `notebooks/05_retention_funnel.ipynb` MUST be created following
   `01`–`04`'s bootstrap-cell pattern. Sections, in order: risk-signal
   construction + `risk_signal_diagnostic()` table (disclosure) →
   stage-assignment logic explanation (At-Risk priority, then tenure/service
   split) → `lifecycle_summary()` table → stage-distribution chart (with the
   non-funnel-shape caveat) → churn-rate-by-stage chart and narration of the
   At-Risk finding (52.42% vs. next-highest 27.71%) and the New-vs-tenure-only
   finding (27.71% vs. cohorts.py's 47.44% for the same tenure band before
   the At-Risk carve-out) → key findings summary matching `01`–`04`'s
   closing-cell style.
10. `tests/test_lifecycle.py` MUST cover: `compute_risk_signal_count` range
    (0–4) and no-`Churn`/no-`tenure`-read guard; hand-computed risk-count
    values on a small fixture; `assign_lifecycle_stage` returns an ordered
    categorical covering every row with no `NaN`; a fixture proving the
    At-Risk override (a low-tenure row with 3+ risk signals is classified
    `At-Risk`, not `New`) and a fixture proving the Loyal
    tenure-AND-service-count requirement (long tenure alone, low
    `ServiceCount`, is classified `Engaged`, not `Loyal`); `lifecycle_summary`
    is in `LIFECYCLE_STAGE_LABELS` order with `customers.sum() ==
    len(clean_df)`; the verified stage counts and churn rates from
    Problem/motivation finding 2 on current data; `At-Risk` has the highest
    churn rate of the four stages; `risk_signal_diagnostic`'s `churn_rate` is
    strictly increasing across `risk_count` 0→4 (locks in finding 1).
11. None of the above may change `src/data/eda.py`, `src/data/cohorts.py`,
    `src/models/segmentation.py`, `clean_data()`'s output, or any existing
    test/figure — all current tests must keep passing unmodified.

### Data & model impact

New descriptive classification, not persisted and not wired into any
existing pipeline. `LifecycleStage`/`RiskSignalCount` are not written back
into `load_clean_data()`'s output — `lifecycle.py` computes them internally
per call, mirroring `cohorts.py`/`segmentation.py`'s precedent. `src/features/`
(Phase 2, not yet built) is unaffected.

### ML guardrails (mandatory check)

- **No target leakage in the shipped code:** `compute_risk_signal_count`
  (Requirement 2) and `assign_lifecycle_stage` (Requirement 3) read only
  `Contract`, `PaymentMethod`, `TechSupport`, `InternetService`, `tenure`,
  and `ServiceCount` — never `Churn` and never a churn probability. `Churn`
  is read only afterward, in `lifecycle_summary`/`risk_signal_diagnostic`, to
  *report* a descriptive churn rate per already-assigned stage — directly
  tested (Requirement 10).
- **Disclosed design-time use of `Churn`, distinct from runtime leakage:**
  the four threshold constants (`AT_RISK_SIGNAL_THRESHOLD`, `NEW_TENURE_MAX`,
  `LOYAL_TENURE_MIN`, `LOYAL_SERVICE_MIN`) were chosen by the implementer
  inspecting churn-rate separation across candidate values on the full
  dataset (Problem/motivation) — a one-time, offline constant-selection
  process, not a parameter any shipped function fits from `Churn` at call
  time. This is analogous to a human picking a decision threshold by
  eyeballing a plot. It is flagged explicitly, per CLAUDE.md §2 rule 1's
  spirit, so a future reviewer doesn't mistake it for runtime leakage — and
  so that **if these constants are ever re-tuned against new data**, that
  re-tuning happens outside any Phase 2 cross-validation loop and outside
  any function that reads `Churn` at runtime, exactly like a manually-set
  hyperparameter would be.
- **Reproducibility:** no randomness — pure rule-based thresholding, so no
  `random_state` is needed (there is nothing to seed).
- **Splitting / resampling:** not applicable — descriptive classification
  over the full customer base, no train/test split, no SMOTE, matching
  `cohorts.py`/`segmentation.py`'s precedent. Not a substitute for, or input
  to, Phase 2's supervised evaluation.
- **Imbalance / metric reporting:** `lifecycle_summary` and
  `risk_signal_diagnostic` report `churn_rate` (not accuracy), consistent
  with CLAUDE.md §2 rule 3's spirit.

### API / UI surface

None — no FastAPI endpoint or Streamlit view. `src/data/lifecycle.py` stays
notebook/report-facing, consistent with `cohorts.py`/`segmentation.py`.

### Edge cases & failure states

- **Stage counts are not monotonically decreasing** (New=1,079 <
  Engaged=1,720 < Loyal=1,764 < At-Risk=2,480), so a literal "funnel" chart
  (each stage a strict subset, visually narrowing) would misrepresent the
  data — these are four mutually-exclusive categories of a single snapshot,
  not sequential drop-off stages. Mitigated by Requirement 6's mandatory
  caption, the same rigor `03-cohort-analysis.md` applied to its retention
  curve's censoring-bias disclosure.
- **A customer meets both the New tenure condition and the Loyal service
  condition** — cannot happen: `NEW_TENURE_MAX = 12 < LOYAL_TENURE_MIN = 24`,
  and stage assignment checks New before Loyal, so this is structurally
  unreachable, not just empirically absent.
- **A customer with `risk_count` exactly at `AT_RISK_SIGNAL_THRESHOLD`**
  (exactly 3 of 4 signals): included in At-Risk (`>=`, not `>`) — verified
  behavior, not an ambiguous boundary, since the comparison operator is
  explicit in Requirement 3.
- **`TechSupport`/other sentinel values** (`"No internet service"` for
  customers without internet): never equal `TECHSUPPORT_RISK_VALUE ==
  "No"`, so such rows correctly don't get that risk point — same sentinel
  handling precedent as `segmentation.py`'s `ServiceCount`.
- **Empty input DataFrame**: `lifecycle_summary` returns 4 rows (one per
  stage label) with `customers == 0` and rate columns as `NaN` (mean of an
  empty group), not a raise — matches `cohorts.py`/`segmentation.py`'s
  documented empty-frame behavior; not separately tested since no current
  caller passes an empty frame.

### Security notes

None — no new untrusted input, secret, network call, or dependency. All new
functions operate on the already-loaded, already-cleaned local DataFrame
using pandas/matplotlib, both already in `requirements.txt`.

### Success criteria

- `pytest -q` passes: all existing tests + the new `tests/test_lifecycle.py`
  module, all green.
- `notebooks/05_retention_funnel.ipynb` runs top-to-bottom without error and
  renders every section named in Functional Requirement 9.
- `reports/figures/` gains `lifecycle_stage_distribution.png` and
  `lifecycle_churn_rate.png`, with no existing filename changed.
- `python -m src.data.lifecycle` is a working, idempotent entry point,
  documented in CLAUDE.md §5.
- `quality-reviewer` and `security-reviewer` report no unresolved findings
  on the diff.

### Out of scope

- Wiring `LifecycleStage` into `src/features/`, the supervised churn model,
  or any FastAPI endpoint — a separate, explicit future decision, subject to
  the design-time-vs-runtime leakage distinction flagged above.
- True time-series lifecycle transitions — this dataset has no repeated
  per-customer observations.
- Statistical optimization of the four threshold constants — fixed,
  disclosed business-rule constants, not grid-searched or fit.
- Model persistence / MLflow logging.
- A new row in CLAUDE.md §14's phase tracker.
- Any change to `01_eda.ipynb`, `02_churn_patterns.ipynb`,
  `03_cohort_analysis.ipynb`, or `04_customer_segmentation.ipynb`.

---

## PART 2 — PLAN

### Approach

Add `src/data/lifecycle.py` as a self-contained sibling to `cohorts.py`,
reusing its exact function-pair pattern (`compute_*`/`assign_*` →
`pd.DataFrame`/`pd.Series`, `plot_*` → `Path` via `save_fig` imported from
`eda.py`), but importing `compute_service_count` from
`src.models.segmentation` rather than duplicating it — cross-module reuse of
a pure function, not a fitted pipeline, so no coupling to segmentation's
K-Means state is introduced. Placed in `src/data/` (not `src/models/`) per
CLAUDE.md §4 and the `04` spec's own precedent: this module has no fitted
parameters (unlike K-Means's cluster centers) — every threshold is a fixed
constant, so it is pure business-rule classification like `cohorts.py`'s
binning, not a trained model.

**Alternative rejected:** implementing At-Risk as a fifth independent
boolean flag layered on top of three tenure-only stages (`New`/`Engaged`/
`Loyal` computed first, `IsAtRisk` computed separately), rather than a
single mutually-exclusive `LifecycleStage` column. Rejected because the
literal request asks for a single funnel/lifecycle **view** — one stage per
customer is easier to chart, summarize, and eventually filter on in a future
dashboard than two overlapping columns, and the override priority
(Requirement 3) already captures "at-risk behavior takes precedence over
tenure-based staging" without losing information (the underlying
`risk_signal_count` and `tenure`/`ServiceCount` remain available via
`risk_signal_diagnostic` and `lifecycle_summary` for anyone who wants the
finer-grained view).

### Task breakdown

- [ ] **1. Create `src/data/lifecycle.py`** — constants
      (`CONTRACT_RISK_VALUE`, `PAYMENT_RISK_VALUE`, `TECHSUPPORT_RISK_VALUE`,
      `INTERNET_RISK_VALUE`, `AT_RISK_SIGNAL_THRESHOLD`, `NEW_TENURE_MAX`,
      `LOYAL_TENURE_MIN`, `LOYAL_SERVICE_MIN`, `LIFECYCLE_STAGE_LABELS`),
      `compute_risk_signal_count`, `assign_lifecycle_stage`,
      `lifecycle_summary`, `risk_signal_diagnostic`,
      `plot_lifecycle_stage_distribution`, `plot_lifecycle_churn_rate`,
      `generate_lifecycle_figures`, `main()`. Import `save_fig`,
      `CHURN_PALETTE`, `FIGURES_DIR` from `src.data.eda`/`src.data.config`
      and `compute_service_count` from `src.models.segmentation`.
- [ ] **2. Run `python -m src.data.lifecycle`** to generate
      `reports/figures/lifecycle_stage_distribution.png` and
      `lifecycle_churn_rate.png`; confirm via `git status`/`git diff --stat`
      that no existing figure file changed.
- [ ] **3. Create `notebooks/05_retention_funnel.ipynb`** — bootstrap cell
      copied from `04_customer_segmentation.ipynb`; sections per Functional
      Requirement 9.
- [ ] **4. Add `tests/test_lifecycle.py`** — cover Functional Requirement 10
      (risk-signal range/leakage guard, hand-computed fixtures for the
      At-Risk override and the Loyal tenure-AND-service requirement,
      verified stage counts/churn rates, monotonic risk-signal diagnostic).
- [ ] **5. Document the new entry point** — add `python -m src.data.lifecycle`
      to CLAUDE.md §5's command list, next to `python -m src.models.segmentation`.
- [ ] **6. Run the full suite** — `pytest -q`, confirm all existing + new
      tests pass.
- [ ] **7. Commit** — `src/data/lifecycle.py`,
      `notebooks/05_retention_funnel.ipynb`, `tests/test_lifecycle.py`,
      `reports/figures/lifecycle_stage_distribution.png`,
      `reports/figures/lifecycle_churn_rate.png`, `.claude/CLAUDE.md`, commit
      message `feat: rule-based customer lifecycle stages (retention funnel)`.

### Tests to write (hand to test-writer)

- `tests/test_lifecycle.py::test_compute_risk_signal_count_range_and_no_leakage_read` —
  on `clean_df`, `compute_risk_signal_count` returns values in `{0,1,2,3,4}`,
  no nulls; result is identical whether or not `Churn`/`tenure` columns are
  present in the input frame (leakage/independence guard).
- `tests/test_lifecycle.py::test_compute_risk_signal_count_matches_hand_computed` —
  small fixture with rows engineered to have exactly 0, 2, and 4 risk
  signals; asserts the exact counts.
- `tests/test_lifecycle.py::test_assign_lifecycle_stage_at_risk_overrides_new` —
  fixture row with `tenure=3` (would be New) but 3+ risk signals present →
  asserts stage is `"At-Risk"`, not `"New"`.
- `tests/test_lifecycle.py::test_assign_lifecycle_stage_requires_both_tenure_and_service_for_loyal` —
  fixture row with `tenure=50` (meets `LOYAL_TENURE_MIN`) but
  `ServiceCount < LOYAL_SERVICE_MIN` and 0 risk signals → asserts stage is
  `"Engaged"`, not `"Loyal"`.
- `tests/test_lifecycle.py::test_assign_lifecycle_stage_covers_every_row` —
  on `clean_df`, no `NaN` stages; `assign_lifecycle_stage` returns an ordered
  `pd.CategoricalDtype` with categories exactly `LIFECYCLE_STAGE_LABELS`.
- `tests/test_lifecycle.py::test_lifecycle_summary_order_and_customer_total` —
  `lifecycle_summary(clean_df)["stage"].tolist() == LIFECYCLE_STAGE_LABELS`;
  `customers.sum() == len(clean_df)`.
- `tests/test_lifecycle.py::test_lifecycle_summary_matches_verified_counts_and_rates` —
  matches `{"New": 1079, "Engaged": 1720, "Loyal": 1764, "At-Risk": 2480}`
  and churn rates `{27.71, 6.28, 9.18, 52.42}` on current data.
- `tests/test_lifecycle.py::test_at_risk_has_highest_churn_rate` — locks in
  the headline finding: `At-Risk`'s `churn_rate` is strictly greater than
  the other three stages'.
- `tests/test_lifecycle.py::test_risk_signal_diagnostic_is_strictly_increasing` —
  `risk_signal_diagnostic(clean_df)["churn_rate"]` strictly increases across
  `risk_count` 0→4 (locks in 2.60/13.02/24.14/43.52/62.92 on current data).

### Quality gates

1. **test-runner** — run `pytest -q`, confirm all tests (existing + new)
   pass, fix any straightforward regression.
2. **quality-reviewer** — review `lifecycle.py`'s priority-ordered
   if/elif staging logic (At-Risk short-circuits before New/Loyal are
   evaluated), the reuse of `compute_service_count` (no duplicated logic),
   the leakage guard (`Churn` never read by the classification functions),
   the notebook's reuse-not-duplicate structure, and CLAUDE.md §8 adherence
   (named threshold constants, type hints, docstrings).
3. **security-reviewer** — confirm no new untrusted input path or dependency
   is introduced (there isn't one).
4. Commit only after both reviews are clean.

### Risks / rollback

- **Risk:** a future Kaggle re-download shifts the churn-rate separation
  across risk-signal counts or tenure/service thresholds enough that the
  locked-in verified counts/rates no longer hold. **Mitigation:**
  intentional, matching `02`/`03`/`04`'s brittleness philosophy — these
  tests should fail loudly on real distributional shift so the thresholds
  get re-examined, not silently trusted.
- **Risk:** a reviewer reads the At-Risk-overrides-tenure design as
  arbitrary. **Mitigation:** the risk-signal diagnostic table and its
  explicit monotonic disclosure (Requirement 5, Problem/motivation finding
  1) make the threshold choice traceable to real churn-rate separation, not
  a guess.
- **Risk:** someone later feeds `LifecycleStage` into the Phase 2 supervised
  model without re-deriving the threshold constants outside any CV loop.
  **Mitigation:** the design-time-vs-runtime leakage distinction in this
  spec's ML Guardrails section exists specifically so that decision, when
  made, is made with this constraint visible.
- **Rollback:** single commit (Task 7) covering only additive files (new
  module, new notebook, new tests, new PNGs, one CLAUDE.md doc line) — `git
  revert` is clean since nothing existing is modified in place.

### Definition of done

- All 7 tasks checked off.
- `pytest -q` green (all existing tests + 9 new `test_lifecycle.py` tests).
- `notebooks/05_retention_funnel.ipynb` executes top-to-bottom without
  error.
- `reports/figures/` gains 2 new PNGs, no existing one altered.
- CLAUDE.md §5 documents `python -m src.data.lifecycle`.
- `quality-reviewer` and `security-reviewer` report no unresolved findings.
- All Success Criteria in Part 1 are met.
