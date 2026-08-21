# Spec + Plan: Customer Segmentation — RFM-Style Features + K-Means Value Tiers

> Location note: this file lives at `.claude/specs/` per CLAUDE.md §4's repo
> tree, which marks `.claude` as "claude.md and also Specs folder" — consistent
> with `01-clean-dataset-eda-notebook-kpi-charts.md`, `02-churn-patterns.md`,
> and `03-cohort-analysis.md`. Spec and plan are combined in one file for the
> same reason as those three.
>
> Scope note: unlike `02`/`03`, this is genuinely **new capability**, not a
> Phase 1 EDA deep-dive — it fits a real unsupervised model (K-Means) rather
> than binning or descriptive aggregation. It is also **not** CLAUDE.md §14's
> Phase 2 ("Feature pipeline, model training + comparison, MLflow"): Phase 2
> is the *supervised* churn classifier (LogReg → RF → XGBoost → LightGBM,
> compared and tracked in MLflow), and this feature touches none of that. Per
> CLAUDE.md §4 ("anything that trains or scores" → `src/models/`), the code
> lives in `src/models/segmentation.py` because it is a real fitted model, but
> no phase-tracker row changes and Phase 2 remains untouched and unblocked by
> this work.
>
> Methodology note: classic RFM (Recency / Frequency / Monetary) assumes
> repeat-purchase transaction history, which this dataset does not have (one
> row per customer, no order log). The feature request explicitly names the
> three axes to substitute: **tenure** stands in for Recency/loyalty (how
> established the relationship is), **MonthlyCharges** stands in for Monetary
> (recurring spend), and a derived **ServiceCount** (breadth of subscribed
> services) stands in for Frequency (depth of engagement). This mapping is
> stated here explicitly, once, rather than left implicit, since it's a
> reinterpretation of a retail concept for a subscription business — a
> reviewer should not have to guess why "Frequency" became "service count."

---

## PART 1 — SPEC

### Feature

A new `src/models/segmentation.py` module that derives a `ServiceCount`
RFM-style feature, fits a K-Means model (`k=3`, `random_state=42`) on
standardized `(tenure, MonthlyCharges, ServiceCount)`, ranks the three
resulting clusters by a composite centroid score into ordered
**Low / Mid / High** value segments (K-Means cluster IDs are arbitrary and
unordered — the ranking step is what turns them into a business-meaningful
label), reports per-segment profile and churn rate, and renders two charts —
consumed by a new `notebooks/04_customer_segmentation.ipynb`.

### Problem / motivation

Nothing in the repo today groups customers by commercial value. `src/data/eda.py`
and `src/data/cohorts.py` cut the population by existing categorical columns
or by `tenure` alone; neither combines spend, tenure, and service breadth into
a single value tier. Verified directly against `load_clean_data()`'s current
7,043 rows with the exact pipeline this feature implements
(`StandardScaler` → `KMeans(n_clusters=3, random_state=42, n_init=10)` on
`[tenure, MonthlyCharges, ServiceCount]`, clusters ranked by mean standardized
centroid):

1. **The three segments split roughly evenly and are stable/reproducible.**
   Low = 2,151 customers, Mid = 2,575, High = 2,317 (sums to 7,043). Refitting
   the same pipeline twice on the same data with `random_state=42` produces
   bit-identical labels and cluster centers — confirmed by direct
   double-fit comparison, not assumed.
2. **Segment profiles are commercially sensible**: High (tenure 56.9 mo,
   MonthlyCharges $89.81, ServiceCount 6.59) reads as long-tenured,
   heavily-subscribed customers; Low (tenure 27.7 mo, MonthlyCharges $26.21,
   ServiceCount 1.56) reads as light, low-spend accounts; Mid (tenure 14.2 mo,
   MonthlyCharges $74.43, ServiceCount 4.10) reads as newly-acquired,
   higher-bill customers who haven't yet built up tenure or service depth.
3. **The churn-by-segment finding is the headline insight, and it's
   counter-intuitive**: churn rate is **not** monotonic in value. Low =
   14.41%, High = 15.80%, but **Mid = 46.33%** — nearly 3x the other two
   segments. The segment carrying the most churn risk is not the cheapest
   one, it's the newly-acquired, higher-paying one that hasn't stuck around
   long enough to prove out. This directly sharpens CLAUDE.md §6's existing
   tenure/contract signals into an actionable retention target: Mid-segment
   customers are simultaneously valuable *and* at high risk, which Low
   customers are not.
4. **`k=3` is a business constraint (High/Mid/Low), not a
   silhouette-optimal choice, and that's disclosed rather than hidden**:
   silhouette scores for `k=2..6` on this feature set are 0.4216, 0.4200,
   0.4267, 0.3850, 0.4012 — essentially flat, so no `k` in a plausible range
   is meaningfully better than `k=3` by this metric. The notebook reports
   this diagnostic honestly instead of implying `k=3` was chosen by an
   elbow/silhouette search it wasn't.
5. **`ServiceCount` (0–9 possible; observed range is 1–9 — no customer has
   zero services) is well-defined and leakage-free**: it counts `"Yes"`
   across `PhoneService`, `MultipleLines`, `OnlineSecurity`, `OnlineBackup`,
   `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`, plus
   1 if `InternetService != "No"`. `"No phone service"` / `"No internet
   service"` sentinel values never match `"Yes"`, so no special-casing is
   needed.

### Goals / non-goals

**Goals**
- Add `src/models/segmentation.py` with a leakage-safe feature builder
  (`tenure`, `MonthlyCharges`, `ServiceCount` — never `Churn`), a fitted
  `Pipeline(StandardScaler, KMeans)`, a composite-centroid ranking step that
  turns arbitrary cluster IDs into ordered `Low`/`Mid`/`High` labels, a
  per-segment profile/churn summary, and a silhouette diagnostic table.
- Add two charts: churn rate by value segment (Low/Mid/High order), and a
  segment profile chart (mean tenure/MonthlyCharges/ServiceCount by segment).
- Create `notebooks/04_customer_segmentation.ipynb` narrating: RFM-style
  feature construction → K-Means fit → silhouette diagnostic (disclosure, not
  a k-selection search) → segment profile table → churn-by-segment chart and
  the Mid-segment finding → key findings, following `01`–`03`'s bootstrap-cell
  notebook pattern.
- Add `python -m src.models.segmentation` as a standalone runnable entry
  point, documented in CLAUDE.md §5.
- Add pytest coverage in `tests/test_segmentation.py`, including a
  reproducibility test (two independent fits, `random_state=42`, produce
  identical labels) and a leakage guard (`Churn` never appears among the
  clustering feature columns).

**Non-goals**
- No wiring of `ValueSegment` into Phase 2's supervised churn pipeline
  (`src/features/`) or the FastAPI contract (CLAUDE.md §10 lists `/predict`,
  `/batch-predict`, `/explain`, `/recommend` — segmentation isn't one of
  them). If a future phase wants `ValueSegment` as a churn-model input, the
  K-Means fit (unlike a fixed tenure-bin edge) is data-dependent and MUST be
  refit inside each CV fold / on the train split only, not precomputed once
  on the full population and joined in as a static column — noted here for
  whoever makes that call later, not acted on.
- No true RFM with a transaction/order log — this dataset has none; the
  substitution (tenure/MonthlyCharges/ServiceCount) is documented above, not
  presented as textbook RFM.
- No silhouette-driven or elbow-driven selection of `k` — `k=3` is fixed to
  match the requested High/Mid/Low business tiers; the silhouette table is
  reported for transparency only, per Problem/motivation finding 4.
- No model persistence (`models/*.pkl`/MLflow logging) — this is a
  reporting/analysis module today, matching `cohorts.py`'s precedent; if a
  later phase needs the fitted pipeline served live, that's a separate,
  explicit decision.
- No new phase-tracker row in CLAUDE.md §14, and no change to
  `src/data/eda.py`, `src/data/cohorts.py`, or any existing test/figure.

### User stories

- As the **engineer (Priyabrata)**, I want a reusable, tested value-segment
  assignment (not one-off notebook clustering) so the same logic can back a
  future "segment" filter on the Phase 5 dashboard without rewriting it.
- As a **retention manager**, I want to see that the highest-churn segment is
  the mid-value, newly-acquired one — not the cheapest customers — so
  retention spend gets targeted at the segment that's both valuable and at
  risk, rather than assumed to track spend level.
- As a **recruiter/reviewer**, I want the RFM-substitution and the
  business-driven (not metric-optimal) choice of `k=3` stated plainly, so the
  analysis reads as a deliberate, disclosed design choice rather than
  overclaiming a data-driven cluster count it doesn't have.

### Functional requirements

1. `src/models/segmentation.py` MUST define named constants:
   `SERVICE_YESNO_COLUMNS` (the 8 columns listed in Problem/motivation
   finding 5), `N_VALUE_SEGMENTS = 3`, `VALUE_SEGMENT_LABELS = ["Low", "Mid",
   "High"]`, `RANDOM_STATE = 42` — no magic numbers inlined elsewhere
   (CLAUDE.md §8).
2. MUST gain `compute_service_count(df: pd.DataFrame) -> pd.Series` returning
   an integer count per row: `sum(df[col] == "Yes" for col in
   SERVICE_YESNO_COLUMNS) + (df["InternetService"] != "No")`. MUST NOT read
   `Churn`.
3. MUST gain `build_segmentation_features(df: pd.DataFrame) -> pd.DataFrame`
   returning exactly the 3 columns `["tenure", "MonthlyCharges",
   "ServiceCount"]` (using `compute_service_count`). MUST NOT include `Churn`
   or any churn-derived column — this is the leakage guard surface tested
   directly.
4. MUST gain
   `fit_segmentation_pipeline(df, n_clusters=N_VALUE_SEGMENTS, random_state=RANDOM_STATE) -> sklearn.pipeline.Pipeline`
   — a 2-step `Pipeline([("scaler", StandardScaler()), ("kmeans",
   KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10))])`
   fit on `build_segmentation_features(df)`. `n_init=10` MUST be passed
   explicitly (not left to library default) so behavior doesn't drift across
   scikit-learn versions.
5. MUST gain
   `rank_segments_by_value(pipeline: Pipeline) -> dict[int, str]` mapping
   each raw K-Means cluster ID to a label in `VALUE_SEGMENT_LABELS`, ordered
   by each cluster center's mean value across the 3 standardized dimensions
   (`pipeline.named_steps["kmeans"].cluster_centers_.mean(axis=1)`, ascending
   → `Low`, `Mid`, `High`). This is what makes the label meaningful despite
   K-Means cluster IDs being arbitrary.
6. MUST gain `assign_value_segment(df, pipeline: Pipeline | None = None) ->
   pd.Series` returning an ordered `pd.Categorical` (categories in
   `VALUE_SEGMENT_LABELS` order) aligned to `df`'s index. If `pipeline` is
   `None`, fits one via `fit_segmentation_pipeline(df)` first.
7. MUST gain `segment_summary(df: pd.DataFrame) -> pd.DataFrame` with one row
   per segment **in `VALUE_SEGMENT_LABELS` order**, columns `segment`,
   `customers`, `avg_tenure`, `avg_monthly_charges`, `avg_service_count`
   (all rounded 2dp), `churn_rate` (%, 2dp). Must not drop or reorder
   segments even if one is empty.
8. MUST gain
   `silhouette_diagnostic(df: pd.DataFrame, k_values: list[int] = [2, 3, 4, 5, 6]) -> pd.DataFrame`
   with columns `k`, `silhouette_score`, fit independently per `k` on the
   same standardized features — a disclosure table, not used to override
   `N_VALUE_SEGMENTS`.
9. MUST gain `plot_segment_churn_rate(df, out_dir=FIGURES_DIR) -> Path` — a
   bar chart of `churn_rate` per segment in `Low, Mid, High` order, saved as
   `reports/figures/segment_churn_rate.png`.
10. MUST gain `plot_segment_profile(df, out_dir=FIGURES_DIR) -> Path` — a
    grouped bar chart of `avg_tenure`, `avg_monthly_charges`,
    `avg_service_count` (each normalized to its own max for readability, or
    plotted on a secondary axis — implementer's call, document whichever is
    used) per segment, saved as `reports/figures/segment_profile.png`.
11. MUST gain `generate_segmentation_figures(df, out_dir=FIGURES_DIR) ->
    list[Path]` and a `main()` calling it via `load_clean_data()`, runnable
    as `python -m src.models.segmentation`.
12. `notebooks/04_customer_segmentation.ipynb` MUST be created following
    `01`–`03`'s bootstrap-cell pattern (sys.path setup, imports from
    `src.models.segmentation`/`src.data.load_data`, `IPython.display.Image`
    for saved PNGs). Sections, in order: RFM-style feature construction
    (with the methodology note on the tenure/MonthlyCharges/ServiceCount
    substitution) → K-Means fit + `silhouette_diagnostic()` table (disclosed
    as non-decisive for `k`) → `segment_summary()` table → segment-profile
    chart → churn-by-segment chart and narration of the Mid-segment finding
    (46.33% vs. 14.41%/15.80%) → key findings summary matching `01`–`03`'s
    closing-cell style.
13. `tests/test_segmentation.py` MUST cover: `compute_service_count` range
    and no-`Churn`-read guard; `build_segmentation_features` columns are
    exactly `{tenure, MonthlyCharges, ServiceCount}` (leakage guard);
    reproducibility (two independent `fit_segmentation_pipeline` calls on
    `clean_df` produce identical `.labels_` and `.cluster_centers_`);
    `segment_summary` is in `Low, Mid, High` order with `customers.sum() ==
    len(clean_df)`; the verified segment counts
    (`Low=2151, Mid=2575, High=2317`) and the verified, counter-intuitive
    finding that `Mid` has the **highest** churn rate of the three segments
    (locking in 46.33% vs. 14.41%/15.80% on current data).
14. None of the above may change `src/data/eda.py`, `src/data/cohorts.py`,
    `clean_data()`'s output, or any existing test/figure — all current tests
    must keep passing unmodified.

### Data & model impact

New model artifact class (K-Means), but not persisted and not wired into any
existing pipeline. `ValueSegment` is not written back into
`load_clean_data()`'s output — `segmentation.py` computes it internally per
call, exactly mirroring `cohorts.py`'s `TenureCohort` precedent. `src/features/`
(Phase 2, not yet built) is unaffected. `MonthlyCharges` and `tenure` are
read, not transformed, by anything outside this module.

### ML guardrails (mandatory check)

- **No target/probability leakage:** `build_segmentation_features` (Functional
  Requirement 3) is limited to exactly `[tenure, MonthlyCharges,
  ServiceCount]`; `Churn` is used only afterward, in `segment_summary`, to
  *report* a descriptive churn rate per already-assigned segment — never as
  a clustering input. Directly tested (Functional Requirement 13).
- **Reproducibility:** `RANDOM_STATE = 42` is passed to `KMeans`, and
  `n_init=10` is pinned explicitly rather than left to a library default that
  could vary across scikit-learn versions. Verified: two independent fits on
  identical data produce bit-identical labels and cluster centers.
- **Splitting / resampling:** not applicable — this is unsupervised,
  descriptive segmentation over the full customer base (no
  train/test split, no SMOTE), matching `cohorts.py`'s precedent of
  operating directly on `load_clean_data()`'s output. This is explicitly
  **not** a substitute for, or an input to, Phase 2's supervised
  train/test-split model evaluation.
- **Imbalance / metric reporting:** `segment_summary` reports `churn_rate`
  (not accuracy) per segment, consistent with CLAUDE.md §2 rule 3's spirit,
  even though this isn't a classifier evaluation.
- **Forward-looking leakage flag (not acted on here):** if a future phase
  uses `ValueSegment` as a Phase-2 model feature, the K-Means fit is
  data-dependent (unlike `cohorts.py`'s fixed tenure-bin edges) and MUST be
  refit inside each CV fold on the training split only — precomputing it
  once on the full population and joining it in as a static feature would
  leak test-fold distributional information into training. Flagged here per
  Non-goals; not implemented in this feature.

### API / UI surface

None — no FastAPI endpoint or Streamlit view. `src/models/segmentation.py`
stays notebook/report-facing for this feature, consistent with `cohorts.py`.

### Edge cases & failure states

- **A cluster ends up empty for some `k` in the silhouette diagnostic**: not
  observed for `k=2..6` on current data; scikit-learn's `KMeans` handles this
  internally (re-seeds an empty cluster) rather than raising, so no special
  handling is added.
- **Fewer rows than `n_clusters`** (e.g., a filtered subset with < 3 rows
  passed to `fit_segmentation_pipeline`): scikit-learn raises its own
  `ValueError` (`n_samples < n_clusters`); not caught or special-cased here —
  acceptable since no current caller passes such a small frame, mirroring
  `cohorts.py`'s treatment of the empty-DataFrame edge case as "must not
  crash the primary path, not separately tested for a degenerate input no
  caller produces."
- **Tied composite centroid scores** in `rank_segments_by_value` (two
  clusters with identical mean standardized value): would produce an
  ambiguous `Low`/`Mid` or `Mid`/`High` split; not observed on current data
  (the three composite scores are well-separated: 0.96, -0.86, -0.15) and not
  defended against explicitly — `dict` ranking would still produce 3 distinct
  labels via a stable sort, just not a meaningfully-ordered one, in the
  unlikely event of an exact tie.
- **`ServiceCount == 0` (a customer with no phone and no internet)**: not
  observed in current data (min is 1) but mathematically reachable; such a
  row would simply pull toward the `Low` end of the value axis, no special
  handling needed.

### Security notes

None — no new untrusted input, secret, network call, or dependency.
`StandardScaler`, `KMeans`, `Pipeline`, and `silhouette_score` are all part
of `scikit-learn`, already in `requirements.txt`. All functions operate on
the already-loaded, already-cleaned local DataFrame.

### Success criteria

- `pytest -q` passes: all existing tests + the new `tests/test_segmentation.py`
  module, all green.
- `notebooks/04_customer_segmentation.ipynb` runs top-to-bottom without error
  and renders every section named in Functional Requirement 12.
- `reports/figures/` gains `segment_churn_rate.png` and `segment_profile.png`,
  with no existing filename changed.
- `python -m src.models.segmentation` is a working, idempotent entry point,
  documented in CLAUDE.md §5.
- `quality-reviewer` and `security-reviewer` report no unresolved findings on
  the diff.

### Out of scope

- Wiring `ValueSegment` into `src/features/`, the supervised churn model, or
  any FastAPI endpoint — a separate, explicit future decision, subject to the
  refit-inside-CV guardrail flagged above.
- True RFM with real transaction/order-level Recency and Frequency — this
  dataset has no such log.
- Silhouette- or elbow-driven selection of `k` — fixed at 3 per the explicit
  business request.
- Model persistence / MLflow logging of the K-Means pipeline.
- A new row in CLAUDE.md §14's phase tracker.
- Any change to `01_eda.ipynb`, `02_churn_patterns.ipynb`, or
  `03_cohort_analysis.ipynb`.

---

## PART 2 — PLAN

### Approach

Add `src/models/segmentation.py` as a self-contained module reusing the
project's established `compute_* -> DataFrame`, `plot_* -> Path` (via
`src.data.eda.save_fig`) pattern, but built around a real `sklearn.Pipeline`
(`StandardScaler` → `KMeans`) rather than pure pandas, since this is genuinely
a fitted model rather than a binning/aggregation helper. Placed in
`src/models/` (not `src/data/`) per CLAUDE.md §4's directory purpose, and kept
independent of any future Phase 2 supervised-model code — same file, but a
distinct concern (unsupervised value segmentation vs. supervised churn
prediction) with no shared state between them.

**Alternative rejected:** placing this in `src/data/segmentation.py` next to
`cohorts.py`, on the grounds that both are "reporting-only, full-dataset"
features. Rejected because `cohorts.py` is pure `pd.cut` binning with no
fitted parameters, while this feature fits and persists cluster centers in
memory (`Pipeline.named_steps["kmeans"].cluster_centers_`) that later code
(e.g. `rank_segments_by_value`) depends on — that's a model artifact by
CLAUDE.md §4's own definition ("anything that trains or scores" →
`src/models/`), even though it isn't persisted to disk in this feature.

### Task breakdown

- [ ] **1. Create `src/models/segmentation.py`** — constants
      (`SERVICE_YESNO_COLUMNS`, `N_VALUE_SEGMENTS`, `VALUE_SEGMENT_LABELS`,
      `RANDOM_STATE`), `compute_service_count`, `build_segmentation_features`,
      `fit_segmentation_pipeline`, `rank_segments_by_value`,
      `assign_value_segment`, `segment_summary`, `silhouette_diagnostic`,
      `plot_segment_churn_rate`, `plot_segment_profile`,
      `generate_segmentation_figures`, `main()`. Import `save_fig`,
      `FIGURES_DIR` from `src.data.eda`/`src.data.config` to avoid duplicating
      the figure-saving helper (same reuse pattern as `cohorts.py`).
- [ ] **2. Run `python -m src.models.segmentation`** to generate
      `reports/figures/segment_churn_rate.png` and `segment_profile.png`;
      confirm via `git status`/`git diff --stat` that no existing figure file
      changed.
- [ ] **3. Create `notebooks/04_customer_segmentation.ipynb`** — bootstrap
      cell copied from `03_cohort_analysis.ipynb`; sections per Functional
      Requirement 12.
- [ ] **4. Add `tests/test_segmentation.py`** — cover Functional Requirement
      13 (service-count correctness, leakage guard, reproducibility, verified
      segment counts and the Mid-segment churn-rate finding).
- [ ] **5. Document the new entry point** — add
      `python -m src.models.segmentation` to CLAUDE.md §5's command list.
- [ ] **6. Run the full suite** — `pytest -q`, confirm all existing + new
      tests pass.
- [ ] **7. Commit** — `src/models/segmentation.py`,
      `notebooks/04_customer_segmentation.ipynb`,
      `tests/test_segmentation.py`,
      `reports/figures/segment_churn_rate.png`,
      `reports/figures/segment_profile.png`, `.claude/CLAUDE.md`, commit
      message `feat: RFM-style K-Means customer value segmentation`.

### Tests to write (hand to test-writer)

- `tests/test_segmentation.py::test_compute_service_count_range_and_no_zero` —
  on `clean_df`, `compute_service_count` returns values in `[0, 9]`, no
  nulls, and matches the verified observed range `[1, 9]` on current data.
- `tests/test_segmentation.py::test_build_segmentation_features_excludes_churn` —
  `set(build_segmentation_features(clean_df).columns) ==
  {"tenure", "MonthlyCharges", "ServiceCount"}` — `Churn` is never present
  (leakage guard).
- `tests/test_segmentation.py::test_fit_segmentation_pipeline_is_reproducible` —
  two independent calls to `fit_segmentation_pipeline(clean_df)` produce
  `np.array_equal` labels and `np.allclose` cluster centers.
- `tests/test_segmentation.py::test_segment_summary_order_and_customer_total` —
  `segment_summary(clean_df)["segment"].tolist() ==
  VALUE_SEGMENT_LABELS` (`["Low", "Mid", "High"]`); `customers.sum() ==
  len(clean_df)`.
- `tests/test_segmentation.py::test_segment_summary_matches_verified_counts` —
  `segment_summary(clean_df)` customers equal
  `{"Low": 2151, "Mid": 2575, "High": 2317}` on current data.
- `tests/test_segmentation.py::test_mid_segment_has_highest_churn_rate` —
  locks in the counter-intuitive finding: `segment_summary(clean_df)`'s
  `churn_rate` for `"Mid"` is strictly greater than both `"Low"` and
  `"High"` (`46.33 > {14.41, 15.80}` on current data).
- `tests/test_segmentation.py::test_silhouette_diagnostic_shape` —
  `silhouette_diagnostic(clean_df, k_values=[2,3,4])` returns one row per
  `k` with `silhouette_score` in `(-1, 1)`.

### Quality gates

1. **test-runner** — run `pytest -q`, confirm all tests (existing + new)
   pass, fix any straightforward regression.
2. **quality-reviewer** — review `segmentation.py`'s pipeline construction
   (scaling before K-Means, `n_init` pinned, the centroid-ranking logic that
   turns arbitrary cluster IDs into `Low`/`Mid`/`High`), the leakage guard
   (`Churn` never in `build_segmentation_features`), the notebook's
   reuse-not-duplicate structure, and CLAUDE.md §8 adherence (named
   constants, type hints, docstrings).
3. **security-reviewer** — confirm no new untrusted input path or dependency
   is introduced (there isn't one — `scikit-learn` is already approved).
4. Commit only after both reviews are clean.

### Risks / rollback

- **Risk:** a future Kaggle re-download shifts cluster composition enough
  that the verified segment counts or the Mid-segment-highest-churn finding
  no longer holds. **Mitigation:** intentional, matching `02`/`03`'s
  brittleness philosophy — these tests should fail loudly on a real
  distributional shift so the finding gets re-verified, not silently trusted.
- **Risk:** a reviewer mistakes the `k=3` business choice for a
  metric-optimal one. **Mitigation:** the silhouette diagnostic table and
  its explicit "disclosed, not decisive" framing (Functional Requirement 8,
  Non-goals) make this an intentional, stated design choice.
- **Risk:** someone later feeds `ValueSegment` into the Phase 2 supervised
  model without refitting K-Means inside CV, introducing subtle leakage.
  **Mitigation:** the forward-looking guardrail note in this spec's ML
  Guardrails section and Non-goals exists specifically so that decision, when
  made, is made with this constraint visible.
- **Rollback:** single commit (Task 7) covering only additive files (new
  module, new notebook, new tests, new PNGs, one CLAUDE.md doc line) — `git
  revert` is clean since nothing existing is modified in place.

### Definition of done

- All 7 tasks checked off.
- `pytest -q` green (all existing tests + 7 new `test_segmentation.py`
  tests).
- `notebooks/04_customer_segmentation.ipynb` executes top-to-bottom without
  error.
- `reports/figures/` gains 2 new PNGs, no existing one altered.
- CLAUDE.md §5 documents `python -m src.models.segmentation`.
- `quality-reviewer` and `security-reviewer` report no unresolved findings.
- All Success Criteria in Part 1 are met.
