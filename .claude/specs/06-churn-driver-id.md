# Spec + Plan: Churn Driver ID — Correlation, Chi-Square, SHAP Global Importance

> Location note: this file lives at `.claude/specs/` per CLAUDE.md §4's repo
> tree, which marks `.claude` as "claude.md and also Specs folder" — consistent
> with `01`–`05`. Spec and plan are combined in one file for the same reason.
>
> Scope note: this is genuinely **new capability**, and it is the first
> feature in the repo to fit a real *supervised* classifier. It is
> deliberately **not** CLAUDE.md §14's Phase 2 ("Feature pipeline, model
> training + comparison, MLflow") and **not** Phase 3 ("SHAP + LIME
> explainability, plain-English reasons"): Phase 2 is the production churn
> classifier (LogReg → RF → XGBoost → LightGBM, compared and tracked in
> MLflow); Phase 3 is per-customer *local* SHAP/LIME explanations wired to
> that production model. Neither exists yet. This feature fits its own small,
> throwaway diagnostic classifier — one XGBoost model, no comparison, no
> MLflow, no persistence — for the single purpose of ranking **global** churn
> drivers, exactly the way `segmentation.py` fits its own K-Means without
> touching Phase 2. No phase-tracker row changes, and Phase 2/3 remain fully
> unblocked and unaffected by this work.
>
> Methodology note: the requested three methods answer three different
> questions and are reported side by side rather than merged into one score:
> **correlation matrix** (linear association, numeric features only),
> **chi-square tests** (independence, *services* categoricals only — CLAUDE.md
> §6's own "Services" column bucket: `PhoneService`, `MultipleLines`,
> `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`,
> `TechSupport`, `StreamingTV`, `StreamingMovies`), and **SHAP global
> importance** (nonlinear, interaction-aware, covers every feature — numeric
> and categorical, service and non-service). SHAP is the only one of the
> three that can see `Contract` and `PaymentMethod` at all, since those are
> Account columns, not Services — this is why SHAP surfaces them as top
> drivers while the chi-square table (by design) does not test them. All
> numbers below are verified against `load_clean_data()`'s current 7,043 rows
> with the exact pipeline this feature implements, not estimated.
>
> Post-commit hardening note: this feature was implemented, reviewed, and
> committed (`72e00aa`), then given a second `quality-reviewer` pass on the
> committed diff, which found the first pass's SHAP-aggregation fix had no
> regression test, plus several robustness gaps. That second round of fixes
> — landed in a follow-up commit, not a rewrite of `72e00aa` — replaced the
> `_original_column_for` string-prefix-matching heuristic (and its
> prefix-collision tests) with `_feature_group_columns`, an exact mapping
> built from the fitted `OneHotEncoder`'s own `categories_` rather than
> parsing generated feature-name strings; pinned `shap==0.52.0` and
> `xgboost==3.4.1` in `requirements.txt` (CLAUDE.md §12); made
> `OneHotEncoder`'s `sparse_output=False` explicit rather than relying on
> `ColumnTransformer`'s density-threshold default; added a `shap_values.ndim
> == 2` guard so a future `shap`/`xgboost` version returning a different
> output shape fails loudly instead of silently computing wrong importances;
> logged the diagnostic model's AUC/PR-AUC on the `python -m
> src.explain.driver_analysis` CLI path (previously only visible via the
> notebook); and extracted the aggregation math into
> `_aggregate_shap_by_group`, unit-tested directly against a hand-computed
> matrix so a regression to the buggy method fails loudly. All verified
> numbers below were re-derived after these changes and are unchanged (the
> exact mapping is mathematically equivalent to the corrected heuristic it
> replaced — confirmed by direct before/after comparison, not assumed).

---

## PART 1 — SPEC

### Feature

A new `src/explain/driver_analysis.py` module that identifies global churn
drivers three ways: a numeric-feature correlation-with-`Churn` ranking, a
chi-square test of independence (with Cramér's V effect size) between each
of the 9 CLAUDE.md §6 "Services" categorical columns and `Churn`, and SHAP
`TreeExplainer` global feature importance from a dedicated diagnostic XGBoost
classifier (features aggregated back from one-hot dummies to their original
column names). Reports one table and one chart per method — consumed by a
new `notebooks/06_churn_driver_id.ipynb`.

### Problem / motivation

Nothing in the repo today ranks *which* features drive churn. `eda.py`'s
`plot_correlation_heatmap` shows a numeric heatmap but doesn't rank or
isolate the `Churn` row; nothing tests categorical association statistically;
nothing has fit a supervised model or computed SHAP values. Verified directly
on current data:

1. **Numeric correlation with `Churn`** (Pearson / point-biserial, on
   `tenure`, `MonthlyCharges`, `TotalCharges`): `tenure` = **−0.3522**,
   `TotalCharges` = **−0.1983**, `MonthlyCharges` = **+0.1934**. Longer
   tenure and higher cumulative spend both associate with *staying*; a higher
   *monthly* bill associates with leaving — consistent with CLAUDE.md §6's
   tenure signal.
2. **Chi-square tests on the 9 Services columns** (χ², p-value, Cramér's V),
   sorted by effect size: `OnlineSecurity` (V=0.3474), `TechSupport`
   (V=0.3429), `InternetService` (V=0.3225), `OnlineBackup` (V=0.2923),
   `DeviceProtection` (V=0.2816), `StreamingMovies` (V=0.2310), `StreamingTV`
   (V=0.2305), `MultipleLines` (V=0.0401), `PhoneService` (V=0.0114). All are
   statistically significant at α=0.05 **except `PhoneService`**
   (p=0.3388) — expected, since ~90% of customers have phone service
   regardless of churn status, so it carries almost no discriminating signal.
   This directly confirms CLAUDE.md §6's tech-support and internet-type
   signals with a formal test, not just an eyeballed bar chart.
3. **SHAP global importance** from a diagnostic XGBoost classifier
   (`n_estimators=200, max_depth=4, learning_rate=0.1, random_state=42`,
   `scale_pos_weight` computed from the train split for imbalance — no SMOTE,
   since CLAUDE.md §7 explicitly allows class weights as an alternative and
   this is a single throwaway diagnostic fit, not the Phase 2 production
   pipeline), fit on a stratified 80/20 split (`random_state=42`), scored
   honestly on the held-out 20%: **AUC-ROC = 0.8360, PR-AUC = 0.6569** — the
   AUC is *just under* CLAUDE.md §2's ~0.85–0.88 honest range (expected,
   since this is an untuned single model, not Phase 2's eventual
   compared-and-tuned one; PR-AUC has no equivalent named band) and far from
   the >0.95 leakage red flag. Top SHAP-ranked drivers (one-hot dummies
   aggregated back to their original column by summing each row's *signed*
   SHAP values across a column's dummies first, then taking the mean
   absolute value across rows — not the other way around, since summing
   each dummy's own `mean(|shap|)` independently overcounts features with
   more levels whenever dummy levels partially offset each other within a
   row): `Contract` (0.8763), `tenure` (0.6096), `MonthlyCharges` (0.3304),
   `OnlineSecurity` (0.2699), `TotalCharges` (0.2694), `InternetService`
   (0.2322), `PaymentMethod` (0.2128), `TechSupport` (0.2014) — **5 of
   CLAUDE.md §6's 5 documented sanity-check signals** (`Contract`, `tenure`,
   `TechSupport`, `PaymentMethod`, `InternetService`) land in the top 8,
   cross-validating the model against real domain knowledge exactly as
   CLAUDE.md §6 asks: *"if these don't show up in SHAP, something is
   wrong."* Cross-checked independently with a 300-tree
   `RandomForestClassifier` (not shipped, verification only) — same top
   signals in a different order (`Contract`, `tenure`, `TotalCharges`,
   `InternetService`, `MonthlyCharges`, `OnlineSecurity`, `TechSupport`,
   `PaymentMethod`), confirming the finding isn't an XGBoost-specific
   artifact.
4. **XGBoost was chosen over RandomForest for the shipped diagnostic model
   specifically because `TreeExplainer` is dramatically faster on it**:
   0.1s for SHAP on the 200-tree XGBoost model that produced the numbers
   above vs. several minutes for a 300-tree RandomForest on the same data —
   a practical engineering choice, not just an accuracy one, disclosed here
   since RandomForest is listed earlier in CLAUDE.md §7's model-comparison
   order.

### Goals / non-goals

**Goals**
- Add `src/explain/driver_analysis.py` with: a numeric correlation-ranking
  function, a chi-square/Cramér's V function scoped to the 9 Services
  columns, a leakage-safe diagnostic-model fitter with an explicit
  `check_auc_leakage_guard` (CLAUDE.md §2 rule 2, directly tested — not just
  documented), and a SHAP global-importance function that aggregates one-hot
  dummy importances back to human-readable original column names.
- Add three charts (one per method) and `generate_driver_figures`.
- Create `notebooks/06_churn_driver_id.ipynb` narrating all three methods,
  their agreement with CLAUDE.md §6's documented signals, and the
  XGBoost-vs-RandomForest `TreeExplainer` speed note, following `01`–`05`'s
  bootstrap-cell pattern.
- Add `python -m src.explain.driver_analysis`, documented in CLAUDE.md §5.
- Add pytest coverage in `tests/test_driver_analysis.py`, including a
  leakage guard on the model input, the `check_auc_leakage_guard` behavior
  itself, and a reproducibility check on the diagnostic model fit.
- Add `scipy` explicitly to `requirements.txt`: it's already an installed
  transitive dependency of `scikit-learn`, but this module is the first to
  `import scipy.stats` directly, so it should be pinned as a direct
  dependency rather than relied on implicitly (CLAUDE.md §3).

**Non-goals**
- No Phase 2 production model, no model comparison (LogReg/RF/XGBoost/
  LightGBM), no MLflow logging, no model persistence to `models/` — the
  diagnostic XGBoost model here exists only to produce SHAP values and is
  refit fresh every call, matching `segmentation.py`'s K-Means precedent.
- No Phase 3 local/per-customer explanations, no LIME, no plain-English
  natural-language reason strings, no `/explain` FastAPI endpoint — this
  feature is **global** driver ranking only.
- No hyperparameter tuning of the diagnostic XGBoost model — `n_estimators`,
  `max_depth`, `learning_rate` are fixed, disclosed constants chosen to be
  "reasonably good and fast," not grid-searched.
- No chi-square testing of non-Services categorical columns (`Contract`,
  `PaymentMethod`, `PaperlessBilling`, demographics) — deliberately scoped to
  the literal "categorical services" wording via CLAUDE.md §6's own bucket;
  SHAP still covers those columns since it runs on the full feature set.
- No new phase-tracker row in CLAUDE.md §14, and no change to
  `src/data/eda.py`, `src/data/cohorts.py`, `src/models/segmentation.py`,
  `src/data/lifecycle.py`, `clean_data()`'s output, or any existing
  test/figure.

### User stories

- As the **engineer (Priyabrata)**, I want a reusable, tested global
  driver-ranking module (not one-off notebook modeling) so the same
  diagnostic pattern — and its leakage guard — can inform how Phase 2's real
  feature set gets designed later.
- As a **churn analyst**, I want three independent methods (linear
  correlation, categorical association test, nonlinear SHAP importance) to
  agree on the top drivers, so I can trust the ranking isn't an artifact of
  any single method's assumptions.
- As a **recruiter/reviewer**, I want the honest AUC (0.8360, not
  suspiciously perfect), the explicit >0.95 leakage guard, and the
  XGBoost-vs-RandomForest speed disclosure stated plainly, so the analysis
  reads as rigorous and self-checking rather than a black-box importance
  plot.

### Functional requirements

1. `src/explain/driver_analysis.py` MUST define named constants:
   `SERVICE_COLUMNS` (the 9 columns listed above, CLAUDE.md §6's "Services"
   bucket), `RANDOM_STATE = 42`, `TEST_SIZE = 0.2`, `CHI_SQUARE_ALPHA =
   0.05`, `LEAKAGE_AUC_THRESHOLD = 0.95`, `XGB_N_ESTIMATORS = 200`,
   `XGB_MAX_DEPTH = 4`, `XGB_LEARNING_RATE = 0.1`, `TOP_N_SHAP_FEATURES =
   10` — no magic numbers inlined elsewhere (CLAUDE.md §8). MUST import
   `NUMERIC_COLUMNS`, `CHURN_PALETTE`, `save_fig` from `src.data.eda` rather
   than redefining the numeric-column list a second time.
2. MUST gain `numeric_correlation_with_churn(df: pd.DataFrame) ->
   pd.DataFrame` with columns `column`, `correlation`, `abs_correlation`
   (both rounded 4dp), one row per `NUMERIC_COLUMNS` entry, sorted by
   `abs_correlation` descending. MUST NOT include a `Churn`-vs-`Churn`
   self-correlation row.
3. MUST gain `chi_square_service_association(df: pd.DataFrame) ->
   pd.DataFrame` with columns `column`, `chi2`, `p_value`, `dof`,
   `cramers_v` (4dp), `significant` (bool, `p_value < CHI_SQUARE_ALPHA`),
   one row per `SERVICE_COLUMNS` entry, sorted by `cramers_v` descending.
   MUST use `scipy.stats.chi2_contingency` on `pd.crosstab(df[column],
   df[TARGET_COLUMN])`; Cramér's V computed as `sqrt(chi2 / (n * (min(rows,
   cols) - 1)))`.
4. MUST gain `build_driver_features(df: pd.DataFrame) -> tuple[pd.DataFrame,
   pd.Series]` returning `(X, y)` where `y = df[TARGET_COLUMN]` and `X` is
   `df` with `TARGET_COLUMN` dropped (strictly — must raise if absent) and
   `ID_COLUMN` dropped defensively (`errors="ignore"`, since
   `load_clean_data()` already drops it on the normal call path) — the
   leakage-guard surface, directly tested (`TARGET_COLUMN` MUST NOT appear
   in `X.columns`).
5. MUST gain `check_auc_leakage_guard(auc: float, threshold: float =
   LEAKAGE_AUC_THRESHOLD) -> None` that raises `ValueError` if `auc >
   threshold` — a mechanical, tested enforcement of CLAUDE.md §2 rule 2, not
   just a comment.
6. MUST gain `fit_driver_diagnostic_model(df: pd.DataFrame, random_state:
   int = RANDOM_STATE, test_size: float = TEST_SIZE) -> dict` returning
   `{"pipeline": Pipeline, "auc": float, "pr_auc": float, "X_test":
   pd.DataFrame, "y_test": pd.Series}`. MUST: split via
   `train_test_split(..., stratify=y, random_state=random_state)`; build a
   `Pipeline([("pre", ColumnTransformer([("num", StandardScaler(),
   NUMERIC_COLUMNS), ("cat", OneHotEncoder(handle_unknown="ignore"),
   <remaining columns>)])), ("clf", XGBClassifier(n_estimators=
   XGB_N_ESTIMATORS, max_depth=XGB_MAX_DEPTH, learning_rate=
   XGB_LEARNING_RATE, random_state=random_state, eval_metric="logloss",
   scale_pos_weight=<neg/pos ratio computed from y_train only>))])`; fit on
   the train split only; score `auc` (`roc_auc_score`) and `pr_auc`
   (`average_precision_score`) on the held-out test split; call
   `check_auc_leakage_guard(auc)` before returning.
7. MUST gain `shap_global_importance(df: pd.DataFrame, result: dict | None =
   None, top_n: int = TOP_N_SHAP_FEATURES) -> pd.DataFrame` with columns
   `column`, `mean_abs_shap` (4dp), top `top_n` rows sorted descending. If
   `result` is `None`, calls `fit_driver_diagnostic_model(df)` first.
   Computes `shap.TreeExplainer` values on the *transformed test split*
   (`result["X_test"]` through the fitted `pipeline`'s `"pre"` step) —
   evaluating global importance on held-out data, not the training fit.
   MUST aggregate one-hot dummy columns (`cat__<original>_<level>`) back to
   their original column name so `Contract` appears once, not fragmented
   across `Contract_Month-to-month` / `Contract_One year` / `Contract_Two
   year`. The aggregation MUST sum each row's *signed* SHAP values across a
   column's dummies first (SHAP values are additive, so this row's total is
   that feature's actual contribution to that prediction), and only then
   take the mean absolute value across rows — summing each dummy's own
   `mean(|shap value|)` independently instead would overcount features with
   more levels whenever dummy levels partially offset each other within a
   row (`|a| + |b| >= |a + b|`).
8. MUST gain `plot_numeric_correlation(df, out_dir=FIGURES_DIR) -> Path` —
   horizontal bar chart of `correlation` per numeric column (signed, not
   absolute), saved as `reports/figures/driver_correlation_with_churn.png`.
9. MUST gain `plot_chi_square_service_association(df, out_dir=FIGURES_DIR)
   -> Path` — bar chart of `cramers_v` per service column sorted descending,
   with the non-significant column (`PhoneService`) visually distinguished
   (e.g. different color/hatch) from the significant ones, saved as
   `reports/figures/driver_chi_square_service_association.png`.
10. MUST gain `plot_shap_global_importance(df, out_dir=FIGURES_DIR, result:
    dict | None = None) -> Path` — horizontal bar chart of the top
    `TOP_N_SHAP_FEATURES` from `shap_global_importance`, saved as
    `reports/figures/driver_shap_global_importance.png`.
11. MUST gain `generate_driver_figures(df, out_dir=FIGURES_DIR) ->
    list[Path]` — fits the diagnostic model once via
    `fit_driver_diagnostic_model` and reuses the `result` across the SHAP
    plot (not refit per chart), and a `main()` calling it via
    `load_clean_data()`, runnable as `python -m src.explain.driver_analysis`.
12. `notebooks/06_churn_driver_id.ipynb` MUST be created following
    `01`–`05`'s bootstrap-cell pattern. Sections, in order: numeric
    correlation table + chart → chi-square/Cramér's V table + chart (with
    the `PhoneService` non-significance callout) → diagnostic model AUC/PR-AUC
    (with the leakage-guard explanation) → SHAP global importance table +
    chart → cross-method agreement narration against CLAUDE.md §6's 5
    documented signals → key findings summary matching `01`–`05`'s
    closing-cell style.
13. `tests/test_driver_analysis.py` MUST cover: numeric correlation values
    match the verified figures (tenure ≈ −0.3522, TotalCharges ≈ −0.1983,
    MonthlyCharges ≈ +0.1934) and `Churn` is excluded from the output;
    chi-square Cramér's V values match the verified 9-column table (within
    `pytest.approx`) and are sorted descending; `PhoneService.significant ==
    False` and every other service column's `significant == True`;
    `build_driver_features` excludes `TARGET_COLUMN` from `X`;
    `check_auc_leakage_guard` raises `ValueError` above threshold and is a
    no-op at/below it; `fit_driver_diagnostic_model` is reproducible
    (`pytest.approx` on two independent fits' `auc`/`pr_auc`) and its `auc`
    is both `> 0.75` (honest-floor sanity check) and `<
    LEAKAGE_AUC_THRESHOLD`; `shap_global_importance` never includes
    `TARGET_COLUMN`, is sorted descending, all values `>= 0`, and its top-10
    set contains at least 4 of CLAUDE.md §6's 5 documented signal columns
    (`Contract`, `tenure`, `TechSupport`, `PaymentMethod`,
    `InternetService`); `generate_driver_figures` returns exactly 3 existing
    file paths.
14. None of the above may change `src/data/eda.py`, `src/data/cohorts.py`,
    `src/models/segmentation.py`, `src/data/lifecycle.py`, `clean_data()`'s
    output, or any existing test/figure — all current tests must keep
    passing unmodified.

### Data & model impact

First feature to fit a real supervised classifier, but it is a throwaway
diagnostic model: not persisted (`models/`/`*.pkl` stay git-ignored and
untouched), not logged to MLflow, not reused by any other module, and refit
fresh on every call — exactly like `segmentation.py`'s K-Means. `X` is built
from every column in `load_clean_data()`'s output except `TARGET_COLUMN`;
no new column is written back into the cleaned dataset. `src/features/`
(Phase 2, not yet built) is unaffected and shares no code with this module.

### ML guardrails (mandatory check)

- **No target/probability leakage:** `build_driver_features` (Requirement 4)
  drops only `TARGET_COLUMN` from `X`; nothing derived from `Churn` or a
  churn probability is ever a model input. Directly tested (Requirement 13).
- **Honest-AUC guard, mechanically enforced:** `check_auc_leakage_guard`
  (Requirement 5) raises `ValueError` if the diagnostic model's test AUC
  exceeds `LEAKAGE_AUC_THRESHOLD = 0.95`, per CLAUDE.md §2 rule 2 — not just
  documented, but called automatically inside `fit_driver_diagnostic_model`
  and directly tested with a fabricated over-threshold value.
- **Reproducibility:** `random_state=42` passed to both `train_test_split`
  and `XGBClassifier`. Verified: two independent fits on identical data
  produce bit-identical `auc`/`pr_auc` (confirmed by direct double-fit
  comparison during spec research, not assumed).
- **Splitting / preprocessing:** stratified `train_test_split` (80/20,
  `random_state=42`); `ColumnTransformer` (scale numeric, one-hot
  categorical) wrapped in a `Pipeline`, fit on the train split only —
  exactly CLAUDE.md §7's preprocessing convention, even though this
  pipeline is never persisted or reused elsewhere.
- **Imbalance handling:** `scale_pos_weight` computed from the train split
  only (never the full dataset) — CLAUDE.md §7 explicitly permits "class
  weights or SMOTE"; `scale_pos_weight` is XGBoost's native class-weighting
  mechanism, chosen over SMOTE because this is a single throwaway diagnostic
  fit, not the Phase 2 production pipeline that will need full
  SMOTE-inside-CV treatment.
- **Metric reporting:** `fit_driver_diagnostic_model` reports both
  `auc` (AUC-ROC) and `pr_auc` (PR-AUC) — never accuracy — consistent with
  CLAUDE.md §2 rule 3 and §7.
- **SHAP computed on held-out data:** `shap_global_importance` (Requirement
  7) explains the *test* split, not the training fit, so the reported global
  importance reflects generalization behavior rather than training-set
  overfitting artifacts.

### API / UI surface

None — no FastAPI endpoint or Streamlit view. `src/explain/driver_analysis.py`
stays notebook/report-facing, consistent with `cohorts.py`/`segmentation.py`/
`lifecycle.py`. (A future Phase 3 `/explain` endpoint, if built, would wire
up *local* per-customer SHAP against the real Phase 2 model — a separate,
unrelated code path from this global-driver module.)

### Edge cases & failure states

- **Diagnostic model AUC exceeds 0.95** (e.g. after a future data
  re-download introduces an accidental leak): `check_auc_leakage_guard`
  raises `ValueError` immediately inside `fit_driver_diagnostic_model`,
  surfacing the problem loudly at call time rather than silently reporting
  suspicious SHAP importances — directly tested with a fabricated value
  since it isn't reachable on current data.
- **One-hot column whose original name is a prefix of another's** (e.g. none
  currently collide, but `InternetService` and a hypothetical
  `InternetServiceType` would both start with `InternetService`): the
  aggregation in Requirement 7 matches a dummy column against the
  *longest* original column name whose name plus a trailing underscore
  (`"<col>_"`) the dummy's remainder starts with — not a bare `startswith`,
  since the underscore is what actually encodes the sklearn
  column/level-name boundary — to avoid mis-attributing SHAP mass; directly
  tested with a synthetic `InternetService`/`InternetServiceType` fixture,
  not just noted as a deliberate rule. A dummy name that matches no known
  categorical column raises `ValueError` rather than failing with an opaque
  `max() arg is an empty sequence` error — also directly tested.
- **`PhoneService` chi-square test is not statistically significant**
  (p=0.3388): reported as-is via the `significant` column rather than
  hidden or excluded — the honest negative result is itself informative
  (phone service isn't a differentiator) and directly tested (Requirement
  13).
- **`scale_pos_weight` computed from an already-stratified split**: uses
  `y_train` only (never `y_test` or the full `y`), so no test-fold
  information leaks into the class-weighting term.

### Security notes

None — no new untrusted input, secret, network call. One dependency
clarification: `scipy` moves from an implicit transitive dependency (via
`scikit-learn`) to an explicit, pinned direct dependency in
`requirements.txt`, since this module is the first to `import scipy.stats`
directly (CLAUDE.md §3). `shap` and `xgboost` are already listed in
`requirements.txt` (previously unused by any shipped module) — no new
package added, just the first real usage.

### Success criteria

- `pytest -q` passes: all existing tests + the new
  `tests/test_driver_analysis.py` module, all green.
- `notebooks/06_churn_driver_id.ipynb` runs top-to-bottom without error and
  renders every section named in Functional Requirement 12.
- `reports/figures/` gains `driver_correlation_with_churn.png`,
  `driver_chi_square_service_association.png`, and
  `driver_shap_global_importance.png`, with no existing filename changed.
- `python -m src.explain.driver_analysis` is a working, idempotent entry
  point, documented in CLAUDE.md §5.
- The diagnostic model's AUC is confirmed honest (< 0.95, reported not
  hidden) and its top SHAP drivers are confirmed to include at least 4 of
  CLAUDE.md §6's 5 documented signals.
- `quality-reviewer` and `security-reviewer` report no unresolved findings
  on the diff.

### Out of scope

- Phase 2's production model, model comparison, MLflow logging, or model
  persistence.
- Phase 3's local/per-customer SHAP+LIME explanations, plain-English reason
  strings, or the `/explain` FastAPI endpoint.
- Hyperparameter tuning of the diagnostic XGBoost model.
- Chi-square testing of non-Services categorical columns.
- A new row in CLAUDE.md §14's phase tracker.
- Any change to `01_eda.ipynb`, `02_churn_patterns.ipynb`,
  `03_cohort_analysis.ipynb`, `04_customer_segmentation.ipynb`, or
  `05_retention_funnel.ipynb`.

---

## PART 2 — PLAN

### Approach

Add `src/explain/driver_analysis.py` as the first module in `src/explain/`
(currently an empty package), following the established `compute_*`/`plot_*`
→ `Path` pattern (`save_fig` reused from `eda.py`) but adding one genuinely
new piece — a throwaway supervised model fit purely to produce SHAP values,
mirroring `segmentation.py`'s precedent of fitting its own unsupervised model
without touching Phase 2. XGBoost is chosen over RandomForest for the shipped
model specifically for `TreeExplainer` speed (0.1s vs. several minutes on
this dataset, verified during spec research), not just accuracy.

**Alternative rejected:** splitting this feature across two files — classical
statistics (correlation, chi-square) in `src/data/`, SHAP in `src/explain/`
— on the grounds that `src/data/` is EDA and `src/explain/` is explainability.
Rejected because all three methods answer the same question ("what drives
churn") and are meant to be read and compared together in one notebook; a
two-file split would force the notebook to import from two modules for one
narrative with no reuse benefit, unlike `segmentation.py`'s split from
`cohorts.py` (which was justified by a real fitted-model boundary, not a
topical one). The whole module lives in `src/explain/` since CLAUDE.md §4
earmarks that directory for "explanation logic," and global driver ranking
is explanation, even where two of the three methods are classical stats
rather than SHAP.

### Task breakdown

- [ ] **1. Add `scipy` to `requirements.txt`** — explicit, pinned direct
      dependency (was previously only an implicit transitive dependency of
      `scikit-learn`).
- [ ] **2. Create `src/explain/driver_analysis.py`** — constants
      (`SERVICE_COLUMNS`, `RANDOM_STATE`, `TEST_SIZE`, `CHI_SQUARE_ALPHA`,
      `LEAKAGE_AUC_THRESHOLD`, `XGB_N_ESTIMATORS`, `XGB_MAX_DEPTH`,
      `XGB_LEARNING_RATE`, `TOP_N_SHAP_FEATURES`),
      `numeric_correlation_with_churn`, `chi_square_service_association`,
      `build_driver_features`, `check_auc_leakage_guard`,
      `fit_driver_diagnostic_model`, `shap_global_importance` (with the
      one-hot-to-original aggregation helper), `plot_numeric_correlation`,
      `plot_chi_square_service_association`, `plot_shap_global_importance`,
      `generate_driver_figures`, `main()`. Import `NUMERIC_COLUMNS`,
      `CHURN_PALETTE`, `save_fig` from `src.data.eda`, `FIGURES_DIR` from
      `src.data.config`.
- [ ] **3. Run `python -m src.explain.driver_analysis`** to generate the 3
      new figures; confirm via `git status`/`git diff --stat` that no
      existing figure file changed.
- [ ] **4. Create `notebooks/06_churn_driver_id.ipynb`** — bootstrap cell
      copied from `05_retention_funnel.ipynb`; sections per Functional
      Requirement 12.
- [ ] **5. Add `tests/test_driver_analysis.py`** — cover Functional
      Requirement 13 (verified correlation/Cramér's V values, significance
      flags, leakage guards, reproducibility, honest-AUC bounds, SHAP
      top-driver membership).
- [ ] **6. Document the new entry point** — add
      `python -m src.explain.driver_analysis` to CLAUDE.md §5's command
      list.
- [ ] **7. Run the full suite** — `pytest -q`, confirm all existing + new
      tests pass.
- [ ] **8. Commit** — `src/explain/driver_analysis.py`,
      `notebooks/06_churn_driver_id.ipynb`, `tests/test_driver_analysis.py`,
      `reports/figures/driver_correlation_with_churn.png`,
      `reports/figures/driver_chi_square_service_association.png`,
      `reports/figures/driver_shap_global_importance.png`, `requirements.txt`,
      `.claude/CLAUDE.md`, commit message `feat: churn driver identification
      (correlation, chi-square, SHAP global importance)`.

### Tests to write (hand to test-writer)

- `tests/test_driver_analysis.py::test_numeric_correlation_matches_verified_values` —
  `numeric_correlation_with_churn(clean_df)` gives `tenure ≈ -0.3522`,
  `TotalCharges ≈ -0.1983`, `MonthlyCharges ≈ 0.1934` (`pytest.approx`,
  4dp), sorted by `abs_correlation` descending in that order.
- `tests/test_driver_analysis.py::test_numeric_correlation_excludes_churn_row` —
  `"Churn"` never appears in the `column` column.
- `tests/test_driver_analysis.py::test_chi_square_service_association_matches_verified_cramers_v` —
  matches the verified 9-value table (`OnlineSecurity` 0.3474 down to
  `PhoneService` 0.0114, `pytest.approx`), sorted descending.
- `tests/test_driver_analysis.py::test_chi_square_phone_service_not_significant` —
  `PhoneService` row has `significant == False`; every other service column
  has `significant == True`.
- `tests/test_driver_analysis.py::test_build_driver_features_excludes_target` —
  `TARGET_COLUMN not in build_driver_features(clean_df)[0].columns`.
- `tests/test_driver_analysis.py::test_check_auc_leakage_guard_raises_above_threshold` —
  `check_auc_leakage_guard(0.97)` raises `ValueError`;
  `check_auc_leakage_guard(0.83)` and `check_auc_leakage_guard(0.95)` do not
  raise.
- `tests/test_driver_analysis.py::test_fit_driver_diagnostic_model_is_reproducible` —
  two independent `fit_driver_diagnostic_model(clean_df)` calls give
  `pytest.approx`-equal `auc` and `pr_auc`.
- `tests/test_driver_analysis.py::test_fit_driver_diagnostic_model_auc_is_honest` —
  `0.75 < result["auc"] < LEAKAGE_AUC_THRESHOLD` on current data (locks in
  ≈0.8360, with headroom for minor library-version drift).
- `tests/test_driver_analysis.py::test_shap_global_importance_excludes_target_and_is_sorted` —
  `"Churn"` never in `shap_global_importance(clean_df)["column"]`; all
  `mean_abs_shap >= 0`; sorted descending.
- `tests/test_driver_analysis.py::test_shap_global_importance_recovers_domain_signals` —
  top-10 `column` set intersected with `{"Contract", "tenure",
  "TechSupport", "PaymentMethod", "InternetService"}` has size `>= 4`.
- `tests/test_driver_analysis.py::test_original_column_for_resolves_prefix_collision` —
  a synthetic `InternetService`/`InternetServiceType` pair each map a
  `cat__<col>_<level>` name to their own column, not each other's.
- `tests/test_driver_analysis.py::test_original_column_for_raises_on_unmatched_feature` —
  a `cat__` feature name matching no known categorical column raises
  `ValueError` rather than an opaque `max()`-on-empty error.
- `tests/test_driver_analysis.py::test_shap_global_importance_covers_every_original_column_once` —
  `shap_global_importance(clean_df, top_n=<all columns>)` returns exactly
  one row per original `X` column, unique, with no `cat__`/`num__` prefix
  surviving into `column`.
- `tests/test_driver_analysis.py::test_generate_driver_figures_returns_three_existing_paths` —
  returns exactly 3 `Path`s, all `.exists()`; writes to a pytest `tmp_path`
  rather than the tracked `reports/figures/` so the test suite never
  produces a phantom `git status` diff.

### Quality gates

1. **test-runner** — run `pytest -q`, confirm all tests (existing + new)
   pass, fix any straightforward regression.
2. **quality-reviewer** — review the leakage guard (`build_driver_features`,
   `check_auc_leakage_guard` called before returning from
   `fit_driver_diagnostic_model`), the stratified-split-then-fit-on-train-
   only preprocessing, the one-hot-to-original SHAP aggregation logic, and
   CLAUDE.md §8 adherence (named constants, type hints, docstrings).
3. **security-reviewer** — confirm no new untrusted input path; confirm the
   `scipy` requirements.txt addition is the only dependency-surface change
   and is justified (already an implicit transitive dependency).
4. Commit only after both reviews are clean.

### Risks / rollback

- **Risk:** a future Kaggle re-download shifts correlation, Cramér's V, or
  SHAP top-driver membership enough that the verified/pinned test values no
  longer hold. **Mitigation:** intentional, matching `02`–`05`'s
  brittleness philosophy — these tests should fail loudly on real
  distributional shift. SHAP-specific values use looser bounds (honest-floor
  AUC range, top-10 set-membership rather than exact magnitudes) precisely
  because SHAP output is more sensitive to library-version drift than a
  deterministic pandas/scipy calculation — this asymmetry is intentional,
  not an oversight.
- **Risk:** a reviewer assumes this diagnostic XGBoost model *is* (or will
  become) the Phase 2 production model. **Mitigation:** the scope note at
  the top of this spec and the Non-goals section state explicitly that it's
  a throwaway fit, never persisted, never compared against other algorithms,
  and unrelated to whatever Phase 2 builds.
- **Risk:** `check_auc_leakage_guard` fires unexpectedly on a legitimate
  future retrain (e.g. after Phase 2 feature engineering genuinely improves
  signal quality). **Mitigation:** this module is disconnected from Phase 2
  entirely — the guard only ever runs against this module's own
  never-tuned diagnostic model, so a Phase 2 model exceeding 0.95 (which
  would itself be suspicious per CLAUDE.md §2 rule 2) is a separate,
  future concern, not one this guard interferes with.
- **Rollback:** single commit (Task 8) covering only additive files (new
  module, new notebook, new tests, 3 new PNGs, one `requirements.txt` line,
  one CLAUDE.md doc line) — `git revert` is clean since nothing existing is
  modified in place.

### Definition of done

- All 8 tasks checked off.
- `pytest -q` green (all existing tests + `test_driver_analysis.py`'s
  tests, including the ID_COLUMN-drop, `_feature_group_columns`
  exact-mapping, and `_aggregate_shap_by_group` regression tests added
  during the post-commit quality-review hardening pass — see the note at
  the top of this file).
- `notebooks/06_churn_driver_id.ipynb` executes top-to-bottom without error.
- `reports/figures/` gains 3 new PNGs, no existing one altered.
- CLAUDE.md §5 documents `python -m src.explain.driver_analysis`.
- `quality-reviewer` and `security-reviewer` report no unresolved findings.
- All Success Criteria in Part 1 are met.
