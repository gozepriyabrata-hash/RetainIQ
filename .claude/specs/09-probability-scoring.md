# Spec + Plan: Probability Scoring — Per-Customer Churn Probability (0–100%) with Calibrated Confidence

> Location note: this file lives at `.claude/specs/` per CLAUDE.md §4's repo
> tree ("`.claude` — claude.md and also Specs folder"), consistent with
> `01`–`08`. Spec and plan are combined in one file for the same reason.
> Numbered `09` (matching the feature request's own `09-Probability-Scoring`
> label) since `08` is already `08-churn-prediction-model.md`.
>
> Scope note: this is **not** a numbered CLAUDE.md §14 phase row by itself.
> It is a bridge between Phase 2 (closed — a real, compared, persisted,
> MLflow-tracked classifier already exists at `models/churn_model.pkl`) and
> Phase 5 ("FastAPI service + Streamlit dashboard + What-If panel"), in the
> same spirit as how `02`–`07` extended Phase 1 without claiming a phase row.
> This spec turns that persisted model into a *scoring capability* — take a
> customer's attributes in, get a calibrated churn probability out — without
> building the endpoint or UI that will eventually call it. It is
> deliberately **not** Phase 3 (SHAP/LIME *why* a customer is at risk — this
> spec produces a number, not a reason), **not** Phase 4 (risk tiers,
> Next-Best-Action — this spec stops at the probability itself), and **not**
> Phase 5 itself (no FastAPI route, no Streamlit view — those still have
> nothing to call into yet after this spec, exactly like `08` added
> `load_trained_model` "but wired it to nothing").
>
> Interpretation note: the feature request's "calibrated confidence" is
> read here as **probability calibration** in the standard ML sense (a
> predicted 70% should mean ~70% of such customers actually churn — measured
> by Brier score, a reliability diagram, and Expected Calibration Error),
> **not** a separate uncertainty interval (e.g. a bootstrapped confidence
> band around the point estimate). This matches CLAUDE.md's own existing
> vocabulary — `src/models/evaluation.py` already reports Brier score as a
> guardrail metric — and there is no mention anywhere in CLAUDE.md of
> prediction intervals. Flagged explicitly per the create-spec workflow's
> instruction to state an assumption rather than silently pick one; a
> confidence-interval reading is out of scope (see Out of scope).
>
> Research note: every number below (train/test split sizes, raw vs.
> calibrated Brier/AUC/ECE, calibration-method comparison, reproducibility)
> was verified by actually running the calibration against the real
> persisted `models/churn_model.pkl` (XGBoost, `trained_at`
> 2026-08-22T05:33:17Z) and `load_clean_data()`'s current 7,043-row dataset
> during spec research — not estimated.
>
> Verified finding — isotonic beats sigmoid here, and calibration is worth
> doing: on the exact `split_data(df)` test split (1,409 rows, 26.54% churn)
> the currently-persisted XGBoost pipeline's **raw** probabilities score
> Brier **0.1466**, AUC-ROC **0.8434**, 10-bin Expected Calibration Error
> **0.0803**. Wrapping the same pipeline architecture in
> `CalibratedClassifierCV` (5-fold `StratifiedKFold`, `random_state=42`,
> fit on the train split only) and re-scoring the same untouched test split:
> **isotonic** → Brier **0.1376**, AUC-ROC 0.8425, ECE **0.0301**; **sigmoid**
> → Brier 0.1383, AUC-ROC 0.8429, ECE not computed but its reliability curve
> is visibly less linear than isotonic's. Isotonic wins on both calibration
> metrics that matter (Brier and ECE) at a negligible AUC-ROC cost (-0.0009,
> well within noise), and the training-fold size (5,634 rows, ~1,494 churners)
> comfortably clears isotonic regression's usual "don't use on tiny folds"
> caveat. Two independent fits with the same `random_state` produced
> bit-identical calibrated probabilities (`np.allclose` true), confirming
> reproducibility holds through `CalibratedClassifierCV` the same way it
> already does through `train.py`'s own fits.

---

## PART 1 — SPEC

### Feature

A calibration and scoring layer on top of the existing persisted churn
model: (1) `CalibratedClassifierCV`-wraps the winning model architecture
(currently XGBoost) using isotonic regression, cross-validated on the
training split only, to produce probabilities that are honestly close to
observed frequencies rather than just rank-ordered; (2) a `score_customers`
/ `score_single_customer` API that takes one or many customers' raw
attributes and returns a calibrated churn probability as both a 0–1 float
and a 0–100% figure, ready for a future Phase 5 endpoint or dashboard view
to call directly.

### Problem / motivation

`models/churn_model.pkl` (Phase 2) outputs `predict_proba` values that are
**honestly ranked** (AUC-ROC 0.8434, already leakage-checked) but, verified
above, **not well calibrated**: a customer scored "70%" by the raw model is
not actually a ~70%-likely churner — the raw ECE is 0.0803, meaning
predictions are off from observed frequency by ~8 percentage points on
average across probability bins. CLAUDE.md §1 promises "explain why, and
recommend what to do about it" and a dashboard showing "calibrated churn
probabilities" (§14 Definition of done) — a retention manager deciding
whether a 70%-flagged customer is worth an expensive save-offer needs that
70% to actually mean 70%, not an arbitrary score that merely ranks
customers correctly relative to each other. Nothing in the repo today turns
a customer's attributes into a probability number at all — `train.py`
persists a model but nothing calls `.predict_proba()` on new input yet.
This spec is that missing scoring layer, calibrated so the number is
trustworthy on its own, not just as a ranking.

### Goals / non-goals

**Goals**
- Add `expected_calibration_error` (+ its `ECE_BINS` constant) to
  `src/models/evaluation.py`, alongside the existing dependency-light metric
  functions.
- Add `src/models/calibration.py`: fits an isotonic `CalibratedClassifierCV`
  around the currently-winning model architecture (read from
  `churn_model_metadata.json`, not hardcoded), evaluates it against the raw
  persisted model on the untouched test split (Brier, AUC-ROC, ECE, a
  reliability diagram), logs the calibrated run to the same MLflow
  experiment as `train.py`, and persists
  `models/churn_model_calibrated.pkl` + `models/churn_model_calibrated_metadata.json`.
- Add `python -m src.models.calibration` (documented in CLAUDE.md §5,
  matching `08`'s `python -m src.models.train` precedent).
- Commit two tracked artifacts: `reports/calibration_metrics.json` and
  `reports/figures/calibration_reliability.png` — the reliability diagram
  and headline numbers, visible without opening the MLflow UI (mirrors
  `08`'s `reports/model_comparison.csv` precedent).
- Refactor `src/data/load_data.py`: factor whitespace-stripping,
  `TotalCharges` coercion/imputation, and `SeniorCitizen` normalization out
  of `clean_data()` into a shared `_clean_common_fields()` helper, and add
  `prepare_scoring_input(raw_df) -> (features_df, customer_ids)` on top of
  it — a training-schema-free cleaning path for **unlabeled** inference
  input (no `Churn` column expected), so this repeats zero cleaning logic
  rather than re-deriving the same `TotalCharges`-blank handling ad hoc in a
  second place.
- Add `src/models/scoring.py`: `score_customers(raw_df, ...) -> pd.DataFrame`
  and `score_single_customer(customer: dict, ...) -> dict`, using the
  calibrated model by default, with clear validation errors on missing
  required columns.
- Add `notebooks/08_probability_calibration.ipynb` following `01`–`07`'s
  bootstrap-cell pattern.
- Add `tests/test_calibration.py`, `tests/test_scoring.py`; extend
  `tests/test_evaluation.py` (ECE) and `tests/test_data.py`
  (`prepare_scoring_input`).

**Non-goals**
- No FastAPI endpoint, no Streamlit view, no model loaded at any app
  startup — Phase 5, not this spec. `src/api/` and `app/` are untouched.
  `score_customers`/`score_single_customer` are added as the functions a
  future `/predict` route will call, exactly as `08`'s `load_trained_model`
  was added but wired to nothing.
- No SHAP/LIME, no per-customer explanation text — Phase 3, not this spec.
  `src/explain/` is untouched.
- No risk tiers, no Next-Best-Action — Phase 4, not this spec.
  `src/recommend/` stays an empty package.
- No re-comparison of the four `MODEL_SPECS` candidates and no change to
  which one wins — this spec calibrates whichever model
  `churn_model_metadata.json` already names as the winner (currently
  XGBoost); it does not re-run `compare_models`.
- No change to `clean_data()`'s **output** (all of `tests/test_data.py`
  passes unmodified) — only its internals are refactored to share logic
  with the new `prepare_scoring_input()`.
- No bootstrapped/quantile confidence intervals around the point estimate
  (see the Interpretation note above) — "calibrated confidence" means a
  trustworthy probability, not a probability plus an error bar.
- No new third-party dependency: `CalibratedClassifierCV` and
  `calibration_curve` are already part of the pinned `scikit-learn==1.6.1`.

### User stories

- As a **retention manager**, I want a customer's churn score to mean what
  it says (a 70% score customer churns about 70% of the time historically),
  so I can size a retention offer's expected value correctly instead of
  trusting an arbitrary rank.
- As the **engineer (Priyabrata)**, I want one command
  (`python -m src.models.calibration`) that calibrates whichever model
  `train.py` most recently selected and reports the before/after
  improvement, so recalibration is a single reproducible step after any
  retrain, not ad hoc notebook code.
- As the **engineer**, I want `score_customers`/`score_single_customer`
  functions with a stable, documented contract now, so Phase 5's FastAPI
  `/predict` endpoint is a thin wrapper around already-tested logic instead
  of reimplementing scoring inline in the API layer (CLAUDE.md §4: "Keep the
  API... thin").
- As a **recruiter/reviewer**, I want the calibration improvement reported
  honestly with real before/after numbers and a reliability diagram (not
  just "we calibrated it"), so the work reads as measured, not asserted.

### Functional requirements

1. `src/models/evaluation.py` MUST gain `ECE_BINS = 10` and
   `expected_calibration_error(y_true: ArrayLike, y_proba: ArrayLike, n_bins:
   int = ECE_BINS) -> float`: bins `y_proba` into `n_bins` equal-width bins
   over `[0, 1]`, and returns the sample-size-weighted mean absolute gap
   between each bin's mean predicted probability and its observed positive
   rate (`Σ (bin_count / total) * |mean(y_true in bin) - mean(y_proba in
   bin)|`). Empty bins contribute 0, not `NaN`. No MLflow/filesystem
   dependency, matching this module's existing style.
2. `src/data/load_data.py` MUST gain a private `_clean_common_fields(df:
   pd.DataFrame) -> pd.DataFrame` performing exactly the three
   dataset-quirk fixes `clean_data()` already does today — object-column
   whitespace stripping, `TotalCharges` → numeric with blanks imputed to
   `0.0`, and `SeniorCitizen` normalized to the `"Yes"`/`"No"` vocabulary
   (only if its dtype is not already `object`, so a caller that already
   passes `"Yes"`/`"No"` is left untouched — a generalization `clean_data()`
   never previously needed since the raw CSV always encodes it numerically,
   verified not to change `clean_data()`'s output on `data/raw/telco.csv`).
   `clean_data()` MUST be refactored to call this helper for those three
   steps, then continue with `Churn` target-mapping and `ID_COLUMN` drop as
   today — its return value on the real dataset is byte-for-byte unchanged
   (all of `tests/test_data.py` passes with zero edits).
3. `src/data/load_data.py` MUST gain `prepare_scoring_input(raw_df:
   pd.DataFrame) -> tuple[pd.DataFrame, pd.Series | None]`: copies `raw_df`,
   captures `raw_df[ID_COLUMN]` as `customer_ids` if `ID_COLUMN` is present
   (else `None`), runs `_clean_common_fields`, then drops `ID_COLUMN` and
   `TARGET_COLUMN` if either is present (`errors="ignore"` — an inference
   payload has no label and may or may not carry `customerID`), and returns
   `(features_df, customer_ids)`. Does not require a `Churn` column, unlike
   `clean_data()`.
4. `src/models/calibration.py` MUST define `CALIBRATION_METHOD = "isotonic"`
   (the verified winner over `"sigmoid"` — Research note above) and reuse
   `train.CV_FOLDS` (5) and `evaluation.RANDOM_STATE` (42) rather than
   redefining them — no new magic numbers (CLAUDE.md §8).
5. MUST gain `build_calibration_template(model_name: str, categorical_columns:
   list[str]) -> imblearn.pipeline.Pipeline` = `train.build_model_pipeline(
   train.MODEL_SPECS[model_name], categorical_columns)` — a thin,
   one-line reuse of `08`'s existing pipeline builder, never a duplicated
   `ColumnTransformer`/`SMOTE`/estimator construction.
6. MUST gain `fit_calibrated_pipeline(pipeline_template: Pipeline, X_train:
   pd.DataFrame, y_train: pd.Series, method: str = CALIBRATION_METHOD, cv:
   int = train.CV_FOLDS, random_state: int = RANDOM_STATE) ->
   CalibratedClassifierCV`: `CalibratedClassifierCV(estimator=
   pipeline_template, method=method, cv=StratifiedKFold(n_splits=cv,
   shuffle=True, random_state=random_state)).fit(X_train, y_train)`. Fit on
   the train split only — `CalibratedClassifierCV`'s internal CV clones and
   refits `pipeline_template` (preprocessing + SMOTE + estimator) on each
   fold's training portion and calibrates on that fold's held-out portion,
   so `X_test`/`y_test` are never touched by this function, matching
   CLAUDE.md §7's split discipline.
7. MUST gain `evaluate_calibration(y_true: ArrayLike, proba_raw: ArrayLike,
   proba_calibrated: ArrayLike) -> dict` returning `{"brier_raw": float,
   "brier_calibrated": float, "auc_raw": float, "auc_calibrated": float,
   "ece_raw": float, "ece_calibrated": float}` via
   `evaluation.compute_classification_metrics` and
   `evaluation.expected_calibration_error` — never a single "improved" flag
   without the underlying numbers.
8. MUST gain `plot_reliability_curve(y_true: ArrayLike, proba_raw:
   ArrayLike, proba_calibrated: ArrayLike, out_dir: Path = FIGURES_DIR) ->
   Path`: a reliability diagram (`sklearn.calibration.calibration_curve`,
   `n_bins=ECE_BINS`, `strategy="uniform"`, matching Requirement 1's binning)
   plotting the perfect-calibration diagonal plus both the raw and
   calibrated curves, saved to
   `reports/figures/calibration_reliability.png`.
9. MUST gain `run_calibration_pipeline(df: pd.DataFrame) -> dict`
   orchestrating: read `train.DEFAULT_METADATA_PATH` for the currently
   winning `model_name` and its `feature_columns`; `train.split_data(df)`
   for the identical train/test partition `train.py` used (same
   `random_state`/`test_size`, deterministic given the same `df`); assert
   `list(X_train.columns) == metadata["feature_columns"]`, raising a clear
   `ValueError` on mismatch (a staleness guard — protects against
   calibrating against a `df` that no longer matches what
   `churn_model.pkl` was actually trained on); `get_categorical_columns`;
   `build_calibration_template` → `fit_calibrated_pipeline` (Requirements
   5–6); `proba_raw = train.load_trained_model().predict_proba(X_test)[:, 1]`
   and `proba_calibrated = calibrated.predict_proba(X_test)[:, 1]`;
   `evaluate_calibration` (Requirement 7) and `plot_reliability_curve`
   (Requirement 8); log one MLflow run named `f"{model_name}_calibrated"`
   (same `train.MLFLOW_TRACKING_URI`/`MLFLOW_EXPERIMENT_NAME`, params
   `{"base_model_name": model_name, "calibration_method":
   CALIBRATION_METHOD, "cv_folds": train.CV_FOLDS}`, metrics from
   Requirement 7); persist the calibrated pipeline + metadata (Requirement
   10); write `reports/calibration_metrics.json` (Requirement 7's dict plus
   `"base_model_name"` and `"calibration_method"`); return `{"base_model_name":
   str, "calibration_metrics": dict, "model_path": Path, "metadata_path":
   Path, "figure_path": Path, "metrics_json_path": Path}`.
10. MUST gain `save_calibrated_artifact(model_name: str, calibrated:
    CalibratedClassifierCV, calibration_metrics: dict, feature_columns:
    list[str], model_path: Path = DEFAULT_CALIBRATED_MODEL_PATH,
    metadata_path: Path = DEFAULT_CALIBRATED_METADATA_PATH) -> tuple[Path,
    Path]` and `load_calibrated_model(model_path: Path =
    DEFAULT_CALIBRATED_MODEL_PATH) -> CalibratedClassifierCV` — same
    `joblib.dump`/`joblib.load` + JSON-sidecar shape as `train.py`'s
    `save_model_artifact`/`load_trained_model` (Requirement 12–13 of `08`),
    with the metadata JSON additionally carrying `"base_model_name"` and
    `"calibration_method"`. `load_calibrated_model` MUST raise a
    `FileNotFoundError` naming `python -m src.models.calibration` (not
    `src.models.train`) when the artifact is absent.
11. `src/models/scoring.py` MUST gain `PROBABILITY_DECIMALS = 4` and
    `PERCENTAGE_DECIMALS = 1` constants (no magic rounding numbers inline).
12. MUST gain `score_customers(raw_df: pd.DataFrame, pipeline:
    "Pipeline | CalibratedClassifierCV | None" = None, use_calibrated: bool
    = True) -> pd.DataFrame`: calls `prepare_scoring_input(raw_df)`
    (Requirement 3); loads `metadata = json.load(...)` from
    `calibration.DEFAULT_CALIBRATED_METADATA_PATH` if `use_calibrated` else
    `train.DEFAULT_METADATA_PATH`, to get `feature_columns`; if any
    `feature_columns` entry is missing from the prepared features, raises
    `ValueError` naming every missing column; reindexes
    `features_df[feature_columns]`; uses `pipeline` if given (already
    loaded — lets a future caller load once and reuse, CLAUDE.md §10's
    "load the model once, not per request," without this spec building the
    actual server that would do so), else loads via
    `calibration.load_calibrated_model()` or `train.load_trained_model()`
    per `use_calibrated`; computes `proba =
    pipeline.predict_proba(features_df)[:, 1]`; returns a `DataFrame` with
    columns `customerID` (from Requirement 3's `customer_ids`, omitted
    entirely if `None`), `churn_probability` (rounded to
    `PROBABILITY_DECIMALS`), `churn_probability_pct` (`proba * 100`, rounded
    to `PERCENTAGE_DECIMALS`, clipped to `[0.0, 100.0]`) — same row order and
    count as `raw_df`, including the empty-input case (`0` rows in, `0` rows
    out, not an error).
13. MUST gain `score_single_customer(customer: dict, pipeline=None,
    use_calibrated: bool = True) -> dict`: `score_customers(pd.DataFrame(
    [customer]), pipeline, use_calibrated).iloc[0].to_dict()` — the
    single-row convenience wrapper a future `/predict` endpoint will call
    directly.
14. `notebooks/08_probability_calibration.ipynb` MUST follow `01`–`07`'s
    bootstrap-cell pattern. Sections, in order: problem framing (raw model
    ranks well but isn't calibrated — cite the verified ECE) → reliability
    diagram walkthrough (raw vs. isotonic vs. sigmoid) → why isotonic was
    chosen (Brier + ECE both better, fold size comfortably supports it) →
    before/after metrics table → example `score_customers` output on a
    handful of real customers → saved-artifact summary → key findings
    closing cell.
15. `tests/test_calibration.py`, `tests/test_scoring.py` MUST cover the
    Plan's "Tests to write" section in full; `tests/test_evaluation.py` and
    `tests/test_data.py` gain the ECE and `prepare_scoring_input` cases
    listed there.
16. None of the above may change `src/data/eda.py`, `src/data/cohorts.py`,
    `src/models/segmentation.py`, `src/data/lifecycle.py`,
    `src/explain/driver_analysis.py`, `src/models/train.py`'s public
    behavior, `clean_data()`'s output, or any existing test/figure/notebook
    — all current tests must keep passing unmodified.

### Data & model impact

Second model artifact in the repo. `models/` gains
`churn_model_calibrated.pkl` (a fitted `CalibratedClassifierCV` wrapping the
same preprocessing+SMOTE+estimator architecture as `churn_model.pkl`, just
recalibrated) and `churn_model_calibrated_metadata.json`, both git-ignored
like their Phase 2 counterparts. `churn_model.pkl` itself is untouched —
this spec adds a second, better-calibrated artifact rather than overwriting
the first, so anything already depending on the Phase 2 contract keeps
working. No feature column is added, removed, or renamed anywhere;
`feature_columns` for the calibrated model is identical to `churn_model.pkl`'s.
`reports/` gains two **tracked** files: `calibration_metrics.json` and
`figures/calibration_reliability.png`. `mlruns/`/`mlflow.db` gain one more
run per `python -m src.models.calibration` invocation, appended to the same
`retainiq-churn-classifier` experiment `train.py` already logs to.

### ML guardrails (mandatory check)

- **No target/probability leakage:** `prepare_scoring_input` (Requirement
  3) drops `TARGET_COLUMN` if present and never derives a feature from it;
  `score_customers`' output columns (`churn_probability`,
  `churn_probability_pct`) are model *outputs*, never fed back in as
  inputs anywhere in this spec.
- **Honest-AUC guard stays intact:** calibration is fit on top of the
  already-leakage-checked `churn_model.pkl` architecture; `evaluate_calibration`
  (Requirement 7) reports `auc_calibrated` (0.8425, verified) which remains
  comfortably `< 0.95` — no new leakage surface is introduced by wrapping an
  already-checked pipeline in `CalibratedClassifierCV`. This spec does not
  re-invoke `check_auc_leakage_guard` itself (that guard lives in
  `evaluate_candidate`, a `train.py`-only concern per `08`'s scope), but the
  verified AUC is disclosed in the Research note and reliability metrics.
- **No fitting on the full dataset before a split:** `run_calibration_pipeline`
  (Requirement 9) calls `train.split_data(df)` — the identical
  stratified 80/20 split `train.py` used — and `fit_calibrated_pipeline`
  (Requirement 6) fits only on `X_train`/`y_train`; `X_test`/`y_test` are
  used exactly once, for `evaluate_calibration`, never for fitting the
  calibrator.
- **SMOTE stays inside the pipeline:** `build_calibration_template`
  (Requirement 5) reuses `train.build_model_pipeline`, which places `SMOTE`
  inside the same `imblearn.Pipeline` `CalibratedClassifierCV` clones per
  fold — SMOTE is refit fresh on each fold's training portion only, the
  same structural guarantee `08` already established, not a new mechanism.
- **Reproducibility:** `random_state=RANDOM_STATE` (42) passed through
  `fit_calibrated_pipeline`'s internal `StratifiedKFold`. Verified during
  spec research: two independent `fit_calibrated_pipeline` calls on
  identical data produce `np.allclose`-identical calibrated test-set
  probabilities.
- **Metric reporting:** `evaluate_calibration` (Requirement 7) reports
  Brier score, AUC-ROC, and ECE for both raw and calibrated — never a bare
  "it's calibrated now" claim without numbers, consistent with CLAUDE.md §2
  rule 3's "never accuracy alone" spirit extended to calibration claims.

### API / UI surface

None shipped. `score_customers`/`score_single_customer` (Requirements
12–13) are added as the exact functions a future Phase 5 `/predict` route
will call — `raw_df` → `DataFrame`/`dict` out, already matching the shape
CLAUDE.md §10's `POST /predict` response ("churn probability + risk tier,"
minus the risk tier, which is Phase 4's job) will need. No FastAPI route or
Streamlit view is wired up; `src/api/` and `app/` are untouched by this
spec.

### Edge cases & failure states

- **`score_customers` called with an empty `DataFrame` (0 rows):** returns
  an empty `DataFrame` with the correct output columns, not an error or a
  shape mismatch — directly tested.
- **A required feature column is missing from `raw_df`:** `score_customers`
  raises `ValueError` naming every missing column (not just the first) —
  directly tested. This is the "empty/malformed CSV" failure state from
  CLAUDE.md §9's testing expectations, generalized to any missing-column
  case.
- **`raw_df` has blank `TotalCharges` strings**, the same known quirk as
  the 11 training rows: `prepare_scoring_input` → `_clean_common_fields`
  coerces and imputes to `0.0`, identical handling to `clean_data()` —
  directly tested with a synthetic blank-string row.
- **`raw_df` carries extra columns** (e.g. a re-uploaded `clean_df` still
  containing `Churn`, or an unrecognized column): silently dropped by the
  `feature_columns` reindex in `score_customers` — not an error, since an
  inference payload legitimately might carry more than the model needs
  (e.g. a `customerID` used only for output labeling).
- **No `customerID` column in `raw_df`:** `score_customers`' output omits
  the `customerID` column entirely rather than inventing one — the caller
  can still align rows by position (`raw_df.index` is preserved).
- **`models/churn_model_calibrated.pkl` absent** (calibration never run):
  `score_customers(..., use_calibrated=True)` raises the
  `load_calibrated_model` `FileNotFoundError`, pointing at
  `python -m src.models.calibration` — directly tested. `use_calibrated=False`
  still works off the Phase 2 raw artifact alone, so scoring isn't blocked
  on this spec ever having run.
- **`churn_model_metadata.json`'s `feature_columns` no longer matches
  `split_data(df)`'s actual columns** (e.g. `df` is from a different
  dataset version than what trained the persisted model):
  `run_calibration_pipeline` raises a clear `ValueError` before fitting
  anything — a deliberate staleness guard (Requirement 9), directly tested.
- **Re-running `python -m src.models.calibration` twice:** idempotent given
  a fixed dataset and `RANDOM_STATE` — overwrites both calibrated artifacts
  and both tracked `reports/` files with bit-identical content (verified:
  reproducible calibrated probabilities), and appends one more MLflow run,
  same additive-history behavior as `08`.

### Security notes

- **No new dependency.** `CalibratedClassifierCV` and `calibration_curve`
  ship inside the already-pinned `scikit-learn==1.6.1` — nothing added to
  `requirements.txt`.
- **`joblib.dump`/`load` on `churn_model_calibrated.pkl` carries the same
  self-produced-artifact trust boundary already documented for
  `churn_model.pkl` in `08`'s Security notes** — this pipeline is fit and
  saved within the same `run_calibration_pipeline` call, never deserialized
  from an external source. The same forward-looking caveat applies
  unchanged: if this artifact is ever loaded from anywhere other than this
  module's own output (a possible Phase 6 concern), that load path must be
  re-reviewed as untrusted deserialization at that time.
- **New untrusted-input surface: `score_customers`' `raw_df` / `customer`
  dict.** This is the first function in the repo that accepts
  *unlabeled, potentially external* customer data (a future API request
  body or an uploaded CSV) rather than only the known training dataset.
  Mitigations already specified: no `eval`/dynamic code execution on any
  cell value; `feature_columns` validation rejects unexpected shapes with a
  named-column error rather than silently misaligning a `ColumnTransformer`;
  numeric coercion (`pd.to_numeric(..., errors="coerce")`) never raises on
  garbage input, it produces `NaN` → imputed `0.0`, so malformed
  `TotalCharges` values can't crash scoring. No column value is used to
  construct a file path, SQL query, or shell command anywhere in this
  spec, so no injection surface is introduced. Extremely large `raw_df`
  inputs (a memory/DoS concern) are explicitly **not** guarded here — that
  belongs to whatever future Phase 5 endpoint enforces a request-size limit,
  not this library-level scoring function.
- No secrets, no new environment variables, no network call.

### Success criteria

- `pytest -q` passes: all existing tests + `test_calibration.py` +
  `test_scoring.py` + the new ECE/`prepare_scoring_input` cases, all green.
- `python -m src.models.calibration` runs end-to-end, produces
  `models/churn_model_calibrated.pkl`,
  `models/churn_model_calibrated_metadata.json`,
  `reports/calibration_metrics.json`,
  `reports/figures/calibration_reliability.png`, and one new MLflow run
  under `retainiq-churn-classifier`.
- `reports/calibration_metrics.json` shows `brier_calibrated < brier_raw`
  and `ece_calibrated < ece_raw` (verified true today: 0.1376 < 0.1466 and
  0.0301 < 0.0803) with `auc_calibrated` still `< LEAKAGE_AUC_THRESHOLD`.
- `score_customers`/`score_single_customer` correctly score a hand-built
  sample of real customer rows, matching a manually-computed
  `pipeline.predict_proba` reference to within the documented rounding.
- `notebooks/08_probability_calibration.ipynb` runs top-to-bottom without
  error.
- `quality-reviewer` and `security-reviewer` report no unresolved findings
  on the diff.

### Out of scope

- Phase 3 (SHAP/LIME), Phase 4 (risk tiers, Next-Best-Action), Phase 5
  (FastAPI `/predict`, Streamlit What-If panel), Phase 6 (Evidently drift,
  Prefect retraining, Docker).
- Bootstrapped or quantile-based confidence/prediction intervals around the
  point estimate (see Interpretation note).
- Re-comparing or re-selecting among the four `MODEL_SPECS` candidates.
- Calibrating every candidate model — only the metadata-declared winner is
  calibrated.
- Any change to `clean_data()`'s output, `src/data/eda.py`,
  `src/data/cohorts.py`, `src/models/segmentation.py`,
  `src/data/lifecycle.py`, `src/explain/driver_analysis.py`,
  `src/models/train.py`'s public functions, or any existing
  test/figure/notebook.
- Request-size / rate limiting on `score_customers` — deferred to whatever
  Phase 5 endpoint eventually wraps it.

---

## PART 2 — PLAN

### Approach

Three small, single-purpose additions layered on top of the existing Phase
2 artifact rather than touching it: a pure-metric addition to
`evaluation.py` (ECE), a calibration module that reuses `train.py`'s own
pipeline builder and split function (`build_model_pipeline`, `split_data`)
instead of re-deriving them, and a thin scoring module that is the only new
piece with a genuinely new input class (untrusted external customer rows).
`clean_data()` is refactored (not rewritten) to share its dataset-quirk
fixes with a new unlabeled-input path, so the well-known `TotalCharges`
blank-string handling exists in exactly one place.

**Alternative rejected:** calibrate via `CalibratedClassifierCV(...,
cv="prefit")` against the already-trained `churn_model.pkl` directly,
using the existing test split as the calibration set. Rejected because
`cv="prefit"` calibrates against whatever split it's given and that data
can then never be used again for an honest final evaluation — this would
either burn the one held-out test split on calibration-fitting (leaving no
honest number to report) or require carving out a third split
(train/calibrate/test) purely for this spec, complicating `train.py`'s
already-settled 80/20 contract from `08`. The chosen `cv=5` internal-CV
approach needs no third split and reuses the exact `X_train`/`y_train`
`train.py` already produces.

### Task breakdown

- [ ] **1. Add `ECE_BINS`/`expected_calibration_error` to
      `src/models/evaluation.py`** (Requirement 1).
- [ ] **2. Refactor `src/data/load_data.py`** — extract
      `_clean_common_fields`, update `clean_data()` to call it, add
      `prepare_scoring_input` (Requirements 2–3). Run `pytest -q
      tests/test_data.py` immediately after to confirm zero behavior
      change before moving on.
- [ ] **3. Create `src/models/calibration.py`** — constants, path
      constants (`DEFAULT_CALIBRATED_MODEL_PATH`,
      `DEFAULT_CALIBRATED_METADATA_PATH`, `CALIBRATION_METRICS_PATH`,
      `RELIABILITY_FIGURE_FILENAME`), `build_calibration_template`,
      `fit_calibrated_pipeline`, `evaluate_calibration`,
      `plot_reliability_curve`, `save_calibrated_artifact`,
      `load_calibrated_model`, `run_calibration_pipeline`, `main()`
      (Requirements 4–10).
- [ ] **4. Create `src/models/scoring.py`** — constants,
      `score_customers`, `score_single_customer` (Requirements 11–13).
- [ ] **5. Run `python -m src.models.calibration`** — confirm
      `models/churn_model_calibrated.pkl`,
      `models/churn_model_calibrated_metadata.json`,
      `reports/calibration_metrics.json`,
      `reports/figures/calibration_reliability.png` are produced; confirm
      via `mlflow ui --backend-store-uri sqlite:///mlflow.db` that one new
      `XGBoost_calibrated` run lands in `retainiq-churn-classifier`;
      confirm `git status`/`git diff --stat` shows no existing
      Phase-2 artifact changed.
- [ ] **6. Create `notebooks/08_probability_calibration.ipynb`** —
      bootstrap cell copied from `07_model_training.ipynb`; sections per
      Functional Requirement 14.
- [ ] **7. Add `tests/test_calibration.py`, `tests/test_scoring.py`**;
      extend `tests/test_evaluation.py` and `tests/test_data.py` — see
      Tests to write below.
- [ ] **8. Add `python -m src.models.calibration` to CLAUDE.md §5's
      command list**, right after the existing `python -m
      src.models.train` line.
- [ ] **9. Run the full suite** — `pytest -q`, confirm all existing + new
      tests pass.
- [ ] **10. Commit** — `src/models/evaluation.py`, `src/data/load_data.py`,
      `src/models/calibration.py`, `src/models/scoring.py`,
      `notebooks/08_probability_calibration.ipynb`,
      `tests/test_calibration.py`, `tests/test_scoring.py`,
      `tests/test_evaluation.py`, `tests/test_data.py`,
      `reports/calibration_metrics.json`,
      `reports/figures/calibration_reliability.png`, `.claude/CLAUDE.md`,
      commit message `feat: calibrated probability scoring — isotonic
      calibration + score_customers API`. `models/` and `mlruns/` stay
      untracked.

### Tests to write (hand to test-writer)

- `tests/test_evaluation.py::test_expected_calibration_error_zero_on_perfect_calibration` —
  synthetic `y_proba` whose bin means exactly match observed bin rates →
  ECE `== 0`.
- `tests/test_evaluation.py::test_expected_calibration_error_positive_on_miscalibration` —
  a hand-built miscalibrated example → ECE matches a manually-computed
  expected value.
- `tests/test_data.py::test_prepare_scoring_input_drops_id_and_returns_it_separately` —
  a `clean_df`-shaped row plus `customerID` → returned `features_df` has no
  `customerID`/`Churn`, returned `customer_ids` matches the input.
- `tests/test_data.py::test_prepare_scoring_input_handles_missing_customer_id` —
  no `customerID` column present → returns `None` for `customer_ids`, no
  error.
- `tests/test_data.py::test_prepare_scoring_input_imputes_blank_total_charges` —
  a synthetic row with `TotalCharges=""` → coerced to `0.0`, matching
  `clean_data()`'s behavior.
- `tests/test_data.py::test_clean_data_output_unchanged_after_refactor` —
  regression guard: `clean_data(load_raw_data())` output is identical
  (`pd.testing.assert_frame_equal`) to a frozen reference computed before
  the refactor (or simply that all pre-existing `test_data.py` assertions
  still hold — satisfied by Task 2 already running the full file, this
  test makes the "no behavior change" claim explicit and permanent).
- `tests/test_calibration.py::test_fit_calibrated_pipeline_uses_only_train_split` —
  fit on a small `X_train`/`y_train` slice; assert the returned
  `CalibratedClassifierCV`'s `predict_proba` shape matches `X_test`'s row
  count without ever having seen `X_test`.
- `tests/test_calibration.py::test_fit_calibrated_pipeline_reproducible` —
  two independent fits on identical data → `np.allclose` on calibrated
  test-set probabilities.
- `tests/test_calibration.py::test_evaluate_calibration_returns_all_six_keys` —
  synthetic `y_true`/`proba_raw`/`proba_calibrated` → all of
  `brier_raw/brier_calibrated/auc_raw/auc_calibrated/ece_raw/ece_calibrated`
  present and in `[0, 1]`.
- `tests/test_calibration.py::test_isotonic_improves_brier_and_ece_on_real_data` —
  locks in the verified spec-research finding on the real dataset:
  `brier_calibrated < brier_raw` and `ece_calibrated < ece_raw`
  (intentionally brittle, matching `08`'s precedent for real-data
  regression guards).
- `tests/test_calibration.py::test_plot_reliability_curve_returns_existing_path` —
  writes to `tmp_path`, returned `Path.exists()`.
- `tests/test_calibration.py::test_save_and_load_calibrated_artifact_roundtrip` —
  same shape as `08`'s `test_save_and_load_model_artifact_roundtrip`, for
  `save_calibrated_artifact`/`load_calibrated_model`.
- `tests/test_calibration.py::test_load_calibrated_model_raises_actionable_error_when_missing` —
  message mentions `python -m src.models.calibration`.
- `tests/test_calibration.py::test_run_calibration_pipeline_raises_on_feature_column_mismatch` —
  fabricated metadata with stale `feature_columns` → `ValueError` before
  any fitting happens.
- `tests/test_calibration.py::test_run_calibration_pipeline_writes_all_artifacts` —
  end-to-end on a real `clean_df`, `tmp_path`-scoped paths monkeypatched
  (mirrors `08`'s `test_run_training_pipeline_writes_all_artifacts`); all
  returned paths exist.
- `tests/test_scoring.py::test_score_customers_returns_probability_and_percentage_columns` —
  a few real customer rows → both columns present, `churn_probability_pct
  == round(churn_probability * 100, 1)` for every row.
- `tests/test_scoring.py::test_score_customers_matches_pipeline_predict_proba` —
  scores computed by `score_customers` match a direct
  `pipeline.predict_proba(...)` call on the same rows, within the
  documented rounding.
- `tests/test_scoring.py::test_score_customers_empty_input_returns_empty_output` —
  0-row `DataFrame` in → 0-row `DataFrame` out with the right columns, no
  error.
- `tests/test_scoring.py::test_score_customers_raises_on_missing_required_column` —
  a row missing e.g. `Contract` → `ValueError` naming `Contract`.
- `tests/test_scoring.py::test_score_customers_omits_customer_id_when_absent` —
  input with no `customerID` → output has no `customerID` column.
- `tests/test_scoring.py::test_score_customers_uses_calibrated_by_default` —
  `use_calibrated=True` (default) output differs from `use_calibrated=False`
  output on at least one customer whose raw vs. calibrated probability
  provably differ (from the verified real-data gap).
- `tests/test_scoring.py::test_score_customers_raises_when_calibrated_model_missing` —
  `use_calibrated=True` against a `tmp_path` with no calibrated artifact →
  the `load_calibrated_model` `FileNotFoundError` surfaces.
- `tests/test_scoring.py::test_score_single_customer_returns_flat_dict` —
  a single customer `dict` → returns a `dict` (not a `DataFrame`) with the
  expected keys.

### Quality gates

1. **test-runner** — run `pytest -q`, confirm all tests (existing + new)
   pass, fix any straightforward regression. Flag if
   `test_run_calibration_pipeline_writes_all_artifacts` is slow (one
   5-fold `CalibratedClassifierCV` fit over the full training pipeline)
   and worth noting alongside `08`'s already-flagged slow tests.
2. **quality-reviewer** — review the train/test split reuse (calibration
   never touches `X_test` during fitting), the `feature_columns`
   staleness guard in `run_calibration_pipeline`, the shared
   `_clean_common_fields` refactor's behavioral equivalence to the old
   `clean_data()`, and CLAUDE.md §8 adherence (named constants, type
   hints, docstrings).
3. **security-reviewer** — focus on `score_customers`'/`score_single_customer`'s
   new untrusted-input surface (the first function in the repo that scores
   arbitrary external customer data): missing-column handling, numeric
   coercion safety, no dynamic code execution or path/query construction
   from input values; confirm no new dependency was added; confirm
   `joblib.dump`/`load` on the calibrated artifact stays within the
   self-produced-artifact trust boundary already accepted for
   `churn_model.pkl`.
4. Commit only after both reviews are clean.

### Risks / rollback

- **Risk:** a future retrain (`python -m src.models.train`) changes which
  model wins (e.g. LightGBM overtakes XGBoost on `cv_auc_mean`), silently
  leaving `churn_model_calibrated.pkl` calibrated against the *old* winner.
  **Mitigation:** `run_calibration_pipeline`'s `feature_columns` staleness
  guard (Requirement 9) catches a schema mismatch but not a same-schema
  different-model swap — flagged here explicitly as a known gap:
  `python -m src.models.calibration` must be re-run after every retrain
  that changes the winning model, and this is not automated in this spec
  (that would be a Phase 6 Prefect-retraining-flow concern). Rollback is
  simply re-running the calibration command.
- **Risk:** isotonic regression can overfit on small calibration folds.
  **Mitigation:** verified fold size (5,634 train rows / 5 folds ≈ 1,127
  per fold, ~300 churners per fold) is well above isotonic's usual
  small-sample caution threshold, and the reproducibility check (identical
  probabilities across two independent fits) is evidence against fold-level
  instability, not just a theoretical argument.
- **Risk:** a future Kaggle re-download shifts the verified Brier/AUC/ECE
  numbers enough that `test_isotonic_improves_brier_and_ece_on_real_data`
  no longer holds. **Mitigation:** intentional, matching `08`'s brittleness
  philosophy — this test should fail loudly on real distributional shift,
  not silently pass on stale assumptions.
- **Risk:** a reviewer assumes `score_customers` is already reachable via
  an HTTP endpoint. **Mitigation:** the scope note at the top of this spec
  and the API/UI surface section state explicitly that no endpoint exists
  yet.
- **Rollback:** single commit (Task 10) covering only additive files, plus
  one refactor-in-place (`load_data.py`, output-preserving, covered by
  Task 2's immediate regression check) and one CLAUDE.md command-list
  addition. `git revert` is clean; `models/`, `mlruns/` are untracked, so
  nothing to roll back there. `churn_model.pkl` itself is never modified,
  so any consumer of the Phase 2 artifact is unaffected by a revert either
  way.

### Definition of done

- All 10 tasks checked off.
- `pytest -q` green (all existing tests + `test_calibration.py` +
  `test_scoring.py` + the new ECE/`prepare_scoring_input` cases).
- `python -m src.models.calibration` runs end-to-end and produces every
  artifact listed in Success Criteria, with `brier_calibrated < brier_raw`
  and `ece_calibrated < ece_raw` both holding.
- `notebooks/08_probability_calibration.ipynb` executes top-to-bottom
  without error.
- `quality-reviewer` and `security-reviewer` report no unresolved findings.
- All Success Criteria in Part 1 are met.
