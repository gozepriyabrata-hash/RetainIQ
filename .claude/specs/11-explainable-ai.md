# Spec + Plan: Explainable AI — Local SHAP + LIME Per-Customer Explanations

> Location note: this file lives at `.claude/specs/` per CLAUDE.md §4's repo
> tree ("`.claude` — claude.md and also Specs folder"), consistent with
> `01`–`10`. Spec and plan are combined in one file for the same reason.
> Numbered `11` (next after `10-risk-classification.md`).
>
> Scope note: unlike `06`, `09`, and `10` (each explicitly a bridge or a
> *different* phase's throwaway diagnostic), this spec **is** CLAUDE.md §14
> Phase 3 itself: "SHAP + LIME explainability, plain-English reasons." It is
> not `06`'s `src/explain/driver_analysis.py`, which fits its own disposable
> diagnostic XGBoost model (never persisted, never the production model)
> purely to rank **global** drivers. This spec explains the **real,
> persisted, production `models/churn_model.pkl`** (`08`) with **local**
> (per-customer, top-3) SHAP `TreeExplainer` output plus a **global**
> importance ranking *on that same production model* (a second, more
> production-honest ranking than `06`'s diagnostic one), and adds LIME as
> CLAUDE.md §7's mandated "alternative local view." It stops at explanation
> generation — no FastAPI `/explain` route, no Streamlit view (Phase 5).
>
> Interpretation note — which model is explained: `09` showed the raw
> `churn_model.pkl` is honestly ranked (AUC 0.8434) but not well calibrated,
> so `score_customers` serves the **calibrated** `CalibratedClassifierCV`
> probability by default. `CalibratedClassifierCV` has no single tree
> structure `shap.TreeExplainer` can open — it internally holds 5 cloned
> pipeline fits plus an isotonic regressor — so it cannot be explained by
> `TreeExplainer` directly. This spec's SHAP **and** LIME explanations are
> therefore both computed against the **raw** `churn_model.pkl` pipeline
> (`train.load_trained_model()`), matching CLAUDE.md §7's literal
> "SHAP TreeExplainer" instruction, kept internally consistent by explaining
> both methods against the same raw model rather than mixing a
> raw-model SHAP number with a calibrated-model LIME number. Isotonic
> calibration is a monotonic reshaping of probability, so it changes *how
> confident* the number is, not *which features* drove the prediction or
> their *direction* — the displayed probability (from `09`'s
> `score_customers`) and this spec's driver explanations are complementary,
> not required to numerically reconcile, and this distinction is stated
> directly in each explanation's output rather than left implicit.
>
> Interpretation note — SHAP value scale: `shap.TreeExplainer` on an
> `XGBClassifier` returns values in the model's raw margin (log-odds) space,
> not probability-percentage-point space (verified below: `expected_value`
> ≈ -0.0009, i.e. ~50% baseline in log-odds, not 0%). This spec reports the
> **sign** of each SHAP value as the reliable signal ("increases" /
> "decreases" risk) and the raw margin value as a secondary, comparable-only-
> to-itself number — never phrased as "+12 percentage points," which would
> misstate what TreeExplainer actually returns. `06` made the same choice
> implicitly (never converts to probability space either); here it's stated
> explicitly since Phase 3's explanations are user-facing.
>
> Research note: every number below was verified by running SHAP and LIME
> against the real, currently-persisted `models/churn_model.pkl`
> (`model_name="XGBoost"`, `trained_at` 2026-08-22T05:33:17Z, test AUC
> 0.8434) and `load_clean_data()`'s current 7,043-row dataset during spec
> research — not estimated. `lime` (listed unpinned in `requirements.txt`
> since `01`, never previously installed in this environment) was installed
> and resolved to `lime==0.2.0.1`.
>
> Verified finding — global SHAP importance on the **production** model
> (`shap.TreeExplainer` on `churn_model.pkl`'s `clf` step, evaluated on the
> held-out 1,409-row test split, one-hot dummies aggregated back to their
> original column exactly as `06` established): **Contract** (0.8253),
> **tenure** (0.4628), OnlineSecurity (0.2834), **InternetService** (0.2750),
> **PaymentMethod** (0.2366), **TechSupport** (0.1932), PaperlessBilling
> (0.1482), MonthlyCharges (0.1264), StreamingMovies (0.1155), OnlineBackup
> (0.1012). All 5 of CLAUDE.md §6's documented signals land in the top 6 —
> confirming the *production* model, not just `06`'s throwaway one, agrees
> with real domain knowledge. `TreeExplainer` runs in **0.71s** for the full
> 1,409-row test split — no performance concern.
>
> Verified finding — two real local explanations, cross-checked between
> methods: (1) a test-split customer with `tenure=1`, `Contract=Month-to-
> month`, `InternetService=Fiber optic`, `PaymentMethod=Electronic check`,
> `TechSupport=No`, `MonthlyCharges=$95.10` scores **93.3%** raw churn
> probability; SHAP top-3 = `tenure` (+0.778), `Contract` (+0.585),
> `InternetService` (+0.307) — all *increasing* risk, all 3 matching
> CLAUDE.md §6's documented signal directions exactly. (2) A different
> test-split customer scores **1.8%**; SHAP top-3 = `Contract` (-1.562),
> `tenure` (-1.285), `OnlineSecurity` (-0.419) — all *decreasing* risk.
> LIME on that same low-risk customer (`num_samples=5000`,
> `random_state=42`) independently returns `Contract=Two year` (-0.212),
> `tenure > 55.00` (-0.103), `InternetService=Fiber optic` (+0.093) — LIME's
> top-2 agree in direction with SHAP's top-2 (`Contract`, `tenure`, both
> negative/risk-decreasing) even though the two methods use unrelated
> mechanics (linear local surrogate on perturbed samples vs. exact tree
> attribution), a genuine cross-method agreement check, not merely asserted.
> `LimeTabularExplainer.explain_instance` (`num_samples=5000`) took **1.4s**
> for this one customer — noted as a real per-call cost, flagged in Risks,
> not a blocker since no live endpoint exists yet.

---

## PART 1 — SPEC

### Feature

A new `src/explain/local_explainer.py` module that explains the real,
persisted production churn model (`models/churn_model.pkl`, `08`) two ways:
`shap.TreeExplainer` for both a production-model global importance ranking
and per-customer local top-3 drivers, and `lime.lime_tabular
.LimeTabularExplainer` as CLAUDE.md §7's mandated alternative local view —
each driver paired with a plain-English, human-readable reason string. A
single `explain_customer(customer: dict) -> dict` convenience function takes
raw customer attributes in and returns both methods' top-3 drivers, ready
for a future Phase 5 `POST /explain` route (CLAUDE.md §10) to call directly.

### Problem / motivation

CLAUDE.md §1 promises explanations for *why* a customer is at risk, and §7
commits to "SHAP TreeExplainer for global importance + local top-3 drivers
per customer; LIME as the alternative local view. Explanations must be
human-readable." Nothing in the repo today explains an individual
customer's prediction: `06`'s `driver_analysis.py` is explicitly scoped to
global rankings from a disposable diagnostic model unrelated to production
(its own spec's Non-goals say so), and `09`/`10` turn the production model
into a probability and a tier but never *why*. A retention manager staring
at "this customer is 93% Critical" has no way to act on that number without
knowing it's driven by 1-month tenure, a month-to-month contract, and fiber
internet — this spec is that missing "why," verified to reproduce CLAUDE.md
§6's documented churn signals on real customers, in both directions
(risk-increasing and risk-decreasing).

### Goals / non-goals

**Goals**
- Add `src/explain/local_explainer.py`: a `build_explainer_context(df)`
  orchestrator that loads the persisted production pipeline once (guarding
  that its winning model is tree-based, and that its `feature_columns`
  haven't gone stale relative to `df`, mirroring `09`'s staleness guard),
  builds a `shap.TreeExplainer` and a `LimeTabularExplainer` against it, and
  returns a reusable context dict so a caller explaining many customers
  pays the setup cost once (CLAUDE.md §10: "load the model once, not per
  request").
- Add global SHAP importance (`global_shap_importance`,
  `plot_global_shap_importance`) computed on the **production** model's
  held-out test split — a second, production-honest ranking distinct from
  `06`'s diagnostic-model one, aggregating one-hot dummies back to their
  original column exactly as `06` established.
- Add local SHAP (`local_shap_top_drivers`) and local LIME
  (`local_lime_top_drivers`) per-customer top-3 driver functions, each
  returning `(feature, value, direction, reason)`-shaped entries.
- Add `humanize_reason`, a small template-based plain-English sentence
  generator (CLAUDE.md §7: "must be human-readable"), with natural phrasing
  for CLAUDE.md §6's 5 documented signal columns and a generic fallback for
  every other column.
- Add `explain_customer(customer: dict, context=None) -> dict`: raw
  customer attributes in, both methods' top-3 drivers out — the function a
  future `POST /explain` endpoint calls directly (CLAUDE.md §10).
- Add `notebooks/10_explainability.ipynb` following `01`–`09`'s
  bootstrap-cell pattern.
- Add `tests/test_local_explainer.py`.
- Install and pin `lime==0.2.0.1` in `requirements.txt` (verified during
  spec research — was listed unpinned since `01` but never actually
  installed in this environment).

**Non-goals**
- No FastAPI `/explain` endpoint, no Streamlit view, no model loaded at app
  startup — Phase 5, not this spec. `src/api/` and `app/` stay untouched;
  `explain_customer` is added as the exact function a future route will
  call, matching `09`'s/`10`'s "added but wired to nothing" precedent.
- No change to `src/explain/driver_analysis.py` (`06`'s throwaway-model
  global ranking stays exactly as-is — this spec's global ranking is a
  separate, additional one on the real model, not a replacement) or to any
  of its tests/figures.
- No change to `src/models/scoring.py`, `src/models/calibration.py`,
  `src/models/train.py`, or `src/recommend/risk_tiers.py` — this spec only
  *reads* `train.load_trained_model()`/`train.DEFAULT_METADATA_PATH`/
  `train.split_data`, never modifies them.
- No explanation of the calibrated model — see the Interpretation note
  above; both methods explain the raw `churn_model.pkl` only.
- No batch/multi-customer explain function — CLAUDE.md §10's `/explain`
  contract is singular ("Customer → SHAP top-3 churn drivers"), matching
  `09`'s `score_single_customer`, not `score_customers`.
- No SHAP/LIME result caching, no async/background explanation
  computation — a future Phase 5 performance concern, flagged in Risks, not
  solved here.
- No new dependency beyond pinning the already-listed `lime`.

### User stories

- As a **retention manager**, I want a customer's Critical/High risk score
  paired with the top-3 specific reasons in plain English, so I know
  whether to offer a contract incentive, a tech-support add-on, or a
  payment-method nudge — not just a bare number.
- As the **engineer (Priyabrata)**, I want one function
  (`explain_customer`) that goes from raw customer attributes straight to
  both SHAP and LIME top-3 drivers, so Phase 5's `/explain` endpoint is a
  thin wrapper (CLAUDE.md §4: "keep the API... thin"), not a place that
  re-derives explanation logic.
- As the **engineer**, I want SHAP and LIME to explain the *same* underlying
  model (not one explaining the raw model and the other the calibrated
  one), so the two "alternative views" CLAUDE.md §7 asks for are actually
  comparable, not silently answering two different questions.
- As a **recruiter/reviewer**, I want two independent local-explanation
  methods to agree in direction on real customers (verified above, not
  merely asserted), so the explainability layer reads as genuinely
  validated, not a black-box importance plot with no cross-check.

### Functional requirements

1. `src/explain/local_explainer.py` MUST define named constants:
   `TOP_N_LOCAL_DRIVERS = 3` (CLAUDE.md §7's "top-3"), `TOP_N_GLOBAL_FEATURES
   = 10` (matching `06`'s precedent), `LIME_NUM_SAMPLES = 5000` (LIME's own
   default, made an explicit named constant per CLAUDE.md §8 rather than an
   implicit library default), `TREE_BASED_MODELS = frozenset({"RandomForest",
   "XGBoost", "LightGBM"})`, `PRODUCTION_SHAP_FIGURE_FILENAME =
   "production_shap_global_importance.png"`, and
   `FEATURE_DISPLAY_TEMPLATES` (a `dict[str, str]` with `"{value}"`-style
   templates for at minimum CLAUDE.md §6's 5 signal columns: `Contract`,
   `tenure`, `TechSupport`, `PaymentMethod`, `InternetService` — verified
   phrasing e.g. `"Contract": "{value} contract"`, `"tenure":
   "{value}-month tenure"`). MUST reuse `evaluation.RANDOM_STATE` (42) for
   the `LimeTabularExplainer`'s `random_state` — no re-declared constant.
2. MUST gain `build_explainer_context(df: pd.DataFrame) -> dict` that: reads
   `train.DEFAULT_METADATA_PATH` for `model_name`/`feature_columns`; raises
   `ValueError` naming `model_name` if it is not in `TREE_BASED_MODELS`
   (before doing any other work); calls `train.split_data(df)` and raises
   `ValueError` (mirroring `09`'s `run_calibration_pipeline` staleness
   guard wording) if `list(X_train.columns) != feature_columns`; loads
   `train.load_trained_model()`, extracts its `"pre"` (preprocessor) and
   `"clf"` (fitted estimator) steps; computes `categorical_columns` via
   `get_categorical_columns(X_train)` and a `feature_group_map` (an exact
   one-hot-dummy-to-original-column mapping built from the fitted
   `OneHotEncoder`'s `categories_`, the same mechanism as `06`'s
   `_feature_group_columns` — duplicated into this module rather than
   imported, see Approach); builds `shap.TreeExplainer(clf)`; fits one
   `sklearn.preprocessing.LabelEncoder` per categorical column on
   `X_train`; builds one `LimeTabularExplainer` (Requirement 5) against the
   label-encoded `X_train`. Returns a dict with (at minimum) `pipeline`,
   `preprocessor`, `clf`, `model_name`, `feature_columns`,
   `categorical_columns`, `feature_group_map`, `X_train`, `X_test`,
   `y_test`, `shap_explainer`, `lime_explainer`, `label_encoders`.
3. MUST gain `global_shap_importance(context: dict, top_n: int =
   TOP_N_GLOBAL_FEATURES) -> pd.DataFrame`: transforms `context["X_test"]`
   through `context["preprocessor"]`, computes `context["shap_explainer"]
   .shap_values(...)`, raises `ValueError` if the result is not 2D
   (mirroring `06`'s guard against an unexpected `shap`/`xgboost` output
   shape), aggregates one-hot dummies back to original columns by summing
   each row's *signed* SHAP values per group first and then taking
   `mean(abs(.))` across rows (`06`'s established, tested aggregation
   order — never the reverse, which overcounts multi-level columns),
   returns `column`/`mean_abs_shap` (4dp) sorted descending, top `top_n`
   rows.
4. MUST gain `plot_global_shap_importance(context: dict, top_n: int =
   TOP_N_GLOBAL_FEATURES, out_dir: Path = FIGURES_DIR) -> Path`: horizontal
   bar chart of `global_shap_importance`'s output, saved as
   `reports/figures/production_shap_global_importance.png` — a distinct
   filename from `06`'s `driver_shap_global_importance.png` so neither
   figure overwrites the other.
5. MUST gain a private `LimeTabularExplainer` builder used inside
   `build_explainer_context`: encodes `X_train`'s categorical columns to
   integers via `context["label_encoders"]`, and constructs
   `LimeTabularExplainer(training_data=<encoded X_train as float64 array>,
   feature_names=feature_columns, class_names=["No Churn", "Churn"],
   categorical_features=<indices of categorical_columns within
   feature_columns>, categorical_names=<index -> list(encoder.classes_)>,
   mode="classification", random_state=RANDOM_STATE)`. Operates in the
   **raw, pre-one-hot feature space** (verified design: LIME then reports
   one row per original column, e.g. `"Contract=Two year"`, with no
   one-hot-dummy aggregation needed — unlike SHAP, which must aggregate
   because `TreeExplainer` only sees the transformed matrix).
6. MUST gain `local_shap_top_drivers(context: dict, features_df: pd.DataFrame,
   top_n: int = TOP_N_LOCAL_DRIVERS) -> list[dict]`: `features_df` is
   exactly 1 row in raw (pre-transform) feature space, already reindexed to
   `context["feature_columns"]`. Transforms the row, computes
   `context["shap_explainer"].shap_values(...)`, raises `ValueError` if not
   shape `(1, n_transformed_features)`, aggregates to original columns
   (signed sum per group, Requirement 3's rule applied to one row), ranks
   by `abs(value)` descending, returns the top `top_n` entries each shaped
   `{"feature": str, "customer_value": <raw value from features_df>,
   "shap_value": float (4dp, margin-space per the Interpretation note),
   "direction": "increases" | "decreases" | "neutral", "reason": str}`.
7. MUST gain `local_lime_top_drivers(context: dict, features_df: pd.DataFrame,
   top_n: int = TOP_N_LOCAL_DRIVERS) -> list[dict]`: label-encodes
   `features_df`'s single row via `context["label_encoders"]`, raising a
   clear `ValueError` naming the offending column and value (not a bare
   `sklearn` "unseen labels" traceback) if a categorical value was never
   seen in `X_train`; builds a `predict_fn` that decodes an array of
   encoded rows back to raw dtypes and calls
   `context["pipeline"].predict_proba` (skipping `SMOTE` automatically —
   `imblearn.Pipeline` only invokes it during `.fit()`, per `train.py`'s
   own documented behavior); calls `context["lime_explainer"]
   .explain_instance(encoded_row, predict_fn, num_features=top_n,
   num_samples=LIME_NUM_SAMPLES, labels=(1,))`; for each
   `(description, weight)` in `exp.as_list(label=1)`, resolves `description`
   back to its original column name via exact word-boundary matching against
   `context["feature_columns"]` (never a bare `startswith`, to avoid a
   prefix-collision misattribution the way `06` avoids it for SHAP; raises
   `ValueError` if no column matches), and returns entries shaped
   `{"feature": description (LIME's own human-readable condition string,
   e.g. "Contract=Two year" or "tenure > 55.00"), "lime_weight": float
   (4dp), "direction": "increases" | "decreases" | "neutral", "reason":
   str}`.
8. MUST gain `humanize_reason(column: str, customer_value, signed_value:
   float) -> str`: `direction = "increases" if signed_value > 0 else
   "decreases" if signed_value < 0 else "has no measurable effect on"`;
   looks up `column` in `FEATURE_DISPLAY_TEMPLATES` (falling back to
   `"{column} = {value}"` for any column not in that dict) and formats it
   with `customer_value`; returns `f"{phrase} {direction} this customer's
   predicted churn risk."`. MUST NOT ever surface a raw transformed-feature
   name (e.g. `cat__Contract_Month-to-month`) in the output — only the
   original column name or its display phrase.
9. MUST gain `explain_customer(customer: dict, context: dict | None = None)
   -> dict`: if `context` is `None`, builds one via
   `build_explainer_context(load_clean_data())` (documented as expensive —
   a caller explaining more than one customer should build and reuse a
   context, mirroring `09`'s `pipeline` reuse parameter). Cleans `customer`
   via `prepare_scoring_input(pd.DataFrame([customer]))`; raises
   `ValueError` naming every missing column (mirroring `09`'s
   `score_customers` wording) if any of `context["feature_columns"]` is
   absent; reindexes to `feature_columns`; calls
   `local_shap_top_drivers`/`local_lime_top_drivers`; returns
   `{"shap_top_drivers": list[dict], "lime_top_drivers": list[dict]}`, plus
   `"customerID"` if the input carried one.
10. MUST gain `generate_explainability_figures(df: pd.DataFrame, out_dir:
    Path = FIGURES_DIR) -> list[Path]` (builds one context, calls
    `plot_global_shap_importance`) and a `main()` calling it via
    `load_clean_data()`, runnable as `python -m src.explain.local_explainer`
    (documented in CLAUDE.md §5, right after `driver_analysis`'s entry).
11. `notebooks/10_explainability.ipynb` MUST follow `01`–`09`'s
    bootstrap-cell pattern. Sections, in order: problem framing (why local
    explanations, and the raw-vs-calibrated-model Interpretation note) →
    `build_explainer_context` + production-model global SHAP importance
    table/chart, compared against `06`'s diagnostic-model ranking → the two
    verified example customers (high-risk and low-risk) with SHAP top-3 →
    the same low-risk customer's LIME top-3, with an explicit
    direction-agreement callout against SHAP's top-2 → `explain_customer`
    demoed end-to-end on a raw customer dict → key findings closing cell.
12. `tests/test_local_explainer.py` MUST cover the Plan's "Tests to write"
    section in full.
13. None of the above may change `src/explain/driver_analysis.py`,
    `src/models/scoring.py`, `src/models/calibration.py`,
    `src/models/train.py`, `src/recommend/risk_tiers.py`,
    `src/data/load_data.py`, `src/data/eda.py`, `src/features/
    preprocessing.py`, or any existing test/figure/notebook — all current
    tests must keep passing unmodified.

### Data & model impact

No new model is fit and no existing artifact is modified — this spec only
*reads* `models/churn_model.pkl` and `models/churn_model_metadata.json`
(`08`'s artifacts). No new column is written back into any DataFrame that
feeds `src/features/preprocessing.py` or `src/models/train.py`;
`explain_customer`'s output (`shap_top_drivers`/`lime_top_drivers`) is
reporting-only, one-way, exactly like `09`/`10`'s outputs. `reports/`
gains one new tracked figure,
`reports/figures/production_shap_global_importance.png`.

### ML guardrails (mandatory check)

- **No target/probability leakage:** neither SHAP nor LIME is ever fed
  `Churn`, `churn_probability`, or `risk_tier` as an input — both explain
  `context["clf"]`/`context["pipeline"]`'s prediction from raw customer
  attributes only, the same `feature_columns` `08`'s model was trained on.
- **Honest-AUC guard is unaffected:** this spec fits no new model and calls
  no training/evaluation code path; it explains an already leakage-checked
  persisted model (`08`'s AUC 0.8434, disclosed again in the Research note
  above).
- **Fitting/splitting/SMOTE:** `build_explainer_context` calls
  `train.split_data(df)` (the identical stratified 80/20 split `08` used)
  only to obtain `X_train` (as `LimeTabularExplainer`'s background
  distribution and the `LabelEncoder`s' fit data) and `X_test` (as the
  global-SHAP evaluation set) — it fits nothing new on either split; SMOTE
  is never invoked here since `.predict_proba()` on the already-fitted
  pipeline skips it automatically.
- **Reproducibility:** `random_state=RANDOM_STATE` (42) passed to
  `LimeTabularExplainer`; `shap.TreeExplainer` on a fixed fitted model is
  itself deterministic (no internal randomness). `train.split_data(df)` is
  called with its own default `random_state=42`, reproducing the identical
  partition `08`/`09` already use.
- **Metric reporting:** N/A — this spec produces explanations, not
  classification metrics; no accuracy claim is made anywhere.

### API / UI surface

None shipped. `explain_customer` (Requirement 9) is added as the exact
function a future Phase 5 `POST /explain` route (CLAUDE.md §10: "Customer →
SHAP top-3 churn drivers") will call — raw customer `dict` in,
`{"shap_top_drivers": [...], "lime_top_drivers": [...]}` out. No FastAPI
route or Streamlit view is wired up; `src/api/` and `app/` are untouched.

### Edge cases & failure states

- **The persisted winning model is not tree-based** (e.g. a future retrain
  selects `LogisticRegression`): `build_explainer_context` raises
  `ValueError` naming the actual `model_name`, before any SHAP/LIME
  machinery is built — directly tested against a monkeypatched metadata
  file naming `"LogisticRegression"`.
- **`churn_model_metadata.json`'s `feature_columns` no longer matches
  `train.split_data(df)`'s actual columns:** `build_explainer_context`
  raises `ValueError` before fitting/loading anything else — mirrors `09`'s
  staleness guard, directly tested.
- **A customer's categorical value was never seen during training** (e.g. a
  hand-typed typo, or a genuinely new category): `local_lime_top_drivers`
  raises `ValueError` naming the column and the offending value, rather
  than surfacing scikit-learn's bare "y contains previously unseen labels"
  — directly tested with a synthetic out-of-vocabulary value.
- **A required feature column is missing from the `customer` dict passed to
  `explain_customer`:** raises `ValueError` naming every missing column —
  directly tested, mirroring `09`'s `score_customers` behavior.
- **LIME's `description` string doesn't match any known column** (should be
  unreachable given `feature_names=feature_columns`, but defensively
  guarded): raises `ValueError` rather than silently mislabeling or
  crashing on an unmatched lookup — directly tested with a synthetic
  description string.
- **`top_n` requested larger than the number of available features:**
  `local_shap_top_drivers`/`local_lime_top_drivers`/`global_shap_importance`
  simply return however many exist (no error) — directly tested.
- **A SHAP value of exactly `0.0`** (a feature with zero attribution for
  this customer): `direction` is reported as `"neutral"`, not misclassified
  as increasing or decreasing — directly tested.

### Security notes

- **No new dependency added** — `lime` was already listed (unpinned) in
  `requirements.txt` since `01`; this spec installs it for the first time
  in this environment and pins the resolved version (`lime==0.2.0.1`),
  matching `06`'s precedent of pinning `shap`/`xgboost` on first real use.
  `shap==0.52.0` is already pinned and already a real dependency (`06`).
- **No new untrusted-input surface beyond what `09` already accepts and
  mitigates:** `explain_customer`'s `customer` dict is the same
  externally-sourced customer-row shape `score_customers`'s `raw_df`
  already validates (missing-column errors, safe numeric coercion via
  `prepare_scoring_input`) — reused here, not reimplemented. The one new
  input-derived failure mode (an out-of-vocabulary categorical value
  reaching `LabelEncoder.transform`) is caught and re-raised as a named,
  actionable `ValueError` rather than an opaque library exception
  (Edge cases above) — it cannot crash the process or execute anything.
- **`joblib.load` on `models/churn_model.pkl`** carries the same
  self-produced-artifact trust boundary already documented in `08`'s and
  `09`'s Security notes — this spec adds no new load path beyond the
  existing `train.load_trained_model()`.
- No secrets, no new environment variables, no network call.

### Success criteria

- `pytest -q` passes: all existing tests + `test_local_explainer.py`, all
  green.
- `global_shap_importance` on the production model reproduces the verified
  Research-note ranking (Contract highest, tenure second, at least 4 of
  CLAUDE.md §6's 5 signals in the top 6).
- The two verified example customers' SHAP top-3 drivers (high-risk:
  tenure/Contract/InternetService all increasing; low-risk: Contract/
  tenure/OnlineSecurity all decreasing) are reproduced exactly by
  `local_shap_top_drivers` on those same rows.
- LIME's top-2 drivers for the low-risk example customer agree in
  direction with SHAP's top-2 on that same customer (both negative), a
  cross-method check, not just a bare feature-importance list.
- Every `reason` string produced is non-empty, contains an explicit
  direction word (`"increases"`/`"decreases"`/`"has no measurable effect
  on"`), and never contains a raw transformed-feature artifact (e.g.
  `"cat__"`, `"num__"`).
- `notebooks/10_explainability.ipynb` runs top-to-bottom without error.
- `reports/figures/production_shap_global_importance.png` is produced,
  with no existing figure filename changed.
- `quality-reviewer` and `security-reviewer` report no unresolved findings
  on the diff.

### Out of scope

- Phase 5 (`POST /explain` FastAPI route, Streamlit "why" panel), Phase 6
  (Evidently drift, Prefect retraining, Docker).
- Any change to `06`'s diagnostic-model global ranking, or to `09`/`10`'s
  scoring/tiering behavior.
- Explaining the calibrated `CalibratedClassifierCV` model (see the
  Interpretation note).
- Batch/multi-customer explanation.
- Caching, async computation, or performance tuning of LIME's ~1.4s-per-
  customer cost — flagged in Risks, addressed only if/when a live endpoint
  is built.

---

## PART 2 — PLAN

### Approach

One new module, `src/explain/local_explainer.py`, built around a single
`build_explainer_context(df)` orchestrator (mirroring `09`'s
`run_calibration_pipeline` and `06`'s `fit_driver_diagnostic_model` shape)
that loads the real persisted model once and constructs both explainers
against it, so per-customer calls (`local_shap_top_drivers`,
`local_lime_top_drivers`) stay cheap. LIME operates in the raw,
pre-one-hot feature space (via label-encode/decode around the full
pipeline's `predict_proba`) specifically so its output needs no
one-hot-aggregation step, unlike SHAP, which must aggregate because
`TreeExplainer` only sees the `ColumnTransformer`'s output. Both methods
explain the same raw `churn_model.pkl`, keeping the two "alternative views"
CLAUDE.md §7 asks for genuinely comparable.

**Alternative rejected:** explain the **calibrated** model instead (since
that's the probability actually shown to users via `09`). Rejected because
`shap.TreeExplainer` cannot open a `CalibratedClassifierCV`'s internal
5-fold structure at all — a `KernelExplainer`/`PermutationExplainer`
workaround would be far slower (no longer "TreeExplainer" per CLAUDE.md
§7's literal instruction) and LIME could technically wrap the calibrated
model's `predict_proba` directly, but doing so would make SHAP and LIME
silently explain two different models, defeating the "alternative view"
comparison this spec's cross-method verification relies on. The chosen
approach explains the raw model with both methods and states the
calibration relationship explicitly instead (Interpretation note).

**Alternative rejected:** duplicate LIME's per-column reporting logic by
running it on the one-hot **transformed** feature space (like SHAP), then
aggregating dummy weights back to original columns the same way `06`
aggregates SHAP. Rejected because LIME natively supports mixed raw
categorical/numeric data via `categorical_features`/`categorical_names`
(verified above: `explain_instance` already returns one row per original
column, e.g. `"Contract=Two year"`, not per dummy) — building a second
aggregation path would duplicate `06`'s SHAP-specific aggregation math for
no benefit and risk the two diverging.

### Task breakdown

- [ ] **1. Install `lime` and pin the resolved version in
      `requirements.txt`** — `lime==0.2.0.1` (verified during spec
      research), replacing the current unpinned `lime` line.
- [ ] **2. Create `src/explain/local_explainer.py`** — constants
      (Requirement 1), `build_explainer_context` (Requirement 2, including
      the duplicated `_feature_group_columns` one-hot-mapping helper),
      `global_shap_importance`/`plot_global_shap_importance` (Requirements
      3–4), the private `LimeTabularExplainer` builder (Requirement 5),
      `local_shap_top_drivers` (Requirement 6), `local_lime_top_drivers`
      plus its `predict_fn`/encode/decode/description-parsing helpers
      (Requirement 7), `humanize_reason` (Requirement 8),
      `explain_customer` (Requirement 9), `generate_explainability_figures`
      + `main()` (Requirement 10).
- [ ] **3. Run `python -m src.explain.local_explainer`** — confirm
      `reports/figures/production_shap_global_importance.png` is produced;
      confirm via `git status`/`git diff --stat` that no existing figure
      changed.
- [ ] **4. Create `notebooks/10_explainability.ipynb`** — bootstrap cell
      copied from `09_risk_classification.ipynb`; sections per Functional
      Requirement 11.
- [ ] **5. Add `tests/test_local_explainer.py`** — see Tests to write
      below.
- [ ] **6. Document the new entry point and flip the phase tracker** — add
      `python -m src.explain.local_explainer` to CLAUDE.md §5 (right after
      `driver_analysis`'s line); change CLAUDE.md §14's Phase 3 row status
      from `☐` to `☑` (this spec, unlike `06`/`09`/`10`, *is* Phase 3
      itself — SHAP + LIME explainability, plain-English reasons).
- [ ] **7. Run the full suite** — `pytest -q`, confirm all existing + new
      tests pass.
- [ ] **8. Commit** — `src/explain/local_explainer.py`,
      `notebooks/10_explainability.ipynb`, `tests/test_local_explainer.py`,
      `reports/figures/production_shap_global_importance.png`,
      `requirements.txt`, `.claude/CLAUDE.md`, commit message `phase 3:
      explainable AI -- local SHAP + LIME per-customer explanations`.

### Tests to write (hand to test-writer)

- `tests/test_local_explainer.py::test_build_explainer_context_raises_on_non_tree_model` —
  monkeypatched metadata naming `"LogisticRegression"` → `ValueError`
  mentioning `"LogisticRegression"`, before any SHAP/LIME object is built.
- `tests/test_local_explainer.py::test_build_explainer_context_raises_on_stale_feature_columns` —
  monkeypatched metadata with a feature-column list that doesn't match
  `train.split_data(df)`'s actual columns → `ValueError`, before fitting
  anything.
- `tests/test_local_explainer.py::test_build_explainer_context_returns_expected_keys` —
  on the real `clean_df`/persisted model, the returned dict has every key
  listed in Requirement 2, and `context["model_name"] == "XGBoost"`.
- `tests/test_local_explainer.py::test_global_shap_importance_matches_verified_production_ranking` —
  top-2 are `Contract` then `tenure` (exact order), and CLAUDE.md §6's 5
  documented signals have at least 4 present in the top 6 — intentionally
  brittle, matching `06`/`08`/`09`/`10`'s precedent for real-data
  regression guards.
- `tests/test_local_explainer.py::test_global_shap_importance_excludes_target_and_is_sorted` —
  `"Churn"` never in the `column` output; `mean_abs_shap` sorted
  descending, all `>= 0`.
- `tests/test_local_explainer.py::test_plot_global_shap_importance_returns_existing_path` —
  writes to `tmp_path`, returned `Path.exists()`.
- `tests/test_local_explainer.py::test_local_shap_top_drivers_matches_verified_high_risk_example` —
  the exact verified high-risk test-split customer (`tenure=1`,
  `Contract=Month-to-month`, `InternetService=Fiber optic`, ...) →
  `local_shap_top_drivers` returns `tenure`, `Contract`, `InternetService`
  as the top-3 (any order within `pytest.approx` tolerance on values), all
  three `direction == "increases"`.
- `tests/test_local_explainer.py::test_local_shap_top_drivers_matches_verified_low_risk_example` —
  the exact verified low-risk test-split customer → top-3 =
  `Contract`/`tenure`/`OnlineSecurity`, all three `direction == "decreases"`.
- `tests/test_local_explainer.py::test_local_shap_top_drivers_zero_value_is_neutral` —
  a synthetic case forcing a `0.0` aggregated SHAP value →
  `direction == "neutral"`.
- `tests/test_local_explainer.py::test_local_shap_top_drivers_respects_top_n` —
  `top_n=1` and `top_n=50` (larger than available columns) both return
  without error, with `len(result) <= top_n` and `len(result) <=
  len(feature_columns)`.
- `tests/test_local_explainer.py::test_local_lime_top_drivers_matches_verified_low_risk_example` —
  the same low-risk customer → LIME's returned `feature` strings include
  `Contract` and `tenure`-derived descriptions among the top drivers, with
  `direction == "decreases"` for both, matching the verified SHAP agreement.
- `tests/test_local_explainer.py::test_local_lime_top_drivers_raises_on_unseen_category` —
  a customer with a categorical value never seen in `X_train` (e.g.
  `Contract="Lifetime"`) → `ValueError` naming `"Contract"` and the bad
  value, not a bare sklearn traceback.
- `tests/test_local_explainer.py::test_column_from_lime_description_resolves_categorical_and_numeric_forms` —
  unit test against the exact verified description shapes
  (`"Contract=Two year"`, `"tenure > 55.00"`, `"InternetService=Fiber
  optic"`) each resolve to the correct original column.
- `tests/test_local_explainer.py::test_column_from_lime_description_raises_on_unmatched` —
  a synthetic description matching no known column → `ValueError`.
- `tests/test_local_explainer.py::test_humanize_reason_known_signal_columns` —
  for each of CLAUDE.md §6's 5 signal columns, the generated reason string
  contains that column's templated phrase (not the generic `"{column} =
  {value}"` fallback) and an explicit direction word.
- `tests/test_local_explainer.py::test_humanize_reason_unknown_column_uses_generic_fallback` —
  a column not in `FEATURE_DISPLAY_TEMPLATES` → falls back to `"{column} =
  {value}"` phrasing, still with a direction word, no exception.
- `tests/test_local_explainer.py::test_humanize_reason_never_leaks_transformed_feature_names` —
  no output for any real feature column contains `"cat__"` or `"num__"`.
- `tests/test_local_explainer.py::test_explain_customer_returns_both_methods` —
  a real raw customer dict (the verified high-risk example, re-supplied as
  raw attributes) → result has non-empty `shap_top_drivers` and
  `lime_top_drivers`, each entry has a non-empty `reason`.
- `tests/test_local_explainer.py::test_explain_customer_raises_on_missing_required_column` —
  a customer dict missing e.g. `Contract` → `ValueError` naming `Contract`.
- `tests/test_local_explainer.py::test_explain_customer_includes_customer_id_when_present` —
  a customer dict carrying `customerID` → result includes it unchanged.
- `tests/test_local_explainer.py::test_explain_customer_reuses_supplied_context` —
  building `context` once and passing it explicitly to two different
  customers does not rebuild `shap_explainer`/`lime_explainer` (asserted
  via `is` identity on `context["shap_explainer"]` before/after both
  calls), confirming the reuse contract Requirement 9 documents.
- `tests/test_local_explainer.py::test_generate_explainability_figures_returns_one_existing_path` —
  returns exactly 1 `Path`, `.exists()`; writes to a pytest `tmp_path`
  rather than the tracked `reports/figures/`.

### Quality gates

1. **test-runner** — run `pytest -q`, confirm all tests (existing + new)
   pass, fix any straightforward regression. Flag if the LIME-based tests
   are slow (verified ~1.4s per `explain_instance` call; several tests call
   it) and worth noting alongside `08`'s/`09`'s already-flagged slow tests.
2. **quality-reviewer** — review `build_explainer_context`'s staleness and
   tree-based-model guards, the SHAP one-hot aggregation order (must match
   `06`'s verified-correct signed-sum-then-abs-mean rule, never the
   reverse), the LIME encode/decode round-trip correctness, the
   description-to-column resolution's prefix-collision safety, and
   CLAUDE.md §8 adherence (named constants, type hints, docstrings).
3. **security-reviewer** — confirm the unseen-category `ValueError` path
   never leaks into an unhandled exception; confirm no dynamic code
   execution or path/query construction from any customer-supplied value;
   confirm the `lime` version pin is the only dependency-surface change and
   is justified (already listed, first real installation/usage).
4. Commit only after both reviews are clean.

### Risks / rollback

- **Risk:** LIME's `explain_instance` costs ~1.4s per customer
  (`num_samples=5000` perturbed rows through the full pipeline) — too slow
  for a synchronous per-request Phase 5 endpoint at scale. **Mitigation:**
  explicitly out of scope for this spec (no endpoint exists yet); flagged
  here as a known future tuning point (`LIME_NUM_SAMPLES` is already a
  named, adjustable constant) rather than silently deferred with no trace.
- **Risk:** a future retrain (`python -m src.models.train`) selects a
  non-tree model (e.g. `LogisticRegression` overtakes on `cv_auc_mean`),
  breaking `build_explainer_context`. **Mitigation:** the explicit
  `TREE_BASED_MODELS` guard (Requirement 2) fails loudly and immediately
  with the actual model name, rather than a confusing SHAP-internal error
  three calls later — directly tested. This is a known, disclosed
  limitation of `TreeExplainer`, not a bug to silently work around within
  this spec's scope.
- **Risk:** a future Kaggle re-download or retrain shifts the verified
  Research-note numbers (global ranking, the two example customers' exact
  SHAP/LIME values) enough that the brittle regression tests fail.
  **Mitigation:** intentional, matching `06`/`08`/`09`/`10`'s brittleness
  philosophy — these tests should fail loudly on real distributional
  shift or a different winning model, prompting deliberate re-verification.
- **Risk:** a reviewer assumes the SHAP/LIME explanations describe the
  *calibrated* probability shown elsewhere in the app. **Mitigation:** the
  Interpretation note at the top of this spec, and `explain_customer`'s
  own scope (it never returns a probability at all, only drivers), make
  this explicit rather than implying false numeric consistency.
- **Rollback:** single commit (Task 8) covering only additive files (new
  module, new notebook, new tests, one new PNG, one `requirements.txt`
  pin, one CLAUDE.md doc/status line) — `git revert` is clean since nothing
  existing is modified in place.

### Definition of done

- All 8 tasks checked off.
- `pytest -q` green (all existing tests + `test_local_explainer.py`).
- `notebooks/10_explainability.ipynb` executes top-to-bottom without error.
- `reports/figures/` gains `production_shap_global_importance.png`, no/
  existing figure altered.
- CLAUDE.md §5 documents `python -m src.explain.local_explainer`; CLAUDE.md
  §14's Phase 3 row is marked `☑`.
- `quality-reviewer` and `security-reviewer` report no unresolved findings.
- All Success Criteria in Part 1 are met.
