# Spec + Plan: Model Comparison / AutoML-Style Leaderboard

> Location note: combined spec+plan at `.claude/specs/`, following `01`–`11`'s
> established convention (CLAUDE.md §4 marks `.claude` as "claude.md and also
> Specs folder"), not the generic `/create-spec` default of
> `specs/<slug>/{spec,plan}.md` — `.claude/specs/08-churn-prediction-model.md`
> made and documented this same override.
>
> Scope note: this is **not** a new CLAUDE.md §14 phase row. It pulls forward
> one narrow slice of Phase 5 ("FastAPI service + Streamlit dashboard +
> What-If panel") — a read-only leaderboard view in the existing dashboard —
> without touching FastAPI, the What-If panel, or any other Phase 5 surface.
> No phase status flips on completion.
>
> Clarified scope (answered via `/create-spec`'s clarifying questions before
> this file was written, recorded here rather than left implicit):
> 1. **Surface:** Streamlit only — a new tab in `app/dashboard.py`. No new
>    FastAPI endpoint in this feature.
> 2. **"AutoML-style" means presentation, not search:** the same 4
>    fixed-hyperparameter candidates `src/models/train.py` already fits
>    (`LogisticRegression`, `RandomForest`, `XGBoost`, `LightGBM`) — this
>    feature adds no new candidate models, no hyperparameter sweep, no new
>    dependency. CLAUDE.md §2 rule 5 (reproducibility) and §13 ("add heavy new
>    dependencies without asking") both cut against silently broadening the
>    search inside a "just add a leaderboard view" ask.
> 3. **Data source: latest run only.** Reads the single
>    `reports/model_comparison.csv` produced by the most recent
>    `python -m src.models.train` run — not a live query over MLflow's full
>    run history. Simpler, no new MLflow-read dependency in the dashboard, and
>    matches what `train.py` already produces and commits.
>
> Research note: `src/models/train.py` (Phase 2, `.claude/specs/08-churn-prediction-model.md`,
> already on `main`) already computes and commits exactly the leaderboard
> data this feature needs — `compare_models()` returns a DataFrame with
> `cv_auc_mean`, `cv_auc_std`, `test_auc`, `test_pr_auc`, `test_precision`,
> `test_recall`, `test_f1`, `test_brier`, `tuned_threshold`,
> `test_f1_at_tuned_threshold`, `meets_target_auc` per candidate, sorted by
> `cv_auc_mean` descending (the train-only selection metric — see `08`'s
> post-implementation hardening note on why `test_auc` is not used for
> selection) — verified live, current committed values:
>
> | name | cv_auc_mean | test_auc | test_pr_auc | test_precision | test_recall | test_f1 | test_brier | meets_target_auc |
> |---|---|---|---|---|---|---|---|---|
> | XGBoost | 0.8466 | 0.8434 | 0.6466 | 0.5616 | 0.6952 | 0.6213 | 0.1466 | False |
> | LightGBM | 0.8463 | 0.8437 | 0.6490 | 0.5558 | 0.6925 | 0.6167 | 0.1462 | False |
> | LogisticRegression | 0.8449 | 0.8391 | 0.6233 | 0.5017 | 0.7914 | 0.6141 | 0.1686 | False |
> | RandomForest | 0.8195 | 0.8189 | 0.5907 | 0.5788 | 0.5695 | 0.5741 | 0.1537 | False |
>
> No candidate meets `TARGET_AUC = 0.85` today — this feature's job is to
> surface that honestly (CLAUDE.md §2 rule 2's "report the finding, don't
> hide it"), not to reach 0.85 by adding candidates.

---

## PART 1 — SPEC

### Feature

A read-only "Model Leaderboard" tab in the Streamlit dashboard that renders
`src/models/train.py`'s existing 4-candidate comparison (from the most recent
`python -m src.models.train` run) as a sortable table plus a grouped bar
chart, highlights the model `select_best_model` would pick, and discloses
plainly whether it clears the project's 0.85 AUC-ROC target.

### Problem / motivation

`reports/model_comparison.csv` already holds the honest, leakage-guarded
4-model comparison, but today the only way to see it is opening the raw CSV,
`reports/figures/model_comparison_auc.png` (a static PNG), or the MLflow UI.
There is no in-product view — the dashboard (`app/dashboard.py`) currently
shows only the KPI Overview page. A recruiter/reviewer opening the live
dashboard has no way to see the model comparison that backs CLAUDE.md §1's
"Churn classifier with AUC-ROC ≥ 0.85" goal, or to see — honestly — that the
current best model doesn't yet clear it.

### Goals / non-goals

**Goals**
- Add `src/models/leaderboard.py`: `load_leaderboard`, `best_model_name`,
  `format_leaderboard_table`, `plot_leaderboard_metrics` — pure
  functions/figure-builders, no Streamlit import, following `src/data/kpi.py`'s
  compute-then-plot pattern so dashboard code stays thin.
- Add a "Model Leaderboard" tab to `app/dashboard.py` (via `st.tabs`,
  alongside the existing KPI content, now "KPI Overview") that renders the
  table, the chart, a best-model callout, and an honest met/not-met badge
  against `TARGET_AUC`.
- Handle the case where `reports/model_comparison.csv` doesn't exist yet
  (fresh clone, no training run) with a friendly `st.info` pointing at
  `python -m src.models.train`, not a crash.
- Add `tests/test_leaderboard.py` and extend `tests/test_dashboard.py`
  (or add `tests/test_dashboard_leaderboard.py`) covering both modules.

**Non-goals**
- No new FastAPI endpoint — no `src/api/` change (clarified scope #1).
- No new candidate models, no hyperparameter sweep, no new ML dependency —
  the leaderboard renders exactly the 4 candidates `MODEL_SPECS` already
  fits (clarified scope #2). `src/models/train.py` is not modified.
- No live MLflow-history view — reads `reports/model_comparison.csv` only,
  not `mlflow.search_runs` (clarified scope #3).
- No What-If panel, no `/predict` wiring, no risk-tier or SHAP content in
  this tab — those are separate existing/future features.
- No change to `reports/model_comparison.csv`'s schema, to
  `src/models/train.py`, `src/models/evaluation.py`, or
  `src/models/calibration.py`.
- No re-run of training from the dashboard (no "retrain" button) — this is a
  read-only view of the last run's output.

### User stories

- As a **recruiter/reviewer**, I want to open the live dashboard and see the
  4-model comparison table and chart without touching a notebook or the
  MLflow UI, so I can evaluate the modeling work in the same place as the KPIs.
- As the **engineer (Priyabrata)**, I want the dashboard to show, honestly,
  whether the currently-selected model clears the 0.85 AUC target, so the
  shortfall documented in `.claude/specs/08-churn-prediction-model.md` stays
  visible rather than only living in a spec file.
- As a **churn analyst**, I want to see precision/recall/F1/Brier alongside
  AUC-ROC/PR-AUC for every candidate, not just accuracy, so I can judge the
  precision/recall trade-off myself instead of trusting one number.

### Functional requirements

1. `src/models/leaderboard.py` MUST define
   `load_leaderboard(path: Path = train.COMPARISON_TABLE_PATH) -> pd.DataFrame`
   — `pd.read_csv(path)`, re-sorted by `cv_auc_mean` descending (defensive;
   the source CSV is already sorted this way, but this function must not
   assume it stays that way). MUST raise `FileNotFoundError` with a message
   naming `python -m src.models.train` if `path` doesn't exist — same pattern
   as `train.load_trained_model`. MUST raise `ValueError` naming the missing
   column(s) if any of `{"name", "cv_auc_mean", "cv_auc_std", "test_auc",
   "test_pr_auc", "test_precision", "test_recall", "test_f1", "test_brier",
   "tuned_threshold", "meets_target_auc"}` is absent from the loaded CSV —
   an actionable failure instead of a `KeyError` deep inside a plotting call.
2. MUST define `best_model_name(df: pd.DataFrame) -> str` — the `name` of the
   row with the max `cv_auc_mean` (`df.loc[df["cv_auc_mean"].idxmax(), "name"]`),
   matching `train.select_best_model`'s selection metric exactly (not
   `test_auc`) so the dashboard's "best model" badge can never disagree with
   which model `python -m src.models.train` actually persisted.
3. MUST define `format_leaderboard_table(df: pd.DataFrame) -> pd.DataFrame` —
   a display copy: renamed, human-readable columns (`Model`, `CV AUC-ROC`,
   `CV AUC-ROC Std`, `Test AUC-ROC`, `Test PR-AUC`, `Precision`, `Recall`,
   `F1`, `Brier Score`, `Tuned Threshold`, `Meets 0.85 Target`), all numeric
   columns rounded to 4 decimal places, `meets_target_auc` rendered as
   `"Yes"`/`"No"` strings (not raw booleans). MUST NOT mutate the input `df`
   (returns a new frame; original numeric precision stays intact for the
   chart function).
4. MUST define `plot_leaderboard_metrics(df: pd.DataFrame) -> go.Figure` — a
   grouped bar chart, one group per model (`name`), one bar per metric in
   `("test_auc", "test_pr_auc", "test_precision", "test_recall", "test_f1")`
   — all five are "higher is better," so grouping them together is not
   misleading. `test_brier` MUST NOT appear in this chart (lower-is-better,
   different scale/semantics — mixing it into a "higher is better" grouped
   bar would misrepresent it). Uses `plotly.graph_objects`, matching
   `src/data/kpi.py`'s existing chart style (not matplotlib — this is a live
   dashboard widget, not a saved notebook figure, mirroring `plot_mrr_breakdown`'s
   own docstring rationale).
5. `app/dashboard.py` MUST wrap existing KPI content in `st.tabs(["KPI
   Overview", "Model Leaderboard"])`'s first tab, unchanged in behavior
   (`test_dashboard_renders_five_metrics_without_exception` and the other two
   existing dashboard tests MUST keep passing with no test-body changes
   beyond locating widgets inside the new tab structure if `AppTest`
   requires it).
6. The "Model Leaderboard" tab MUST, on success: call `load_leaderboard()`,
   render `st.dataframe(format_leaderboard_table(df))`, render
   `st.plotly_chart(plot_leaderboard_metrics(df))`, and show a callout naming
   `best_model_name(df)` plus that row's `meets_target_auc` value rendered
   via `st.success` (if `True`) or `st.warning` (if `False`) with the actual
   `test_auc` value and the `0.85` target both stated in the message text
   (not hidden behind a boolean icon alone).
7. The "Model Leaderboard" tab MUST, if `load_leaderboard()` raises
   `FileNotFoundError`, render `st.info(...)` naming
   `python -m src.models.train` as the fix — MUST NOT raise an unhandled
   exception up through `AppTest` (mirrors `app/dashboard.py`'s existing
   `FileNotFoundError`/`OSError`/... handling for `load_clean_data`).
8. `tests/test_leaderboard.py` and the dashboard test additions MUST cover
   Plan's "Tests to write" section in full.
9. None of the above may change `src/models/train.py`,
   `src/models/evaluation.py`, `src/models/calibration.py`,
   `src/data/kpi.py`, or any existing test/figure/notebook — all current
   tests must keep passing unmodified in behavior (only relocated inside the
   new `st.tabs` structure where `AppTest` requires it).

### Data & model impact

None. No column added to or removed from any DataFrame; no model retrained,
no new artifact under `models/`; `reports/model_comparison.csv` is read-only
input, never written by this feature. Purely a presentation layer over data
Phase 2 already produces and already commits to the repo.

### ML guardrails (mandatory check)

N/A for new modeling — no model is trained, tuned, or scored by this
feature. The guardrail relevance here is **disclosure, not computation**:
Functional Requirement 6 mandates that the dashboard state the actual
`test_auc` against the `0.85` target in plain text (not just a colored icon),
so CLAUDE.md §2 rule 2's "report the finding, don't hide it" carries through
from `.claude/specs/08-churn-prediction-model.md`'s honest-AUC finding into
the live UI, not just the spec file and CSV. `format_leaderboard_table`
(Requirement 3) surfaces precision, recall, F1, and Brier score alongside
AUC-ROC/PR-AUC for every candidate — never accuracy alone (CLAUDE.md §2 rule 3),
consistent with what `compare_models` already computes.

### API / UI surface

**UI only.** `app/dashboard.py` gains a second tab. No FastAPI route added —
`src/api/` is untouched (clarified scope #1). No new Streamlit page file;
implemented as `st.tabs` inside the existing single `dashboard.py` (simplest
structure for two sections; a `pages/` multi-page app is Phase 5's full scope,
not needed for one additional tab).

### Edge cases & failure states

- **`reports/model_comparison.csv` absent** (fresh clone before any
  `python -m src.models.train` run, or a wiped `reports/` directory): handled
  by Requirement 7 — friendly `st.info`, no crash. Directly tested.
- **CSV present but missing an expected column** (e.g. a stale CSV from a
  version of `train.py` before a column was added): `load_leaderboard` raises
  `ValueError` naming the missing column(s) (Requirement 1) rather than
  letting a `KeyError` surface later inside `format_leaderboard_table` or
  `plot_leaderboard_metrics` — caught by the same `except` clause pattern the
  dashboard tab uses for `FileNotFoundError`, surfaced as `st.error` with a
  message suggesting a fresh `python -m src.models.train` run.
- **CSV present but empty (zero data rows)**: `best_model_name`'s
  `idxmax()` on an empty column raises `ValueError` (pandas' native
  behavior) — allowed to propagate up to the same `st.error` handler as the
  malformed-column case above, rather than silently rendering an empty table
  with no best-model callout.
- **All four candidates tied exactly on `cv_auc_mean`** (not reachable on
  real data, verified in the research table above — max shared margin is
  ~0.0003 between XGBoost/LightGBM, not an exact tie): `idxmax()` returns the
  first tied row, consistent with `train.select_best_model`'s
  already-established `comparison.iloc[0]`-on-sorted-frame tie-break
  behavior — same convention, not a new one invented here.
- **Re-running `python -m src.models.train` while the dashboard is open**:
  out of scope to auto-refresh live — the tab reads the CSV fresh on every
  Streamlit rerun (no `st.cache_data` on `load_leaderboard`, unlike
  `_load_data`'s cached KPI load, since the comparison table is 4 rows and
  cheap to re-read every time), so a manual page reload picks up a new run's
  output without needing a dedicated refresh button.

### Security notes

- **No new untrusted input.** `reports/model_comparison.csv` is generated
  exclusively by this repo's own `python -m src.models.train` (Phase 2,
  trusted, already reviewed) and is a **tracked** file in the repo, not a
  user upload or request body — no CSV-injection or deserialization concern
  distinct from any other tracked-and-read CSV in this codebase (e.g.
  `src/data/kpi.py` reading `load_clean_data()`'s output).
- **No new dependency.** `pandas`, `streamlit`, and `plotly` are all already
  pinned in `requirements.txt` and already imported by `app/dashboard.py` /
  `src/data/kpi.py`.
- **No secret handling.** No env var, no API key, no network call introduced.

### Success criteria

- `pytest -q` passes: all existing tests + `tests/test_leaderboard.py` +
  the new/extended dashboard leaderboard tests, all green.
- `streamlit run app/dashboard.py` shows two tabs; "Model Leaderboard" renders
  the 4-candidate table, the chart, and a best-model callout that correctly
  states whether the 0.85 AUC target is met.
- Deleting/renaming `reports/model_comparison.csv` and reloading the
  dashboard shows the friendly `st.info` message, not a stack trace.
- `quality-reviewer` and `security-reviewer` report no unresolved findings on
  the diff.

### Out of scope

- FastAPI leaderboard endpoint, live MLflow-history view, broadened
  model/hyperparameter search (all explicitly deferred per the clarifying
  answers above).
- What-If panel, `/predict` wiring, SHAP/risk-tier content in this tab.
- A dedicated "retrain" button or any write path from the dashboard.
- Any change to `src/models/train.py`, `src/models/evaluation.py`,
  `src/models/calibration.py`, or `reports/model_comparison.csv`'s schema.

---

## PART 2 — PLAN

### Approach

One new pure module, `src/models/leaderboard.py`, following
`src/data/kpi.py`'s established compute-then-Plotly-figure pattern (no
Streamlit import inside it — keeps it independently unit-testable, matching
CLAUDE.md §4's "keep the API/dashboard thin" split), plus a minimal
`app/dashboard.py` change wrapping the current single view in `st.tabs` and
adding the second tab's rendering calls.

**Alternative rejected:** a separate `pages/1_Leaderboard.py` Streamlit
multi-page file instead of `st.tabs` in one file. Rejected because a full
`pages/` directory is Phase 5's eventual full-dashboard scope (KPI, What-If,
Leaderboard, etc. as separate pages) — introducing that structure now for one
extra tab is more surface than this feature needs, and `st.tabs` keeps the
existing `AppTest`-based test file working against the same single
`DASHBOARD_PATH` with minimal disruption.

### Task breakdown

- [ ] **1. Create `src/models/leaderboard.py`** — `load_leaderboard`,
      `best_model_name`, `format_leaderboard_table`, `plot_leaderboard_metrics`
      (Functional Requirements 1–4). Imports `train.COMPARISON_TABLE_PATH` and
      `train.TARGET_AUC`-equivalent (`evaluation.TARGET_AUC`) rather than
      redefining either constant.
- [ ] **2. Edit `app/dashboard.py`** — wrap existing body in
      `st.tabs(["KPI Overview", "Model Leaderboard"])`; add the second tab's
      render logic per Requirements 5–7 (try/except around
      `load_leaderboard()`, success path renders table + chart + callout,
      `FileNotFoundError`/`ValueError` path renders `st.info`/`st.error`).
- [ ] **3. Add `tests/test_leaderboard.py`** — see Tests to write below.
- [ ] **4. Extend `tests/test_dashboard.py`** (or add
      `tests/test_dashboard_leaderboard.py` if that reads cleaner once
      written — decide at implementation time based on file length) — new
      tests for the leaderboard tab; confirm the three existing tests still
      pass with the `st.tabs` wrapper.
- [ ] **5. Manual check** — `streamlit run app/dashboard.py`, click into
      "Model Leaderboard," confirm the table/chart/callout render correctly
      against the real `reports/model_comparison.csv`; temporarily
      rename the CSV and reload to confirm the friendly-error path.
- [ ] **6. Run the full suite** — `pytest -q`, confirm all existing + new
      tests pass.
- [ ] **7. Commit** — `src/models/leaderboard.py`, `app/dashboard.py`,
      `tests/test_leaderboard.py`, dashboard test changes,
      `.claude/CLAUDE.md` (only if any command-list update is warranted —
      `streamlit run app/dashboard.py` is already documented, so likely no
      change needed here), commit message
      `feat: model comparison leaderboard in Streamlit dashboard`.

### Tests to write (hand to test-writer)

- `tests/test_leaderboard.py::test_load_leaderboard_returns_sorted_by_cv_auc_desc` —
  write a small synthetic CSV (4 rows, all required columns, deliberately
  out of order) to `tmp_path`, assert `load_leaderboard(path)["cv_auc_mean"]`
  is non-increasing.
- `tests/test_leaderboard.py::test_load_leaderboard_raises_actionable_error_when_missing` —
  `load_leaderboard(tmp_path / "nonexistent.csv")` raises `FileNotFoundError`
  mentioning `python -m src.models.train`.
- `tests/test_leaderboard.py::test_load_leaderboard_raises_on_missing_column` —
  a synthetic CSV missing e.g. `test_brier` raises `ValueError` naming that
  column.
- `tests/test_leaderboard.py::test_best_model_name_matches_max_cv_auc_mean` —
  fabricated DataFrame, asserts the returned name matches the row with the
  highest `cv_auc_mean`, not the highest `test_auc` (a row deliberately
  constructed where these two disagree, to lock in Requirement 2's
  train-only selection metric).
- `tests/test_leaderboard.py::test_format_leaderboard_table_columns_are_readable_and_input_unmutated` —
  asserts renamed columns present, `Meets 0.85 Target` is `"Yes"`/`"No"`
  strings, and the original input DataFrame's `meets_target_auc` column is
  still boolean after the call (no in-place mutation).
- `tests/test_leaderboard.py::test_plot_leaderboard_metrics_has_one_trace_per_metric_and_excludes_brier` —
  asserts the returned `go.Figure` has exactly 5 bar traces (one per
  higher-is-better metric) and no trace's data corresponds to `test_brier`.
- `tests/test_leaderboard.py::test_plot_leaderboard_metrics_covers_all_models` —
  asserts every model `name` from the input DataFrame appears on the chart's
  categorical axis.
- `tests/test_dashboard.py::test_dashboard_kpi_tab_still_renders_five_metrics` —
  update/rename the existing five-metrics test if `AppTest` widget lookup
  needs to target the first tab explicitly after the `st.tabs` wrap; MUST
  still pass unchanged in assertion intent.
- `tests/test_dashboard.py::test_dashboard_leaderboard_tab_renders_table_chart_and_best_model_callout` —
  point `train.COMPARISON_TABLE_PATH` (via `monkeypatch`) at a `tmp_path` CSV
  fixture with 4 synthetic rows; `AppTest` run, assert `at.exception == []`,
  a dataframe widget is present, a plotly chart widget is present, and
  `at.success` or `at.warning` (whichever the fixture's best row implies)
  is non-empty.
- `tests/test_dashboard.py::test_dashboard_leaderboard_tab_shows_friendly_message_when_csv_missing` —
  point `train.COMPARISON_TABLE_PATH` at a nonexistent `tmp_path` file;
  assert `at.exception == []` and `at.info` (or equivalent) is non-empty.
- `tests/test_dashboard.py::test_dashboard_refresh_button_still_reloads_kpi_tab_without_exception` —
  confirm the existing refresh-button test still passes with the `st.tabs`
  wrapper in place.

### Quality gates

1. **test-runner** — run `pytest -q`, confirm all tests (existing + new)
   pass, fix any straightforward regression from the `st.tabs` restructuring.
2. **quality-reviewer** — review `best_model_name`'s selection metric matches
   `train.select_best_model` exactly (train-only `cv_auc_mean`, not
   `test_auc`), the missing-column/missing-file error handling, Brier
   score's exclusion from the higher-is-better bar chart, and CLAUDE.md §8
   adherence (named constants reused from `train`/`evaluation` rather than
   re-defined, type hints, docstrings).
3. **security-reviewer** — confirm no new untrusted input surface (CSV is
   repo-generated and tracked, not user-supplied) and no new dependency.
4. Commit only after both reviews are clean.

### Risks / rollback

- **Risk:** `AppTest`'s widget-lookup API behaves differently once content
  moves inside `st.tabs` (e.g. `at.metric` might need tab-scoped lookup).
  **Mitigation:** Task 5's manual check plus Task 4 explicitly re-validates
  the three pre-existing dashboard tests before considering the feature done;
  if `AppTest` can't address widgets inside a specific tab cleanly, fall back
  to `st.expander`/`st.container` sections instead of `st.tabs` (a
  Streamlit-API-compatibility decision made at implementation time, not
  pre-committed here).
- **Risk:** a stale `reports/model_comparison.csv` from before this feature
  (e.g. checked out from an older commit) is missing a column added later to
  `train.py`'s output. **Mitigation:** Requirement 1's explicit
  missing-column `ValueError` turns that into a clear, actionable dashboard
  message instead of a silent wrong render or a raw `KeyError` traceback.
- **Rollback:** single commit (Task 7) adding one new module, one new test
  file, and one contained edit to `app/dashboard.py` (wrap-in-tabs + new tab
  body) — `git revert` is clean; `reports/model_comparison.csv` itself is
  untouched by this feature either way.

### Definition of done

- All 7 tasks checked off.
- `pytest -q` green (all existing tests + `tests/test_leaderboard.py` + the
  extended dashboard tests).
- `streamlit run app/dashboard.py` manually confirmed: both tabs render, the
  leaderboard's honest met/not-met badge is correct against the real CSV,
  and the missing-CSV path shows a friendly message.
- `quality-reviewer` and `security-reviewer` report no unresolved findings.
- All Success Criteria in Part 1 are met.
