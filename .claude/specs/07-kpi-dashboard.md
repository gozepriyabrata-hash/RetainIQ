# Spec + Plan: KPI Dashboard — Real-Time Churn Rate, MRR Loss, CLV, Avg Tenure, Service-Attach Rate

> Location note: this file lives at `.claude/specs/` per CLAUDE.md §4's repo
> tree, which marks `.claude` as "claude.md and also Specs folder" —
> consistent with `01`–`06`. Spec and plan are combined in one file for the
> same reason.
>
> Scope note: this is the **first UI-facing feature in the repo** — `app/` is
> currently an empty directory in the tree diagram, and no Streamlit code
> exists anywhere. It is deliberately **not** the full CLAUDE.md §14 Phase 5
> ("FastAPI service + Streamlit dashboard + What-If panel"): Phase 5's
> What-If panel and prediction views depend on a trained model
> (`models/`) and a `/predict` endpoint, and neither Phase 2 nor Phase 5's
> API exists yet. This feature ships a narrower, self-contained **KPI-only**
> Streamlit page that reads only the already-cleaned dataset — five
> descriptive metrics computed live on load, no model dependency, no
> FastAPI dependency. No phase-tracker row changes; Phase 5 remains fully
> unblocked and unaffected (a future Phase 5 dashboard can compose this
> page's `src/data/kpi.py` logic into a multi-page app rather than
> rewriting it).
>
> Scope decisions (asked and answered before writing this spec, since none
> were inferable from the code or CLAUDE.md):
> 1. **Ship a real Streamlit app now**, not compute-only functions — the
>    user confirmed a minimal `app/dashboard.py` over deferring UI to a
>    later phase.
> 2. **CLV = ARPU / churn rate** (the classic model-free formula), not
>    historical `TotalCharges` — chosen because no predictive/survival model
>    exists yet to base a forward-looking CLV on, and this is the standard
>    substitute in that situation.
> 3. **Service-attach rate = mean(`ServiceCount`)** (a continuous
>    "services per customer" number reusing `compute_service_count` from
>    `segmentation.py`), not a binary "% with 2+ services" threshold rate —
>    avoids inventing and justifying an arbitrary threshold.
>
> "Real-time" note: this dataset is a static, single-snapshot CSV with no
> streaming source and no timestamped billing periods — there is nothing to
> stream. "Real-time" is interpreted honestly as **recomputed live from the
> current cleaned dataset on every dashboard load**, plus a manual "Refresh
> data" control that clears the cache and rebuilds from the raw CSV — not a
> literal live data feed. This interpretation is stated explicitly rather
> than silently narrowed, the same way `03-cohort-analysis.md` disclosed its
> retention curve wasn't censoring-corrected.
>
> All figures below are verified directly against `load_clean_data()`'s
> current 7,043 rows with the exact formulas this feature implements, not
> estimated.

---

## PART 1 — SPEC

### Feature

A new `src/data/kpi.py` module computing five headline retention KPIs —
**Churn Rate**, **MRR Loss**, **CLV**, **Average Tenure**, **Service-Attach
Rate** — plus two supporting breakdown charts, consumed by a new
`app/dashboard.py` Streamlit page that renders them as live metric tiles
with a manual refresh control.

### Problem / motivation

Nothing in the repo today gives a single-glance business summary of churn
impact — `eda.py`/`cohorts.py`/`segmentation.py`/`lifecycle.py` all produce
segment- or cohort-level tables and static PNGs consumed from notebooks, not
a live top-line view. Verified on current data:

1. **Churn rate = 26.537%** (1,869 of 7,043 customers) — matches CLAUDE.md
   §1's documented ~26.5% positive class.
2. **MRR Loss:** total monthly recurring revenue across all customers is
   **$456,116.60**; churned customers alone account for **$139,130.85/mo**
   (**30.50%** of total MRR) — a share *larger* than their 26.54% population
   share, because churners skew toward higher-`MonthlyCharges` plans. This is
   consistent with CLAUDE.md §6's documented fiber-optic/no-tech-support
   churn signals, both of which correlate with pricier plans.
3. **CLV (ARPU / churn rate) = $244.04** — average `MonthlyCharges`
   ($64.76) divided by the churn-rate fraction (0.26537), the standard
   model-free expected-lifetime-value estimate. **Caveat (found during
   `quality-reviewer`):** this churn rate is a cross-sectional share of the
   customer base that has ever churned, not a true per-period (monthly)
   hazard rate — dividing by it implies an average customer lifetime of
   `244.04 / 64.76 ≈ 3.77 months`, which visibly disagrees with the
   `avg_tenure_kpi` tile rendered right next to it (32.37 months), an
   ~8.6x gap. This is disclosed in `clv_kpi`'s docstring and via an
   `st.caption` on the dashboard rather than silently presented as
   precise — read CLV as a rough order-of-magnitude estimate, not a
   monthly-cohort-based lifetime value, and not reconcilable with
   `avg_tenure_kpi` without a true per-period churn rate (which would need
   time-series billing data this snapshot dataset doesn't have).
4. **Average tenure = 32.37 months.**
5. **Service-attach rate = 4.15 of 9 possible services (46.07%)** — mean
   `ServiceCount` (8 Yes/No add-ons + an internet flag, reusing
   `compute_service_count` from `segmentation.py` rather than redefining it).

### Goals / non-goals

**Goals**
- Add `src/data/kpi.py` with one `*_kpi` function per metric, a combined
  `kpi_summary()` table, and two Plotly breakdown figures — following the
  `compute_*`/`plot_*` pattern of `cohorts.py`/`segmentation.py`/
  `lifecycle.py`, importing `compute_service_count` from
  `src.models.segmentation` rather than duplicating it.
- Add `app/dashboard.py`: a Streamlit page rendering the 5 KPIs as
  `st.metric` tiles, the 2 breakdown charts, and a manual refresh control —
  calling only into `src/data/kpi.py` and `src/data/load_data.py` (CLAUDE.md
  §4: keep the UI thin, no business logic inline).
- Add pytest coverage: `tests/test_kpi.py` for the compute functions
  (verified values, a zero-churn-rate division guard) and
  `tests/test_dashboard.py` using `streamlit.testing.v1.AppTest` for a
  dashboard smoke test.

**Non-goals**
- No FastAPI endpoint, no `/predict`/What-If panel, no model dependency —
  those are Phase 5's remaining scope, gated on Phase 2's trained model,
  which doesn't exist yet.
- No CSV upload / customer search — this page is a fixed, dataset-wide KPI
  summary, not the eventual per-customer or batch-scoring views.
- No new notebook — unlike `01`–`06`, the deliverable here *is* the live
  page itself, not a static notebook narrative; the two charts render only
  inside `app/dashboard.py` (as `plotly.graph_objects.Figure` objects, not
  saved PNGs) rather than joining `reports/figures/`.
- No literal streaming/live data source — see the "Real-time" note above.
- No new row in CLAUDE.md §14's phase tracker, and no change to
  `src/data/eda.py`, `src/data/cohorts.py`, `src/models/segmentation.py`,
  `src/data/lifecycle.py`, `clean_data()`'s output, or any existing
  test/figure.

### User stories

- As a **retention manager**, I want the five headline KPIs on one screen
  when I open the dashboard, so I don't have to run notebooks or query
  tables to see current churn exposure.
- As the **engineer (Priyabrata)**, I want the KPI math in a tested,
  reusable `src/data/kpi.py` module (not inline in the Streamlit script), so
  a future Phase 5 multi-page dashboard can reuse it without rewriting.
- As a **recruiter/reviewer**, I want the CLV formula and the "real-time"
  interpretation both stated plainly (not silently assumed), so the
  dashboard reads as a deliberate, disclosed design rather than a
  hand-wavy metric.

### Functional requirements

1. `src/data/kpi.py` MUST import `compute_service_count` and
   `SERVICE_YESNO_COLUMNS` from `src.models.segmentation` (no redefinition)
   and derive `SERVICE_ATTACH_MAX = len(SERVICE_YESNO_COLUMNS) + 1` as a
   module constant (9 today, but computed, not hardcoded) rather than a
   magic number (CLAUDE.md §8).
2. MUST gain `churn_rate_kpi(df: pd.DataFrame) -> float` = `df[TARGET_COLUMN].mean()
   * 100`, rounded 2dp. Verified ≈ **26.54**.
3. MUST gain `mrr_loss_kpi(df: pd.DataFrame) -> dict` with keys
   `total_mrr`, `lost_mrr`, `retained_mrr`, `lost_mrr_pct` (all rounded 2dp;
   `total_mrr = df["MonthlyCharges"].sum()`, `lost_mrr =
   df.loc[df[TARGET_COLUMN] == 1, "MonthlyCharges"].sum()`,
   `retained_mrr = total_mrr - lost_mrr`, `lost_mrr_pct = lost_mrr /
   total_mrr * 100`). Verified: `total_mrr` ≈ **456116.60**, `lost_mrr` ≈
   **139130.85**, `lost_mrr_pct` ≈ **30.50**.
4. MUST gain `clv_kpi(df: pd.DataFrame) -> float` = `df["MonthlyCharges"].mean()
   / df[TARGET_COLUMN].mean()`, rounded 2dp — dividing by the **unrounded**
   churn fraction (not the 2dp-rounded `churn_rate_kpi()` output) to avoid
   compounding rounding error. MUST raise `ValueError` (not return
   `inf`/`NaN`) if the churn fraction is `0` **or `NaN`** — both
   unreachable on real data today but real hazards on a filtered,
   synthetic, or empty frame; both directly tested. Verified ≈ **244.04**
   (dividing by the rounded `churn_rate_kpi()` output instead gives the
   subtly wrong `244.02` — a real rounding bug found and fixed during
   implementation).
5. MUST gain `avg_tenure_kpi(df: pd.DataFrame) -> float` =
   `df["tenure"].mean()`, rounded 2dp. Verified ≈ **32.37**.
6. MUST gain `service_attach_rate_kpi(df: pd.DataFrame) -> dict` with keys
   `avg_services` (`compute_service_count(df).mean()`, rounded 2dp) and
   `pct_of_max` (`avg_services / SERVICE_ATTACH_MAX * 100`, rounded 2dp).
   MUST NOT read `Churn`. Verified: `avg_services` ≈ **4.15**, `pct_of_max`
   ≈ **46.07**.
7. MUST gain `kpi_summary(df: pd.DataFrame) -> pd.DataFrame` with exactly 5
   rows in this fixed order — `Churn Rate`, `MRR Loss`, `CLV`, `Avg Tenure`,
   `Service-Attach Rate` — columns `kpi`, `value`, `unit`, where `value` is
   each metric's single headline number (`churn_rate_kpi`;
   `mrr_loss_kpi(df)["lost_mrr"]`; `clv_kpi`; `avg_tenure_kpi`;
   `service_attach_rate_kpi(df)["avg_services"]`) and `unit` is a short
   display string (`"%"`, `"$/mo"`, `"$"`, `"months"`, `"of 9 services"`).
   No `NaN` values in `value`.
8. MUST gain `plot_mrr_breakdown(df: pd.DataFrame) -> go.Figure` — a Plotly
   bar or donut chart of `retained_mrr` vs. `lost_mrr` from
   `mrr_loss_kpi(df)`. Returns the figure object directly; does **not**
   call `save_fig` or write to `reports/figures/` (deliberate deviation
   from `eda.py`'s PNG-saving convention — see Non-goals).
9. MUST gain `plot_service_attach_distribution(df: pd.DataFrame) -> go.Figure`
   — a Plotly histogram of `compute_service_count(df)` over `0..SERVICE_ATTACH_MAX`.
10. `app/dashboard.py` MUST: call `st.set_page_config(page_title="RetainIQ —
    KPI Dashboard", layout="wide")`; load data through a `@st.cache_data`-
    wrapped thin wrapper around `load_clean_data()`; render the 5
    `kpi_summary()` rows as `st.metric` tiles (label = `kpi`, value = `f"{value}{unit}"`
    formatted per row); render both Plotly figures via `st.plotly_chart`;
    provide a "Refresh data" `st.button` that clears the cache
    (`st.cache_data.clear()`) and reloads via `load_clean_data(rebuild=True)`
    before rerunning. MUST wrap the data-load call in `try/except
    FileNotFoundError` and render `st.error(...)` with guidance (place
    `telco.csv` in `data/raw/`) instead of crashing with a raw traceback.
    MUST NOT contain KPI math inline — every number comes from
    `src/data/kpi.py` (CLAUDE.md §4).
11. `tests/test_kpi.py` MUST cover: each `*_kpi` function's verified value
    (§ Problem/motivation) via `pytest.approx`; `clv_kpi` raises `ValueError`
    on an all-`Churn=0` fixture; `service_attach_rate_kpi` unaffected by
    dropping `Churn` from the input frame (leakage/independence guard);
    `kpi_summary` row order, column names, and no-`NaN` guarantee; both
    plot functions return a `go.Figure` with at least one trace.
12. `tests/test_dashboard.py` MUST use `streamlit.testing.v1.AppTest.from_file("app/dashboard.py").run()`
    and assert: no exception (`at.exception` empty), exactly 5 `st.metric`
    elements present, and their labels match `kpi_summary()`'s `kpi` column
    in order.
13. None of the above may change `src/data/eda.py`, `src/data/cohorts.py`,
    `src/models/segmentation.py`, `src/data/lifecycle.py`, `clean_data()`'s
    output, or any existing test/figure — all current tests must keep
    passing unmodified.

### Data & model impact

Purely descriptive; no model is fit or loaded. All 5 KPIs are simple
pandas aggregations over `load_clean_data()`'s existing columns — no new
column is written back into the cleaned dataset, and nothing here is
persisted to `models/`. `src/features/` (Phase 2, not yet built) is
unaffected.

### ML guardrails (mandatory check)

N/A — no model path affected. One disclosure carried forward from
`05-retention-funnel.md`'s precedent: if a future phase feeds any of
`churn_rate_kpi`, `clv_kpi`, or `service_attach_rate_kpi`'s outputs back in
as a Phase 2 model *feature* (none do today — these are report-only), that
would need the same design-time-vs-runtime leakage discipline
`lifecycle.py` already documents, since `churn_rate_kpi` and `clv_kpi` are
themselves derived from `Churn`.

### API / UI surface

New `app/dashboard.py` Streamlit page (no FastAPI endpoint — out of scope,
see Non-goals). Layout: page title → 5-column `st.metric` row (Churn Rate,
MRR Loss, CLV, Avg Tenure, Service-Attach Rate) → 2-column row with the MRR
breakdown chart and the service-attach histogram → a "Refresh data" button.
Run via the existing documented command `streamlit run app/dashboard.py`
(already listed in CLAUDE.md §5 — no doc change needed).

### Edge cases & failure states

- **`data/raw/telco.csv` missing** (cold start / recruiter clone without
  the dataset): `load_clean_data()` raises `FileNotFoundError`; the
  dashboard catches it and shows a friendly `st.error` instead of a raw
  traceback (Requirement 10).
- **Zero churn rate** (unreachable on real data, reachable on a filtered or
  synthetic frame): `clv_kpi` raises `ValueError` rather than silently
  returning `inf` (Requirement 4), directly tested.
- **`TechSupport`/other sentinel values** ("No internet service"):
  `service_attach_rate_kpi` inherits `compute_service_count`'s existing
  sentinel-safe handling — never miscounted as an attached service, same
  precedent as `segmentation.py`.
- **`st.cache_data` staleness after "Refresh data"**: explicitly cleared
  before the rebuild call (Requirement 10), so a click always reflects the
  freshly rebuilt frame, not a stale cached one.
- **Empty input DataFrame** (defensive, not reachable via the dashboard's
  own load path): `churn_rate_kpi`/`avg_tenure_kpi`/`service_attach_rate_kpi`
  return `NaN` (pandas `mean()` of an empty series) rather than raising;
  `clv_kpi` raises `ValueError` via the zero/`NaN`-churn-rate guard --
  directly tested (`test_clv_kpi_raises_on_nan_churn_rate`), unlike
  `cohorts.py`/`segmentation.py`'s untested empty-frame precedent, since
  this guard's correctness (not just its existence) mattered enough to
  verify directly once the NaN case was found during review.

### Security notes

No new untrusted input — the dashboard reads only the local cleaned
dataset, same trust boundary as every existing module; no CSV upload, no
user-supplied query, no network call. `streamlit` and `plotly` are both
already in `requirements.txt`, now pinned (`streamlit==1.51.0`,
`plotly==6.3.0`, matching the versions actually installed and exercised —
per CLAUDE.md §12, unpinned was a gap this feature's first real usage of
both packages should close, not just inherit).

**Correction (found during `security-reviewer`):** `streamlit run` does
**not** bind to `localhost` by default — it binds all interfaces, so
anyone on the same network could reach the dashboard while it runs, with
no authentication. Fixed via `.streamlit/config.toml` (`server.address =
"127.0.0.1"`), not by relying on Streamlit's default. The same file also
sets `client.showErrorDetails = "none"` so an unanticipated exception
never renders a full traceback (absolute paths, library versions) in the
browser, as defense in depth alongside `app/dashboard.py`'s broadened
`except (FileNotFoundError, OSError, ValueError, KeyError)` clause. No
authentication is added or claimed — still appropriate only for local,
single-user use.

### Success criteria

- `pytest -q` passes: all existing tests + `tests/test_kpi.py` +
  `tests/test_dashboard.py`, all green.
- `streamlit run app/dashboard.py` opens without error and renders all 5
  KPI tiles with the verified values above, plus both charts.
- The "Refresh data" button works without raising.
- `quality-reviewer` and `security-reviewer` report no unresolved findings
  on the diff.

### Out of scope

- FastAPI `/predict`/`/explain`/`/recommend` endpoints, the What-If panel,
  CSV upload, per-customer search — remaining Phase 5 scope, gated on
  Phase 2's model.
- A new row in CLAUDE.md §14's phase tracker.
- Any change to `01_eda.ipynb`–`06_churn_driver_id.ipynb` or their modules.

---

## PART 2 — PLAN

### Approach

Add `src/data/kpi.py` as a self-contained sibling to `cohorts.py`/
`lifecycle.py` (same `compute_*`/`plot_*` pattern, reusing
`compute_service_count` rather than duplicating it), then a thin
`app/dashboard.py` that only calls into it and Streamlit's own rendering
primitives — no KPI math inline in the UI layer, per CLAUDE.md §4. Plotly
(already a listed dependency, previously unused) is used for the two
dashboard charts instead of matplotlib/seaborn, since these are interactive
widgets embedded live in Streamlit rather than static PNGs destined for
`reports/figures/` — a deliberate, disclosed split from `eda.py`'s
convention, not an inconsistency.

**Alternative rejected:** computing the KPIs inline inside
`app/dashboard.py` with no separate `src/data/kpi.py` module, on the
grounds that five simple aggregations don't need their own module.
Rejected because CLAUDE.md §4 is explicit that the API/dashboard layer
"calls into `src/`, doesn't reimplement logic," and because a testable,
importable `kpi_summary()` is what lets a future Phase 5 multi-page
dashboard (or a `/kpi`-style endpoint, if ever added) reuse this exact
logic instead of re-deriving it.

### Task breakdown

- [ ] **1. Create `src/data/kpi.py`** — `SERVICE_ATTACH_MAX` constant,
      `churn_rate_kpi`, `mrr_loss_kpi`, `clv_kpi` (with the zero-churn
      guard), `avg_tenure_kpi`, `service_attach_rate_kpi`, `kpi_summary`,
      `plot_mrr_breakdown`, `plot_service_attach_distribution`. Import
      `compute_service_count`, `SERVICE_YESNO_COLUMNS` from
      `src.models.segmentation`; `TARGET_COLUMN` from `src.data.load_data`.
- [ ] **2. Create `app/dashboard.py`** — page config, cached data loader
      with the `FileNotFoundError` guard, 5-tile `st.metric` row from
      `kpi_summary()`, both Plotly charts via `st.plotly_chart`, "Refresh
      data" button clearing `st.cache_data` and calling
      `load_clean_data(rebuild=True)`.
- [ ] **3. Run `streamlit run app/dashboard.py`** manually, confirm all 5
      tiles render the verified values and both charts load without error;
      click "Refresh data" once to confirm it doesn't raise.
- [ ] **4. Add `tests/test_kpi.py`** — cover Functional Requirement 11
      (verified values, zero-churn `ValueError`, leakage/independence
      guard, `kpi_summary` shape/order, both plots return non-empty
      `go.Figure`).
- [ ] **5. Add `tests/test_dashboard.py`** — `AppTest` smoke test per
      Functional Requirement 12 (no exception, exactly 5 metrics, correct
      labels/order).
- [ ] **6. Run the full suite** — `pytest -q`, confirm all existing + new
      tests pass.
- [ ] **7. Commit** — `src/data/kpi.py`, `app/dashboard.py`,
      `tests/test_kpi.py`, `tests/test_dashboard.py`, commit message
      `feat: KPI dashboard (churn rate, MRR loss, CLV, avg tenure,
      service-attach rate)`. (No CLAUDE.md §5 change needed — `streamlit
      run app/dashboard.py` is already documented there.)

### Tests to write (hand to test-writer)

- `tests/test_kpi.py::test_churn_rate_kpi_matches_verified_value` — ≈26.54
  on `clean_df`.
- `tests/test_kpi.py::test_mrr_loss_kpi_matches_verified_values` —
  `total_mrr` ≈456116.60, `lost_mrr` ≈139130.85, `lost_mrr_pct` ≈30.50;
  `retained_mrr == total_mrr - lost_mrr` exactly.
- `tests/test_kpi.py::test_clv_kpi_matches_verified_value` — ≈244.04 on
  `clean_df`.
- `tests/test_kpi.py::test_clv_kpi_raises_on_zero_churn_rate` — an
  all-`Churn=0` fixture raises `ValueError`, not `inf`/`NaN`.
- `tests/test_kpi.py::test_avg_tenure_kpi_matches_verified_value` — ≈32.37.
- `tests/test_kpi.py::test_service_attach_rate_kpi_matches_verified_values` —
  `avg_services` ≈4.15, `pct_of_max` ≈46.07.
- `tests/test_kpi.py::test_service_attach_rate_kpi_independent_of_churn_column` —
  identical result whether or not `Churn` is present in the input frame.
- `tests/test_kpi.py::test_kpi_summary_order_columns_and_no_nulls` —
  `kpi_summary(clean_df)["kpi"].tolist() == ["Churn Rate", "MRR Loss",
  "CLV", "Avg Tenure", "Service-Attach Rate"]`; no `NaN` in `value`.
- `tests/test_kpi.py::test_plot_functions_return_nonempty_figures` —
  `plot_mrr_breakdown` and `plot_service_attach_distribution` each return a
  `go.Figure` with `len(fig.data) >= 1`.
- `tests/test_dashboard.py::test_dashboard_renders_five_metrics_without_exception` —
  `AppTest.from_file("app/dashboard.py").run()`; `at.exception == []`;
  `len(at.metric) == 5`; metric labels match `kpi_summary()`'s order.

### Quality gates

1. **test-runner** — run `pytest -q`, confirm all tests (existing + new)
   pass, fix any straightforward regression.
2. **quality-reviewer** — review the zero-churn `ValueError` guard, the
   `compute_service_count` reuse (no duplicated logic), the dashboard's
   thin-UI adherence (no inline KPI math in `app/dashboard.py`), and
   CLAUDE.md §8 style (named constants, type hints, docstrings).
3. **security-reviewer** — confirm no new untrusted input path; confirm
   `streamlit`/`plotly` are pre-existing pinned dependencies with no
   version change.
4. Commit only after both reviews are clean.

### Risks / rollback

- **Risk:** a future Kaggle re-download shifts the verified KPI values
  enough that pinned test values no longer hold. **Mitigation:**
  intentional, matching `02`–`06`'s brittleness philosophy — these tests
  should fail loudly on real distributional shift.
- **Risk:** a reviewer expects "real-time" to mean a live/streaming data
  source. **Mitigation:** the "Real-time note" at the top of this spec and
  Non-goals state the interpretation explicitly, with the manual refresh
  control as the honest substitute.
- **Risk:** this page is later mistaken for the full Phase 5 dashboard.
  **Mitigation:** the scope note at the top and Non-goals state plainly
  that What-If/prediction/API views remain separate, gated on Phase 2.
- **Rollback:** single commit (Task 7) covering only additive files (new
  module, new app page, new tests) — `git revert` is clean since nothing
  existing is modified in place.

### Definition of done

- All 7 tasks checked off.
- `pytest -q` green (all existing tests + 9 new `test_kpi.py` tests + 1
  `test_dashboard.py` test).
- `streamlit run app/dashboard.py` runs cleanly with all 5 tiles and both
  charts rendering the verified values.
- `quality-reviewer` and `security-reviewer` report no unresolved findings.
- All Success Criteria in Part 1 are met.
