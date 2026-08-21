# CLAUDE.md

Guidance for Claude Code when working in this repository. Read this fully before making changes.

---

## 1. Project overview

**RetainIQ** is an end-to-end Customer Churn Analytics & Prediction System. It predicts which
customers are about to leave, explains *why*, and recommends what to do about it — exposed through
a REST API and an interactive dashboard, with MLOps for monitoring and retraining.

- **Type:** ML + Analytics portfolio project (production-style).
- **Domain:** SaaS / Telecom / FinTech subscription churn.
- **Dataset:** IBM Telco Customer Churn — 7,043 customers, 21 columns, target `Churn` (Yes/No),
  ~26.5% positive class. https://www.kaggle.com/datasets/blastchar/telco-customer-churn
- **Primary goal:** Churn classifier with **AUC-ROC ≥ 0.85**, per-customer explanations (SHAP),
  a Next-Best-Action engine, a live dashboard + API, and drift-monitored retraining.

When a task is ambiguous, prefer the choice that matches this goal and the conventions below.
Ask before introducing a new framework, a new data source, or a breaking change to the API.

---

## 2. Golden rules (read first)

1. **Never let churn probability, or anything derived from it, become a model input.** This is the
   #1 leakage risk. Features come only from customer attributes/behaviour, never from the label.
2. **Honest AUC on this dataset is ~0.85–0.88.** If any model scores **> 0.95 AUC**, STOP and
   investigate leakage before celebrating. Report the finding, don't hide it.
3. **Churn is imbalanced (26.5%).** Always report **PR-AUC and recall**, never accuracy alone.
   Use stratified splits everywhere.
4. **One phase at a time.** Do the requested phase, show results, and stop. Don't scaffold future
   phases unprompted. Keep diffs reviewable.
5. **Reproducibility:** set `random_state=42` on every split, model, and resampler.
6. **Secrets never touch git.** API keys load from environment variables only. Never hardcode them,
   never commit `.env`.
7. **Data and model artifacts are git-ignored.** Never commit files under `data/` or `models/`.

---

## 3. Tech stack

| Layer | Tools |
|---|---|
| Language | Python 3.11 |
| Data | pandas, numpy |
| Modeling | scikit-learn, XGBoost, LightGBM, imbalanced-learn (SMOTE) |
| Explainability | SHAP (TreeExplainer), LIME |
| Experiment tracking | MLflow |
| API | FastAPI + Uvicorn |
| Dashboard | Streamlit |
| Viz | Plotly, Seaborn |
| MLOps | Evidently (drift), Prefect (scheduling), Docker |
| LLM insights (optional) | Anthropic Claude / OpenAI API — configurable, skips if no key |
| Testing | pytest |

Don't add dependencies outside this list without asking. If a new package is genuinely needed,
explain why and add it to `requirements.txt` in the same change.

---

## 4. Repository structure

```
retainiq/
|__.claude              # claude.md and also Specs folder 
├── data/
│   ├── raw/            # telco.csv lives here (git-ignored)
│   └── processed/      # cleaned data (git-ignored)
├── notebooks/          # EDA and exploration
├── src/
│   ├── data/           # loading, cleaning, validation
│   ├── features/       # preprocessing pipeline (ColumnTransformer)
│   ├── models/         # training, comparison, evaluation
│   ├── explain/        # SHAP + LIME explainers
│   ├── recommend/      # risk tiers + Next-Best-Action engine
│   └── api/            # FastAPI service
├── app/                # Streamlit dashboard
├── models/             # saved model artifacts (git-ignored)
├── mlops/              # drift detection, retraining flows
├── tests/              # pytest suite
├── requirements.txt
├── README.md

```

**Where things go:** data logic in `src/data/`, feature transforms in `src/features/`, anything
that trains or scores in `src/models/`, explanation logic in `src/explain/`, business/retention
logic in `src/recommend/`. Keep the API (`src/api/`) and dashboard (`app/`) thin — they call into
`src/`, they don't reimplement logic.

---

## 5. Commands

```bash
# Setup
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Tests (run before every commit)
pytest -q

# Regenerate EDA figures (distributions, outliers, segment churn rates)
python -m src.data.eda

# Regenerate tenure-cohort figures (retention curve, churn trend by band)
python -m src.data.cohorts

# Regenerate customer value-segmentation figures (segment profile, churn rate by segment)
python -m src.models.segmentation

# Regenerate retention-funnel (lifecycle stage) figures
python -m src.data.lifecycle

# Regenerate churn-driver-ID figures (correlation, chi-square, SHAP global importance)
python -m src.explain.driver_analysis

# Train + compare models (logs to MLflow, saves best to models/)
python -m src.models.train

# MLflow UI
mlflow ui                      # http://localhost:5000

# Run the API
uvicorn src.api.main:app --reload --port 8000     # docs at /docs

# Run the dashboard
streamlit run app/dashboard.py

# Drift report
python -m mlops.drift_check

# Docker (Phase 6)
docker compose up --build
```

If you create a script that should be runnable, expose it as `python -m src.<module>` and note it
here so this list stays the source of truth.

---

## 6. Dataset reference

**Target:** `Churn` — map `Yes → 1`, `No → 0`.

**Columns (21):**
- Identifier: `customerID` — **drop before modeling**.
- Demographics: `gender`, `SeniorCitizen`, `Partner`, `Dependents`.
- Account: `tenure`, `Contract`, `PaperlessBilling`, `PaymentMethod`.
- Services: `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`,
  `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`.
- Financial: `MonthlyCharges`, `TotalCharges`.

**Known data quirk:** `TotalCharges` has 11 blank strings (new customers with `tenure = 0`).
Convert to numeric and impute sensibly (0 or median). Handle this in `src/data/`, not ad hoc.

**Sanity-check the model against these real signals** (if these don't show up in SHAP, something
is wrong):

| Signal | Pattern | Retention lever |
|---|---|---|
| Contract | Month-to-month churns ~47% vs ~3% for two-year | Incentivize longer contracts |
| Tenure | Churn concentrated in first 12 months | Strengthen onboarding |
| Tech support | No tech support → higher churn | Offer support add-on |
| Payment method | Electronic-check payers churn ~45% | Nudge to auto-pay |
| Internet | Fiber-optic users churn more | Proactive quality outreach |

---

## 7. ML conventions & guardrails

- **Splitting:** stratified train/test (e.g. 80/20) + stratified cross-validation. `random_state=42`.
- **Imbalance:** class weights or SMOTE — apply SMOTE **inside** the CV fold / pipeline, never
  before the split (that leaks).
- **Preprocessing:** one `ColumnTransformer` (one-hot categoricals, scale numerics) wrapped in a
  scikit-learn `Pipeline` so train and serve use identical transforms. Fit on train only.
- **Models to compare:** Logistic Regression (baseline) → Random Forest → XGBoost → LightGBM.
- **Metrics table:** AUC-ROC, PR-AUC, precision, recall, F1, Brier score. Tune the decision
  threshold, don't assume 0.5.
- **Tracking:** log params, metrics, and the model to MLflow for every run.
- **Risk tiers:** Critical > 70%, High 50–70%, Medium 30–50%, Low < 30%.
- **Explainability:** SHAP TreeExplainer for global importance + local top-3 drivers per customer;
  LIME as the alternative local view. Explanations must be human-readable.

---

## 8. Code style

- Follow PEP 8. Use type hints on function signatures. Add concise docstrings on public functions.
- Prefer small, pure, testable functions over long scripts. No business logic inside notebooks —
  notebooks import from `src/`.
- No hardcoded absolute paths; use paths relative to the project root (or a small `config`).
- No magic numbers — name thresholds (risk tiers, drift limits) as constants.
- Keep functions deterministic where possible; isolate randomness behind `random_state`.
- Don't leave commented-out code or `print` debugging in committed files; use logging.

---

## 9. Testing

- Every phase that adds logic adds at least one `pytest` test.
- Minimum coverage expectations:
  - `src/data/`: cleaned data has no missing values, `Churn` is binary, `customerID` dropped.
  - `src/features/`: transformer output shape/columns are stable; no NaNs post-transform.
  - `src/models/`: a smoke test that training runs and returns a model scoring above a floor.
  - `src/api/`: each endpoint returns the expected schema on a sample payload.
- `pytest -q` must pass before any commit. If a test can't pass yet, mark it `xfail` with a reason,
  don't delete it.

---

## 10. API contract

FastAPI service in `src/api/`. Load the model once at startup, not per request.

| Method | Path | Purpose |
|---|---|---|
| POST | `/predict` | Single customer → churn probability + risk tier |
| POST | `/batch-predict` | CSV / list of customers → scored list |
| POST | `/explain` | Customer → SHAP top-3 churn drivers |
| POST | `/recommend` | Customer → ranked Next-Best-Actions |

Use Pydantic models for request/response. Validate input and return clear 4xx errors on bad data.
Keep the response schema stable — if it must change, update tests and README in the same change.

---

## 11. Git workflow

- Small, focused commits. One phase → one (or few) commits.
- Commit message style: `phase N: short description` (e.g. `phase 2: model training and comparison`).
- Run `pytest -q` before committing.
- Never commit: `data/`, `models/`, `.venv/`, `.env`, `__pycache__/`, MLflow run artifacts.
- If a change is risky, describe the rollback (`git reset`) in your summary.

---

## 12. Environment & secrets

- Python dependencies pinned in `requirements.txt`.
- Optional LLM insight generator reads its key from an env var (e.g. `ANTHROPIC_API_KEY` or
  `OPENAI_API_KEY`). If unset, the feature **degrades gracefully** — the rest of the app still works.
- Provide a `.env.example` listing variable names (no values). Never commit real `.env`.

---

## 13. Common pitfalls — do NOT

- ❌ Fit any transformer or SMOTE on the full dataset before splitting.
- ❌ Feed `Churn`, its probability, or any post-outcome field back in as a feature.
- ❌ Report accuracy as the headline metric on this imbalanced data.
- ❌ Commit the dataset, trained models, or secrets.
- ❌ Put model/business logic inside the Streamlit or FastAPI layer — call into `src/`.
- ❌ Silently accept a > 0.95 AUC. Investigate and report it.
- ❌ Add heavy new dependencies without asking.

---

## 14. Build phases (status tracker)

Work top to bottom. Update the status as phases complete.

| Phase | Scope | Status |
|---|---|---|
| 0 | Project scaffold, requirements, git init | ☑ |
| 1 | Data loading, cleaning, EDA, first tests | ☑ |
| 2 | Feature pipeline, model training + comparison, MLflow | ☐ |
| 3 | SHAP + LIME explainability, plain-English reasons | ☐ |
| 4 | Risk tiers + Next-Best-Action engine, optional LLM | ☐ |
| 5 | FastAPI service + Streamlit dashboard + What-If panel | ☐ |
| 6 | Evidently drift, retraining flow, Docker, deploy notes | ☐ |

Phase 1 was extended (not a new phase row, per `.claude/specs/02-churn-patterns.md`'s
own scope decision) with a churn-patterns deep-dive: distribution analysis, IQR-based
outlier detection, and a documented investigation of the 11 missing `TotalCharges`
rows, in `notebooks/02_churn_patterns.ipynb` and `src/data/eda.py`.

Phase 1 was further extended (again not a new phase row, per
`.claude/specs/03-cohort-analysis.md`) with tenure-cohort analysis: binning
customers into 0-12/12-24/24-48/48-72-month bands, per-cohort churn/retention
rates, and an empirical (non-censoring-corrected) retention curve, in
`notebooks/03_cohort_analysis.ipynb` and `src/data/cohorts.py`.

**Definition of done:** a recruiter can open the dashboard, upload a customer CSV, see calibrated
churn probabilities with SHAP explanations and next-best-action recommendations, and call the REST
API — backed by a model at AUC ≥ 0.85, with drift monitoring and automated retraining in place.