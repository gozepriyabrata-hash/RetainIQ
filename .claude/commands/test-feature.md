---
description: End-to-end test a single RetainIQ feature — write tests (test-writer), run + fix them (test-runner), then report coverage. Feature-scoped, not the whole suite.
argument-hint: <spec slug | module path | short feature description>
model: sonnet
---

# /test-feature — RetainIQ feature test loop

You are the test coordinator for **RetainIQ** (Python 3.11; pandas, scikit-learn,
XGBoost, LightGBM, imbalanced-learn/SMOTE, SHAP/LIME, MLflow, FastAPI, Streamlit,
Evidently, Prefect, Docker; pytest). Your job for this command is to make sure ONE
feature is properly tested and green — by delegating to the project's subagents,
not by doing everything yourself.

Follow @CLAUDE.md throughout (repo layout, ML guardrails, testing rules).

## Feature to test
$ARGUMENTS

If empty, ask which feature (a `specs/<slug>` spec, a path like
`src/features/encode.py`, or a one-line description) and stop.

## Step 1 — Resolve the scope
Turn the argument into concrete targets before doing anything else:
- If it's a spec slug, read `specs/<slug>/spec.md` (and `plan.md` if present) —
  the spec's functional requirements, edge cases, and success criteria ARE the
  test checklist.
- If it's a path, read that module and its direct collaborators.
- If it's a description, use Grep/Glob to locate the relevant module(s) in
  src/data, src/features, src/models, src/explain, src/recommend, src/api, app/,
  or mlops/.
- Run `git diff HEAD --stat` to see what recently changed for this feature.

State clearly: the source file(s) under test, the matching `tests/` path(s), and
whether tests already exist for them.

## Step 2 — Write / extend tests  → delegate to `test-writer`
Invoke the **test-writer** subagent to author the missing tests for this feature.
Hand it a concrete checklist derived from the scope, covering the categories that
apply:
- **Data** (src/data): schema, no-missing, `Churn` binary, deterministic output.
- **Features** (src/features): transformer output shape/columns stable, no NaNs,
  fit on TRAIN ONLY.
- **Models** (src/models): smoke test — training returns a fitted estimator and
  clears a LOW AUC floor on a tiny seeded fixture; never assert exact metrics.
- **Leakage guard** (mandatory if the feature touches data/features/models):
  assert `Churn`/probability/derivatives and `customerID` are NOT in the feature
  matrix; assert SMOTE/scaling run inside CV on train folds only.
- **API** (src/api): FastAPI `TestClient` — status codes, response schema, 422 on
  bad input, model mocked via `app.dependency_overrides`.
- **Edge cases** from the spec: bad/empty input (e.g. the 11 blank TotalCharges,
  empty CSV), model unavailable, empty results.
- Mock external calls (LLM insight generator, MLflow). Seed `random_state=42`.
  Prefer tiny fixtures over the full 7,043-row dataset.

Do not write these tests yourself — that's the test-writer's job.

## Step 3 — Run & fix  → delegate to `test-runner`
Invoke the **test-runner** subagent to run the feature's tests, classify any
failures (regression / flaky / environment / outdated), and fix straightforward
regressions. Scope its run with `-k` to this feature's tests first, then confirm
nothing else broke with a full `pytest -q`.

Hard rule (enforce it): never weaken, skip, or delete a test to make it pass, and
never widen a metric tolerance to go green. If a test is genuinely wrong, the
runner reports it — you surface it, don't bury it. A metric threshold failing is
investigated, not silenced. AUC > 0.95 is a leakage flag even if green.

## Step 4 — Coverage report
Run coverage scoped to the feature's modules:
`pytest --cov=<feature module path(s)> --cov-report=term-missing -q`
Report the percentage and the notable uncovered lines/branches. Target ~80% on the
feature's code; if it's well below, list the specific gaps and loop back to Step 2
for those cases (don't chase 100% — meaningful coverage over vanity numbers).

## Step 5 — Final report
End with a tight summary:

```
## /test-feature — <feature>
Scope: <source file(s)> ↔ <test file(s)>
Tests: A written/updated by test-writer
Run: X passed, Y failed→fixed, Z flaky→isolated (pytest -q green? yes/no)
Coverage: N% on <modules>  (gaps: <...> or "none material")
Leakage/metric flags: <none | details>
Still open: <anything needing your decision — outdated test, new gap, design Q>
```

Stop after the report. Do not commit — that's the user's call after review, and
the quality-reviewer / security-reviewer pass. If the suite can't be made green
without weakening a test, STOP and explain rather than forcing it.
