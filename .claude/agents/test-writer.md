# Building a `test-writer` Subagent for RetainIQ: Claude Code Format + ML/pytest Testing Best Practices

## TL;DR
- **The subagent file is a Markdown file with YAML frontmatter** stored at `.claude/agents/test-writer.md` (project scope, version-controlled); only `name` and `description` are required, `tools` and `model` are the two levers you should set explicitly, and everything below the second `---` becomes the agent's system prompt verbatim. For a test-writer, grant `tools: Read, Grep, Glob, Edit, Write, Bash` and set `model: sonnet`.
- **Make the description a trigger, not a label:** put phrases like "Use PROACTIVELY" / "MUST BE USED" plus concrete conditions ("after writing or modifying any code in src/ or app/") so Claude auto-delegates. The body should encode a fixed workflow and the ML-specific invariants (no target/probability leakage into features; fit transformers/SMOTE on training folds only; report AUC-ROC/PR-AUC/recall not accuracy; treat AUC > 0.95 as a leakage red flag).
- **For ML testing quality, the agent must prefer tiny synthetic/fixture data over the full IBM Telco dataset (7,043 records, 26.5% churn), write smoke tests that assert the pipeline runs and clears a low performance floor rather than exact metrics, use `random_state=42` everywhere, mock external calls (LLM insight generator, MLflow), test FastAPI endpoints with `TestClient` (status codes + schemas + 422 validation), and always run `pytest -q` to verify green before finishing.**

## Key Findings

### A. Claude Code subagent format (from official docs)
1. **Location & scope.** Custom subagents live as Markdown files in `.claude/agents/` (project) or `~/.claude/agents/` (user). Project agents take precedence over user agents on name collision, and Anthropic recommends checking project agents into version control. For RetainIQ, use project scope so the agent ships with the repo.
2. **Required vs optional fields.** Only `name` and `description` are required. Optional fields include `tools`, `disallowedTools`, `model`, `permissionMode`, `skills`, `maxTurns`, `mcpServers`, `hooks`, `memory`, `effort`, `isolation`, `color`, and more.
3. **Tools inheritance.** If `tools` is omitted the subagent inherits every tool available to subagents (including MCP tools). Specifying `tools` makes it an allowlist. `disallowedTools` is a denylist applied before `tools`.
4. **Model field.** `model` accepts `sonnet`, `opus`, `haiku`, a full model ID, or `inherit`; it defaults to `inherit` (uses the main conversation's model).
5. **Body = system prompt.** Everything below the frontmatter becomes the subagent's system prompt verbatim; subagents get only this prompt plus environment details (working directory), not the full Claude Code system prompt. Non-Explore/Plan subagents DO load the CLAUDE.md hierarchy, so RetainIQ's existing CLAUDE.md guardrails reach the test-writer automatically.
6. **Automatic delegation is driven by `description`.** Claude matches the task against the description; adding "use proactively" / "MUST BE USED" encourages auto-delegation. You can also invoke explicitly by name or with `@agent-test-writer`.
7. **`/agents` command.** As of recent versions the `/agents` command no longer opens the interactive wizard; you ask Claude to write the file or create it by hand. Claude Code watches the agents directories and picks up changes within seconds (a restart is needed only when creating a scope's first agent file in a brand-new directory).
8. **Anthropic's four design best practices:** (1) design focused single-responsibility subagents; (2) write detailed descriptions; (3) limit tool access; (4) check project subagents into version control.

### B. What good ML/pytest tests look like for this stack
1. **Structure & discovery.** Tests live in `tests/`, files named `test_*.py`, functions `test_*`, classes `Test*`; shared fixtures go in `conftest.py` (auto-discovered, no import needed). Use `tmp_path`/`tmp_path_factory` for file I/O, `@pytest.mark.parametrize` for input sets, custom markers (register with `--strict-markers`), and `monkeypatch` for env/attr patching.
2. **Data-layer tests** validate schema/dtypes/no-missing/binary-target/deterministic output. `pandera` is the recommended lightweight validator (`DataFrameSchema`/`DataFrameModel`, `nullable`, `coerce`, `Check.isin`, `Check.ge`, `lazy=True`); by default pandera treats all columns as non-nullable, which is exactly the strictness you want for churn-clean data.
3. **Transformer/pipeline tests** assert stable output shape/columns (`get_feature_names_out()`), no NaNs after transform, and — critically — no leakage: transformers must be fit on training folds only.
4. **Leakage is the headline risk for churn.** Use `imblearn.pipeline.Pipeline` so SMOTE and scaling run *inside* CV on training folds only; never `fit_resample` the whole dataset before splitting. Tests should assert the target/probability and their derivatives are absent from the feature matrix, and treat AUC > 0.95 as a red flag.
5. **Model tests are "smoke tests," not accuracy assertions.** Train on a tiny fixed-seed sample, assert training returns a fitted model and clears a low performance floor; use `np.isclose`/bounds rather than exact metric equality because of stochasticity. The probabl-ai ML skill formalizes this with a hard structural assertion (exact prediction row-count) plus a soft "metric within N× CV mean" bound.
6. **FastAPI endpoint tests** use `fastapi.testclient.TestClient`: assert status codes, response JSON schema, and 422 on invalid input; use `app.dependency_overrides` to swap the real model for a stub and mock external services.
7. **Reproducibility & speed.** Seed everything (`random_state=42`, `np.random`), keep tests offline (auto-block network in `conftest.py`), mock the LLM insight generator and MLflow logging so tests are fast and deterministic. Property-based testing (`hypothesis`, and `pandera`'s data-synthesis strategies) is useful for data-cleaning invariants.
8. **Coverage expectations.** ~80% is the widely accepted target; ML practitioners mostly cite 70–90% as acceptable, and coverage in Python has measurable runtime overhead. Prioritize meaningful invariant tests over chasing 100%.

## Details

### 1. The exact subagent file format
Anthropic's official Claude Code docs ("Create custom subagents") state: "Subagents are Markdown files with YAML frontmatter." Files are stored by scope, resolved highest-priority-first:

| Location | Scope | Priority |
|---|---|---|
| Managed settings `.claude/agents/` | Organization-wide | 1 (highest) |
| `--agents` CLI flag (JSON) | Current session | 2 |
| `.claude/agents/` | Current project | 3 |
| `~/.claude/agents/` | All your projects | 4 |
| Plugin `agents/` directory | Where plugin enabled | 5 (lowest) |

The docs describe project subagents (`.claude/agents/`) as "ideal for subagents specific to a codebase. Check them into version control so your team can use and improve them collaboratively." For RetainIQ (a solo portfolio project), project scope is right: it makes the agent part of the repo you're showcasing.

**Supported frontmatter fields** (from the docs' "Supported frontmatter fields" table; only `name` and `description` are required):
- `name` — "Unique identifier using lowercase letters and hyphens." The filename doesn't have to match; names can't contain `:`.
- `description` — "When Claude should delegate to this subagent." This is the trigger for automatic delegation.
- `tools` — "Tools the subagent can use. Inherits every tool available to subagents if omitted." An allowlist.
- `disallowedTools` — "Tools to deny, removed from inherited or specified list." Applied before `tools`.
- `model` — "`sonnet`, `opus`, `haiku`, `fable`, a full model ID …, or `inherit`. Defaults to `inherit`."
- `permissionMode` — `default`, `acceptEdits`, `auto`, `dontAsk`, `bypassPermissions`, `plan`.
- `skills` — Skills to preload into the subagent's context at startup (full content injected).
- `maxTurns`, `mcpServers`, `hooks`, `memory` (`user`/`project`/`local`), `effort` (`low`/`medium`/`high`/`max`), `isolation` (`worktree`), `color`, `initialPrompt` — all optional.

The docs note the body role explicitly: "The body becomes the system prompt that guides the subagent's behavior. Subagents receive only this system prompt plus basic environment details like the working directory, not the full Claude Code system prompt." And on context loading: a non-fork subagent's initial context includes "CLAUDE.md files: every level of the CLAUDE.md hierarchy the main conversation loads" — so RetainIQ's CLAUDE.md engineering source-of-truth reaches the test-writer without extra wiring. (Only the built-in Explore and Plan agents skip CLAUDE.md.)

**Available tools & least privilege.** The docs give a canonical read-only example: `tools: Read, Grep, Glob, Bash`. A test-writer needs to *write* test files and *run* pytest, so it also needs `Edit` and `Write`. Recommended set: `Read, Grep, Glob, Edit, Write, Bash`. The docs' own examples confirm the pattern — the read-only "code-reviewer" uses `Read, Grep, Glob, Bash`, while the "debugger" that must modify code adds `Edit` ("Unlike the code reviewer, this one includes Edit because fixing bugs requires modifying code").

**Model choice.** The docs list `model: inherit` in the code-reviewer example and `model: sonnet` in the data-scientist example. Two real-world test subagents both pin `model: sonnet`: the FlorianBruniaux `test-writer.md` uses `model: sonnet` with `tools: Read, Write, Edit, Grep, Glob, Bash`, and the VoltAgent `test-automator.md` uses `model: sonnet` with `tools: Read, Write, Edit, Bash, Glob, Grep`. Sonnet is the right default for test writing: capable enough for correct ML reasoning, cheaper than Opus.

**Managing it.** The docs state: "As of v2.1.198, the `/agents` command no longer opens the interactive creation wizard; running it prints a reminder to ask Claude or edit `.claude/agents/` directly." So the workflow is: ask Claude to write the file (it will scaffold `name`, `description`, `tools`, `model`, and a system prompt), then hand-edit. Claude Code watches these directories and reloads within a few seconds; only creating the very first file in a new `agents/` directory needs a restart.

### 2. Writing the `description` for proactive triggering
The docs are explicit: "Claude automatically delegates tasks based on the task description in your request, the `description` field in subagent configurations, and current context. To encourage proactive delegation, include phrases like 'use proactively' in your subagent's description field." Anthropic's own doc examples model this: the code-reviewer's description reads "Expert code review specialist. Proactively reviews code … Use immediately after writing or modifying code." Community convention (VoltAgent, the Sathish Raju guide) is to write the description as a *routing rule*: state the exact conditions and add "MUST BE USED"/"use PROACTIVELY." For a test-writer, a strong description is:

> `Writes and runs pytest tests for the RetainIQ churn project. MUST BE USED PROACTIVELY immediately after writing or modifying any code in src/, app/, or mlops/, or when the user mentions tests, coverage, pytest, fixtures, or a failing test. Specializes in data/ML pipeline tests, leakage guards, and FastAPI endpoint tests.`

### 3. ML-specific testing content the body must encode

**Leakage guards (the single most important thing for churn).** Multiple sources converge: SMOTE and scaling must run inside CV on the training folds only, via `imblearn.pipeline.Pipeline`, because "if you apply SMOTE or Standardization to your entire X_train before passing it to `cross_val_score`, you are leaking data across your folds." The canonical fix:
```python
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
pipe = ImbPipeline([("preprocessor", preprocessor),
                    ("smote", SMOTE(random_state=42)),
                    ("classifier", clf)])
scores = cross_val_score(pipe, X_train, y_train, cv=StratifiedKFold(5))
```
The test-writer should generate tests that (a) assert the churn label and any probability/derivative columns are NOT in the feature matrix; (b) assert transformers/SMOTE are fit only on training data; and (c) assert a leakage-canary — a synthetic feature that perfectly encodes the target produces an implausibly high AUC, proving the guard works. Domain sources are blunt that "a churn model that scores 0.99 in validation is usually a leakage bug, not a triumph." A concrete published example of the failure mode: the arXiv case study "When Administrative Networks Fail" (arXiv 2511.17736, §5.4) reports discovering "models with F1 = 1.00 and ROC–AUC = 1.00 across all folds" as "a clear red flag"; the audit "revealed two post-outcome administrative variables—ongoing enrolment status and graduation flag—that had inadvertently entered the feature set and effectively encoded the outcome. Removing these variables … produced realistic performance levels (F1 ≈ 0.94)."

Calibrating the red-flag threshold: published, leakage-audited XGBoost results on the IBM Telco dataset land high but not perfect — e.g., Frontiers in Artificial Intelligence (2026, doi 10.3389/frai.2026.1748799) reports "XGBoost attaining the best discriminative ability (AUC-ROC: 0.932)" with F1 0.84 under stratified 5-fold CV, while other honest baselines (logistic regression/random forest) sit around 0.79–0.82. That spread is precisely why the sensible tripwire is AUC > 0.95 (well above any legitimately reported result), not merely "high" — the RetainIQ target of ~0.85–0.88 is comfortably inside the plausible band.

**Metric choice.** On this imbalanced dataset (26.5% churn), tests and the model must report AUC-ROC, PR-AUC, and recall, not accuracy: "A churn dataset with 5% positive class will score 95% accuracy if the model predicts 'retained' for every single customer." A dummy classifier scores AUC 0.50, which is the floor any real model must beat.

**Smoke tests over exact metrics.** The probabl-ai `smoke-test-ml-pipeline` skill formalizes the pattern the agent should follow. Its "hard assertion is exact row-count equality" — `assert len(predictions) == n_predict_grid_rows` — described as "Not 'approximately equal', not 'at least 80% of expected rows'. A row-count mismatch is the failure mode the smoke test exists to catch." It pairs this with a *soft* assertion `assert smoke_mae < 3 * cv_mae_mean` (an opt-out bound that catches NaN-poisoned predictions), noting "the 3× bound is a starting heuristic; adjust per task." For classification (RetainIQ), the analogue is: assert training returns a fitted estimator and that AUC on a held-out tiny fixture clears a low floor (e.g., ≥ 0.6) rather than asserting an exact value, using `np.isclose` with tolerance when a value must be pinned because it accounts for "the stochastic nature of many ML models." The probabl skill's underlying leakage rule (its `build-ml-pipeline` skill, rule S5) is worth mirroring in spirit: "for any cross-row step (lag, rolling, group-agg, target shift, side-join …) the X marker goes UPSTREAM of that step," and "Don't loosen the smoke-test assertion. Don't wrap the predictor."

**Tiny fixtures, not the full dataset.** Speed and determinism demand small fixtures. The probabl skill warns "Don't synthesize the fixture … Synthetic fixtures look fine but skip the loaders that actually break in production" for *end-to-end* smoke tests that read real `data/`; but for unit tests of transformers/models, a tiny synthetic frame or a small sampled slice with a fixed seed is the norm (`make_classification(..., random_state=42)` or a 50-row fixture in `conftest.py`).

**FastAPI tests.** Use `from fastapi.testclient import TestClient`. Per the FastAPI docs the test functions are "normal `def`, not `async def`." For `/predict`, `/batch-predict`, `/explain`, `/recommend`: assert `status_code == 200`, assert the response JSON matches the expected schema/keys, and assert `422` for invalid bodies (FastAPI auto-validates via Pydantic). Override the model dependency with `app.dependency_overrides` so tests never load the real trained artifact or hit MLflow.

**Mocking external services.** The LLM insight/Next-Best-Action generator and MLflow logging must be mocked so tests stay fast and offline. Use `monkeypatch.setattr(...)` or `unittest.mock.patch`/`pytest-mock`; a common pattern is an autouse `conftest.py` fixture that blocks real network calls and patches the Anthropic/OpenAI client (`monkeypatch.setattr("openai.resources.chat.completions.Completions.create", mock_fn)`). For MLflow, patch the logging calls or point `mlflow` at a `tmp_path` tracking URI.

**Reproducibility.** Seed `random_state=42` on splits/models/SMOTE and seed `np.random`. Google's ML Crash Course ("Production ML systems: Deployment testing") advises: "Deterministically seed the random number generator … Initialize model components in a fixed order to ensure the components get the same random number from the random number generator on every run." An autouse seeding fixture in `conftest.py` is a clean way to enforce this suite-wide.

**Property-based & schema testing.** `hypothesis` (used by numpy, pandas, scikit-learn's ecosystem) plus `pandera`'s schema-driven data-synthesis strategies let the agent test data-cleaning invariants across generated inputs; pandera schemas act as "fancy assertion statements" and integrate with pytest.

### 4. Coverage & pitfalls
Target ~80% coverage (`pytest --cov=src --cov-report=term-missing`, optionally `--cov-fail-under=80`). The ML-CI practitioner survey (Bernardo et al., arXiv 2502.17378) found acceptable coverage rates concentrated at "the 70–80% range (30.3%), the 80–90% range (16.8%), or even the 90–100% range (16.1%)," with a further "13.5% selecting 60–70% and 6.5% choosing 50–60%" — a plurality at 70–80%, tempered by the reality (respondent P10) that it is "generally harder, more time-consuming, and costly to test ML projects … compared to non-ML projects." Common ML testing pitfalls the agent should avoid: asserting brittle exact metrics; fitting preprocessing on the full dataset; using accuracy on imbalanced data; hitting real APIs/network; and flaky nondeterministic tests (seed everything; mark genuinely flaky tests explicitly rather than loosening assertions).

### 5. A ready-to-use `test-writer.md`
```markdown
---
name: test-writer
description: >
  Writes and runs pytest tests for the RetainIQ churn-prediction project.
  MUST BE USED PROACTIVELY immediately after writing or modifying any code in
  src/, app/, or mlops/, and whenever the user mentions tests, pytest, coverage,
  fixtures, or a failing test. Specializes in data/ML pipeline tests, data-leakage
  guards, imbalanced-data handling, and FastAPI endpoint tests.
tools: Read, Grep, Glob, Edit, Write, Bash
model: sonnet
---

You are a senior test engineer for RetainIQ, an end-to-end customer-churn
prediction system (Python 3.11; pandas, numpy, scikit-learn, XGBoost, LightGBM,
imbalanced-learn/SMOTE, SHAP/LIME, MLflow, FastAPI, Streamlit, Evidently,
Prefect, Docker). You write fast, deterministic, offline pytest tests and verify
they pass before finishing.

## Workflow (always)
1. Read the target module and its neighbours; read tests/conftest.py and any
   existing tests for that area; read CLAUDE.md and follow its rules.
2. Reuse or add fixtures in conftest.py; prefer a TINY synthetic/fixture dataset
   (tens of rows, random_state=42) over the full 7,043-row Telco dataset.
3. Write tests under tests/ mirroring src/ layout (test_<module>.py, test_*
   functions, Arrange-Act-Assert, descriptive names).
4. Run `pytest -q` (and `pytest --cov=src --cov-report=term-missing` when asked).
   Fix failures while preserving intent. Report only the summary + failures.

## What to test
- Data (src/data): schema/dtypes, no unexpected missing values, target Churn is
  binary, deterministic cleaning output. Use pandera schemas where helpful.
- Features (src/features): ColumnTransformer output shape/columns stable
  (get_feature_names_out), no NaNs after transform, fit on TRAIN ONLY.
- Models (src/models): SMOKE tests — training returns a fitted estimator and
  clears a LOW performance floor (e.g. AUC >= 0.6) on a tiny seeded fixture.
  Never assert exact metric values; use tolerances/bounds. Report AUC-ROC,
  PR-AUC, recall — never accuracy on this ~26.5% imbalanced target.
- API (src/api): use fastapi.testclient.TestClient for /predict, /batch-predict,
  /explain, /recommend. Assert status codes, response schema, and 422 on bad
  input. Override the model via app.dependency_overrides; do not load real
  artifacts or hit MLflow.

## Non-negotiable invariants (leakage & correctness)
- Assert the churn label / predicted probability and their derivatives are NOT in
  the feature matrix.
- Assert SMOTE and scaling run INSIDE CV via imblearn.pipeline.Pipeline (train
  folds only); never fit_resample the whole dataset before splitting.
- Include a leakage canary: a feature that perfectly encodes the target yields an
  implausibly high AUC — flag AUC > 0.95 as a leakage red flag (honest ~0.85-0.88).
- Use stratified splits with random_state=42; seed np.random.

## Speed / isolation
- Mock external services: the LLM insight/Next-Best-Action generator and MLflow
  logging (monkeypatch / unittest.mock). No network. No real API keys.
- Keep unit tests fast (<100ms where possible); use tmp_path for file I/O.
- Use @pytest.mark.parametrize for input sets; register custom markers.

## Definition of done
`pytest -q` passes; tests are deterministic and offline; each new/changed public
behaviour has a test; leakage and imbalance invariants are asserted.
```

## Recommendations
1. **Create the file now at `.claude/agents/test-writer.md`** using the template above, and commit it — project-scoped and version-controlled is Anthropic's recommended pattern and it becomes part of your portfolio artifact. Ask Claude to generate it, then hand-tune.
2. **Keep the tool set tight:** `Read, Grep, Glob, Edit, Write, Bash`. It needs Write/Edit (create test files) and Bash (run pytest); withhold everything else. If you later want a review-only variant, clone it with `tools: Read, Grep, Glob, Bash` and `disallowedTools: Write, Edit`.
3. **Pin `model: sonnet`** (both real-world test subagents do). Escalate to `opus` only if you find the agent mis-reasoning about leakage on complex pipelines; drop to `haiku` only for trivial mechanical test edits.
4. **Invest in the description**, since it's the trigger. Include "MUST BE USED PROACTIVELY" and the concrete file paths/keywords. Verify it fires by making a small change in `src/` and watching whether Claude delegates; if it doesn't, sharpen the trigger conditions.
5. **Point the agent at your CLAUDE.md guardrails** — it loads automatically, but restate the AUC red-flag and leakage rules in the agent body too, because the body is the only place you fully control its behaviour.
6. **Add a `SubagentStop` hook (optional, later)** in `.claude/settings.json` that runs `pytest -q` and blocks completion if red — a programmatic enforcement of "definition of done."
7. **Benchmarks that change these choices:** if `pytest -q` regularly exceeds ~a minute, push harder on tiny fixtures and mocking; if coverage sits well below ~80% on `src/`, have the agent target uncovered functions; if any test asserts an exact metric or fits preprocessing on the full dataset, treat it as a bug to fix.

## Caveats
- **Claude Code evolves quickly.** Exact frontmatter fields, the `/agents` wizard behaviour, and version-gated details (e.g., watcher/restart rules, model aliases) change between releases; the details here reflect the official docs as of August 2026. Verify against `code.claude.com/docs/en/sub-agents` for your installed version.
- **The two example test-writer subagents are generic/JS-leaning.** Their frontmatter and structure transfer cleanly, but their code templates are Jest/Vitest — the pytest/ML content above is what makes the agent useful for RetainIQ.
- **The probabl-ai ML skills are a strong reference but not a drop-in.** Their smoke-test row-count assertion is framed for time-series/regression pipelines; adapt it to classification (fitted-estimator + AUC-floor). Note also that dedicated leakage/distribution/schema *test* skills don't yet exist in that repo (they're referenced as future work; the leakage rules currently live inside their `build-ml-pipeline` skill), and that repo's stack guidance prefers HistGradientBoosting and plain (non-stratified) KFold — a deliberate divergence from RetainIQ's XGBoost/LightGBM + stratified-split choices, so don't copy those recommendations blindly.
- **Published Telco AUC varies by study and by leakage discipline.** Reported XGBoost AUC-ROC on this dataset ranges from ~0.79 (some baselines) to ~0.93 (Frontiers 2026, stratified 5-fold). RetainIQ's ~0.85–0.88 target is realistic; the >0.95 tripwire is deliberately set above any legitimately published figure so it flags leakage, not merely a good model.
- **Subagents cost context/tokens.** Anthropic notes subagent-heavy workflows can consume substantially more tokens than single-threaded sessions; a single well-scoped test-writer is worth it, but don't proliferate overlapping agents.
- **Coverage % is a guide, not a goal.** 80% is a reasonable target, but high coverage with brittle or leakage-blind tests is worse than fewer meaningful invariant tests.