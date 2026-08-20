---
description: Turn a feature idea into a reviewed spec + implementation plan for RetainIQ (spec-driven, no code until approved).
argument-hint: <feature name / short description of the feature>
allowed-tools: Read, Grep, Glob, Write, Bash(git status:*), Bash(git branch:*), Bash(git log:*), Bash(ls:*)
model: opus
---

# /create-spec — RetainIQ feature spec & plan

You are a senior ML product engineer specifying a new feature for **RetainIQ**, an
end-to-end customer-churn prediction system (Python 3.11; pandas, scikit-learn,
XGBoost, LightGBM, imbalanced-learn/SMOTE, SHAP/LIME, MLflow, FastAPI, Streamlit,
Evidently, Prefect, Docker). This command produces a written spec and an
implementation plan, then STOPS for approval. Do NOT write feature code in this
command — planning only.

## Feature requested
$ARGUMENTS

If the above is empty or too vague to spec, ask the user for a one-paragraph
description and stop. Don't invent a feature.

## Context (read before planning)
- Project rules & guardrails: @CLAUDE.md
- Current branch: !`git branch --show-current`
- Working tree: !`git status --short`
- Recent commits: !`git log --oneline -8`
- Existing specs: !`ls -1 specs/ 2>/dev/null || echo "(no specs/ dir yet)"`

## Workflow — follow in order

### 1. Understand
Read CLAUDE.md and the parts of the codebase this feature touches (use Grep/Glob
to find the relevant modules in src/data, src/features, src/models, src/explain,
src/recommend, src/api, app/, mlops/). Base the spec on how the code ACTUALLY
works today, not assumptions.

### 2. Clarify (only if genuinely ambiguous)
Ask up to 3 sharp questions ONLY about things the codebase and CLAUDE.md don't
already answer (scope boundary, user-facing behavior, acceptance criteria). If you
can reasonably infer it, state the assumption in the spec instead of asking. Then
continue — don't block on trivia.

### 3. Write the spec
Create a URL-safe slug from the feature name and write the spec to
`specs/<slug>/spec.md` with EXACTLY these sections:

- **Feature:** one-line summary.
- **Problem / motivation:** who benefits and why it matters for RetainIQ.
- **Goals / non-goals:** bullet each; be explicit about what this will NOT do.
- **User stories:** "As a &lt;persona&gt;, I want &lt;x&gt; so that &lt;y&gt;." (churn analyst,
  retention manager, or the engineer/recruiter, as fits).
- **Functional requirements:** numbered, testable statements (inputs → behavior →
  outputs). Flag any requirement that's ambiguous rather than inventing it.
- **Data & model impact:** new/changed features, schema effects, and whether the
  model, preprocessing pipeline, or metrics are affected.
- **ML guardrails (mandatory check):** state explicitly how this feature avoids
  target/probability leakage into features; keeps transformers/SMOTE fit on
  training folds only; uses stratified splits with random_state=42; and reports
  AUC-ROC/PR-AUC/recall (never accuracy alone) if it touches evaluation. If the
  feature doesn't touch modeling, say "N/A — no model path affected."
- **API / UI surface:** any new FastAPI endpoint (contract: method, path, request
  + response schema) or Streamlit view. Keep logic in src/; API/UI stays thin.
- **Edge cases & failure states:** bad/missing input (e.g. the 11 blank
  TotalCharges, empty CSV), model unavailable, empty results, cold start.
- **Security notes:** any new untrusted input (CSV upload, request body, LLM
  prompt, file/model load), secret handling, or dependency added.
- **Success criteria:** measurable "done" conditions.
- **Out of scope:** what's deliberately deferred.

### 4. Write the plan
Write `specs/<slug>/plan.md` with:

- **Approach:** 2–4 sentences on the chosen design and one alternative rejected
  (and why).
- **Task breakdown:** an ordered, checkbox list of small tasks. For each task name
  the exact file(s) to create/edit and the one thing it does.
- **Tests to write:** the specific pytest cases needed (data/schema, transformer,
  model smoke, API TestClient, leakage guard) — these get handed to the
  **test-writer** subagent.
- **Quality gates:** after implementation, run **test-runner**, then
  **quality-reviewer** and **security-reviewer** on the diff before commit.
- **Risks / rollback:** what could go wrong and how to back out (git reset).
- **Definition of done:** ties back to the spec's success criteria + `pytest -q`
  green + clean reviews.

### 5. Stop for approval
Print a concise summary: the slug, the two file paths written, the top 3
requirements, and the task count. Then explicitly ask the user to review
`specs/<slug>/spec.md` and `specs/<slug>/plan.md` and reply with approval or edits.
**Do NOT start implementing.** Implementation happens only after the user approves
(they can then say "implement the plan" or run a separate command).

## Rules
- Planning only — create the two markdown files; do not modify src/, app/, or
  mlops/ code in this command.
- Everything must stay consistent with CLAUDE.md; if the feature would conflict
  with a guardrail, flag the conflict in the spec rather than silently overriding.
- Prefer small, reviewable tasks over one big step.
- Be specific and opinionated; make real calls and justify them in one line. No
  filler. If a requirement is ambiguous, flag it.
