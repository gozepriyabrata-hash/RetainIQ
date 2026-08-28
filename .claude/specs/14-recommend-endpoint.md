# Spec + Plan: `POST /recommend` FastAPI Endpoint ("Offer Suggestions")

> Location note: this file lives at `.claude/specs/` per CLAUDE.md §4's repo
> tree ("`.claude` — claude.md and also Specs folder"), consistent with
> `01`–`13`, overriding the generic `/create-spec` command's
> `specs/<slug>/spec.md`+`plan.md` template — same override `10` and `13`
> already made, for the same reason. Spec and plan are combined in one file.
> Numbered `14` (next after `13-retention-action-engine.md`).
>
> Scope note: this is a **slice** of CLAUDE.md §14 Phase 5 ("FastAPI service
> + Streamlit dashboard + What-If panel"), not the whole phase — it ships
> exactly one of CLAUDE.md §10's four contracted routes, `POST /recommend`,
> mirroring `12-explain-endpoint.md`'s own precedent of shipping one route
> as a thin wrapper over already-tested `src/` logic. Streamlit and
> `/predict`/`/batch-predict` are out of scope (Non-goals); Phase 5's
> tracker row stays `☐`.
>
> Origin note: requested as "Offer Suggestions / Contract upgrade
> incentive, tech-support add-on, loyalty discount, payment-method switch."
> All four of those are not new logic — they already exist, committed, in
> `13-retention-action-engine.md`'s `src/recommend/action_engine.py`
> (`DRIVER_ACTION_RULES`' `contract`/`support`/`payment` categories and
> `TIER_BASE_ACTIONS["High"]`'s loyalty-discount action). Nothing in the
> repo exposes them outside a notebook. This spec wires the existing,
> already-tested `recommend_actions_for_customer` into CLAUDE.md §10's
> `POST /recommend` contract — the same "expose it, don't reinvent it"
> pattern `12` used for `explain_customer`.
>
> Research note: two real, measured findings from spec research (not
> assumed):
> 1. **Thread-safety hazard inherited from `/explain`.**
>    `recommend_actions_for_customer` calls `local_explainer.explain_customer`
>    internally (to get `shap_top_drivers`), which — per `action_engine.py`'s
>    own comment — "also computes `lime_top_drivers` (a 5000-sample LIME fit)
>    as a side effect." `12`'s own `main.py` already documents that
>    `LimeTabularExplainer` mutates its own `numpy.RandomState` on every call
>    and is not thread-safe across FastAPI's sync-route threadpool — that's
>    exactly why `app.state.explain_lock` exists. `/recommend` reaches the
>    same non-thread-safe call transitively, so it MUST serialize on the same
>    lock (Requirement 5) — this is not a new hazard, just an inherited one
>    that would otherwise go unguarded.
> 2. **Calibrated-model load cost, benchmarked directly** (not estimated):
>    `calibration.load_calibrated_model()` measured at **1.78s on first call
>    in a fresh process** (dominated by one-time sklearn/xgboost import
>    resolution inside `joblib.load`, most of which `lifespan`'s existing
>    `build_explainer_context` call already pays) and **0.028s on a
>    subsequent call** with the OS file cache warm. `risk_tiers.
>    classify_scored_customers` → `scoring.score_customers` calls this on
>    every invocation when `pipeline=None` (its default) — currently
>    `action_engine.recommend_actions_for_customer` never passes a
>    `pipeline` through, so every call reloads the pickle from disk. CLAUDE.md
>    §10 states plainly: "Load the model once at startup, not per request."
>    Requirement 1 closes this gap by threading a `pipeline` parameter
>    through, and Requirement 2 loads it once in `lifespan`, exactly `12`'s
>    own `explainer_context` precedent — bringing `/recommend` in line with
>    the rule `/explain` already follows, rather than leaving a second,
>    inconsistent model-loading path in the same app.

---

## PART 1 — SPEC

### Feature

`POST /recommend`: a customer's raw attributes in, their risk tier plus a
ranked list of concrete "Offer Suggestions" — retention actions such as a
contract-upgrade incentive, a tech-support add-on, a loyalty discount, or a
payment-method nudge — out. A thin FastAPI wrapper over `13`'s existing,
already-tested `recommend_actions_for_customer`; no new recommendation
logic.

### Problem / motivation

`13` built the Next-Best-Action engine and explicitly deferred wiring it
into an API ("Non-goals: no FastAPI endpoint... `recommend_actions_for_
customer` is added as the function a future `POST /recommend` route will
call, wired to nothing yet"). Today the only way to see an offer suggestion
for a customer is to run `notebooks/11_retention_action_engine.ipynb` by
hand. CLAUDE.md §10 has contracted `POST /recommend` since before `13`
shipped, and §14's Definition of Done requires a caller to "call the REST
API" and see "next-best-action recommendations." This spec is that missing
wire, matching `12`'s own `POST /explain` precedent exactly.

### Goals / non-goals

**Goals**
- Add `POST /recommend` to `src/api/main.py`, returning
  `recommend_actions_for_customer`'s output verbatim (field-for-field, no
  reshaping — `12`'s own `ExplainResponse` precedent).
- Load the calibrated model once at FastAPI startup (`lifespan`), not per
  request, matching CLAUDE.md §10 and `explainer_context`'s existing
  precedent (Research note #2).
- Thread a `pipeline` parameter through
  `src/recommend/action_engine.recommend_actions_for_customer` so the
  route can supply the once-loaded pipeline instead of forcing a fresh
  disk load per call (backward-compatible: default stays `None`,
  preserving `13`'s existing call sites and tests unchanged).
- Serialize `/recommend` on the same `app.state.explain_lock` `/explain`
  uses, since it transitively hits the same non-thread-safe LIME call
  (Research note #1).

**Non-goals**
- **No Streamlit "Offer Suggestions" panel or any other dashboard view.**
  `app/dashboard.py` currently calls no API route at all (verified: it has
  no `requests`/`explain`/`recommend` references); this spec keeps that
  boundary and adds only the API route, matching `12`'s own precedent of
  shipping the API slice without the dashboard slice.
- **No `/predict` or `/batch-predict` endpoint** — CLAUDE.md §10's other
  two contracted routes, separate specs.
- **No optional-LLM recommendation narrative** — still `13`'s deferred
  Phase 4 half, untouched here.
- **No change to `recommend_actions`'s rule tables, `TIER_BASE_ACTIONS`,
  `DRIVER_ACTION_RULES`, or any of `13`'s existing tests/notebook.** The
  only `action_engine.py` change is the additive `pipeline` parameter
  (Requirement 1).
- **No change to `src/recommend/risk_tiers.py`, `src/models/scoring.py`,
  or `src/models/calibration.py`** — this spec only calls their existing
  public functions, and only adds a way to pass an already-loaded pipeline
  into a call path that already accepts one (`classify_scored_customers`'s
  `pipeline` parameter already exists and is unused end-to-end today).
- **No change to `GET /health`'s existing response fields or meaning.**
  `model_loaded` keeps meaning exactly what it means today (explainer
  context ready) — existing `test_api.py` assertions on it are not
  touched. `/recommend` itself checks `app.state.calibrated_pipeline`
  directly and 503s, exactly how `/explain` checks
  `app.state.explainer_context` directly rather than delegating to
  `/health`.
- No new third-party dependency.

### User stories

- As a **retention manager**, I want to `POST` one customer's attributes
  and get back a risk tier plus a short, concrete offer-suggestion list
  (not a raw driver dump), so I can act on it directly from a UI or script
  without touching a notebook.
- As the **engineer**, I want `/recommend` to reuse `13`'s already-tested
  `recommend_actions_for_customer` unchanged in its core logic, so this
  spec is verified to be a thin wrapper, not a second implementation that
  could drift from the tested one.
- As the **engineer**, I want the calibrated model loaded once at startup
  for `/recommend` exactly as the explainer context already is for
  `/explain`, so the app doesn't silently violate CLAUDE.md §10's own
  "load once, not per request" rule on one route while following it on
  another.
- As a **recruiter/reviewer**, I want to hit a real endpoint and see the
  same two worked examples (`5178-LMXOP`, `9763-GRSKD`) `13`'s tests
  already lock in, end-to-end through HTTP, not just at the function
  level.

### Functional requirements

1. `src/recommend/action_engine.recommend_actions_for_customer` MUST gain
   an additional parameter `pipeline: object | None = None`, inserted
   after `customer` to match `risk_tiers.classify_scored_customers`'s own
   parameter order, forwarded as
   `risk_tiers.classify_scored_customers(pd.DataFrame([customer]),
   pipeline=pipeline)`. Default `None` preserves current behavior (fresh
   load from disk) exactly — every existing call site and test in `13`
   keeps passing unmodified.
2. `src/api/main.py`'s `lifespan` MUST additionally load the calibrated
   pipeline once via `calibration.load_calibrated_model()` and store it as
   `app.state.calibrated_pipeline`, in its own `try`/`except Exception`
   block independent of the existing `explainer_context` block — a
   failure loading one must not prevent the other from loading (the two
   artifacts are independent files on disk; either can be present/absent
   on its own, e.g. a fresh clone before *any* training step has run).
   `create_app()` MUST initialize `app.state.calibrated_pipeline = None`
   before `lifespan` runs, mirroring `explainer_context`'s own
   before-lifespan initialization (so `/health` and `/recommend` never
   hit an unset-state `AttributeError` if called before startup
   completes).
3. `src/api/schemas.py` MUST gain `ActionItem` (`priority: int, action:
   str, category: str, rationale: str, source: str, driver_feature: str |
   None`) and `RecommendResponse` (`customerID: str | None = None,
   churn_probability: float, churn_probability_pct: float, risk_tier:
   str, actions: list[ActionItem]`) — field-for-field identical to
   `recommend_actions_for_customer`'s existing dict output, no reshaping,
   matching `ExplainResponse`'s own precedent. `CustomerPayload` (request
   body) is reused unchanged — identical input shape to `/explain`.
4. `src/api/main.py` MUST gain `POST /recommend`, request body
   `CustomerPayload`, response model `RecommendResponse`:
   - 503 with `MODEL_UNAVAILABLE_MESSAGE` if either
     `app.state.explainer_context` or `app.state.calibrated_pipeline` is
     `None` (checked directly in the route, `/explain`'s own pattern —
     not delegated to `/health`).
   - Otherwise calls `recommend_actions_for_customer(payload.
     to_customer_dict(), explainer_context=context,
     pipeline=app.state.calibrated_pipeline)` inside
     `with request.app.state.explain_lock:` (Requirement 5).
   - `ValueError` → 422 with `str(exc)` truncated to the same length bound
     `/explain` already uses (reuse `EXPLAIN_ERROR_DETAIL_MAX_LENGTH`, or
     rename to a shared constant — implementer's call, document whichever
     is chosen).
   - Any other exception: logged via `logger.exception(...)` and
     re-raised, exactly `/explain`'s own defensive-but-not-expected-to-
     fire pattern.
5. Both `/explain` and `/recommend` MUST serialize on the same single
   `app.state.explain_lock` (not a second, independent lock) — they hit
   the same underlying non-thread-safe `LimeTabularExplainer` instance in
   `app.state.explainer_context`, so two separate locks would not
   actually prevent a concurrent `/explain` + `/recommend` race.
6. `tests/test_api.py` MUST cover the Plan's "Tests to write" section in
   full, added to the existing file (not a new one), reusing its existing
   `client` fixture and `_customer_payload`/`_customer_payload_stub`
   helpers.
7. None of the above may change `src/recommend/risk_tiers.py`,
   `src/explain/local_explainer.py`, `src/models/scoring.py`,
   `src/models/calibration.py`, `src/models/train.py`, `app/dashboard.py`,
   or any existing test/notebook/figure/route behavior — all current
   tests (including every existing `test_api.py` and
   `test_action_engine.py` case) must keep passing unmodified.

### Data & model impact

No new model, no new training feature, no schema change to any training
artifact. `/recommend` reads the same two already-persisted model
artifacts `/explain` (SHAP/LIME context) and the existing-but-currently-
unused calibrated-pipeline path already support; nothing here is written
back into `load_clean_data()`'s output or any training DataFrame.

### ML guardrails (mandatory check)

- **No target/probability leakage:** the route accepts only raw customer
  attributes (`CustomerPayload`, `extra="forbid"` already rejects a stray
  `Churn` field) and returns only already-computed model outputs
  (`risk_tier`, `churn_probability`, actions) — no new input path touches
  `src/features/preprocessing.py` or `src/models/train.py`.
- **Honest-AUC guard is unaffected:** no new model, no training/evaluation
  code path touched; `/recommend` calls the same calibrated model already
  used elsewhere for scoring.
- **Fitting/splitting/SMOTE:** N/A — this spec adds no training code.
- **Imbalance/metric reporting:** N/A — no classifier added, no metric
  reported by this route.
- **Reproducibility:** deterministic given identical input, same as `13`'s
  `recommend_actions_for_customer` — the only new source of run-to-run
  variance is `explain_customer`'s existing LIME sampling, already present
  and already accepted in `/explain` today; unchanged by this spec.

### API / UI surface

| Method | Path | Request | Response | Notes |
|---|---|---|---|---|
| POST | `/recommend` | `CustomerPayload` (identical to `/explain`'s) | `RecommendResponse` | 503 if either model artifact isn't loaded; 422 on validation or `ValueError` |

No Streamlit view. `src/api/` stays the only place with new code besides
the one additive `action_engine.py` parameter; `app/` is untouched.

### Edge cases & failure states

- **Calibrated pipeline missing, explainer context present (or vice
  versa):** `/recommend` 503s (Requirement 4); `/explain` is unaffected
  either way since its own check is independent — directly tested.
- **Neither artifact present (fresh clone, nothing trained yet):** both
  routes 503, `/health` still 200 with `model_loaded: false` — directly
  tested.
- **Unseen categorical value, out-of-range numeric:** rejected at the
  `CustomerPayload` Pydantic boundary (422), same as `/explain` — no new
  validation needed, reused unchanged.
- **Concurrent `/recommend` and `/explain` requests:** serialized
  correctly by the shared lock, no `LimeTabularExplainer` `RandomState`
  race — directly tested with a `ThreadPoolExecutor`, mirroring
  `test_api.py`'s existing concurrency test pattern.
- **Malformed JSON body:** FastAPI's default 422 (unchanged, inherited).

### Security notes

- **No new dependency.** No new untrusted-input surface beyond what
  `/explain` already accepts and validates — same `CustomerPayload`,
  same `extra="forbid"` leakage guard, same Literal-typed closed
  vocabularies.
- `ActionItem`/`RecommendResponse` are output-only Pydantic models; no
  untrusted data flows into them except `action_engine`'s own fixed
  string constants and already-validated derived values (`risk_tier`,
  SHAP `reason` strings) — no `eval`, no template injection.
- No secrets, no new environment variables, no new network call, no file
  I/O beyond the same `joblib.load` of a model artifact `scoring.py`
  already performs elsewhere.

### Success criteria

- `pytest -q` passes: all existing tests + the new `test_api.py` cases,
  all green.
- `POST /recommend` on the real `5178-LMXOP` and `9763-GRSKD` customers
  (the same two worked examples `13` locked in at the function level)
  returns the identical action lists, now verified end-to-end through
  HTTP.
- `/recommend` 503s correctly when either model artifact is absent, and
  `/explain`/`/health` are unaffected by that absence when their own
  artifact is present.
- `quality-reviewer` and `security-reviewer` report no unresolved
  findings on the diff.

### Out of scope

- Streamlit "Offer Suggestions" dashboard view — a later Phase 5 spec.
- `/predict`, `/batch-predict` — separate specs.
- Optional LLM-generated recommendation narrative — still `13`'s deferred
  Phase 4 half.
- Any change to `13`'s rule tables or `risk_tiers.py`/`scoring.py`/
  `calibration.py`'s public behavior.

---

## PART 2 — PLAN

### Approach

Extend the existing FastAPI app and `action_engine.py` additively,
mirroring `12`'s `POST /explain` implementation pattern exactly: one new
route, one new pair of response schemas, one more artifact loaded once in
`lifespan`, and a widened-but-backward-compatible function signature.

**Alternative rejected:** have `/recommend` call `/explain` and a future
`/predict` internally over HTTP (or `TestClient`-style in-process request)
rather than calling `recommend_actions_for_customer` directly. Rejected as
over-engineered for a same-process, same-app call — it would add a network
or Starlette-routing hop for no benefit, and breaks CLAUDE.md §4's "keep
the API thin, call into `src/`" principle by making the API layer depend
on itself instead of on `src/`.

### Task breakdown

- [ ] **1. Edit `src/recommend/action_engine.py`** — add `pipeline: object
      | None = None` to `recommend_actions_for_customer`'s signature,
      forward it to `classify_scored_customers`, update the docstring
      (Requirement 1).
- [ ] **2. Edit `src/api/main.py`** — `lifespan` loads
      `app.state.calibrated_pipeline` in its own try/except;
      `create_app()` initializes it to `None` beforehand; add `POST
      /recommend` reusing `CustomerPayload`, `app.state.explain_lock`,
      and the existing error-mapping pattern (Requirements 2, 4, 5).
- [ ] **3. Edit `src/api/schemas.py`** — add `ActionItem`,
      `RecommendResponse` (Requirement 3).
- [ ] **4. Edit `tests/test_api.py`** — add the cases in "Tests to write"
      below, reusing the existing `client` fixture and payload helpers.
- [ ] **5. Run the full suite** — `pytest -q`, confirm all existing + new
      tests pass.
- [ ] **6. Commit** — `src/recommend/action_engine.py`,
      `src/api/main.py`, `src/api/schemas.py`, `tests/test_api.py`,
      commit message `phase 5: POST /recommend endpoint -- offer-
      suggestion API surface`. No files under `data/`, `models/`, or
      `mlruns/` are touched.

### Tests to write (hand to test-writer)

- `tests/test_api.py::test_recommend_returns_expected_shape` — a real
  test-split customer through `POST /recommend` → 200, response has
  `churn_probability`, `churn_probability_pct`, `risk_tier`, `actions`
  (non-empty list, each item has `priority`, `action`, `category`,
  `rationale`, `source`, `driver_feature`).
- `tests/test_api.py::test_recommend_matches_worked_example_critical` —
  `5178-LMXOP`'s real attributes through `POST /recommend` → actions match
  `13`'s locked-in Critical worked example (tier base + `tenure` rule +
  `Contract` rule, 3 total, `InternetService` rule absent).
- `tests/test_api.py::test_recommend_matches_worked_example_low` —
  `9763-GRSKD`'s real attributes → matches `13`'s locked-in Low worked
  example (tier base + `Contract` rule, 2 total).
- `tests/test_api.py::test_recommend_503_when_calibrated_pipeline_missing` —
  build an app whose `lifespan` calibrated-pipeline load fails (e.g. point
  at a missing path or monkeypatch `calibration.load_calibrated_model`) but
  whose `explainer_context` loads fine → `/recommend` 503,
  `/explain` still 200, `/health` still 200 `model_loaded: true`.
- `tests/test_api.py::test_recommend_503_when_explainer_context_missing` —
  the inverse: `explainer_context` fails to build, calibrated pipeline
  loads fine → `/recommend` 503, `/health` `model_loaded: false`.
- `tests/test_api.py::test_recommend_422_on_unseen_category` — reuse
  `/explain`'s existing unseen-category test payload/assertion shape
  against `/recommend`.
- `tests/test_api.py::test_recommend_and_explain_share_lock_under_concurrency` —
  `ThreadPoolExecutor` firing concurrent `/explain` and `/recommend`
  requests on the same customer, asserting both return 200 with
  internally consistent results (no interleaved-`RandomState` corruption),
  mirroring the existing concurrency test's structure.
- `tests/test_action_engine.py::test_recommend_actions_for_customer_accepts_explicit_pipeline` —
  pass a real loaded `calibration.load_calibrated_model()` result as
  `pipeline=` explicitly → result identical to calling with
  `pipeline=None` (default fresh-load) on the same customer, confirming
  Requirement 1's forwarding is correct and doesn't change output.

### Quality gates

1. **test-runner** — run `pytest -q`, confirm all tests (existing + new)
   pass, fix any straightforward regression.
2. **quality-reviewer** — review `lifespan`'s two-independent-try/except
   structure, the shared-lock reasoning (Requirement 5), and CLAUDE.md §8
   adherence (named constants, type hints, docstrings) against the diff.
3. **security-reviewer** — confirm no new dependency, confirm
   `RecommendResponse`/`ActionItem` introduce no new untrusted-input
   handling, confirm the 422 error-detail truncation is applied
   consistently with `/explain`'s.
4. Commit only after both reviews are clean.

### Risks / rollback

- **Risk:** loading a second artifact in `lifespan` doubles the ways
  startup can partially fail. **Mitigation:** Requirement 2's independent
  try/except blocks mean a missing calibrated model degrades only
  `/recommend`, never `/explain` or `/health` — directly tested.
- **Risk:** reusing `explain_lock` for `/recommend` too could become a
  throughput bottleneck if both routes see heavy concurrent traffic.
  **Mitigation:** intentional and documented (Requirement 5) — correctness
  over throughput for a portfolio-scale service; revisit only if a real
  load test shows contention.
- **Rollback:** one commit touching 4 files, all additive (widened
  `action_engine.py` signature with a backward-compatible default, new
  route, new schemas, new tests) — no existing route, schema, or test is
  modified. `git revert` is clean; nothing under `models/`/`mlruns/`/
  `data/` is touched.

### Definition of done

- All 6 tasks checked off.
- `pytest -q` green (all existing tests + the new `test_api.py`/
  `test_action_engine.py` cases).
- `quality-reviewer` and `security-reviewer` report no unresolved
  findings.
- All Success Criteria in Part 1 are met.
