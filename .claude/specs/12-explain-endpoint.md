# Spec + Plan: `POST /explain` FastAPI Endpoint

> Location note: this file lives at `.claude/specs/` per CLAUDE.md §4's repo
> tree ("`.claude` — claude.md and also Specs folder"), consistent with
> `01`–`11`. Spec and plan are combined in one file for the same reason.
> Numbered `12` (next after `11-explainable-ai.md`).
>
> Scope note: this is a **slice** of CLAUDE.md §14 Phase 5 ("FastAPI service
> + Streamlit dashboard + What-If panel"), not the whole phase — it ships
> exactly one of CLAUDE.md §10's four contracted routes, `POST /explain`,
> plus a minimal `GET /health` liveness check needed to run any FastAPI app
> at all. `/predict`, `/batch-predict`, `/recommend`, and every Streamlit
> view are explicitly out of scope (see Non-goals) and Phase 5's tracker row
> stays `☐` — the same "bridge, not the full phase" pattern `06`, `09`, and
> `10` already established for earlier phases.
>
> Origin note: this spec was requested as "Feature Importance / Top-N
> features globally + top-3 churn reasons per customer." Both pieces already
> exist and are already committed — `shap_global_importance`
> (`src/explain/driver_analysis.py`, `06`) and `global_shap_importance`
> (`src/explain/local_explainer.py`, `11`) rank top-N global drivers;
> `local_shap_top_drivers`/`local_lime_top_drivers`/`explain_customer`
> (`11`) already return top-3 per-customer drivers with plain-English
> reasons. None of that logic is reachable outside a notebook. The user was
> asked what genuinely new work this spec should cover and chose: wire
> `explain_customer` into CLAUDE.md §10's `POST /explain` route. This spec
> covers that route only — it adds no new explanation logic, only a thin
> API layer over what `11` already built.
>
> Research note: `src/api/` is currently an empty package (`__init__.py`
> only) — no FastAPI app exists yet anywhere in the repo, so this is the
> first module in `src/api/`. Verified directly during spec research, on
> the real persisted model already on disk (`models/churn_model.pkl` +
> metadata, git-ignored but present locally from `08`/`09`): building an
> explainer context (`build_explainer_context`) takes **1.895s**; a
> subsequent `explain_customer` call reusing that context takes **0.253s**
> and returns exactly the shape below (real customer, real output, not
> fabricated):
> ```json
> {
>   "shap_top_drivers": [
>     {"feature": "Contract", "customer_value": "Two year", "shap_value": -1.5618, "direction": "decreases", "reason": "Two year contract decreases this customer's predicted churn risk."},
>     {"feature": "tenure", "customer_value": 72, "shap_value": -1.2845, "direction": "decreases", "reason": "72-month tenure decreases this customer's predicted churn risk."},
>     {"feature": "OnlineSecurity", "customer_value": "Yes", "shap_value": -0.4195, "direction": "decreases", "reason": "OnlineSecurity = Yes decreases this customer's predicted churn risk."}
>   ],
>   "lime_top_drivers": [
>     {"feature": "Contract=Two year", "lime_weight": -0.2055, "direction": "decreases", "reason": "..."},
>     {"feature": "tenure > 55.00", "lime_weight": -0.0966, "direction": "decreases", "reason": "..."},
>     {"feature": "InternetService=Fiber optic", "lime_weight": 0.0956, "direction": "increases", "reason": "..."}
>   ],
>   "customerID": "TEST-0001"
> }
> ```
> This confirms: (1) the ~1.9s context-build cost is a one-time startup
> price, never a per-request one — directly motivating Requirement 2's
> load-once-at-startup design; (2) a per-request `/explain` call is fast
> enough (~0.25s) for a synchronous route with no caching/async needed.
>
> Post-implementation hardening note: this feature was implemented, then
> given two rounds of `quality-reviewer`/`security-reviewer` passes before
> commit (mirroring `06`'s precedent), which changed several details below
> from what was originally specified. The code, not this note, is the
> source of truth for these; `tests/test_api.py` covers every one directly:
> - **`SeniorCitizen` is `Literal[0, 1, "Yes", "No"]`, not `int`** (Req. 1) —
>   `context["X_test"]` (built from `load_clean_data()`'s already-cleaned
>   output) actually carries `"Yes"`/`"No"` strings, not the raw CSV's 0/1
>   ints; the schema accepts both forms rather than picking one and
>   rejecting the dataset's own native encoding.
> - **Every other closed-vocabulary categorical field is `Literal[...]`,
>   not bare `str`** — rejects an unseen category at the Pydantic boundary
>   (a genuine `422`, not a later library-specific failure inside
>   `explain_customer`), closes an unbounded-string DoS surface a bare
>   `str` left open, and documents the real allowed values in `/docs`.
>   `explain_customer`'s own `ValueError` mapping (Req. 5) stays in place
>   behind this rather than becoming dead code: LIME's label encoders are
>   fit on `X_train` only (the 80% split), a strict subset of the Literal
>   vocabularies (the full dataset) — see `src/api/schemas.py`'s comment.
> - **`main.py` uses a `create_app()` factory, not a bare module-level
>   `app`** — so a test exercising the startup-failure path can build an
>   independent `app.state` without mutating the singleton every other test
>   depends on. `app = create_app()` at module level is still what
>   `uvicorn src.api.main:app` targets.
> - **`app.state.explainer_context`/`startup_error` are initialized in
>   `create_app()` itself, before `lifespan` ever runs** — Requirement 4's
>   "`/health` is 200 always" wasn't true otherwise (Starlette's `State`
>   raises `AttributeError` on an unset key if `/health` is hit before
>   startup completes).
> - **`lifespan` catches bare `Exception`, not `(FileNotFoundError,
>   ValueError, KeyError)`** — a corrupted or version-mismatched pickle can
>   surface as `pickle.UnpicklingError`, `EOFError`, `AttributeError`, or an
>   xgboost-internal error; this is a startup-degradation boundary where the
>   whole point is that nothing crashes the process (mirrors
>   `app/dashboard.py`'s broad-but-intentional system-boundary handling).
> - **`app.state.explain_lock` (a `threading.Lock`) serializes the
>   `explain_customer` call** — `LimeTabularExplainer` mutates its own
>   `numpy.RandomState` on every call, and FastAPI dispatches sync `def`
>   routes to a shared threadpool, so concurrent `/explain` requests would
>   otherwise race on that shared, non-thread-safe state.
> - **`GET /health` is `async def`, not `def`** — it does no blocking work,
>   so running it on the event loop (rather than Starlette's sync-route
>   threadpool) keeps it responsive even if every threadpool worker is
>   queued behind `explain_lock` during a burst of `/explain` calls.
> - **A custom `RequestValidationError` handler strips the `input` field**
>   from Pydantic's default `422` body — FastAPI's stock handler otherwise
>   echoes the full rejected value back unbounded (e.g. a multi-megabyte
>   string sent for a `Literal`-typed field would be reflected in full);
>   `loc`/`msg`/`type` are kept, so which field failed and why is still
>   reported. `EXPLAIN_ERROR_DETAIL_MAX_LENGTH = 500` (Req. 5) is now a
>   second, narrower defense specifically for `explain_customer`'s own
>   `ValueError` message.
> - **`CustomerPayload` gained `extra="forbid"`, numeric upper bounds
>   (`tenure`, `MonthlyCharges`, `TotalCharges` — with `allow_inf_nan=False`
>   on the floats), and `customerID: max_length=64`** — `extra="forbid"` is
>   also a leakage guard: it rejects a raw Telco row carrying `Churn` rather
>   than silently accepting-then-dropping it.
> - **`requirements.txt` also pins `pydantic==2.12.4`** (Security notes) —
>   it's the entire input-validation boundary for this endpoint, previously
>   only an implicit transitive dependency of `fastapi`.

---

## PART 1 — SPEC

### Feature

A new `src/api/` FastAPI service exposing `POST /explain` (CLAUDE.md §10):
raw customer attributes in, SHAP + LIME top-3 churn drivers with
plain-English reasons out. The explainer context (SHAP `TreeExplainer` +
`LimeTabularExplainer` built against the persisted production model) is
built once when the app starts, not per request (CLAUDE.md §10). A minimal
`GET /health` route reports whether that startup build succeeded, so the
process can start and stay reachable even if model artifacts are missing.

### Problem / motivation

CLAUDE.md §1 promises churn predictions exposed "through a REST API," and
§10 contracts `POST /explain` explicitly. `11` built every piece of
explanation logic this route needs (`explain_customer`) but deliberately
stopped short of any route (its own Non-goals say so). Today there is
*no way* to get an explanation out of this system except by running a
notebook cell — a recruiter or retention manager cannot `curl` it, and no
other service (a future dashboard, a future `/predict` caller) has
anything to call. This spec closes that specific gap.

### Goals / non-goals

**Goals**
- Add `src/api/schemas.py`: Pydantic request/response models for
  `POST /explain` — a `CustomerPayload` covering every raw Telco column
  CLAUDE.md §6 documents except `Churn` (customerID optional), and an
  `ExplainResponse` mirroring `explain_customer`'s existing dict shape
  exactly (no reshaping).
- Add `src/api/main.py`: the `FastAPI` app, a `lifespan` startup hook that
  builds one explainer context via `build_explainer_context(load_clean_data())`
  and stores it on `app.state`, the `POST /explain` route (thin: validate →
  call `explain_customer` → return; no explanation logic reimplemented),
  and `GET /health`.
- Map `explain_customer`'s `ValueError`s (missing required column, unseen
  categorical value) to `422 Unprocessable Entity` with the existing,
  already-actionable message — not a bare `500`.
- Handle a missing/stale persisted model at startup without crashing the
  process: log the failure, leave the app reachable, serve `503 Service
  Unavailable` from `/explain` and `model_loaded: false` from `/health`
  until a real model is trained.
- Explicit, pinned versions for `fastapi`, `uvicorn`, and `httpx` in
  `requirements.txt` (first real usage of all three; `httpx` is currently
  an *implicit* transitive dependency required by `fastapi.testclient
  .TestClient` — same promotion pattern `06`/`11` used for `scipy`/`lime`).
- Add `tests/test_api.py` using `TestClient`.

**Non-goals**
- No `/predict`, `/batch-predict`, or `/recommend` route — CLAUDE.md §10's
  other three contracted endpoints are separate, unstarted work (`/predict`
  and `/batch-predict` would wrap `09`'s `scoring.py`/`10`'s
  `risk_tiers.py`; `/recommend` has no underlying Next-Best-Action module
  yet — Phase 4's second half, per CLAUDE.md §14, is also not done).
- No Streamlit "why" panel, no change to `app/dashboard.py` — this spec is
  API-only.
- No authentication/authorization, no rate limiting, no CORS configuration
  — no such requirement exists anywhere in CLAUDE.md and no auth
  infrastructure exists in the repo yet; flagged explicitly in Security
  notes as a known gap for a future spec, not silently assumed away.
- No caching of explainer contexts or results, no async/background
  computation — the verified ~0.25s per-request cost (Research note) does
  not need it.
- No Docker/deployment config (CLAUDE.md §14 Phase 6).
- No change to `src/explain/local_explainer.py`, `src/explain/
  driver_analysis.py`, `src/models/*`, or `src/recommend/risk_tiers.py` —
  this spec only *calls* `explain_customer`, never modifies it.
- No new row/status flip in CLAUDE.md §14 — Phase 5 stays `☐` (see Scope
  note above).

### User stories

- As a **retention manager or recruiter demoing the project**, I want to
  `curl -X POST http://localhost:8000/explain -d '{...customer...}'` and
  get back plain-English churn reasons, so the explainability work built
  in `11` is actually usable outside a notebook.
- As the **engineer (Priyabrata)**, I want the route to be a thin wrapper
  around `explain_customer` (CLAUDE.md §4: "keep the API thin"), so any
  future change to explanation logic (a new SHAP method, a different
  reason template) needs no route change.
- As the **engineer**, I want the server to start and serve `/health` even
  on a fresh clone with no trained model yet (`models/` is git-ignored per
  CLAUDE.md §7), so a missing artifact fails loudly and specifically at
  request time, not by crashing `uvicorn` on boot with an unreadable
  traceback.

### Functional requirements

1. `src/api/schemas.py` MUST define `CustomerPayload(BaseModel)` with one
   field per CLAUDE.md §6 raw column except `Churn`: `customerID:
   str | None = None`; `gender`, `Partner`, `Dependents`, `PhoneService`,
   `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`,
   `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`,
   `Contract`, `PaperlessBilling`, `PaymentMethod` all `str`; `SeniorCitizen:
   int` (`Field(ge=0, le=1)`, matching the raw CSV's native `0`/`1`
   encoding — `_clean_common_fields` already accepts this form); `tenure:
   int = Field(ge=0)`; `MonthlyCharges: float = Field(ge=0)`;
   `TotalCharges: float | None = None` (matches the known 11-blank-row
   quirk from CLAUDE.md §6: `None` means "not yet billed," normalized to
   `0.0` before calling `explain_customer`, mirroring `clean_data`'s own
   imputation — never left as `None` going into `explain_customer`).
2. `src/api/schemas.py` MUST define `ShapDriver` (`feature: str,
   customer_value: str | int | float, shap_value: float, direction: str,
   reason: str`), `LimeDriver` (`feature: str, lime_weight: float,
   direction: str, reason: str`), and `ExplainResponse` (`customerID: str |
   None, shap_top_drivers: list[ShapDriver], lime_top_drivers:
   list[LimeDriver]`) — field-for-field identical to `explain_customer`'s
   existing dict output (Requirement in `11`), no renaming or reshaping.
3. `src/api/main.py` MUST build `app = FastAPI(...)` with a `lifespan`
   async context manager (not the deprecated `@app.on_event("startup")`)
   that: calls `local_explainer.build_explainer_context(load_clean_data())`
   once; on success, stores the context on `app.state.explainer_context`
   and sets `app.state.startup_error = None`; on `FileNotFoundError`,
   `ValueError`, or `KeyError` (a missing/stale/non-tree-model artifact —
   the same exception surface `build_explainer_context` documents), logs
   the error server-side and sets `app.state.explainer_context = None`,
   `app.state.startup_error = <short, path-free message>` instead of
   raising — the process MUST still start.
4. MUST gain `GET /health` returning `{"status": "ok", "model_loaded":
   bool}` — `model_loaded = app.state.explainer_context is not None` — no
   dependency on any request body, `200` always.
5. MUST gain `POST /explain` accepting a `CustomerPayload` body, returning
   `ExplainResponse`: if `app.state.explainer_context is None`, raise
   `HTTPException(503, "Model is not available. ...")` before touching the
   payload; otherwise build a plain `dict` from the payload (`customerID`
   included only if not `None`, `TotalCharges` defaulted to `0.0` if
   `None`), call `explain_customer(customer, context=app.state
   .explainer_context)`, and return its result coerced into
   `ExplainResponse`. MUST catch `ValueError` raised by `explain_customer`
   (missing feature column, or a categorical value LIME never saw in
   training) and re-raise as `HTTPException(422, detail=str(exc))` — the
   existing message is already specific and safe to surface (verified: no
   absolute path in any `ValueError` message in `local_explainer.py`).
6. `src/api/main.py` MUST NOT import or duplicate any explanation, SHAP,
   LIME, scoring, or risk-tier logic — every request is validated
   (Pydantic) and dispatched to `src.explain.local_explainer` only,
   matching CLAUDE.md §4's "keep the API thin."
7. `tests/test_api.py` MUST use `fastapi.testclient.TestClient` against a
   real app instance (real persisted model, exactly like `11`'s `context`
   fixture — module-scoped, built once and reused, since a fresh
   `TestClient(app)` construction re-runs the `lifespan` startup and the
   ~1.9s context build) and cover: `GET /health` returns `200` with
   `model_loaded: true`; `POST /explain` on the same verified high-risk and
   low-risk customers `11` locked in returns `200` with the exact top-3
   SHAP feature names and directions `11`'s tests already verify (reused,
   not re-derived); a payload missing a required field returns `422`
   (Pydantic's own validation, before the route body even runs); a payload
   with an unseen categorical value (e.g. `Contract="Lifetime"`) returns
   `422` with a message naming `Contract`; a startup failure (monkeypatched
   `build_explainer_context` to raise `FileNotFoundError`) leaves `/health`
   returning `200` with `model_loaded: false` and `/explain` returning
   `503` — the process itself must not crash.
8. None of the above may change `src/explain/local_explainer.py`,
   `src/explain/driver_analysis.py`, `src/models/*`, `src/recommend/
   risk_tiers.py`, `app/dashboard.py`, or any existing test/figure/notebook
   — all current tests must keep passing unmodified.

### Data & model impact

None — this spec fits no model and reads no new data path. It only calls
`local_explainer.build_explainer_context`/`explain_customer` (`11`), which
already read `models/churn_model.pkl` + metadata and `load_clean_data()`.
No new column, artifact, or figure is produced.

### ML guardrails (mandatory check)

N/A — no model path affected. This spec adds an API layer only; every
guardrail relevant to leakage, splitting, SMOTE, or metric reporting
already applies inside `11`'s `build_explainer_context`/`explain_customer`
(unmodified here) and `08`'s training pipeline. The route never computes a
metric, never fits anything, and never derives a feature from `Churn` —
the request payload is customer attributes only, the same shape
`explain_customer` already validates.

### API / UI surface

New (this is the entire feature):

| Method | Path | Request | Response |
|---|---|---|---|
| GET | `/health` | — | `{"status": "ok", "model_loaded": bool}` |
| POST | `/explain` | `CustomerPayload` (JSON body, raw customer attributes) | `ExplainResponse` (`200`); `422` on invalid/unseen input; `503` if no model is loaded |

`/predict`, `/batch-predict`, `/recommend` remain unimplemented (Non-goals).
Interactive docs are FastAPI's free `/docs`/`/redoc`, no extra work needed.

### Edge cases & failure states

- **Fresh clone, no trained model yet** (`models/` is git-ignored per
  CLAUDE.md §7): the app starts (Requirement 3), `/health` reports
  `model_loaded: false`, `/explain` returns `503` with an actionable
  message ("run `python -m src.models.train` and `python -m
  src.models.calibration` first") rather than crashing `uvicorn` on boot —
  directly tested with a monkeypatched startup failure.
- **Payload missing a required field** (e.g. no `Contract`): Pydantic
  itself rejects the request with `422` before the route body runs — no
  custom handling needed, directly tested.
- **Payload has a categorical value never seen during training** (e.g.
  `Contract="Lifetime"`): passes Pydantic (it's a valid string) but
  `explain_customer`'s LIME path raises `ValueError` naming the column and
  value; the route converts it to `422` — directly tested.
- **`TotalCharges` omitted** (mirrors the raw dataset's 11 blank-string new
  customers, CLAUDE.md §6): defaults to `0.0` before calling
  `explain_customer`, matching `clean_data`'s own imputation rather than
  raising — directly tested.
- **Two customers explained back-to-back**: the second call MUST reuse
  `app.state.explainer_context` (`is` identity check, mirroring `11`'s own
  reuse test) — never rebuild the ~1.9s context per request.

### Security notes

- **First genuinely untrusted, network-reachable input surface in the
  repo.** Mitigated by: Pydantic type/range validation on every field
  before any of it reaches `explain_customer`; every unhandled-by-Pydantic
  input error (unseen category, missing feature) is caught and re-raised
  as a named `422` rather than an opaque `500`/traceback; no value from
  the request body is ever used to construct a file path, shell command,
  SQL query, or dynamic import — it flows only into `pandas`/`sklearn`/
  `shap`/`lime` calls already exercised by `11`'s test suite.
- **No authentication.** Explicitly out of scope (Non-goals) — flagged
  here as a real gap: this endpoint would be open to anyone who can reach
  it. Acceptable for a local/portfolio deployment target; a real
  deployment would need this addressed in a future spec before going
  public.
- **Startup-failure messages are path-free.** `app.state.startup_error`
  and the `503` response text are short, fixed strings — never the raw
  exception `str()`, which could otherwise include an absolute file path
  (e.g. `FileNotFoundError`'s default message embeds the missing path).
  The full exception is logged server-side via `logger.exception`, never
  returned to the client.
- **`joblib.load` on `models/churn_model.pkl`** carries the same
  self-produced-artifact trust boundary already documented in `08`'s/`09`'s/
  `11`'s Security notes — this spec adds no new load path, only a new
  network entry point in front of the existing one.
- **New pinned dependencies:** `fastapi`, `uvicorn`, `httpx` move from
  unpinned/implicit to explicit pinned versions in `requirements.txt`
  (verified installed versions: `fastapi==0.138.1`, `uvicorn==0.49.0`,
  `httpx==0.28.1`) — no new package family added, `fastapi`/`uvicorn` were
  already listed unpinned since `01`, `httpx` was already an implicit
  transitive dependency required by `TestClient`.

### Success criteria

- `pytest -q` passes: all existing tests + `tests/test_api.py`, all green.
- `uvicorn src.api.main:app --reload --port 8000` starts successfully both
  with and without trained model artifacts present.
- `POST /explain` on the two customers verified in `11` reproduces their
  exact top-3 SHAP feature names/directions through the live HTTP layer,
  not just the underlying function.
- A malformed or out-of-vocabulary request returns `422` with an
  actionable message; a missing model returns `503`; neither returns a
  bare `500`.
- `quality-reviewer` and `security-reviewer` report no unresolved findings
  on the diff.

### Out of scope

- `/predict`, `/batch-predict`, `/recommend` routes.
- Any Streamlit view or `app/dashboard.py` change.
- Authentication, rate limiting, CORS.
- Caching, async/background explanation computation.
- Docker/deployment (CLAUDE.md §14 Phase 6).
- Flipping CLAUDE.md §14's Phase 5 tracker row.

---

## PART 2 — PLAN

### Approach

Two new files: `src/api/schemas.py` (Pydantic request/response models) and
`src/api/main.py` (the `FastAPI` app, `lifespan` startup, two routes) —
mirroring how `06`/`09`/`10`/`11` each added exactly one new module calling
into the previous phase's functions without modifying them. The explainer
context is built once in `lifespan` and stashed on `app.state`, matching
the verified ~1.9s build / ~0.25s reuse cost (Research note) and CLAUDE.md
§10's explicit "load the model once at startup" instruction.

**Alternative rejected:** build the explainer context lazily on the first
`/explain` request (via a cached dependency) instead of at startup.
Rejected because CLAUDE.md §10 states the load-once-at-startup requirement
explicitly, and a lazy approach would make the *first* real request pay
the ~1.9s cost unpredictably instead of `/health` surfacing readiness up
front — worse for both a demo and a real deployment's readiness-probe
semantics.

**Alternative rejected:** let a missing model raise straight out of
`lifespan`, crashing the app on a fresh clone. Rejected because CLAUDE.md
§7's git-ignore rule for `models/` means a fresh clone *always* starts
without a trained model until the operator runs `train.py`/`calibration
.py` — crashing on boot would make `/health` itself unreachable, which is
worse than a `503` on the one route that actually needs the model.

### Task breakdown

- [ ] **1. Pin `fastapi`, `uvicorn[standard]`, add `httpx`, in
      `requirements.txt`** — `fastapi==0.138.1`, `uvicorn[standard]==0.49.0`,
      `httpx==0.28.1` (verified installed versions).
- [ ] **2. Create `src/api/schemas.py`** — `CustomerPayload`, `ShapDriver`,
      `LimeDriver`, `ExplainResponse` (Requirements 1–2).
- [ ] **3. Create `src/api/main.py`** — `FastAPI` app, `lifespan` startup
      hook, `GET /health`, `POST /explain` (Requirements 3–6). Import
      `build_explainer_context`/`explain_customer` from
      `src.explain.local_explainer`, `load_clean_data` from
      `src.data.load_data`.
- [ ] **4. Add `tests/test_api.py`** — see Tests to write below.
- [ ] **5. Manual smoke test** — run `uvicorn src.api.main:app --port
      8000`, `curl` `/health` and `/explain` with a real customer payload
      (and once more after temporarily renaming `models/` to confirm the
      `503`/`model_loaded: false` path), confirm `/docs` renders.
- [ ] **6. Run the full suite** — `pytest -q`, confirm all existing + new
      tests pass.
- [ ] **7. Commit** — `src/api/schemas.py`, `src/api/main.py`,
      `tests/test_api.py`, `requirements.txt`, commit message `feat: POST
      /explain FastAPI endpoint`.

### Tests to write (hand to test-writer)

- `tests/test_api.py::test_health_returns_ok_and_model_loaded_true` —
  `GET /health` → `200`, `{"status": "ok", "model_loaded": True}`, against
  the real app with real model artifacts present.
- `tests/test_api.py::test_explain_matches_verified_high_risk_example` —
  `POST /explain` with the exact verified high-risk customer attributes
  (reused from `11`'s locked-in example) → `200`, `shap_top_drivers`' top-3
  feature names are `tenure`/`Contract`/`InternetService` (any order),
  all `direction == "increases"`.
- `tests/test_api.py::test_explain_matches_verified_low_risk_example` —
  same, for the verified low-risk customer → top-3 =
  `Contract`/`tenure`/`OnlineSecurity`, all `direction == "decreases"`,
  plus a `lime_top_drivers` entry present for `Contract` and `tenure`.
- `tests/test_api.py::test_explain_missing_required_field_returns_422` — a
  payload omitting `Contract` → `422` (Pydantic-level, before the route
  body runs).
- `tests/test_api.py::test_explain_unseen_category_returns_422` — a
  payload with `Contract="Lifetime"` → `422`, detail mentions `Contract`.
- `tests/test_api.py::test_explain_omitted_total_charges_defaults_to_zero` —
  a payload with `TotalCharges` omitted (new/zero-tenure customer, CLAUDE.md
  §6) → `200`, no exception.
- `tests/test_api.py::test_explain_reuses_context_across_requests` — two
  sequential `POST /explain` calls; `app.state.explainer_context` identity
  (captured via the app instance) is unchanged between them.
- `tests/test_api.py::test_startup_failure_leaves_health_reachable_and_explain_503` —
  monkeypatch `build_explainer_context` to raise `FileNotFoundError` before
  constructing a fresh `TestClient` (forcing `lifespan` to hit the failure
  path) → `GET /health` returns `200` with `model_loaded: false`; `POST
  /explain` (any payload) returns `503`; no exception escapes either call.
- `tests/test_api.py::test_health_and_explain_error_messages_never_leak_paths` —
  every error-path response body (`422`/`503`) is asserted not to contain
  any local filesystem path fragment (e.g. the repo's absolute root path).

### Quality gates

1. **test-runner** — run `pytest -q`, confirm all tests (existing + new)
   pass. Flag the module-scoped `TestClient`/context-build cost (~1.9s
   once) if it noticeably slows the suite, alongside `11`'s already-flagged
   LIME cost.
2. **quality-reviewer** — review the `lifespan` startup-failure handling
   (process must never crash on a missing model), the `ValueError`→`422`
   and startup-failure→`503` mapping, that `main.py` truly reimplements no
   explanation logic (CLAUDE.md §4), and CLAUDE.md §8 adherence (type
   hints, docstrings, no magic status codes — use `fastapi.status`
   constants).
3. **security-reviewer** — confirm the untrusted-input surface (Pydantic
   validation, no path/shell/SQL construction from request data), confirm
   no path leakage in error responses, confirm the no-auth gap is
   explicitly documented rather than silently absent, confirm the
   `fastapi`/`uvicorn`/`httpx` pins are the only dependency-surface change.
4. Commit only after both reviews are clean.

### Risks / rollback

- **Risk:** a reviewer expects `/explain` to return SHAP only, per CLAUDE.md
  §10's literal one-line contract ("Customer → SHAP top-3 churn drivers").
  **Mitigation:** `explain_customer` (`11`) already returns both SHAP and
  LIME by design — CLAUDE.md §7 mandates LIME as the alternative local
  view, and this spec's `ExplainResponse` deliberately mirrors that
  existing output verbatim rather than dropping half of it to match the
  contract table's shorthand wording.
- **Risk:** the `lifespan` context build (~1.9s) noticeably slows
  `pytest -q` if every test function creates its own `TestClient`.
  **Mitigation:** Requirement 7 mandates a module-scoped fixture (mirroring
  `11`'s `context` fixture) so the app/context is built once for the whole
  test module, not once per test.
- **Risk:** a future retrain (`python -m src.models.train`) selects a
  non-tree model, which `build_explainer_context` already rejects with a
  `ValueError` (`11`'s `TREE_BASED_MODELS` guard) — under this spec's
  design that surfaces as a normal startup failure (`503`/`model_loaded:
  false`), not a crash. **Mitigation:** this is the intended, tested
  behavior, not a gap — no separate handling needed.
- **Rollback:** single commit (Task 7) covering only additive files (two
  new `src/api/` modules, one new test file, one `requirements.txt`
  change) — `git revert` is clean since nothing existing is modified in
  place.

### Definition of done

- All 7 tasks checked off.
- `pytest -q` green (all existing tests + `tests/test_api.py`).
- `uvicorn src.api.main:app --reload --port 8000` verified to start and
  serve both routes correctly, with and without model artifacts present
  (manual smoke test, Task 5).
- `quality-reviewer` and `security-reviewer` report no unresolved findings.
- All Success Criteria in Part 1 are met.
