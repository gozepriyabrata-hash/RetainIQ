# Spec + Plan: Retention Action Engine — Rule-Based + ML-Driven Next-Best-Action

> Location note: this file lives at `.claude/specs/` per CLAUDE.md §4's repo
> tree ("`.claude` — claude.md and also Specs folder"), consistent with
> `01`–`12`. Spec and plan are combined in one file for the same reason,
> following `10-risk-classification.md`'s precedent for overriding the
> generic `/create-spec` command's `specs/<slug>/spec.md`+`plan.md` template
> (the command's own workflow step 3 says "if the feature would conflict
> with a guardrail, flag the conflict in the spec rather than silently
> overriding" — flagged here). Numbered `13` (next after `12-explain-
> endpoint.md`/`12-model-comparison-leaderboard.md`, both `12` since they
> shipped as parallel PRs).
>
> Scope note: this is the **second and final half of CLAUDE.md §14 Phase 4**
> ("Risk tiers + Next-Best-Action engine, optional LLM"). `10-risk-
> classification.md` shipped the first half (Critical/High/Medium/Low
> tiers, `src/recommend/risk_tiers.py`) and explicitly deferred "the Next-
> Best-Action engine (recommending what to do about a tier)" as a separate
> spec — this is that spec. Per the feature request's own wording ("Rule-
> based + ML-driven recommended action per risk tier"), the **optional LLM**
> half of Phase 4 is explicitly **not** included here (see Non-goals) — the
> Phase 4 tracker row stays `☐` until a later spec covers it.
>
> Research note: the engine's rule table (Functional Requirement 3) mirrors
> CLAUDE.md §6's own "Sanity-check the model against these real signals"
> table verbatim (Contract, tenure, Tech support, Payment method, Internet)
> — no invented lever. Verified against the real, currently-persisted
> production model (`models/churn_model.pkl`, raw — same model
> `src.explain.local_explainer` explains) and calibrated model
> (`models/churn_model_calibrated.pkl`) during spec research, on two real
> customers:
> - **`5178-LMXOP`** (also `test_local_explainer.py`'s `HIGH_RISK_TEST_INDEX`
>   fixture): calibrated `churn_probability = 1.0` → tier **Critical**.
>   Real top-3 SHAP drivers, all `direction: "increases"`: `tenure=1`
>   (`+0.7781`), `Contract="Month-to-month"` (`+0.5849`),
>   `InternetService="Fiber optic"` (`+0.3072`). Hand-applying this spec's
>   Requirement 4 algorithm (`top_n=3`): tier base action (Critical →
>   escalation, 1 slot) + `tenure` rule (onboarding) + `Contract` rule
>   (contract upgrade) — 3 slots filled before the `InternetService` rule is
>   reached, so it does **not** appear in this customer's list. This is the
>   expected, specified behavior (Requirement 4's slot cap), not a bug.
> - **`9763-GRSKD`**: calibrated `churn_probability = 0.1677` → tier **Low**.
>   Real top-3 SHAP drivers: `Contract="Month-to-month"` (`+0.5188`,
>   increases), `InternetService="DSL"` (`-0.4439`, decreases),
>   `OnlineSecurity="Yes"` (`-0.4225`, decreases). Hand-applying Requirement
>   4: tier base action (Low → monitor) + `Contract` rule (contract
>   upgrade) — the two `direction: "decreases"` drivers are protective
>   factors and correctly produce no action (Requirement 4's direction
>   filter), so only **2** actions are returned even though `top_n=3` — a
>   real example of "fewer than `top_n` when fewer rules match," not padded
>   with an unrelated generic action. This is also the case CLAUDE.md's
>   golden rule against reporting a false sense of coverage motivates:
>   Low-tier customers still get a targeted action when a real risk-
>   increasing driver is present, and a bare "no action" when none is.
>
> Both are real, verified model outputs (not fabricated) — only the
> resulting *action list* is hand-computed here, since the module doesn't
> exist yet; `tests/test_action_engine.py` locks in both examples exactly
> (Tests to write, below).

---

## PART 1 — SPEC

### Feature

A Next-Best-Action engine: given a customer's risk tier (`10`) and their
top-3 SHAP churn drivers (`11`), return a ranked list of concrete retention
actions — one tier-appropriate base action plus up to `top_n - 1`
driver-specific actions, each driver-specific action triggered only by a
real risk-increasing driver matching one of CLAUDE.md §6's five documented
churn-driver → retention-lever pairs. "Rule-based" = the tier→action and
driver→action lookup tables; "ML-driven" = which specific driver-actions
fire and in what order is entirely determined by the trained model's own
per-customer SHAP output, not a static list.

### Problem / motivation

`10` sorts customers into four tiers; `11` explains *why* a customer is at
risk. Neither tells a retention manager *what to do about it*. CLAUDE.md
§6's driver table already names five levers ("Incentivize longer
contracts," "Strengthen onboarding," "Offer support add-on," "Nudge to
auto-pay," "Proactive quality outreach") but nothing in the repo turns them
into an actual per-customer recommendation — `src/recommend/` currently
holds only `risk_tiers.py`. This spec is that missing translation layer,
and the direct prerequisite for CLAUDE.md §10's planned `POST /recommend`
endpoint and the dashboard's "next-best-action recommendations" (§14
Definition of done).

### Goals / non-goals

**Goals**
- Add `src/recommend/action_engine.py`: named tier→action and
  driver→action lookup tables, a pure rule function
  (`recommend_actions`) operating on an already-known tier + already-
  computed SHAP drivers (no model I/O, fully unit-testable), and a thin
  raw-customer-in composition (`recommend_actions_for_customer`) wrapping
  `10`'s `classify_scored_customers` and `11`'s `explain_customer`.
- Add `notebooks/11_retention_action_engine.ipynb` following `01`–`10`'s
  bootstrap-cell pattern, demonstrating the engine on real customers
  including the two worked examples above.
- Add `tests/test_action_engine.py`.

**Non-goals**
- **No optional-LLM insight generation** — CLAUDE.md §14 Phase 4's other
  half. The feature request explicitly scoped this to "rule-based +
  ML-driven"; an LLM-phrased recommendation narrative is a separate,
  later spec (would read `ANTHROPIC_API_KEY`/`OPENAI_API_KEY` per CLAUDE.md
  §12, degrade gracefully if unset — out of scope here entirely, not even
  stubbed).
- **No FastAPI endpoint, no Streamlit view, no model loaded at app
  startup** — Phase 5, not this spec, exactly `10`'s and `11`'s own
  precedent. `src/api/` and `app/` are untouched.
  `recommend_actions_for_customer` is added as the function a future
  `POST /recommend` route will call, wired to nothing yet.
- **No rule for any driver outside CLAUDE.md §6's five-signal table**
  (e.g. `gender`, `Partner`, `Dependents`, `StreamingTV`,
  `StreamingMovies`, `DeviceProtection`, `OnlineBackup`, `MultipleLines`,
  `PaperlessBilling`, `SeniorCitizen`, `MonthlyCharges`, `TotalCharges`).
  If SHAP surfaces one of these as a top-3 driver, no rule fires for it —
  it is silently skipped, never given an invented action. Closed,
  testable catalog over guessed coverage.
- **No batch/DataFrame version.** `10`'s `assign_risk_tiers`/`risk_tier_
  summary` already cover batch tier reporting; CLAUDE.md §10's `/recommend`
  contract is single-customer (like `/explain`), so this spec matches that
  shape. A caller wanting many customers' actions calls
  `recommend_actions_for_customer` (or the pure `recommend_actions`) in a
  loop.
- **No configurability of the rule tables** (env var, YAML, per-deployment
  override) — the five driver rules and four tier-base actions are fixed
  module constants, matching `10`'s fixed-threshold precedent.
- **No change to `src/recommend/risk_tiers.py`, `src/explain/
  local_explainer.py`, `src/models/scoring.py`, `src/models/calibration.py`,
  `src/models/train.py`, or any existing test/notebook/figure** — this
  spec only calls their existing public functions.
- No new third-party dependency.

### User stories

- As a **retention manager**, I want a ranked, concrete action list per
  customer (not just a risk score) so I know exactly what to do — call,
  offer, or nudge — without re-deriving it from a raw driver list myself.
- As the **engineer (Priyabrata)**, I want one function
  (`recommend_actions_for_customer`) that goes from raw customer
  attributes straight to `(risk_tier, ranked actions)`, so Phase 5's
  `/recommend` endpoint is a thin wrapper (CLAUDE.md §4) around already-
  tested logic, exactly the `classify_scored_customers`/`explain_customer`
  precedent.
- As the **engineer**, I want the driver→action rules to fire only on
  genuinely risk-increasing drivers with a value in the documented "bad"
  state (e.g. `Contract == "Month-to-month"`, not any `Contract` value),
  so a customer whose month-to-month-adjacent driver is actually
  protective (e.g. `Contract == "Two year"`, `direction: "decreases"`)
  never gets a nonsensical "upgrade your contract" recommendation.
- As a **recruiter/reviewer**, I want the engine's output traceable to a
  real SHAP driver on a real customer (not a black-box label), so each
  recommendation reads as explainable, not arbitrary.

### Functional requirements

1. `src/recommend/action_engine.py` MUST define `TOP_N_ACTIONS = 3` and
   `EARLY_TENURE_THRESHOLD_MONTHS = 12` (reuses `src/data/cohorts.py`'s
   own `"0-12"`-band boundary — CLAUDE.md §8's "no magic numbers," and no
   independently-chosen threshold that could silently drift from the
   cohort analysis's).
2. MUST define `TIER_BASE_ACTIONS: dict[str, dict]` keyed by `10`'s
   `RISK_TIER_LABELS` (imported from `risk_tiers`, never re-declared),
   each value `{"action": str, "category": str}`:
   - `"Critical"` → `{"action": "Escalate to a retention specialist for a
     personal outreach call within 24 hours.", "category": "escalation"}`
   - `"High"` → `{"action": "Proactively offer a loyalty discount or
     service credit.", "category": "retention_offer"}`
   - `"Medium"` → `{"action": "Send a targeted engagement email
     highlighting underused benefits.", "category": "engagement"}`
   - `"Low"` → `{"action": "No immediate action needed; continue standard
     engagement monitoring.", "category": "monitor"}`
3. MUST define `DRIVER_ACTION_RULES: list[dict]`, one entry per CLAUDE.md
   §6 row, each `{"feature": str, "condition": Callable[[object], bool],
   "action": str, "category": str}`, in this exact order (also each
   feature's tie-break order if more than one ever matched, though today
   each `feature` appears exactly once):
   1. `feature="Contract"`, `condition=lambda v: v == "Month-to-month"`,
      action offering an incentive to upgrade to a 1- or 2-year contract,
      `category="contract"`.
   2. `feature="tenure"`, `condition=lambda v: float(v) <=
      EARLY_TENURE_THRESHOLD_MONTHS`, action enrolling the customer in
      proactive early-tenure onboarding, `category="onboarding"`.
   3. `feature="TechSupport"`, `condition=lambda v: v == "No"`, action
      offering a free/discounted Tech Support add-on, `category="support"`.
   4. `feature="PaymentMethod"`, `condition=lambda v: v == "Electronic
      check"`, action nudging to automatic payment, `category="payment"`.
   5. `feature="InternetService"`, `condition=lambda v: v == "Fiber
      optic"`, action scheduling proactive service-quality outreach,
      `category="service_quality"`.
4. MUST gain `recommend_actions(risk_tier: str, shap_drivers: list[dict],
   top_n: int = TOP_N_ACTIONS) -> list[dict]`. `shap_drivers` is the exact
   shape `local_explainer.local_shap_top_drivers`/
   `explain_customer()["shap_top_drivers"]` returns (list of `{"feature",
   "customer_value", "shap_value", "direction", "reason"}`, already ranked
   by `|shap_value|` descending — **not** `lime_top_drivers`, whose
   `"feature"` is a condition string like `"Contract=Two year"`, not a bare
   column name, and is therefore incompatible with `DRIVER_ACTION_RULES`).
   Algorithm:
   1. Raise `ValueError` naming `risk_tier` if it is not a key of
      `TIER_BASE_ACTIONS`. Raise `ValueError` if `top_n < 1`.
   2. Result starts with exactly one entry: the tier's base action,
      `{"priority": 1, "action": ..., "category": ..., "rationale": f"Customer
      is in the {risk_tier} risk tier.", "source": "tier", "driver_feature":
      None}`.
   3. Iterate `shap_drivers` in the given (already-ranked) order. For each
      driver: stop entirely once `len(result) >= top_n`. Skip if
      `driver["direction"] != "increases"` (a protective driver never
      produces an action). Skip if no rule in `DRIVER_ACTION_RULES` has a
      matching `feature` whose `condition(driver["customer_value"])` is
      `True`. Skip if the matched rule's `action` text is already present
      in `result` (dedup — two different drivers could in principle match
      the same rule text only if `DRIVER_ACTION_RULES` grows non-unique
      features in the future; defensive now). Otherwise append `{"priority":
      len(result) + 1, "action": rule["action"], "category": rule
      ["category"], "rationale": driver["reason"], "source": "driver",
      "driver_feature": driver["feature"]}`.
   4. Return `result` (length between 1 and `top_n`, inclusive — never
      padded to exactly `top_n` if fewer rules match).
5. MUST gain `recommend_actions_for_customer(customer: dict,
   explainer_context: dict | None = None, top_n: int = TOP_N_ACTIONS) ->
   dict`: `risk_tiers.classify_scored_customers(pd.DataFrame([customer]))`
   for `(churn_probability, churn_probability_pct, risk_tier)` +
   `local_explainer.explain_customer(customer, context=explainer_context)`
   for `shap_top_drivers`, piped into `recommend_actions`. Returns
   `{"customerID": ..., "churn_probability": float, "churn_probability_pct":
   float, "risk_tier": str, "actions": list[dict]}` (`customerID` omitted
   if absent from `customer`, matching `explain_customer`'s own
   `exclude_none`-style precedent). Building `explainer_context` when
   `None` is expensive (`11`'s own docstring: ~1.9s) — a caller processing
   more than one customer must build one via
   `local_explainer.build_explainer_context` and pass it in, exactly `11`'s
   own stated contract for `explain_customer`.
6. `notebooks/11_retention_action_engine.ipynb` MUST follow `01`–`10`'s
   bootstrap-cell pattern. Sections, in order: problem framing (cite
   CLAUDE.md §6's five-lever table) → build one shared explainer context →
   run `recommend_actions_for_customer` on the two worked-example
   customers above (`5178-LMXOP`, `9763-GRSKD`), confirming the actions
   match this spec's Research note exactly → a small sample (~5) of
   additional real customers spanning all four tiers → a tier-vs-action-
   category cross-tab sanity check (every Critical/High customer in the
   sample gets a non-monitor action if any risk-increasing rule matched) →
   key findings closing cell.
7. `tests/test_action_engine.py` MUST cover the Plan's "Tests to write"
   section in full.
8. None of the above may change `src/recommend/risk_tiers.py`,
   `src/explain/local_explainer.py`, `src/explain/driver_analysis.py`,
   `src/models/scoring.py`, `src/models/calibration.py`,
   `src/models/train.py`, `src/data/*`, `src/api/`, `app/`, or any
   existing test/figure/notebook — all current tests must keep passing
   unmodified.

### Data & model impact

No new model, no new training feature, no schema change to any existing
artifact. The engine reads two existing model *outputs* — `10`'s
`risk_tier` and `11`'s `shap_top_drivers` — and returns a business-logic
action list; nothing here is written back into `load_clean_data()`'s
output, any training DataFrame, or any feature the model consumes.

### ML guardrails (mandatory check)

- **No target/probability leakage:** `recommend_actions` consumes only a
  `risk_tier` string and a list of driver dicts, both already-computed
  model *outputs* (never the raw `Churn` label). No function in this
  module is reachable from `src/features/preprocessing.py` or
  `src/models/train.py` — same one-way, output-only data flow `10`
  established.
- **Honest-AUC guard is unaffected:** no new model, no training/evaluation
  code path touched.
- **Fitting/splitting/SMOTE:** N/A — `recommend_actions` is pure Python
  (no scikit-learn dependency); `recommend_actions_for_customer` calls
  `10`/`11`'s already-fit pipelines unmodified, fitting nothing itself.
- **Imbalance/metric reporting:** N/A — this spec adds no classifier and
  reports no accuracy/AUC; its own correctness is verified against real,
  hand-traced SHAP driver output (Research note), not a metric.
- **Reproducibility:** deterministic given identical `(risk_tier,
  shap_drivers)` input — `DRIVER_ACTION_RULES` iteration order and the
  dedup/cap logic have no randomness. `recommend_actions_for_customer`
  inherits whatever determinism `11`'s `local_shap_top_drivers` already
  has (exact and reproducible; only `lime_top_drivers`, not used here, has
  LIME's known per-call RNG variance per `11`'s own docstring).

### API / UI surface

None shipped. `recommend_actions_for_customer` (Requirement 5) is added as
the exact function a future Phase 5 `POST /recommend` route (CLAUDE.md
§10) will call — raw customer dict in, ranked action list out. No FastAPI
route or Streamlit view is wired up; `src/api/` and `app/` are untouched.

### Edge cases & failure states

- **`risk_tier` not one of the four labels:** `recommend_actions` raises
  `ValueError` naming it — directly tested.
- **`top_n < 1`:** `recommend_actions` raises `ValueError` — directly
  tested.
- **Empty `shap_drivers`:** returns exactly the one-entry tier-base-action
  list — directly tested.
- **No driver matches any rule** (e.g. all top-3 drivers are outside the
  five-signal table, or all are `direction: "decreases"`): same, list
  length 1 — directly tested with the real `9763-GRSKD` two-decreasing-
  driver case (Research note) plus a synthetic all-outside-table case.
- **Fewer than `top_n` actions available:** list is short, never padded
  with a duplicate or unrelated action — directly tested with the real
  `9763-GRSKD` case (2 of 3 slots filled).
- **Duplicate action text across two drivers:** second one is skipped, not
  double-listed — directly tested with a synthetic case (two hand-built
  drivers both routing to the same rule).
- **`recommend_actions_for_customer` inherits every `10`/`11` edge case**
  (empty/malformed customer dict, missing required feature column, unseen
  categorical value, missing model artifact, non-tree-based persisted
  model) unchanged, since it is a thin composition — directly tested that
  these propagate rather than being swallowed or altered.

### Security notes

- **No new dependency**, no new untrusted-input surface beyond what `10`
  and `11` already accept and validate: `recommend_actions_for_customer`'s
  `customer` dict is the same untrusted-customer-row surface
  `explain_customer`/`classify_scored_customers` already document and
  mitigate (missing-column validation, unseen-category `ValueError`, no
  dynamic code execution) — unchanged here, just reused. `DRIVER_ACTION_
  RULES`' `condition` lambdas do plain `==`/`<=` comparisons against
  already-typed, already-validated customer field values — no `eval`,
  no string formatting of untrusted input into a template that isn't
  already `f"..."`-safe (action/rationale strings interpolate only
  `risk_tier` and `driver["reason"]`, both already-validated/derived
  values, never a raw unsanitized request field).
- No secrets, no new environment variables, no network call, no file I/O.

### Success criteria

- `pytest -q` passes: all existing tests + `test_action_engine.py`, all
  green.
- `recommend_actions` reproduces this spec's two hand-computed worked
  examples (`5178-LMXOP` → 3 actions ending at `Contract`, `InternetService`
  rule never reached; `9763-GRSKD` → 2 actions, both `decreases` drivers
  correctly produce no action) exactly, driven by the real SHAP driver
  data captured in the Research note.
- `recommend_actions_for_customer` runs end-to-end on a real customer and
  returns the documented dict shape.
- `notebooks/11_retention_action_engine.ipynb` runs top-to-bottom without
  error, including the two worked-example confirmations.
- `quality-reviewer` and `security-reviewer` report no unresolved findings
  on the diff.

### Out of scope

- Optional LLM-generated recommendation narrative — CLAUDE.md §14 Phase
  4's other half, a separate spec.
- Phase 5 (`POST /recommend`, Streamlit action-list view), Phase 6
  (Evidently drift, Prefect retraining, Docker).
- Batch/DataFrame recommendation function, configurable rule tables.
- Any change to `src/recommend/risk_tiers.py`'s or
  `src/explain/local_explainer.py`'s public behavior, output, or test.

---

## PART 2 — PLAN

### Approach

One new module (`src/recommend/action_engine.py`) split into a pure,
model-free rule function (`recommend_actions`, the fully unit-testable
core) and a thin composition (`recommend_actions_for_customer`) that
chains `10`'s tier classification and `11`'s SHAP explanation into it —
mirroring `risk_tiers.py`'s own `classify_risk_tier`
(pure)/`classify_scored_customers` (composition) split, so the expensive
model/explainer path is exercised only where genuinely needed.

**Alternative rejected:** score a numeric "priority weight" per rule (e.g.
sum of the CLAUDE.md §6 real-world churn-rate deltas) and re-sort all
candidate actions by that weight, rather than trusting the driver list's
existing SHAP-magnitude order. Rejected because it would require
maintaining a second, independently-tuned weight table that could drift
from CLAUDE.md §6's actual verified numbers, and because the SHAP
magnitude *already* reflects this specific customer's actual driver
strength (not a fixed table's population-average one) — reusing the
driver order the model itself produced is more faithful to "ML-driven"
than re-deriving a static priority score.

### Task breakdown

- [ ] **1. Create `src/recommend/action_engine.py`** — `TOP_N_ACTIONS`,
      `EARLY_TENURE_THRESHOLD_MONTHS`, `TIER_BASE_ACTIONS`,
      `DRIVER_ACTION_RULES`, `recommend_actions`,
      `recommend_actions_for_customer` (Requirements 1–5).
- [ ] **2. Create `notebooks/11_retention_action_engine.ipynb`** —
      bootstrap cell copied from `10_explainability.ipynb`; sections per
      Functional Requirement 6.
- [ ] **3. Add `tests/test_action_engine.py`** — see Tests to write below.
- [ ] **4. Run the full suite** — `pytest -q`, confirm all existing + new
      tests pass.
- [ ] **5. Commit** — `src/recommend/action_engine.py`,
      `notebooks/11_retention_action_engine.ipynb`,
      `tests/test_action_engine.py`, commit message `phase 4: retention
      action engine -- rule-based + ML-driven next-best-action`. No files
      under `data/`, `models/`, or `mlruns/` are touched by this spec.

### Tests to write (hand to test-writer)

- `tests/test_action_engine.py::test_recommend_actions_raises_on_unknown_tier` —
  `risk_tier="Extreme"` → `ValueError` naming it.
- `tests/test_action_engine.py::test_recommend_actions_raises_on_invalid_top_n` —
  `top_n=0` → `ValueError`.
- `tests/test_action_engine.py::test_recommend_actions_empty_drivers_returns_only_tier_base` —
  each of the four tiers with `shap_drivers=[]` → one-entry list matching
  that tier's `TIER_BASE_ACTIONS` entry exactly.
- `tests/test_action_engine.py::test_recommend_actions_skips_decreasing_drivers` —
  a hand-built driver list where every entry has a rule-matching
  `feature`/`customer_value` but `direction="decreases"` → result length 1
  (tier base only).
- `tests/test_action_engine.py::test_recommend_actions_skips_drivers_outside_rule_table` —
  hand-built `direction="increases"` drivers on features with no rule
  (e.g. `gender`, `StreamingTV`) → result length 1.
- `tests/test_action_engine.py::test_recommend_actions_each_rule_fires_on_matching_value` —
  parametrized over all 5 `DRIVER_ACTION_RULES` entries: a single
  `direction="increases"` driver with the exact triggering value (e.g.
  `Contract="Month-to-month"`, `tenure=12`, `tenure=1`,
  `TechSupport="No"`, `PaymentMethod="Electronic check"`,
  `InternetService="Fiber optic"`) → result length 2, second entry's
  `category` matches that rule's.
- `tests/test_action_engine.py::test_recommend_actions_does_not_fire_on_non_triggering_value` —
  same features with a non-triggering value (e.g. `Contract="Two year"`,
  `tenure=13`) but `direction="increases"` → result length 1 (no rule
  matches, since `condition` is `False`).
- `tests/test_action_engine.py::test_recommend_actions_respects_top_n_cap` —
  3 driver dicts all rule-matching + increasing, `top_n=2` → result length
  2 (tier base + exactly 1 driver action), third driver never reached.
- `tests/test_action_engine.py::test_recommend_actions_dedups_repeated_action_text` —
  two driver dicts that both resolve to the same rule's action text (e.g.
  two synthetic `"Contract"` entries) → the second is skipped, not
  double-listed.
- `tests/test_action_engine.py::test_recommend_actions_matches_worked_example_critical` —
  hardcodes the real `5178-LMXOP` SHAP driver data from this spec's
  Research note (`tenure=1`/`+0.7781`/increases,
  `Contract="Month-to-month"`/`+0.5849`/increases,
  `InternetService="Fiber optic"`/`+0.3072`/increases),
  `risk_tier="Critical"`, `top_n=3` → exactly 3 actions: tier base
  (escalation), `tenure` rule (onboarding), `Contract` rule (contract);
  `InternetService` rule absent.
- `tests/test_action_engine.py::test_recommend_actions_matches_worked_example_low` —
  hardcodes the real `9763-GRSKD` SHAP driver data (`Contract="Month-to-
  month"`/`+0.5188`/increases, `InternetService="DSL"`/`-0.4439`/decreases,
  `OnlineSecurity="Yes"`/`-0.4225`/decreases), `risk_tier="Low"`, `top_n=3`
  → exactly 2 actions: tier base (monitor), `Contract` rule (contract).
- `tests/test_action_engine.py::test_recommend_actions_for_customer_returns_expected_shape` —
  a real raw customer row (e.g. `9763-GRSKD`'s own attributes) through
  `recommend_actions_for_customer` with a real shared `explainer_context`
  → result has `customerID`, `churn_probability`, `churn_probability_pct`,
  `risk_tier`, `actions`, and `actions` matches calling `recommend_actions`
  directly on the same customer's independently-computed tier + SHAP
  drivers.
- `tests/test_action_engine.py::test_recommend_actions_for_customer_propagates_scoring_errors` —
  a `customer` dict missing a required feature column → the same
  `ValueError` `classify_scored_customers`/`explain_customer` would raise,
  unaltered.
- `tests/test_action_engine.py::test_tier_base_actions_cover_all_risk_tier_labels` —
  `set(TIER_BASE_ACTIONS) == set(risk_tiers.RISK_TIER_LABELS)` (guards
  against the two constant sets silently drifting apart).

### Quality gates

1. **test-runner** — run `pytest -q`, confirm all tests (existing + new)
   pass, fix any straightforward regression.
2. **quality-reviewer** — review `DRIVER_ACTION_RULES`' condition logic
   for exact agreement with CLAUDE.md §6's five-signal table, the
   `direction == "increases"` filter, the dedup/cap algorithm against the
   two worked examples, and CLAUDE.md §8 adherence (named constants, type
   hints, docstrings).
3. **security-reviewer** — confirm no new dependency, confirm the
   `condition` lambdas do no dynamic evaluation of untrusted input, confirm
   `recommend_actions_for_customer` introduces no new untrusted-input
   handling beyond what `10`/`11` already validate.
4. Commit only after both reviews are clean.

### Risks / rollback

- **Risk:** a future retrain shifts which drivers surface in a given
  customer's top-3 SHAP list, so the two worked-example tests
  (`5178-LMXOP`/`9763-GRSKD`) could fail if run against
  `recommend_actions_for_customer` end-to-end after a model change (they
  hardcode the *driver data*, not a live model call, for the primary
  correctness test — only `test_recommend_actions_for_customer_returns_
  expected_shape` calls the live model, and only checks internal
  consistency, not the exact action list, for that reason).
  **Mitigation:** intentional design choice, stated here; a real
  distributional shift should prompt re-verifying the Research note's
  numbers, not silently loosening the test.
- **Risk:** a reviewer assumes `recommend_actions` should rank by a
  business-value/upsell-potential score rather than raw SHAP magnitude.
  **Mitigation:** the Approach section states and justifies this choice;
  reopen as a future spec if product priorities change.
- **Rollback:** single commit (Task 5) covering only new, additive files
  (`src/recommend/action_engine.py`, one notebook, one test file) — no
  existing file is modified. `git revert` is clean; nothing under
  `models/`/`mlruns/`/`data/` is touched.

### Definition of done

- All 5 tasks checked off.
- `pytest -q` green (all existing tests + `test_action_engine.py`).
- `notebooks/11_retention_action_engine.ipynb` executes top-to-bottom
  without error, including both worked-example confirmations.
- `quality-reviewer` and `security-reviewer` report no unresolved
  findings.
- All Success Criteria in Part 1 are met.
