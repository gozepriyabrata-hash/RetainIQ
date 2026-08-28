# Spec + Plan: Expected Churn-Reduction % on Next-Best-Action Recommendations

> Location note: this file lives at `.claude/specs/` per CLAUDE.md Sec 4's repo
> tree, consistent with `01`-`14`, overriding the generic `/create-spec`
> command's `specs/<slug>/spec.md`+`plan.md` template -- same override `10`,
> `13`, and `14` already made, for the same reason. Spec and plan are combined
> in one file. Numbered `15` (next after `14-recommend-endpoint.md`).
>
> Scope note: requested as "Next Best Action (NBA) / Ranked interventions with
> expected churn-reduction %." The "ranked interventions" half already exists,
> committed, in `13-retention-action-engine.md`'s `recommend_actions` (ranked
> by SHAP-driver magnitude) and `14-recommend-endpoint.md`'s `POST /recommend`.
> Nothing in the repo attaches a quantitative expected-impact number to any
> action -- that gap is this spec's entire scope: enrich each already-ranked
> action with an `expected_churn_reduction_pct`, computed per customer, per
> action. This does **not** reopen `13`'s ranking algorithm (see Non-goals).
>
> Design-choice note (stated as an assumption per `/create-spec`'s own
> workflow, not asked as a question -- see rationale below): the number is
> computed via **counterfactual model re-scoring**, not a static lookup table.
> `13`'s own "Alternative rejected" already rejected adding "a numeric
> priority weight per rule (e.g. sum of the CLAUDE.md Sec 6 real-world
> churn-rate deltas)" as "a second, independently-tuned weight table that
> could drift from CLAUDE.md Sec 6's actual verified numbers." A static
> expected-reduction table would be exactly that rejected pattern. Instead:
> for the subset of actions with a well-defined counterfactual feature value
> (see Requirement 1), flip that one feature on the real customer's data and
> re-score with the same calibrated pipeline already used for their
> `churn_probability` -- an honest, per-customer, model-derived number, never
> a fabricated one. This is plain forward inference (feature values in,
> probability out), the same operation `scoring.score_customers` already
> performs elsewhere; it introduces no new leakage path (Data & model impact,
> ML guardrails below).
>
> Research note: real, measured numbers from spec research (not assumed),
> using the persisted production artifacts (`models/churn_model_calibrated.pkl`)
> and the two worked-example customers `13`/`14` already locked in
> (`5178-LMXOP` Critical, `9763-GRSKD` Low):
>
> | Customer | Base `churn_probability` | Flip | Counterfactual probability | Delta (pp) |
> |---|---|---|---|---|
> | `5178-LMXOP` | 1.0000 | `Contract`: `"Month-to-month"` -> `"Two year"` | 0.4143 | **58.6** |
> | `5178-LMXOP` | 1.0000 | `TechSupport`: `"No"` -> `"Yes"` | 0.7024 | 29.8 |
> | `5178-LMXOP` | 1.0000 | `PaymentMethod`: `"Electronic check"` -> `"Credit card (automatic)"` | 0.8426 | 15.7 |
> | `9763-GRSKD` | 0.1677 | `Contract`: `"Month-to-month"` -> `"Two year"` | 0.0235 | **14.4** |
>
> Bolded rows are the only ones that actually surface in either worked
> example's action list (`13`'s Research note: `5178-LMXOP` -> tier base +
> `tenure` + `Contract`, `InternetService` never reached; `9763-GRSKD` ->
> tier base + `Contract` only -- `TechSupport` wasn't a top-3 SHAP driver for
> either customer, and `9763-GRSKD`'s real `PaymentMethod` is `"Mailed
> check"`, which never triggers that rule's condition in the first place).
> The other two rows are included here only to verify the mechanism produces
> sane, positive deltas across multiple flips, not because they appear in a
> worked example. `tests/test_action_engine.py::CRITICAL_CUSTOMER`/
> `LOW_CUSTOMER` (real raw rows, `Churn` stripped) are reused verbatim to
> reproduce this table exactly.

---

## PART 1 -- SPEC

### Feature

Attach `expected_churn_reduction_pct` (a per-customer, model-computed
percentage-point reduction in churn probability) and `counterfactual_basis`
(a human-readable description of the assumed change) to each driver-sourced
action in `13`'s `recommend_actions`/`recommend_actions_for_customer` output
and `14`'s `POST /recommend` response -- for the subset of actions where a
single, well-defined counterfactual feature value exists. Ranking/priority
order is unchanged.

### Problem / motivation

A retention manager reading `13`/`14`'s existing output sees *what* to do
("offer an incentive to upgrade from month-to-month...") but not *how much
it's likely to matter* for this specific customer. Two customers can get the
same `Contract`-rule action text with very different actual risk exposure
(the Research note's own two customers: 58.6pp vs 14.4pp) -- a flat action
list can't distinguish "this will probably save the account" from "this
helps a little." CLAUDE.md Sec 1's feature request for this session names
this explicitly: "Ranked interventions with expected churn-reduction %."

### Goals / non-goals

**Goals**
- Compute a real, per-customer `expected_churn_reduction_pct` for every
  driver-sourced action whose triggering rule defines a `counterfactual_value`
  (Requirement 1), via one-feature-flip re-scoring against the same
  calibrated pipeline already used for the customer's own `churn_probability`.
- Attach `counterfactual_basis` (e.g. `"Contract: 'Month-to-month' ->
  'Two year'"`) alongside it so the number is traceable, not a bare float --
  matches CLAUDE.md Sec 7's "explanations must be human-readable."
- Keep `recommend_actions` pure/model-I/O-free (`13`'s own architecture,
  reaffirmed, not reopened) -- both new fields default to `None` there;
  only the composition layer (`recommend_actions_for_customer`) fills them
  in, exactly the existing `recommend_actions`/`recommend_actions_for_
  customer` split.
- Surface both fields on `ActionItem`/`RecommendResponse` (`src/api/
  schemas.py`) so `POST /recommend` returns them -- additive, backward-
  compatible response fields (CLAUDE.md Sec 10: "if it must change, update
  tests and README in the same change" -- README currently documents no
  `/recommend` field-level shape at all, verified via grep, so there is
  nothing there to update).
- Resolve `recommend_actions_for_customer`'s `pipeline` parameter to a
  concrete loaded object once, before the base score and every counterfactual
  score, instead of letting each downstream `scoring.score_customers` call
  independently reload the pickle from disk when the caller passed `None`
  (Requirement 3) -- CLAUDE.md Sec 10's "load once" rule, now also relevant
  inside a single call, not just across requests.

**Non-goals**
- **No re-ranking.** Actions stay ordered exactly as `13` produces them
  (tier base first, then SHAP-magnitude order). `expected_churn_reduction_pct`
  is enrichment metadata, not a sort key -- reaffirms `13`'s own rejected
  alternative (a second priority score) rather than reopening it. `priority`
  values are unchanged.
- **No counterfactual for every action.** Only `Contract`, `TechSupport`,
  and `PaymentMethod` (of `13`'s five `DRIVER_ACTION_RULES`) get a
  `counterfactual_value`. `tenure`/onboarding has no valid flip -- tenure
  advances on its own regardless of the intervention, so "what if tenure
  were higher" is not a faithful counterfactual of "enroll in onboarding."
  `InternetService`/service-quality has no valid flip either, for two
  independent reasons: (1) the action is proactive quality outreach on the
  *existing* fiber service, not a downgrade to DSL, so flipping the feature
  would answer the wrong question; (2) `InternetService="No"` is entangled
  with six other columns' `"No internet service"` sentinel value, so a
  faithful flip would require a coordinated multi-column change, not a
  single-field one, which is out of scope. Both get `expected_churn_reduction_
  pct: None` -- never a fabricated or misleading number. The tier-base action
  (`source="tier"`) also always gets `None` -- it names no single feature to
  flip.
- **No change to `TOP_N_ACTIONS`, `EARLY_TENURE_THRESHOLD_MONTHS`,
  `TIER_BASE_ACTIONS`, the dedup/cap algorithm, or any existing `13`/`14`
  rule-matching, priority, or category value.**
- **No new FastAPI route, no Streamlit view, no `src/api/main.py` change.**
  `POST /recommend`'s route body already only calls
  `recommend_actions_for_customer` with a real, non-`None`
  `app.state.calibrated_pipeline` (verified: both are 503-guarded before the
  call) -- this spec's new fields flow through unchanged route code, purely
  via the widened `ActionItem` schema and `recommend_actions_for_customer`'s
  existing return dict.
- **No relative-percent framing** (e.g. "reduces churn risk by 58%
  relative"). `expected_churn_reduction_pct` is an absolute percentage-point
  delta in probability (`churn_probability_pct`'s own units), matching the
  existing field's convention, never a relative-reduction figure that could
  be read as a stronger claim than the model supports.
- No new third-party dependency.

### User stories

- As a **retention manager**, I want to see how much churn risk each
  suggested action is actually expected to remove for *this* customer, so I
  can prioritize effort across a list of at-risk accounts, not just within
  one customer's action list.
- As the **engineer**, I want the number to come from the same trained model
  already used for scoring, not a second hand-maintained table, so it can
  never silently drift from what the model actually believes.
- As a **recruiter/reviewer**, I want a number I can't get from an
  off-the-shelf rule engine -- a genuine counterfactual query against a real
  trained model -- with a visible `counterfactual_basis` so it's auditable,
  not a black box.

### Functional requirements

1. `src/recommend/action_engine.py`'s `DRIVER_ACTION_RULES` gains an
   optional `"counterfactual_value"` key on exactly these three of its five
   entries: `Contract` -> `"Two year"`, `TechSupport` -> `"Yes"`,
   `PaymentMethod` -> `"Credit card (automatic)"`. The `tenure` and
   `InternetService` entries do not gain the key (`.get("counterfactual_
   value")` returns `None` for them). `"Two year"` (not `"One year"`) is
   chosen as `Contract`'s target because it's the stronger, unambiguous
   lever the action text already names ("a 1- or 2-year contract") and
   matches CLAUDE.md Sec 6's own cited two-year churn-rate figure.
   `"Credit card (automatic)"` (not `"Bank transfer (automatic)"`) is an
   arbitrary-but-fixed choice between the two equally-valid automatic
   options; either escapes the electronic-check correlation the rule
   targets -- documented here so a reviewer doesn't read it as significant.
2. `recommend_actions` (pure, unchanged signature) MUST include
   `"expected_churn_reduction_pct": None` and `"counterfactual_basis": None`
   in every entry it builds -- both the tier-base entry and every
   driver-sourced entry -- so the dict shape is stable and self-describing
   regardless of which layer eventually fills the values in. This function
   remains fully I/O-free; it does no model scoring.
3. `recommend_actions_for_customer` MUST:
   a. Resolve `pipeline` to a concrete object before any scoring:
      `pipeline = pipeline if pipeline is not None else
      calibration.load_calibrated_model()`, then pass that resolved object
      explicitly to `classify_scored_customers` and to every counterfactual
      `scoring.score_customers` call below -- never let a `None` default
      cause more than one fresh disk load per invocation.
   b. After computing `actions = recommend_actions(...)`, for each entry
      where `source == "driver"`: look up the matching `DRIVER_ACTION_RULES`
      entry by `driver_feature` (each feature appears exactly once, per
      `13`'s own documented invariant). If it has no `"counterfactual_value"`,
      leave both new fields `None` and continue.
   c. Otherwise build `counterfactual_customer = {**customer, feature:
      rule["counterfactual_value"]}` (the original customer dict with
      exactly that one field replaced), score it via
      `scoring.score_customers(pd.DataFrame([counterfactual_customer]),
      pipeline=pipeline).iloc[0]["churn_probability"]`, and compute
      `delta = max(0.0, base_churn_probability - counterfactual_probability)`
      where `base_churn_probability` is this same customer's already-computed
      `row["churn_probability"]` (not re-derived). Set `expected_churn_
      reduction_pct = round(delta * 100, scoring.PERCENTAGE_DECIMALS)`
      (reuses the existing named constant, CLAUDE.md Sec 8's "no magic
      numbers," rather than a new independently-chosen decimal count) and
      `counterfactual_basis = f"{feature}: {customer[feature]!r} ->
      {rule['counterfactual_value']!r}"`.
   d. The `max(0.0, ...)` floor means `expected_churn_reduction_pct` is
      never negative even if a counterfactual flip happens to raise the
      model's estimate for some customer -- a floored-at-zero "no expected
      benefit" reading, never a claimed negative benefit.
   e. A counterfactual scoring call failing is not caught/swallowed --
      `counterfactual_customer` differs from the already-successfully-scored
      `customer` in exactly one already-Pydantic-validated categorical
      field, so it cannot introduce a new missing-column or dtype failure
      mode; any error still propagates like every other `recommend_actions_
      for_customer` failure (Requirement 3's edge cases, `13`'s own
      precedent).
4. `src/api/schemas.py`'s `ActionItem` gains `expected_churn_reduction_pct:
   float | None = None` and `counterfactual_basis: str | None = None`,
   field-for-field with `recommend_actions_for_customer`'s widened dict --
   no reshaping, `12`/`14`'s own precedent. `RecommendResponse` is
   structurally unchanged (still `actions: list[ActionItem]`).
5. `notebooks/11_retention_action_engine.ipynb` gains one additional cell
   after the two existing worked-example confirmations, printing `expected_
   churn_reduction_pct`/`counterfactual_basis` for both and confirming they
   match this spec's Research note (58.6 for `5178-LMXOP`'s `Contract`
   action, 14.4 for `9763-GRSKD`'s).
6. `tests/test_action_engine.py` and `tests/test_api.py` MUST cover the
   Plan's "Tests to write" section in full.
7. None of the above may change `TOP_N_ACTIONS`, `EARLY_TENURE_THRESHOLD_
   MONTHS`, `TIER_BASE_ACTIONS`, the existing five rules' `feature`/
   `condition`/`action`/`category` values, `priority` assignment,
   `src/recommend/risk_tiers.py`, `src/explain/local_explainer.py`,
   `src/models/scoring.py`, `src/models/calibration.py`,
   `src/models/train.py`, `src/api/main.py`, `app/`, or any existing
   test/notebook/figure/route behavior -- all current tests must keep
   passing unmodified (verified compatible: existing tests either assert
   individual fields, or compare two live-computed dicts to each other --
   see Risks/rollback -- never a hardcoded full-dict literal that new keys
   would break).

### Data & model impact

No new model, no new training feature, no schema change to any training
artifact. `expected_churn_reduction_pct` is computed by calling the existing
calibrated pipeline's `predict_proba` an additional 0-2 times per
`recommend_actions_for_customer` call (once per driver action with a defined
`counterfactual_value` actually present in the result, capped by `top_n`) --
plain forward inference on customer attributes, the same operation
`scoring.score_customers` already performs elsewhere. Nothing is written
back into `load_clean_data()`'s output, any training DataFrame, or any
feature `src/features/preprocessing.py`/`src/models/train.py` consume.

### ML guardrails (mandatory check)

- **No target/probability leakage:** the counterfactual call's *input* is a
  customer attribute dict with one already-validated categorical field
  swapped for another valid value from the same closed vocabulary -- never
  `Churn`, never a probability fed back in as a feature. The counterfactual
  *output* (a probability) is exposed only as response metadata, never
  written back into any feature column or training path.
- **Honest-AUC guard is unaffected:** no new model, no training/evaluation
  code path touched; this reuses the already-verified calibrated model's
  existing `predict_proba` unchanged.
- **Fitting/splitting/SMOTE:** N/A -- no fitting occurs; `scoring.
  score_customers` only calls `.predict_proba` on an already-fit pipeline.
- **Imbalance/metric reporting:** N/A -- no classifier added, no accuracy/
  AUC/recall reported by this feature.
- **Reproducibility:** deterministic given identical `(customer, pipeline)`
  input -- the calibrated pipeline's `predict_proba` has no randomness (only
  `explain_customer`'s LIME half does, untouched here). `random_state=42`
  is inherited from however the loaded pipeline was originally trained/
  calibrated; nothing here refits anything.

### API / UI surface

| Method | Path | Change |
|---|---|---|
| POST | `/recommend` | `ActionItem` gains `expected_churn_reduction_pct: float \| None`, `counterfactual_basis: str \| None`. Additive, backward-compatible -- existing consumers reading only the prior fields are unaffected. |

No new route, no Streamlit view, no `src/api/main.py` change (verified: the
existing route already resolves a non-`None` pipeline before calling
`recommend_actions_for_customer`, so this spec's new fields flow through the
current route body unmodified).

### Edge cases & failure states

- **Rule has no `counterfactual_value`** (`tenure`, `InternetService`) ->
  both new fields stay `None` -- directly tested.
- **Tier-base action** -> both new fields stay `None` -- directly tested.
- **`PaymentMethod`/`TechSupport` rule doesn't actually fire for a given
  customer** (as with the real `9763-GRSKD`, whose real `PaymentMethod` is
  `"Mailed check"`) -> no counterfactual is computed for it since it's never
  in the result list in the first place -- inherited from `13`'s existing
  matching logic, unchanged.
- **Counterfactual probability >= base probability** (model oddity) ->
  floored to `0.0`, never negative -- directly tested via a monkeypatched
  `scoring.score_customers`.
- **`pipeline=None` (caller relies on the default)** -> resolved once
  (Requirement 3a) and reused for the base score and every counterfactual
  score, not reloaded per call -- directly tested that the resolved pipeline
  is the same object across all internal scoring calls (call-count
  assertion via monkeypatch/spy on `calibration.load_calibrated_model`).
- **Every `13`/`14` edge case not touched by this spec** (missing required
  column, unseen category, missing model artifact, 503 when an artifact
  isn't loaded) is unaffected and inherited unchanged, since this spec adds
  a post-processing step after `recommend_actions`/`classify_scored_
  customers` already succeed, never before.

### Security notes

- **No new dependency**, no new untrusted-input surface. The counterfactual
  customer dict is built entirely from already-Pydantic-validated (API path)
  or already-successfully-scored (direct-call path) values plus a fixed
  module constant (`rule["counterfactual_value"]`) -- never a raw, unvalidated
  request field. `counterfactual_basis`'s f-string interpolates only
  `feature` (a fixed constant from `DRIVER_ACTION_RULES`, never user input)
  and `customer[feature]`/the counterfactual value (already-typed,
  Literal-constrained values at the API boundary) -- no `eval`, no template
  injection, same safety argument `13`'s `rationale` string already makes.
- No secrets, no new environment variables, no new network call, no file I/O
  beyond the same `joblib`-backed pipeline `predict_proba` calls `scoring.py`
  already performs elsewhere.

### Success criteria

- `pytest -q` passes: all existing tests + the new `test_action_engine.py`/
  `test_api.py` cases, all green.
- `recommend_actions_for_customer` on the real `5178-LMXOP` and `9763-GRSKD`
  customers reproduces this spec's Research note numbers exactly:
  `expected_churn_reduction_pct` of `58.6` on `5178-LMXOP`'s `Contract`
  action, `None` on its `tenure` action and tier-base action; `14.4` on
  `9763-GRSKD`'s `Contract` action, `None` on its tier-base action.
- `POST /recommend` returns the same two numbers end-to-end over HTTP.
- `notebooks/11_retention_action_engine.ipynb` runs top-to-bottom without
  error, including the new confirmation cell.
- `quality-reviewer` and `security-reviewer` report no unresolved findings
  on the diff.

### Out of scope

- Re-ranking actions by `expected_churn_reduction_pct` -- reaffirmed `13`
  non-goal, not reopened.
- A counterfactual for `tenure`/onboarding or `InternetService`/
  service-quality actions, or the tier-base action -- see Non-goals for the
  specific reasons each is excluded.
- Multi-feature ("what if we changed both Contract and TechSupport")
  counterfactuals -- single-field flips only.
- Any Streamlit "expected impact" view -- API/notebook only, matching `13`/
  `14`'s own phase boundary.
- Any change to `13`'s rule-matching, dedup, or cap algorithm, or to
  `risk_tiers.py`/`scoring.py`/`calibration.py`'s public behavior.

---

## PART 2 -- PLAN

### Approach

Add the counterfactual computation as a post-processing step inside the
existing `recommend_actions_for_customer` composition layer, keeping
`recommend_actions` itself pure and model-I/O-free -- exactly `13`'s own
pure/composition split, extended rather than broken. Each of the three
flippable rules gets a fixed `counterfactual_value`; the re-score reuses
`scoring.score_customers` (already-tested, already-used elsewhere) rather
than calling `pipeline.predict_proba` directly, so preprocessing stays
identical to every other scoring path in the codebase.

**Alternative rejected:** widen `recommend_actions`'s own signature to
accept an optional `customer`/`pipeline` and compute counterfactuals inside
the "pure" function. Rejected because it would blur `13`'s explicit,
already-tested pure/composition boundary (`recommend_actions` is
independently unit-tested today with hand-built driver dicts and no
customer/pipeline at all) for no benefit -- the composition layer already
has everything a post-processing pass needs, and keeping `recommend_actions`
I/O-free means its existing 13 tests never need to change.

### Task breakdown

- [ ] **1. Edit `src/recommend/action_engine.py`** -- add `"counterfactual_
      value"` to the `Contract`/`TechSupport`/`PaymentMethod` rule entries;
      add `import` of `src.models.scoring`; add `"expected_churn_reduction_
      pct": None, "counterfactual_basis": None` to both dict-building sites
      inside `recommend_actions`; add a `_rule_for_feature(feature)` helper;
      extend `recommend_actions_for_customer` per Requirement 3
      (pipeline resolution, counterfactual loop); update docstrings
      (Requirements 1-3).
- [ ] **2. Edit `src/api/schemas.py`** -- add `expected_churn_reduction_pct`,
      `counterfactual_basis` to `ActionItem` (Requirement 4).
- [ ] **3. Edit `notebooks/11_retention_action_engine.ipynb`** -- add the
      confirmation cell (Requirement 5).
- [ ] **4. Edit `tests/test_action_engine.py`** -- add the cases in "Tests
      to write" below.
- [ ] **5. Edit `tests/test_api.py`** -- extend the two worked-example
      `/recommend` tests to assert the new fields end-to-end.
- [ ] **6. Run the full suite** -- `pytest -q`, confirm all existing + new
      tests pass.
- [ ] **7. Commit** -- `src/recommend/action_engine.py`, `src/api/
      schemas.py`, `notebooks/11_retention_action_engine.ipynb`,
      `tests/test_action_engine.py`, `tests/test_api.py`, commit message
      `phase 4: expected churn-reduction % via counterfactual re-scoring on
      next-best-action recommendations`. No files under `data/`, `models/`,
      or `mlruns/` are touched.

### Tests to write (hand to test-writer)

- `tests/test_action_engine.py::test_recommend_actions_always_includes_null_impact_fields` --
  any tier/driver combination through `recommend_actions` directly (no
  customer/pipeline) -> every entry has `expected_churn_reduction_pct: None`
  and `counterfactual_basis: None` (pure function never computes them).
- `tests/test_action_engine.py::test_recommend_actions_for_customer_matches_worked_example_critical_impact` --
  `CRITICAL_CUSTOMER` (`5178-LMXOP`, real fixture already in the file)
  through `recommend_actions_for_customer` -> the `Contract` action has
  `expected_churn_reduction_pct == 58.6` and a `counterfactual_basis`
  containing `"Contract"`, `"Month-to-month"`, `"Two year"`; the `tenure`
  action and the tier-base action both have `expected_churn_reduction_pct
  is None`.
- `tests/test_action_engine.py::test_recommend_actions_for_customer_matches_worked_example_low_impact` --
  `LOW_CUSTOMER` (`9763-GRSKD`) -> `Contract` action's
  `expected_churn_reduction_pct == 14.4`; tier-base action's is `None`.
- `tests/test_action_engine.py::test_recommend_actions_for_customer_floors_negative_delta_at_zero` --
  monkeypatch `scoring.score_customers` so the counterfactual call returns a
  probability higher than the base -> `expected_churn_reduction_pct == 0.0`,
  never negative.
- `tests/test_action_engine.py::test_recommend_actions_for_customer_resolves_pipeline_once` --
  monkeypatch/spy `calibration.load_calibrated_model` with a call counter,
  call `recommend_actions_for_customer` with `pipeline=None` on a customer
  whose action list includes at least one flippable driver action -> the
  loader is called exactly once (not once per counterfactual score).
- `tests/test_action_engine.py::test_recommend_actions_for_customer_no_flip_rules_present` --
  a hand-built scenario (or a customer whose only driver action is
  `tenure`/`InternetService`) -> no counterfactual scoring call is made at
  all (spy on `scoring.score_customers`'s call count: exactly one, the base
  score) and both new fields are `None` on every entry.
- `tests/test_api.py::test_recommend_matches_worked_example_critical` --
  extend the existing test to additionally assert
  `expected_churn_reduction_pct == 58.6` on the `Contract` action.
- `tests/test_api.py::test_recommend_matches_worked_example_low` -- extend
  to assert `expected_churn_reduction_pct == 14.4` on the `Contract` action.

### Quality gates

1. **test-runner** -- run `pytest -q`, confirm all tests (existing + new)
   pass, fix any straightforward regression.
2. **quality-reviewer** -- review the pipeline-resolution logic (Requirement
   3a) for the "load once" guarantee, the floor-at-zero logic, the pure/
   composition boundary (`recommend_actions` stays I/O-free), and CLAUDE.md
   Sec 8 adherence (named constants, type hints, docstrings) against the
   diff.
3. **security-reviewer** -- confirm no new dependency, confirm the
   counterfactual customer dict is built only from already-validated values
   plus fixed constants, confirm no new untrusted-input handling in
   `ActionItem`.
4. Commit only after both reviews are clean.

### Risks / rollback

- **Risk:** a future retrain shifts the exact counterfactual deltas (Research
  note's 58.6/14.4), the same risk `13` already flagged for its own SHAP-
  driver worked examples. **Mitigation:** same accepted tradeoff `13` already
  made -- a real distributional shift should prompt re-verifying this spec's
  Research note against the new model, not silently loosening the test.
- **Risk:** adding two always-present keys to every `recommend_actions`
  dict entry could break an existing test asserting exact dict equality.
  **Mitigation:** verified during spec research -- every existing
  `test_action_engine.py`/`test_api.py` assertion either checks individual
  fields (`result[0]["source"]`, etc.) or compares two independently
  computed dicts to each other (`result["actions"] == expected_actions`,
  `result_with_pipeline == result_default`), never a hardcoded full-dict
  literal -- both styles remain valid with two extra keys present on both
  sides.
- **Risk:** the up-to-2 extra `predict_proba` calls per request add latency
  to `POST /recommend`. **Mitigation:** each call is a single-row
  `predict_proba` on an already-loaded, already-warm pipeline (no disk I/O
  once Requirement 3a's resolve-once fix is in place) -- negligible next to
  `explain_customer`'s existing ~0.25s LIME cost on the same request path;
  revisit only if a real load test shows otherwise.
- **Rollback:** one commit touching 5 files, all additive (widened dict
  shape with a backward-compatible default, widened schema, one notebook
  cell, new tests) -- no existing route, rule value, or ranking behavior is
  modified. `git revert` is clean; nothing under `models/`/`mlruns/`/`data/`
  is touched.

### Definition of done

- All 7 tasks checked off.
- `pytest -q` green (all existing tests + the new `test_action_engine.py`/
  `test_api.py` cases).
- `notebooks/11_retention_action_engine.ipynb` executes top-to-bottom
  without error, including the new confirmation cell.
- `quality-reviewer` and `security-reviewer` report no unresolved findings.
- All Success Criteria in Part 1 are met.
