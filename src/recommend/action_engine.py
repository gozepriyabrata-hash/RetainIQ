"""Next-Best-Action engine: risk tier (src.recommend.risk_tiers) + top-3 SHAP
churn drivers (src.explain.local_explainer) in, a ranked, concrete retention
action list out.

"Rule-based" = the TIER_BASE_ACTIONS / DRIVER_ACTION_RULES lookup tables
below. "ML-driven" = which driver-specific actions fire, and in what order,
is entirely determined by the trained model's own per-customer SHAP output
(shap_drivers), not a static list -- two customers in the same risk tier can
get different action lists if the model attributes their risk to different
drivers.

Every action list is derived purely from a risk_tier string and an
already-computed driver list -- both model *outputs*, never the raw Churn
label -- and nothing here is written back into load_clean_data()'s output or
any DataFrame src.features.preprocessing/src.models.train consume. The data
flow in this module is one-way, toward reporting/serving, never back toward
training. See .claude/specs/13-retention-action-engine.md for the full
methodology and the two verified real-customer worked examples this
module's tests lock in.
"""

from collections.abc import Callable

import pandas as pd

from src.data import cohorts
from src.explain import local_explainer
from src.models import calibration, scoring
from src.recommend import risk_tiers

TOP_N_ACTIONS = 3
# Bound to src.data.cohorts' own "0-12" cohort-band boundary (churn is
# concentrated in a customer's first 12 months) rather than an
# independently chosen literal, so it can't silently drift from the cohort
# analysis's threshold -- guarded directly by
# test_early_tenure_threshold_matches_cohort_boundary.
EARLY_TENURE_THRESHOLD_MONTHS = cohorts.TENURE_COHORT_BOUNDARIES[0]

# One urgency-framed base action per tier. Keys are plain string literals
# (not risk_tiers.RISK_TIER_LABELS itself) but are guarded to stay in sync
# with it by test_tier_base_actions_cover_all_risk_tier_labels, which fails
# loudly if the two constant sets ever diverge.
TIER_BASE_ACTIONS: dict[str, dict[str, str]] = {
    "Critical": {
        "action": "Escalate to a retention specialist for a personal outreach call within 24 hours.",
        "category": "escalation",
    },
    "High": {
        "action": "Proactively offer a loyalty discount or service credit.",
        "category": "retention_offer",
    },
    "Medium": {
        "action": "Send a targeted engagement email highlighting underused benefits.",
        "category": "engagement",
    },
    "Low": {
        "action": "No immediate action needed; continue standard engagement monitoring.",
        "category": "monitor",
    },
}

# One entry per CLAUDE.md Sec 6 row (Contract, tenure, Tech support, Payment
# method, Internet) -- no invented lever. Each `condition` is evaluated only
# against a driver whose `direction` is already "increases" (recommend_actions
# never calls a rule for a protective driver), so a rule only needs to encode
# the "bad" value pattern, not the direction itself.
#
# "counterfactual_value" (present on exactly 3 of these 5 entries) is the
# single feature value recommend_actions_for_customer flips to when computing
# expected_churn_reduction_pct -- see .claude/specs/15-expected-churn-
# reduction.md. tenure and InternetService deliberately have no
# counterfactual_value: tenure advances on its own regardless of the
# intervention (there is no faithful "what if tenure were higher" flip for
# "enroll in onboarding"), and InternetService="Fiber optic" -> "DSL" would
# answer the wrong question (the action is quality outreach on the existing
# service, not a downgrade) and is entangled with six other columns' "No
# internet service" sentinel value.
DRIVER_ACTION_RULES: list[dict[str, object]] = [
    {
        "feature": "Contract",
        "condition": lambda v: v == "Month-to-month",
        "action": (
            "Offer an incentive (discount or loyalty perk) to upgrade "
            "from month-to-month to a 1- or 2-year contract."
        ),
        "category": "contract",
        "counterfactual_value": "Two year",
    },
    {
        "feature": "tenure",
        "condition": lambda v: isinstance(v, (int, float)) and v <= EARLY_TENURE_THRESHOLD_MONTHS,
        "action": (
            "Enroll the customer in a proactive early-tenure onboarding "
            "check-in (first-12-month risk window)."
        ),
        "category": "onboarding",
    },
    {
        "feature": "TechSupport",
        "condition": lambda v: v == "No",
        "action": "Offer a free or discounted Tech Support add-on.",
        "category": "support",
        "counterfactual_value": "Yes",
    },
    {
        "feature": "PaymentMethod",
        "condition": lambda v: v == "Electronic check",
        "action": "Nudge the customer to switch to automatic payment (credit card or bank transfer).",
        "category": "payment",
        "counterfactual_value": "Credit card (automatic)",
    },
    {
        "feature": "InternetService",
        "condition": lambda v: v == "Fiber optic",
        "action": "Schedule a proactive fiber-service-quality outreach call.",
        "category": "service_quality",
    },
]


def _rule_for_feature(feature: str) -> dict[str, object] | None:
    """First DRIVER_ACTION_RULES entry whose feature matches, ignoring
    condition -- guarded by test_driver_action_rules_features_are_unique,
    so "first" and "only" are equivalent in practice."""
    for rule in DRIVER_ACTION_RULES:
        if rule["feature"] == feature:
            return rule
    return None


def _matching_rule(feature: str, customer_value: object) -> dict[str, object] | None:
    """The DRIVER_ACTION_RULES entry for feature, if its condition(customer_value)
    is True, else None. Delegates to _rule_for_feature so a driver action's
    rule-match (used to decide whether it fires) and its later
    counterfactual_value lookup (recommend_actions_for_customer) can never
    disagree about which rule a feature maps to."""
    rule = _rule_for_feature(feature)
    if rule is None:
        return None
    condition: Callable[[object], bool] = rule["condition"]
    return rule if condition(customer_value) else None


def recommend_actions(
    risk_tier: str, shap_drivers: list[dict], top_n: int = TOP_N_ACTIONS
) -> list[dict]:
    """risk_tier + ranked SHAP drivers in, a ranked action list out.

    shap_drivers must be the shape local_explainer.local_shap_top_drivers /
    explain_customer()["shap_top_drivers"] returns (list of dicts with
    "feature", "customer_value", "direction", "reason", already ranked by
    |shap_value| descending) -- not lime_top_drivers, whose "feature" is a
    condition string like "Contract=Two year", not a bare column name, and
    is therefore incompatible with DRIVER_ACTION_RULES.

    Result always starts with the tier's base action, followed by up to
    top_n - 1 driver-specific actions: each shap_drivers entry is skipped
    unless its direction is "increases" (a protective driver never produces
    an action) and its (feature, customer_value) matches a
    DRIVER_ACTION_RULES condition; a repeated action text is deduped, not
    double-listed. Returns between 1 and top_n entries -- never padded if
    fewer rules match.

    Every entry also carries "expected_churn_reduction_pct" and
    "counterfactual_basis", always None here -- this function is pure and
    does no model scoring. recommend_actions_for_customer fills them in
    (for the subset of driver actions whose rule defines a
    counterfactual_value) after calling this function. See
    .claude/specs/15-expected-churn-reduction.md.
    """
    if risk_tier not in TIER_BASE_ACTIONS:
        raise ValueError(
            f"recommend_actions: unknown risk_tier {risk_tier!r}; expected one "
            f"of {sorted(TIER_BASE_ACTIONS)}"
        )
    if top_n < 1:
        raise ValueError(f"recommend_actions: top_n must be >= 1, got {top_n!r}")

    base = TIER_BASE_ACTIONS[risk_tier]
    result = [{
        "priority": 1,
        "action": base["action"],
        "category": base["category"],
        "rationale": f"Customer is in the {risk_tier} risk tier.",
        "source": "tier",
        "driver_feature": None,
        "expected_churn_reduction_pct": None,
        "counterfactual_basis": None,
    }]

    existing_actions = {result[0]["action"]}
    for driver in shap_drivers:
        if len(result) >= top_n:
            break
        if driver.get("direction") != "increases":
            continue
        rule = _matching_rule(driver["feature"], driver.get("customer_value"))
        if rule is None:
            continue
        if rule["action"] in existing_actions:
            continue
        result.append({
            "priority": len(result) + 1,
            "action": rule["action"],
            "category": rule["category"],
            "rationale": driver["reason"],
            "source": "driver",
            "driver_feature": driver["feature"],
            "expected_churn_reduction_pct": None,
            "counterfactual_basis": None,
        })
        existing_actions.add(rule["action"])

    return result


def recommend_actions_for_customer(
    customer: dict,
    *,
    pipeline: object | None = None,
    explainer_context: dict | None = None,
    top_n: int = TOP_N_ACTIONS,
) -> dict:
    """Raw customer attributes in, (risk_tier, ranked actions) out.

    Thin composition of risk_tiers.classify_scored_customers +
    local_explainer.explain_customer -- the function the Phase 5
    POST /recommend endpoint (CLAUDE.md Sec 10, src/api/main.py) calls
    directly. Every classify_scored_customers/explain_customer edge case
    (missing required column, unseen category, missing model artifact)
    propagates unchanged since this never reimplements scoring or
    explanation, only pipes their output onward.

    pipeline is resolved to a concrete loaded object up front -- if the
    caller passes None (the default), calibration.load_calibrated_model()
    is called exactly once here and that same object is then reused for the
    risk-tier score and every counterfactual score below, rather than
    letting each downstream scoring.score_customers call independently
    reload the pickle from disk. A caller processing many customers, or an
    API route that already loaded the pipeline once at startup, should
    still pass it in explicitly to skip even this one load. Note this only
    skips the pickle load -- scoring.score_customers still re-reads and
    re-parses the (small) model metadata JSON on every call regardless of
    whether pipeline is supplied, now up to 3 times per invocation (the
    base score plus up to 2 counterfactual scores) instead of 1; that cost
    is negligible (sub-millisecond each) next to the pickle load, so it's
    left as-is rather than threading a third cached argument through for it.

    pipeline and explainer_context (and top_n) are keyword-only so a
    positional second argument can never be silently misbound to pipeline
    by an existing or future caller.

    Building explainer_context when None is expensive (~1.9s per
    local_explainer's own docstring) -- a caller processing more than one
    customer should build one via local_explainer.build_explainer_context
    and pass it in, exactly local_explainer.explain_customer's own contract.

    Each driver-sourced action whose triggering DRIVER_ACTION_RULES entry
    defines a "counterfactual_value" additionally gets
    "expected_churn_reduction_pct" (a percentage-point drop in
    churn_probability, floored at 0.0) and "counterfactual_basis" (a
    human-readable description of the flip) filled in, computed by
    re-scoring the customer with that one feature changed, holding every
    other feature fixed. Actions whose rule has no counterfactual_value, and
    the tier-base action, keep both fields None -- see
    .claude/specs/15-expected-churn-reduction.md.

    expected_churn_reduction_pct is the trained model's own single-feature
    sensitivity to this specific customer's data, not a causal treatment-
    effect estimate -- it answers "what would the model predict for an
    otherwise-identical customer with this one attribute changed," which is
    an associational query on a model trained on customers' own (self-
    selected) attribute values, not a randomized intervention. Treat it as
    a model-consistency signal for prioritization, the same epistemic status
    as a SHAP value, not a guaranteed real-world outcome.
    """
    # Resolved once so the base score and every counterfactual score below
    # share the same pipeline object instead of each independently
    # reloading the pickle from disk when the caller passed None
    # (calibration.load_calibrated_model()'s ~1.78s cold-load cost, per
    # .claude/specs/14-recommend-endpoint.md's Research note).
    if pipeline is None:
        pipeline = calibration.load_calibrated_model()

    # risk_tier comes from the calibrated model, shap_top_drivers from the
    # raw model -- local_explainer's own documented split (calibration
    # reshapes confidence, not which features drove the prediction), so
    # these two calls are intentionally against different underlying
    # models and never expected to numerically reconcile.
    scored = risk_tiers.classify_scored_customers(pd.DataFrame([customer]), pipeline=pipeline)
    row = scored.iloc[0]
    base_churn_probability = float(row["churn_probability"])
    risk_tier = str(row["risk_tier"])

    # explain_customer also computes lime_top_drivers (a 5000-sample
    # LIME fit) as a side effect; it's discarded below since only SHAP
    # drivers drive DRIVER_ACTION_RULES. Accepted per 13's Requirement 5,
    # which mandates explain_customer's existing contract rather than a
    # SHAP-only variant -- worth revisiting for latency if POST /recommend
    # (src/api/main.py) ever needs this on a tighter SLA.
    explanation = local_explainer.explain_customer(customer, context=explainer_context)
    actions = recommend_actions(risk_tier, explanation["shap_top_drivers"], top_n)
    # Keyed for the loop below so counterfactual_basis quotes the same
    # (already-normalized) value the action's own "rationale" was built
    # from, not customer[feature]'s raw pre-prepare_scoring_input form --
    # the two would otherwise disagree for a whitespace-padded direct-call
    # input (the API path never differs, since CustomerPayload already
    # rejects that).
    drivers_by_feature = {d["feature"]: d for d in explanation["shap_top_drivers"]}

    for item in actions:
        if item["source"] != "driver":
            continue
        rule = _rule_for_feature(item["driver_feature"])
        counterfactual_value = rule.get("counterfactual_value") if rule else None
        if counterfactual_value is None:
            continue
        feature = rule["feature"]
        current_value = drivers_by_feature[feature]["customer_value"]
        counterfactual_customer = {**customer, feature: counterfactual_value}
        counterfactual_scored = scoring.score_customers(
            pd.DataFrame([counterfactual_customer]), pipeline=pipeline
        )
        counterfactual_probability = float(counterfactual_scored.iloc[0]["churn_probability"])
        delta = max(0.0, base_churn_probability - counterfactual_probability)
        item["expected_churn_reduction_pct"] = round(delta * 100, scoring.PERCENTAGE_DECIMALS)
        item["counterfactual_basis"] = f"{feature}: {current_value!r} -> {counterfactual_value!r}"

    result = {
        "churn_probability": base_churn_probability,
        "churn_probability_pct": float(row["churn_probability_pct"]),
        "risk_tier": risk_tier,
        "actions": actions,
    }
    if "customerID" in explanation:
        result["customerID"] = explanation["customerID"]
    return result
