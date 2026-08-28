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
DRIVER_ACTION_RULES: list[dict[str, object]] = [
    {
        "feature": "Contract",
        "condition": lambda v: v == "Month-to-month",
        "action": (
            "Offer an incentive (discount or loyalty perk) to upgrade "
            "from month-to-month to a 1- or 2-year contract."
        ),
        "category": "contract",
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
    },
    {
        "feature": "PaymentMethod",
        "condition": lambda v: v == "Electronic check",
        "action": "Nudge the customer to switch to automatic payment (credit card or bank transfer).",
        "category": "payment",
    },
    {
        "feature": "InternetService",
        "condition": lambda v: v == "Fiber optic",
        "action": "Schedule a proactive fiber-service-quality outreach call.",
        "category": "service_quality",
    },
]


def _matching_rule(feature: str, customer_value: object) -> dict[str, object] | None:
    """First DRIVER_ACTION_RULES entry whose feature matches and whose
    condition(customer_value) is True, else None."""
    for rule in DRIVER_ACTION_RULES:
        if rule["feature"] == feature:
            condition: Callable[[object], bool] = rule["condition"]
            if condition(customer_value):
                return rule
    return None


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
        })
        existing_actions.add(rule["action"])

    return result


def recommend_actions_for_customer(
    customer: dict, explainer_context: dict | None = None, top_n: int = TOP_N_ACTIONS
) -> dict:
    """Raw customer attributes in, (risk_tier, ranked actions) out.

    Thin composition of risk_tiers.classify_scored_customers +
    local_explainer.explain_customer -- the function a future Phase 5
    POST /recommend endpoint (CLAUDE.md Sec 10) calls directly. Every
    classify_scored_customers/explain_customer edge case (missing required
    column, unseen category, missing model artifact) propagates unchanged
    since this never reimplements scoring or explanation, only pipes their
    output onward.

    Building explainer_context when None is expensive (~1.9s per
    local_explainer's own docstring) -- a caller processing more than one
    customer should build one via local_explainer.build_explainer_context
    and pass it in, exactly local_explainer.explain_customer's own contract.
    """
    # risk_tier comes from the calibrated model, shap_top_drivers from the
    # raw model -- local_explainer's own documented split (calibration
    # reshapes confidence, not which features drove the prediction), so
    # these two calls are intentionally against different underlying
    # models and never expected to numerically reconcile.
    scored = risk_tiers.classify_scored_customers(pd.DataFrame([customer]))
    row = scored.iloc[0]

    # explain_customer also computes lime_top_drivers (a 5000-sample
    # LIME fit) as a side effect; it's discarded below since only SHAP
    # drivers drive DRIVER_ACTION_RULES. Accepted per this spec's
    # Requirement 5, which mandates explain_customer's existing contract
    # rather than a SHAP-only variant -- worth revisiting for latency if
    # a future Phase 5 /recommend route needs this on a tight SLA.
    explanation = local_explainer.explain_customer(customer, context=explainer_context)

    result = {
        "churn_probability": float(row["churn_probability"]),
        "churn_probability_pct": float(row["churn_probability_pct"]),
        "risk_tier": str(row["risk_tier"]),
        "actions": recommend_actions(str(row["risk_tier"]), explanation["shap_top_drivers"], top_n),
    }
    if "customerID" in explanation:
        result["customerID"] = explanation["customerID"]
    return result
