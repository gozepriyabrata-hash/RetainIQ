import pytest

import src.recommend.action_engine as action_engine
import src.recommend.risk_tiers as risk_tiers
from src.data import cohorts
from src.explain import local_explainer
from src.models import calibration

# Real raw customer rows (Churn stripped), captured during
# .claude/specs/13-retention-action-engine.md's spec research. Reused
# verbatim rather than re-fetched from the dataset so these tests stay
# fast and self-contained.
CRITICAL_CUSTOMER = {
    "customerID": "5178-LMXOP", "gender": "Male", "SeniorCitizen": 1,
    "Partner": "Yes", "Dependents": "No", "tenure": 1, "PhoneService": "Yes",
    "MultipleLines": "Yes", "InternetService": "Fiber optic",
    "OnlineSecurity": "No", "OnlineBackup": "No", "DeviceProtection": "No",
    "TechSupport": "No", "StreamingTV": "Yes", "StreamingMovies": "Yes",
    "Contract": "Month-to-month", "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check", "MonthlyCharges": 95.1,
    "TotalCharges": "95.1",
}
LOW_CUSTOMER = {
    "customerID": "9763-GRSKD", "gender": "Male", "SeniorCitizen": 0,
    "Partner": "Yes", "Dependents": "Yes", "tenure": 13, "PhoneService": "Yes",
    "MultipleLines": "No", "InternetService": "DSL", "OnlineSecurity": "Yes",
    "OnlineBackup": "No", "DeviceProtection": "No", "TechSupport": "No",
    "StreamingTV": "No", "StreamingMovies": "No", "Contract": "Month-to-month",
    "PaperlessBilling": "Yes", "PaymentMethod": "Mailed check",
    "MonthlyCharges": 49.95, "TotalCharges": "587.45",
}

# Real SHAP driver data verified against the persisted model during spec
# research (see .claude/specs/13-retention-action-engine.md's Research
# note) -- hardcoded so the primary correctness tests below need no live
# model call and stay stable across an unrelated retrain.
CRITICAL_CUSTOMER_SHAP_DRIVERS = [
    {"feature": "tenure", "customer_value": 1, "shap_value": 0.7781, "direction": "increases", "reason": "1-month tenure increases this customer's predicted churn risk."},
    {"feature": "Contract", "customer_value": "Month-to-month", "shap_value": 0.5849, "direction": "increases", "reason": "Month-to-month contract increases this customer's predicted churn risk."},
    {"feature": "InternetService", "customer_value": "Fiber optic", "shap_value": 0.3072, "direction": "increases", "reason": "Fiber optic internet service increases this customer's predicted churn risk."},
]
LOW_CUSTOMER_SHAP_DRIVERS = [
    {"feature": "Contract", "customer_value": "Month-to-month", "shap_value": 0.5188, "direction": "increases", "reason": "Month-to-month contract increases this customer's predicted churn risk."},
    {"feature": "InternetService", "customer_value": "DSL", "shap_value": -0.4439, "direction": "decreases", "reason": "DSL internet service decreases this customer's predicted churn risk."},
    {"feature": "OnlineSecurity", "customer_value": "Yes", "shap_value": -0.4225, "direction": "decreases", "reason": "OnlineSecurity = Yes decreases this customer's predicted churn risk."},
]


@pytest.fixture(scope="module")
def module_monkeypatch():
    mp = pytest.MonkeyPatch()
    yield mp
    mp.undo()


@pytest.fixture(scope="module")
def explainer_context(clean_df, module_monkeypatch):
    """One real build_explainer_context() call, shared by every test below
    that needs the full raw-customer-in pipeline."""
    return local_explainer.build_explainer_context(clean_df)


def _driver(feature: str, customer_value: object, direction: str = "increases") -> dict:
    return {
        "feature": feature, "customer_value": customer_value,
        "shap_value": 0.5, "direction": direction,
        "reason": f"{feature} {direction} this customer's predicted churn risk.",
    }


# --- recommend_actions: validation ------------------------------------------


def test_recommend_actions_raises_on_unknown_tier():
    with pytest.raises(ValueError, match="Extreme"):
        action_engine.recommend_actions("Extreme", [])


def test_recommend_actions_raises_on_invalid_top_n():
    with pytest.raises(ValueError):
        action_engine.recommend_actions("Low", [], top_n=0)


# --- recommend_actions: base cases ------------------------------------------


@pytest.mark.parametrize("tier", list(action_engine.TIER_BASE_ACTIONS))
def test_recommend_actions_empty_drivers_returns_only_tier_base(tier):
    result = action_engine.recommend_actions(tier, [])
    assert len(result) == 1
    assert result[0]["action"] == action_engine.TIER_BASE_ACTIONS[tier]["action"]
    assert result[0]["category"] == action_engine.TIER_BASE_ACTIONS[tier]["category"]
    assert result[0]["source"] == "tier"
    assert result[0]["driver_feature"] is None
    assert result[0]["priority"] == 1


def test_recommend_actions_skips_decreasing_drivers():
    drivers = [
        _driver("Contract", "Month-to-month", direction="decreases"),
        _driver("TechSupport", "No", direction="decreases"),
    ]
    result = action_engine.recommend_actions("Medium", drivers)
    assert len(result) == 1


def test_recommend_actions_skips_drivers_outside_rule_table():
    drivers = [_driver("gender", "Male"), _driver("StreamingTV", "Yes")]
    result = action_engine.recommend_actions("Medium", drivers)
    assert len(result) == 1


# --- recommend_actions: rule firing ------------------------------------------


@pytest.mark.parametrize("feature,value,expected_category", [
    ("Contract", "Month-to-month", "contract"),
    ("tenure", 1, "onboarding"),
    ("tenure", 12, "onboarding"),
    ("TechSupport", "No", "support"),
    ("PaymentMethod", "Electronic check", "payment"),
    ("InternetService", "Fiber optic", "service_quality"),
])
def test_recommend_actions_each_rule_fires_on_matching_value(feature, value, expected_category):
    result = action_engine.recommend_actions("Medium", [_driver(feature, value)])
    assert len(result) == 2
    assert result[1]["category"] == expected_category
    assert result[1]["source"] == "driver"
    assert result[1]["driver_feature"] == feature


@pytest.mark.parametrize("feature,value", [
    ("Contract", "Two year"),
    ("tenure", 13),
    ("TechSupport", "Yes"),
    ("PaymentMethod", "Credit card (automatic)"),
    ("InternetService", "DSL"),
])
def test_recommend_actions_does_not_fire_on_non_triggering_value(feature, value):
    result = action_engine.recommend_actions("Medium", [_driver(feature, value)])
    assert len(result) == 1


def test_recommend_actions_respects_top_n_cap():
    drivers = [
        _driver("Contract", "Month-to-month"),
        _driver("tenure", 1),
        _driver("TechSupport", "No"),
    ]
    result = action_engine.recommend_actions("Medium", drivers, top_n=2)
    assert len(result) == 2
    assert result[1]["driver_feature"] == "Contract"


def test_recommend_actions_dedups_repeated_action_text():
    drivers = [
        _driver("Contract", "Month-to-month"),
        _driver("Contract", "Month-to-month"),
    ]
    result = action_engine.recommend_actions("Medium", drivers, top_n=5)
    assert len(result) == 2


def test_recommend_actions_top_n_larger_than_available_matches():
    drivers = [_driver("Contract", "Month-to-month"), _driver("gender", "Male")]
    result = action_engine.recommend_actions("Medium", drivers, top_n=10)
    assert len(result) == 2


def test_recommend_actions_driver_entries_have_sequential_priority_and_rationale():
    drivers = [_driver("Contract", "Month-to-month"), _driver("tenure", 1)]
    result = action_engine.recommend_actions("Medium", drivers, top_n=5)
    assert [a["priority"] for a in result] == [1, 2, 3]
    assert result[1]["rationale"] == drivers[0]["reason"]
    assert result[2]["rationale"] == drivers[1]["reason"]


def test_recommend_actions_tenure_rule_skips_non_numeric_customer_value():
    """The tenure condition must be total (never raise) on a malformed
    customer_value -- see .claude/specs/13-retention-action-engine.md's
    quality/security review findings."""
    drivers = [_driver("tenure", None), _driver("tenure", "unknown")]
    result = action_engine.recommend_actions("Medium", drivers, top_n=5)
    assert len(result) == 1


def test_early_tenure_threshold_matches_cohort_boundary():
    assert action_engine.EARLY_TENURE_THRESHOLD_MONTHS == cohorts.TENURE_COHORT_BOUNDARIES[0]


# --- recommend_actions: worked examples (real, verified SHAP data) ----------


def test_recommend_actions_matches_worked_example_critical():
    result = action_engine.recommend_actions("Critical", CRITICAL_CUSTOMER_SHAP_DRIVERS, top_n=3)
    assert len(result) == 3
    assert result[0]["source"] == "tier" and result[0]["category"] == "escalation"
    assert result[1]["driver_feature"] == "tenure" and result[1]["category"] == "onboarding"
    assert result[2]["driver_feature"] == "Contract" and result[2]["category"] == "contract"
    assert not any(a["driver_feature"] == "InternetService" for a in result)


def test_recommend_actions_matches_worked_example_low():
    result = action_engine.recommend_actions("Low", LOW_CUSTOMER_SHAP_DRIVERS, top_n=3)
    assert len(result) == 2
    assert result[0]["source"] == "tier" and result[0]["category"] == "monitor"
    assert result[1]["driver_feature"] == "Contract" and result[1]["category"] == "contract"


# --- recommend_actions_for_customer ------------------------------------------


def test_recommend_actions_for_customer_returns_expected_shape(explainer_context):
    result = action_engine.recommend_actions_for_customer(LOW_CUSTOMER, explainer_context=explainer_context)

    assert result["customerID"] == "9763-GRSKD"
    assert isinstance(result["churn_probability"], float)
    assert isinstance(result["churn_probability_pct"], float)
    assert result["risk_tier"] in risk_tiers.RISK_TIER_LABELS
    assert isinstance(result["actions"], list) and len(result["actions"]) >= 1

    explanation = local_explainer.explain_customer(LOW_CUSTOMER, context=explainer_context)
    expected_actions = action_engine.recommend_actions(result["risk_tier"], explanation["shap_top_drivers"])
    assert result["actions"] == expected_actions


def test_recommend_actions_for_customer_omits_customer_id_when_absent(explainer_context):
    customer_without_id = {k: v for k, v in LOW_CUSTOMER.items() if k != "customerID"}
    result = action_engine.recommend_actions_for_customer(customer_without_id, explainer_context=explainer_context)
    assert "customerID" not in result


def test_recommend_actions_for_customer_propagates_scoring_errors(explainer_context):
    incomplete_customer = dict(CRITICAL_CUSTOMER)
    del incomplete_customer["Contract"]
    with pytest.raises(ValueError):
        action_engine.recommend_actions_for_customer(incomplete_customer, explainer_context=explainer_context)


def test_recommend_actions_for_customer_accepts_explicit_pipeline(explainer_context):
    """pipeline forwarding (.claude/specs/14-recommend-endpoint.md Req. 1) must
    not change output versus the default fresh-load-from-disk path."""
    pipeline = calibration.load_calibrated_model()
    result_with_pipeline = action_engine.recommend_actions_for_customer(
        LOW_CUSTOMER, pipeline=pipeline, explainer_context=explainer_context,
    )
    result_default = action_engine.recommend_actions_for_customer(
        LOW_CUSTOMER, explainer_context=explainer_context,
    )
    assert result_with_pipeline == result_default


# --- constants ----------------------------------------------------------------


def test_tier_base_actions_cover_all_risk_tier_labels():
    assert set(action_engine.TIER_BASE_ACTIONS) == set(risk_tiers.RISK_TIER_LABELS)
