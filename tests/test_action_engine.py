import pandas as pd
import pytest

import src.recommend.action_engine as action_engine
import src.recommend.risk_tiers as risk_tiers
from src.data import cohorts
from src.explain import local_explainer
from src.models import calibration, scoring

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
    # recommend_actions() itself is pure and always leaves
    # expected_churn_reduction_pct/counterfactual_basis at None
    # (.claude/specs/15-expected-churn-reduction.md); only the composition
    # layer under test fills them in, so those two keys are expected to
    # differ here -- everything else (rule matching, ranking, dedup) must
    # still agree exactly.
    impact_fields = ("expected_churn_reduction_pct", "counterfactual_basis")

    def _without_impact_fields(actions):
        return [{k: v for k, v in a.items() if k not in impact_fields} for a in actions]

    assert _without_impact_fields(result["actions"]) == _without_impact_fields(expected_actions)


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


# --- expected_churn_reduction_pct / counterfactual_basis ---------------------
# .claude/specs/15-expected-churn-reduction.md


def test_recommend_actions_always_includes_null_impact_fields():
    """recommend_actions is pure/model-I/O-free -- it never computes these,
    only recommend_actions_for_customer does."""
    result = action_engine.recommend_actions("Critical", CRITICAL_CUSTOMER_SHAP_DRIVERS, top_n=3)
    assert len(result) == 3
    for item in result:
        assert item["expected_churn_reduction_pct"] is None
        assert item["counterfactual_basis"] is None


def test_recommend_actions_for_customer_matches_worked_example_critical_impact(explainer_context):
    """Real, verified number from .claude/specs/15-expected-churn-reduction.md's
    Research note: flipping 5178-LMXOP's Contract to "Two year" drops
    churn_probability from 1.0 to 0.4143 -> 58.6pp."""
    result = action_engine.recommend_actions_for_customer(CRITICAL_CUSTOMER, explainer_context=explainer_context)
    actions_by_key = {a["driver_feature"] or "tier": a for a in result["actions"]}

    contract_action = actions_by_key["Contract"]
    assert contract_action["expected_churn_reduction_pct"] == 58.6
    assert "Contract" in contract_action["counterfactual_basis"]
    assert "Month-to-month" in contract_action["counterfactual_basis"]
    assert "Two year" in contract_action["counterfactual_basis"]

    assert actions_by_key["tenure"]["expected_churn_reduction_pct"] is None
    assert actions_by_key["tier"]["expected_churn_reduction_pct"] is None


def test_recommend_actions_for_customer_matches_worked_example_low_impact(explainer_context):
    """Real, verified number: flipping 9763-GRSKD's Contract to "Two year"
    drops churn_probability from 0.1677 to 0.0235 -> 14.4pp."""
    result = action_engine.recommend_actions_for_customer(LOW_CUSTOMER, explainer_context=explainer_context)
    actions_by_key = {a["driver_feature"] or "tier": a for a in result["actions"]}

    assert actions_by_key["Contract"]["expected_churn_reduction_pct"] == 14.4
    assert actions_by_key["tier"]["expected_churn_reduction_pct"] is None


def test_recommend_actions_for_customer_floors_negative_delta_at_zero(explainer_context, monkeypatch):
    """scoring is a single shared module object (see
    test_recommend_actions_for_customer_no_driver_actions_present's
    docstring), so a fake that inflates every row would inflate the base
    score too and always yield a 0.0 delta regardless of the max(0.0, ...)
    floor -- a vacuous test. Only inflate rows whose Contract is already
    "Two year" (i.e. only the counterfactual row, never LOW_CUSTOMER's own
    real "Month-to-month" base row), so the unfloored delta
    (real base ~0.1677 minus faked counterfactual 1.0) is genuinely
    negative and the floor is what makes the assertion pass."""
    real_score_customers = action_engine.scoring.score_customers

    def _inflate_only_counterfactual_row(raw_df, pipeline=None, use_calibrated=True):
        scored = real_score_customers(raw_df, pipeline, use_calibrated)
        if raw_df.iloc[0]["Contract"] == "Two year":
            scored["churn_probability"] = 1.0
        return scored

    monkeypatch.setattr(action_engine.scoring, "score_customers", _inflate_only_counterfactual_row)
    result = action_engine.recommend_actions_for_customer(LOW_CUSTOMER, explainer_context=explainer_context)
    contract_action = next(a for a in result["actions"] if a["driver_feature"] == "Contract")
    assert contract_action["expected_churn_reduction_pct"] == 0.0


def test_recommend_actions_for_customer_resolves_pipeline_once(explainer_context, monkeypatch):
    """pipeline=None must be resolved exactly once and reused for the base
    score and every counterfactual score, not reloaded per call."""
    calls = []
    real_loader = calibration.load_calibrated_model

    def _counting_loader(*args, **kwargs):
        calls.append(1)
        return real_loader(*args, **kwargs)

    monkeypatch.setattr(action_engine.calibration, "load_calibrated_model", _counting_loader)
    result = action_engine.recommend_actions_for_customer(CRITICAL_CUSTOMER, explainer_context=explainer_context)

    assert len(calls) == 1
    assert any(a["expected_churn_reduction_pct"] is not None for a in result["actions"])


def test_recommend_actions_for_customer_no_driver_actions_present(explainer_context, monkeypatch):
    """top_n=1 caps the result to the tier-base action only -- no driver
    action fires at all, so no counterfactual scoring call should happen.
    scoring is a single shared module object (both action_engine.py and
    risk_tiers.py do `from src.models import scoring`), so patching
    action_engine.scoring.score_customers also intercepts
    risk_tiers.classify_scored_customers's one base-score call -- the
    expected count is exactly 1 (that base call), not 0; a count of 2
    would mean a counterfactual call happened despite no driver action
    firing. See test_recommend_actions_for_customer_driver_action_without_
    counterfactual_value for the separate case of a driver action firing
    whose own rule has no counterfactual_value."""
    calls = []
    real_score_customers = action_engine.scoring.score_customers

    def _counting_score_customers(*args, **kwargs):
        calls.append(1)
        return real_score_customers(*args, **kwargs)

    monkeypatch.setattr(action_engine.scoring, "score_customers", _counting_score_customers)
    result = action_engine.recommend_actions_for_customer(
        CRITICAL_CUSTOMER, explainer_context=explainer_context, top_n=1,
    )
    assert len(result["actions"]) == 1
    assert result["actions"][0]["expected_churn_reduction_pct"] is None
    assert len(calls) == 1


def _explain_customer_returning(drivers: list[dict]):
    """A fake local_explainer.explain_customer replacement that reports
    exactly the given (already-hand-built) shap_top_drivers, for tests that
    need a specific driver to fire deterministically rather than relying on
    it naturally ranking in a real customer's top-3 SHAP output."""

    def _fake(customer, context=None):
        result = {"shap_top_drivers": drivers, "lime_top_drivers": []}
        if "customerID" in customer:
            result["customerID"] = customer["customerID"]
        return result

    return _fake


def test_recommend_actions_for_customer_driver_action_without_counterfactual_value(
    explainer_context, monkeypatch
):
    """InternetService's rule fires (real value "Fiber optic") but defines
    no counterfactual_value (Non-goals,
    .claude/specs/15-expected-churn-reduction.md) -- the resulting action
    must keep both new fields None, never a fabricated number, even though
    a driver action did fire (distinct from the top_n=1 case above, where
    no driver action fires at all)."""
    driver = _driver("InternetService", "Fiber optic")
    monkeypatch.setattr(action_engine.local_explainer, "explain_customer", _explain_customer_returning([driver]))

    result = action_engine.recommend_actions_for_customer(CRITICAL_CUSTOMER, explainer_context=explainer_context)
    internet_action = next(a for a in result["actions"] if a["driver_feature"] == "InternetService")
    assert internet_action["expected_churn_reduction_pct"] is None
    assert internet_action["counterfactual_basis"] is None


@pytest.mark.parametrize("feature,real_value", [("TechSupport", "No"), ("PaymentMethod", "Electronic check")])
def test_recommend_actions_for_customer_computes_flip_for_uncovered_rules(
    explainer_context, monkeypatch, feature, real_value
):
    """5178-LMXOP's/9763-GRSKD's worked examples (the two tests above) only
    exercise the Contract rule's counterfactual_value -- TechSupport and
    PaymentMethod never surface in either customer's real top-3 SHAP
    drivers. Forces each to fire via a fake explain_customer instead, on
    CRITICAL_CUSTOMER, whose real TechSupport/PaymentMethod values match
    real_value, and independently recomputes the expected delta via a
    direct scoring.score_customers call rather than hardcoding a number,
    since neither is one of the spec's locked-in worked examples."""
    rule = action_engine._rule_for_feature(feature)
    counterfactual_value = rule["counterfactual_value"]
    driver = _driver(feature, real_value)
    monkeypatch.setattr(action_engine.local_explainer, "explain_customer", _explain_customer_returning([driver]))

    result = action_engine.recommend_actions_for_customer(CRITICAL_CUSTOMER, explainer_context=explainer_context)
    action = next(a for a in result["actions"] if a["driver_feature"] == feature)

    pipeline = calibration.load_calibrated_model()
    base_probability = scoring.score_customers(
        pd.DataFrame([CRITICAL_CUSTOMER]), pipeline=pipeline
    ).iloc[0]["churn_probability"]
    flipped_customer = {**CRITICAL_CUSTOMER, feature: counterfactual_value}
    counterfactual_probability = scoring.score_customers(
        pd.DataFrame([flipped_customer]), pipeline=pipeline
    ).iloc[0]["churn_probability"]
    expected_pct = round(max(0.0, base_probability - counterfactual_probability) * 100, scoring.PERCENTAGE_DECIMALS)

    assert action["expected_churn_reduction_pct"] == expected_pct
    assert action["expected_churn_reduction_pct"] > 0  # sanity: real model data, not a degenerate flip
    assert action["counterfactual_basis"] == f"{feature}: {real_value!r} -> {counterfactual_value!r}"


# --- constants ----------------------------------------------------------------


def test_tier_base_actions_cover_all_risk_tier_labels():
    assert set(action_engine.TIER_BASE_ACTIONS) == set(risk_tiers.RISK_TIER_LABELS)


def test_driver_action_rules_features_are_unique():
    """_rule_for_feature/_matching_rule both return the *first* matching
    entry -- only correct if each feature appears at most once."""
    features = [rule["feature"] for rule in action_engine.DRIVER_ACTION_RULES]
    assert len(features) == len(set(features))


def test_driver_action_rules_counterfactual_values_are_known_categories():
    """Every DRIVER_ACTION_RULES counterfactual_value must be a category the
    fitted OneHotEncoder actually saw for that column. The encoder is
    handle_unknown="ignore" (src/features/preprocessing.py), so a typo'd or
    stale counterfactual_value would not raise -- it would silently encode
    as all-zeros and produce a plausible-looking but meaningless
    expected_churn_reduction_pct instead of failing loudly."""
    pipeline = calibration.load_calibrated_model()
    preprocessor = pipeline.calibrated_classifiers_[0].estimator.named_steps["pre"]
    cat_name, cat_transformer, cat_columns = next(t for t in preprocessor.transformers_ if t[0] == "cat")
    assert cat_name == "cat"

    for rule in action_engine.DRIVER_ACTION_RULES:
        counterfactual_value = rule.get("counterfactual_value")
        if counterfactual_value is None:
            continue
        known_categories = cat_transformer.categories_[cat_columns.index(rule["feature"])]
        assert counterfactual_value in known_categories, (
            f"{rule['feature']}'s counterfactual_value {counterfactual_value!r} is not a category "
            f"the fitted encoder saw; known categories are {list(known_categories)}"
        )
