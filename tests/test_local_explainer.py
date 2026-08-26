import json

import numpy as np
import pandas as pd
import pytest

import src.explain.local_explainer as local_explainer
import src.models.train as train

# Located during .claude/specs/11-explainable-ai.md's spec research by
# filtering X_test on the spec's described attribute combination -- each
# uniquely matches one test-split row. Re-derive via that same filter if the
# dataset or split ever changes and these tests start failing.
HIGH_RISK_TEST_INDEX = 3380  # customerID 5178-LMXOP, verified raw proba 93.3%
LOW_RISK_TEST_INDEX = 437    # customerID 4376-KFVRS, verified raw proba 1.8%


@pytest.fixture(scope="module")
def module_monkeypatch():
    """pytest's built-in monkeypatch fixture is function-scoped; this is the
    documented workaround for using it from a module-scoped fixture."""
    mp = pytest.MonkeyPatch()
    yield mp
    mp.undo()


@pytest.fixture(scope="module")
def context(clean_df, module_monkeypatch):
    """One real build_explainer_context() call, reused by every test below
    that needs a genuine SHAP TreeExplainer + LimeTabularExplainer -- the
    single expensive construction in this module."""
    return local_explainer.build_explainer_context(clean_df)


def _row(context: dict, index: int) -> pd.DataFrame:
    return context["X_test"].loc[[index]]


class _StubShapExplainer:
    def __init__(self, values: np.ndarray):
        self._values = values

    def shap_values(self, _X):
        return self._values


class _IdentityPreprocessor:
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        return X.to_numpy()


# ---- build_explainer_context ----------------------------------------------


def test_build_explainer_context_raises_on_non_tree_model(clean_df, monkeypatch, tmp_path):
    fake_metadata_path = tmp_path / "churn_model_metadata.json"
    fake_metadata_path.write_text(json.dumps({
        "model_name": "LogisticRegression",
        "feature_columns": [],
    }))
    monkeypatch.setattr(train, "DEFAULT_METADATA_PATH", fake_metadata_path)

    with pytest.raises(ValueError, match="LogisticRegression"):
        local_explainer.build_explainer_context(clean_df)


def test_build_explainer_context_raises_on_stale_feature_columns(clean_df, monkeypatch, tmp_path):
    fake_metadata_path = tmp_path / "churn_model_metadata.json"
    fake_metadata_path.write_text(json.dumps({
        "model_name": "XGBoost",
        "feature_columns": ["this_column_does_not_exist"],
    }))
    monkeypatch.setattr(train, "DEFAULT_METADATA_PATH", fake_metadata_path)

    with pytest.raises(ValueError, match="no longer match"):
        local_explainer.build_explainer_context(clean_df)


def test_build_explainer_context_returns_expected_keys(context):
    expected_keys = {
        "pipeline", "preprocessor", "clf", "model_name", "feature_columns",
        "categorical_columns", "feature_group_map", "X_train", "X_test",
        "y_test", "shap_explainer", "lime_explainer", "label_encoders",
    }
    assert expected_keys.issubset(context.keys())
    assert context["model_name"] == "XGBoost"


# ---- global_shap_importance / plot_global_shap_importance -----------------


def test_global_shap_importance_matches_verified_production_ranking(context):
    ranked = local_explainer.global_shap_importance(context)
    assert ranked["column"].tolist()[:2] == ["Contract", "tenure"]

    signals = {"Contract", "tenure", "TechSupport", "PaymentMethod", "InternetService"}
    top_6 = set(ranked["column"].tolist()[:6])
    assert len(signals & top_6) >= 4


def test_global_shap_importance_excludes_target_and_is_sorted(context):
    ranked = local_explainer.global_shap_importance(context)
    assert "Churn" not in ranked["column"].tolist()
    assert (ranked["mean_abs_shap"] >= 0).all()
    assert ranked["mean_abs_shap"].tolist() == sorted(ranked["mean_abs_shap"].tolist(), reverse=True)


def test_plot_global_shap_importance_returns_existing_path(context, tmp_path):
    path = local_explainer.plot_global_shap_importance(context, out_dir=tmp_path)
    assert path.exists()


# ---- local_shap_top_drivers -------------------------------------------------


def test_local_shap_top_drivers_matches_verified_high_risk_example(context):
    result = local_explainer.local_shap_top_drivers(context, _row(context, HIGH_RISK_TEST_INDEX))
    by_feature = {d["feature"]: d for d in result}

    assert set(by_feature) == {"tenure", "Contract", "InternetService"}
    for driver in by_feature.values():
        assert driver["direction"] == "increases"
    assert by_feature["tenure"]["shap_value"] == pytest.approx(0.778, abs=1e-2)
    assert by_feature["Contract"]["shap_value"] == pytest.approx(0.585, abs=1e-2)
    assert by_feature["InternetService"]["shap_value"] == pytest.approx(0.307, abs=1e-2)


def test_local_shap_top_drivers_matches_verified_low_risk_example(context):
    result = local_explainer.local_shap_top_drivers(context, _row(context, LOW_RISK_TEST_INDEX))
    by_feature = {d["feature"]: d for d in result}

    assert set(by_feature) == {"Contract", "tenure", "OnlineSecurity"}
    for driver in by_feature.values():
        assert driver["direction"] == "decreases"
    assert by_feature["Contract"]["shap_value"] == pytest.approx(-1.5618, abs=1e-2)
    assert by_feature["tenure"]["shap_value"] == pytest.approx(-1.2845, abs=1e-2)
    assert by_feature["OnlineSecurity"]["shap_value"] == pytest.approx(-0.4195, abs=1e-2)


def test_local_shap_top_drivers_zero_value_is_neutral():
    fake_context = {
        "shap_explainer": _StubShapExplainer(np.array([[0.0, 3.0]])),
        "feature_group_map": np.array(["A", "B"]),
        "preprocessor": _IdentityPreprocessor(),
    }
    features_df = pd.DataFrame([{"A": 1, "B": 2}])

    result = local_explainer.local_shap_top_drivers(fake_context, features_df, top_n=2)
    by_feature = {d["feature"]: d for d in result}

    assert by_feature["A"]["direction"] == "neutral"
    assert by_feature["B"]["direction"] == "increases"


def test_local_shap_top_drivers_preserves_signed_sum_not_mean_abs():
    """A multi-dummy group whose individual dummies partially offset must
    return their *signed sum*, not sum(|.|) or mean(|.|) -- the whole point
    of local_shap_top_drivers not reusing _aggregate_shap_by_group as-is."""
    fake_context = {
        "shap_explainer": _StubShapExplainer(np.array([[2.0, -0.5, -0.5]])),
        "feature_group_map": np.array(["Multi", "Multi", "Single"]),
        "preprocessor": _IdentityPreprocessor(),
    }
    features_df = pd.DataFrame([{"Multi": "A", "Single": 1}])

    result = local_explainer.local_shap_top_drivers(fake_context, features_df, top_n=2)
    by_feature = {d["feature"]: d for d in result}

    # signed sum: 2.0 + (-0.5) = 1.5 -- not sum(|.|)=2.5, not mean(|.|)=0.75
    assert by_feature["Multi"]["shap_value"] == pytest.approx(1.5)
    assert by_feature["Multi"]["direction"] == "increases"


def test_local_shap_top_drivers_respects_top_n(context):
    row = _row(context, LOW_RISK_TEST_INDEX)

    top1 = local_explainer.local_shap_top_drivers(context, row, top_n=1)
    assert len(top1) == 1

    top_many = local_explainer.local_shap_top_drivers(context, row, top_n=50)
    assert len(top_many) <= 50
    assert len(top_many) <= len(context["feature_columns"])


# ---- local_lime_top_drivers -------------------------------------------------


def test_local_lime_top_drivers_matches_verified_low_risk_example(context):
    # LimeTabularExplainer's internal RandomState advances on every
    # explain_instance call, so results drift depending on how many prior
    # calls happened on this module-scoped, shared explainer (verified
    # during review -- tenure can even drop out of the top-3 after enough
    # prior calls). Reseeding here makes this specific assertion
    # deterministic regardless of test execution order/history.
    context["lime_explainer"].random_state = np.random.RandomState(local_explainer.RANDOM_STATE)

    result = local_explainer.local_lime_top_drivers(context, _row(context, LOW_RISK_TEST_INDEX))
    directions_by_column = {}
    for driver in result:
        column = local_explainer._column_from_lime_description(driver["feature"], context["feature_columns"])
        directions_by_column[column] = driver["direction"]

    assert directions_by_column.get("Contract") == "decreases"
    assert directions_by_column.get("tenure") == "decreases"


def test_local_lime_top_drivers_raises_on_unseen_category(context):
    bad_row = _row(context, LOW_RISK_TEST_INDEX).copy()
    bad_row["Contract"] = "Lifetime"

    with pytest.raises(ValueError, match="Contract") as exc_info:
        local_explainer.local_lime_top_drivers(context, bad_row)
    assert "Lifetime" in str(exc_info.value)


def test_column_from_lime_description_resolves_categorical_and_numeric_forms(context):
    feature_columns = context["feature_columns"]
    resolve = local_explainer._column_from_lime_description

    assert resolve("Contract=Two year", feature_columns) == "Contract"
    assert resolve("tenure > 55.00", feature_columns) == "tenure"
    assert resolve("12.00 < tenure <= 24.00", feature_columns) == "tenure"
    assert resolve("InternetService=Fiber optic", feature_columns) == "InternetService"


def test_column_from_lime_description_raises_on_unmatched(context):
    with pytest.raises(ValueError):
        local_explainer._column_from_lime_description("foobar > 5", context["feature_columns"])


# ---- humanize_reason ---------------------------------------------------------


def test_humanize_reason_known_signal_columns():
    signals = {
        "Contract": "Month-to-month",
        "tenure": 1,
        "TechSupport": "No",
        "PaymentMethod": "Electronic check",
        "InternetService": "Fiber optic",
    }
    for column, value in signals.items():
        reason = local_explainer.humanize_reason(column, value, 1.0)
        assert f"{column} = {value}" not in reason
        assert "increases" in reason


def test_humanize_reason_unknown_column_uses_generic_fallback():
    reason = local_explainer.humanize_reason("SomeUnknownColumn", "X", 1.0)
    assert "SomeUnknownColumn = X" in reason
    assert "increases" in reason

    neutral_reason = local_explainer.humanize_reason("SomeUnknownColumn", "X", 0.0)
    assert "has no measurable effect on" in neutral_reason


def test_humanize_reason_never_leaks_transformed_feature_names(context):
    for column in context["feature_columns"]:
        reason = local_explainer.humanize_reason(column, "dummy_value", -1.0)
        assert "cat__" not in reason
        assert "num__" not in reason


# ---- explain_customer ---------------------------------------------------------


def test_explain_customer_returns_both_methods(context):
    customer = _row(context, HIGH_RISK_TEST_INDEX).iloc[0].to_dict()
    result = local_explainer.explain_customer(customer, context=context)

    assert result["shap_top_drivers"]
    assert result["lime_top_drivers"]
    for driver in result["shap_top_drivers"] + result["lime_top_drivers"]:
        assert driver["reason"]


def test_explain_customer_output_is_json_serializable(context):
    """The payload a future Phase 5 POST /explain route returns must survive
    a plain json.dumps -- numpy scalar types (np.int64, np.str_, ...) leaking
    into the dict would break a raw JSONResponse even though a Pydantic
    response_model would silently coerce them."""
    customer = _row(context, HIGH_RISK_TEST_INDEX).iloc[0].to_dict()
    result = local_explainer.explain_customer(customer, context=context)
    json.dumps(result)  # raises TypeError if any value isn't JSON-serializable


def test_explain_customer_raises_on_missing_required_column(context):
    customer = _row(context, HIGH_RISK_TEST_INDEX).iloc[0].to_dict()
    del customer["Contract"]

    with pytest.raises(ValueError, match="Contract"):
        local_explainer.explain_customer(customer, context=context)


def test_explain_customer_raises_clean_error_on_missing_total_charges(context):
    """prepare_scoring_input's _clean_common_fields touches TotalCharges/
    SeniorCitizen unconditionally, so the missing-column check must run
    before it -- otherwise these two columns would raise a bare KeyError
    instead of the documented ValueError."""
    customer = _row(context, HIGH_RISK_TEST_INDEX).iloc[0].to_dict()
    del customer["TotalCharges"]

    with pytest.raises(ValueError, match="TotalCharges"):
        local_explainer.explain_customer(customer, context=context)


def test_explain_customer_includes_customer_id_when_present(context):
    customer = _row(context, LOW_RISK_TEST_INDEX).iloc[0].to_dict()
    customer["customerID"] = "4376-KFVRS"

    result = local_explainer.explain_customer(customer, context=context)
    assert result["customerID"] == "4376-KFVRS"


def test_explain_customer_reuses_supplied_context(context):
    shap_explainer_before = context["shap_explainer"]
    lime_explainer_before = context["lime_explainer"]

    high_customer = _row(context, HIGH_RISK_TEST_INDEX).iloc[0].to_dict()
    low_customer = _row(context, LOW_RISK_TEST_INDEX).iloc[0].to_dict()
    local_explainer.explain_customer(high_customer, context=context)
    local_explainer.explain_customer(low_customer, context=context)

    assert context["shap_explainer"] is shap_explainer_before
    assert context["lime_explainer"] is lime_explainer_before


# ---- generate_explainability_figures ------------------------------------------


def test_generate_explainability_figures_returns_one_existing_path(clean_df, tmp_path):
    paths = local_explainer.generate_explainability_figures(clean_df, out_dir=tmp_path)
    assert len(paths) == 1
    assert paths[0].exists()
