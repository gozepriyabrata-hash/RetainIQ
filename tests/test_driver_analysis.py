import pytest

from src.data.load_data import TARGET_COLUMN
from src.explain.driver_analysis import (
    LEAKAGE_AUC_THRESHOLD,
    _original_column_for,
    build_driver_features,
    check_auc_leakage_guard,
    chi_square_service_association,
    fit_driver_diagnostic_model,
    generate_driver_figures,
    numeric_correlation_with_churn,
    shap_global_importance,
)

VERIFIED_NUMERIC_CORRELATION = {
    "tenure": -0.3522, "TotalCharges": -0.1983, "MonthlyCharges": 0.1934,
}
VERIFIED_CRAMERS_V = {
    "OnlineSecurity": 0.3474, "TechSupport": 0.3429, "InternetService": 0.3225,
    "OnlineBackup": 0.2923, "DeviceProtection": 0.2816, "StreamingMovies": 0.2310,
    "StreamingTV": 0.2305, "MultipleLines": 0.0401, "PhoneService": 0.0114,
}
DOMAIN_SIGNAL_COLUMNS = {"Contract", "tenure", "TechSupport", "PaymentMethod", "InternetService"}


def test_numeric_correlation_matches_verified_values(clean_df):
    result = numeric_correlation_with_churn(clean_df).set_index("column")
    for column, expected in VERIFIED_NUMERIC_CORRELATION.items():
        assert result.loc[column, "correlation"] == pytest.approx(expected, abs=1e-3)
    ordered = numeric_correlation_with_churn(clean_df)["column"].tolist()
    assert ordered == ["tenure", "TotalCharges", "MonthlyCharges"]


def test_numeric_correlation_excludes_churn_row(clean_df):
    result = numeric_correlation_with_churn(clean_df)
    assert TARGET_COLUMN not in result["column"].values


def test_chi_square_service_association_matches_verified_cramers_v(clean_df):
    result = chi_square_service_association(clean_df).set_index("column")
    for column, expected in VERIFIED_CRAMERS_V.items():
        assert result.loc[column, "cramers_v"] == pytest.approx(expected, abs=1e-3)
    values = chi_square_service_association(clean_df)["cramers_v"].tolist()
    assert values == sorted(values, reverse=True)


def test_chi_square_phone_service_not_significant(clean_df):
    result = chi_square_service_association(clean_df).set_index("column")
    assert result.loc["PhoneService", "significant"] == False  # noqa: E712
    others = result.drop(index="PhoneService")
    assert others["significant"].all()


def test_build_driver_features_excludes_target(clean_df):
    X, y = build_driver_features(clean_df)
    assert TARGET_COLUMN not in X.columns
    assert len(y) == len(clean_df)


def test_check_auc_leakage_guard_raises_above_threshold():
    with pytest.raises(ValueError):
        check_auc_leakage_guard(0.97)
    check_auc_leakage_guard(0.83)
    check_auc_leakage_guard(LEAKAGE_AUC_THRESHOLD)


def test_fit_driver_diagnostic_model_is_reproducible(clean_df):
    result_a = fit_driver_diagnostic_model(clean_df)
    result_b = fit_driver_diagnostic_model(clean_df)
    assert result_a["auc"] == pytest.approx(result_b["auc"])
    assert result_a["pr_auc"] == pytest.approx(result_b["pr_auc"])


def test_fit_driver_diagnostic_model_auc_is_honest(clean_df):
    result = fit_driver_diagnostic_model(clean_df)
    assert 0.75 < result["auc"] < LEAKAGE_AUC_THRESHOLD


def test_original_column_for_resolves_prefix_collision():
    # A hypothetical InternetService / InternetServiceType pair: the "_"
    # boundary check must stop InternetService from stealing
    # InternetServiceType's dummy.
    columns = ["InternetService", "InternetServiceType"]
    assert _original_column_for("cat__InternetService_Fiber optic", columns) == "InternetService"
    assert _original_column_for("cat__InternetServiceType_Business", columns) == "InternetServiceType"


def test_original_column_for_raises_on_unmatched_feature():
    with pytest.raises(ValueError):
        _original_column_for("cat__NotAKnownColumn_Yes", ["Contract", "PaymentMethod"])


def test_shap_global_importance_excludes_target_and_is_sorted(clean_df):
    result = shap_global_importance(clean_df)
    assert TARGET_COLUMN not in result["column"].values
    assert (result["mean_abs_shap"] >= 0).all()
    diffs = result["mean_abs_shap"].diff().dropna()
    assert (diffs <= 0).all()


def test_shap_global_importance_recovers_domain_signals(clean_df):
    top_columns = set(shap_global_importance(clean_df, top_n=10)["column"])
    assert len(top_columns & DOMAIN_SIGNAL_COLUMNS) >= 4


def test_shap_global_importance_covers_every_original_column_once(clean_df):
    X, _ = build_driver_features(clean_df)
    full_ranking = shap_global_importance(clean_df, top_n=len(X.columns))
    assert len(full_ranking) == len(X.columns)
    assert full_ranking["column"].is_unique
    assert not full_ranking["column"].str.startswith(("cat__", "num__")).any()


def test_generate_driver_figures_returns_three_existing_paths(clean_df, tmp_path):
    paths = generate_driver_figures(clean_df, out_dir=tmp_path)
    assert len(paths) == 3
    assert all(path.exists() for path in paths)
