import numpy as np
import pytest

from src.data.eda import NUMERIC_COLUMNS
from src.data.load_data import ID_COLUMN, TARGET_COLUMN
from src.explain.driver_analysis import (
    LEAKAGE_AUC_THRESHOLD,
    _aggregate_shap_by_group,
    _feature_group_columns,
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


class _FakeEncoder:
    def __init__(self, categories):
        self.categories_ = categories


class _FakePreprocessor:
    """Duck-typed stand-in for a fitted ColumnTransformer's "cat" step.

    _feature_group_columns only reads named_transformers_["cat"].categories_,
    so a real ColumnTransformer/OneHotEncoder fit isn't needed to unit-test it.
    """

    def __init__(self, categories):
        self.named_transformers_ = {"cat": _FakeEncoder(categories)}


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


def test_build_driver_features_drops_id_column_when_present(clean_df):
    with_id = clean_df.copy()
    with_id[ID_COLUMN] = [f"C{i}" for i in range(len(with_id))]
    X, _ = build_driver_features(with_id)
    assert ID_COLUMN not in X.columns


def test_build_driver_features_raises_when_target_missing(clean_df):
    without_target = clean_df.drop(columns=[TARGET_COLUMN])
    with pytest.raises(KeyError):
        build_driver_features(without_target)


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


def test_feature_group_columns_maps_dummies_to_original_columns():
    categorical_columns = ["InternetService", "Contract"]
    preprocessor = _FakePreprocessor([
        np.array(["DSL", "Fiber optic", "No"]),
        np.array(["Month-to-month", "One year", "Two year"]),
    ])
    groups = _feature_group_columns(preprocessor, categorical_columns)
    expected = list(NUMERIC_COLUMNS) + ["InternetService"] * 3 + ["Contract"] * 3
    assert list(groups) == expected


def test_aggregate_shap_by_group_sums_signed_values_before_taking_abs():
    # Two rows, one group "A" with two dummy columns. Row 0's dummies
    # partially offset (+3, -1 -> true per-row contribution +2); row 1's
    # don't (+1, +1 -> true per-row contribution +2). The correct
    # aggregation (sum signed values per row, then mean(|.|)) gives
    # (|2| + |2|) / 2 = 2.0. The bug this regression-guards against --
    # summing each dummy's own mean(|shap|) independently -- would instead
    # give (3+1)/2 + (1+1)/2 = 3.0, silently overcounting the group.
    shap_values = np.array([
        [3.0, -1.0],
        [1.0, 1.0],
    ])
    original_columns = np.array(["A", "A"])

    result = _aggregate_shap_by_group(shap_values, original_columns).set_index("column")
    assert result.loc["A", "mean_abs_shap"] == pytest.approx(2.0)

    naive_overcounted = float(np.abs(shap_values).mean(axis=0).sum())
    assert naive_overcounted == pytest.approx(3.0)
    assert result.loc["A", "mean_abs_shap"] != pytest.approx(naive_overcounted)


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
