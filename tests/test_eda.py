import numpy as np
import pandas as pd
import pytest

from src.data.eda import NUMERIC_COLUMNS, detect_outliers_iqr, distribution_stats, summarize_outliers

SUMMARY_COLUMNS = ["column", "q1", "q3", "iqr", "lower_bound", "upper_bound", "n_outliers", "pct_outliers"]
STATS_COLUMNS = ["column", "mean", "median", "std", "skew", "kurtosis", "min", "max"]


@pytest.fixture
def synthetic_df() -> pd.DataFrame:
    # Hand-computed Tukey fence: Q1=13, Q3=19, IQR=6, bounds=(4, 28).
    # 100 is the only value outside the fence.
    return pd.DataFrame({"value": [10, 12, 14, 16, 18, 20, 100]})


@pytest.mark.parametrize("column", NUMERIC_COLUMNS)
def test_detect_outliers_iqr_finds_none_in_current_data(clean_df, column):
    assert len(detect_outliers_iqr(clean_df, column)) == 0


def test_detect_outliers_iqr_finds_known_outlier(synthetic_df):
    result = detect_outliers_iqr(synthetic_df, "value")
    assert list(result["value"]) == [100]


def test_detect_outliers_iqr_k_changes_result(clean_df):
    # The default k=1.5 finds nothing in TotalCharges (verified above); a
    # much tighter fence on the same column must find something, proving
    # `k` actually reaches the comparison rather than being ignored.
    assert len(detect_outliers_iqr(clean_df, "TotalCharges", k=1.5)) == 0
    assert len(detect_outliers_iqr(clean_df, "TotalCharges", k=0.1)) > 0


def test_summarize_outliers_bounds_match_hand_computed_fence(synthetic_df):
    summary = summarize_outliers(synthetic_df, columns=["value"])
    row = summary.iloc[0]
    assert row["q1"] == pytest.approx(13.0)
    assert row["q3"] == pytest.approx(19.0)
    assert row["iqr"] == pytest.approx(6.0)
    assert row["lower_bound"] == pytest.approx(4.0)
    assert row["upper_bound"] == pytest.approx(28.0)
    assert row["n_outliers"] == 1
    assert row["pct_outliers"] == pytest.approx(100 / 7, rel=1e-2)


def test_summarize_outliers_shape(clean_df):
    summary = summarize_outliers(clean_df)
    assert list(summary["column"]) == NUMERIC_COLUMNS
    assert (summary["n_outliers"] >= 0).all()
    assert summary["pct_outliers"].between(0, 100).all()


def test_summarize_outliers_empty_columns_returns_headers_only(clean_df):
    result = summarize_outliers(clean_df, columns=[])
    assert list(result.columns) == SUMMARY_COLUMNS
    assert len(result) == 0


def test_distribution_stats_no_nan_or_inf(clean_df):
    stats = distribution_stats(clean_df)
    assert np.isfinite(stats.drop(columns="column").to_numpy()).all()


def test_distribution_stats_empty_columns_returns_headers_only(clean_df):
    result = distribution_stats(clean_df, columns=[])
    assert list(result.columns) == STATS_COLUMNS
    assert len(result) == 0
