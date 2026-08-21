import numpy as np
import pandas as pd
import pytest

from src.data.load_data import TARGET_COLUMN
from src.models.segmentation import (
    VALUE_SEGMENT_LABELS,
    assign_value_segment,
    build_segmentation_features,
    compute_service_count,
    fit_segmentation_pipeline,
    rank_segments_by_value,
    segment_summary,
    silhouette_diagnostic,
)

VERIFIED_SEGMENT_COUNTS = {"Low": 2151, "Mid": 2575, "High": 2317}
VERIFIED_CHURN_RATES = {"Low": 14.41, "Mid": 46.33, "High": 15.80}


@pytest.fixture
def service_sentinel_df() -> pd.DataFrame:
    # Hand-computed: row 0 has PhoneService+MultipleLines+internet (DSL) with
    # no add-ons -> 1 (phone) + 1 (multiple lines) + 1 (has internet) = 3.
    # Row 1 has no phone at all ("No phone service" sentinel never counts as
    # "Yes") and no internet ("No internet service" sentinels never count,
    # InternetService == "No" doesn't add the has-internet flag) -> 0.
    # Row 2 has every add-on plus phone and internet -> 9 (max possible).
    return pd.DataFrame({
        "PhoneService": ["Yes", "No", "Yes"],
        "MultipleLines": ["Yes", "No phone service", "Yes"],
        "InternetService": ["DSL", "No", "Fiber optic"],
        "OnlineSecurity": ["No", "No internet service", "Yes"],
        "OnlineBackup": ["No", "No internet service", "Yes"],
        "DeviceProtection": ["No", "No internet service", "Yes"],
        "TechSupport": ["No", "No internet service", "Yes"],
        "StreamingTV": ["No", "No internet service", "Yes"],
        "StreamingMovies": ["No", "No internet service", "Yes"],
    })


@pytest.fixture
def value_ordered_groups_df() -> pd.DataFrame:
    # Three well-separated groups, low/mid/high on every one of the 3
    # clustering axes at once, so the composite ranking is unambiguous
    # regardless of how the 3 dimensions are weighted -- unlike the real
    # dataset, where "Low" and "Mid" trade off tenure vs. spend.
    n = 10
    low = pd.DataFrame({
        "tenure": [2] * n, "MonthlyCharges": [20.0] * n,
        "PhoneService": ["Yes"] * n, "MultipleLines": ["No"] * n,
        "InternetService": ["No"] * n,
        "OnlineSecurity": ["No internet service"] * n, "OnlineBackup": ["No internet service"] * n,
        "DeviceProtection": ["No internet service"] * n, "TechSupport": ["No internet service"] * n,
        "StreamingTV": ["No internet service"] * n, "StreamingMovies": ["No internet service"] * n,
        "group": ["low"] * n,
    })
    mid = pd.DataFrame({
        "tenure": [30] * n, "MonthlyCharges": [55.0] * n,
        "PhoneService": ["Yes"] * n, "MultipleLines": ["Yes"] * n,
        "InternetService": ["DSL"] * n,
        "OnlineSecurity": ["Yes"] * n, "OnlineBackup": ["No"] * n,
        "DeviceProtection": ["No"] * n, "TechSupport": ["No"] * n,
        "StreamingTV": ["No"] * n, "StreamingMovies": ["No"] * n,
        "group": ["mid"] * n,
    })
    high = pd.DataFrame({
        "tenure": [70] * n, "MonthlyCharges": [110.0] * n,
        "PhoneService": ["Yes"] * n, "MultipleLines": ["Yes"] * n,
        "InternetService": ["Fiber optic"] * n,
        "OnlineSecurity": ["Yes"] * n, "OnlineBackup": ["Yes"] * n,
        "DeviceProtection": ["Yes"] * n, "TechSupport": ["Yes"] * n,
        "StreamingTV": ["Yes"] * n, "StreamingMovies": ["Yes"] * n,
        "group": ["high"] * n,
    })
    return pd.concat([low, mid, high], ignore_index=True)


def test_compute_service_count_range_and_no_zero(clean_df):
    counts = compute_service_count(clean_df)
    assert counts.isna().sum() == 0
    assert counts.min() == 1
    assert counts.max() == 9


def test_compute_service_count_matches_hand_computed(service_sentinel_df):
    counts = compute_service_count(service_sentinel_df)
    assert counts.tolist() == [3, 0, 9]


def test_compute_service_count_independent_of_churn_column(clean_df):
    with_churn = compute_service_count(clean_df)
    without_churn = compute_service_count(clean_df.drop(columns=[TARGET_COLUMN]))
    assert with_churn.equals(without_churn)


def test_build_segmentation_features_excludes_churn(clean_df):
    features = build_segmentation_features(clean_df)
    assert set(features.columns) == {"tenure", "MonthlyCharges", "ServiceCount"}


def test_fit_segmentation_pipeline_is_reproducible(clean_df):
    pipeline_a = fit_segmentation_pipeline(clean_df)
    pipeline_b = fit_segmentation_pipeline(clean_df)
    labels_a = pipeline_a.named_steps["kmeans"].labels_
    labels_b = pipeline_b.named_steps["kmeans"].labels_
    assert np.array_equal(labels_a, labels_b)
    assert np.allclose(
        pipeline_a.named_steps["kmeans"].cluster_centers_,
        pipeline_b.named_steps["kmeans"].cluster_centers_,
    )


def test_fit_segmentation_pipeline_wrong_cluster_count_raises(clean_df):
    # Regression test: rank_segments_by_value must reject a pipeline that
    # wasn't fit with exactly len(VALUE_SEGMENT_LABELS) clusters, rather than
    # silently mapping only the first 3 cluster ids and dropping the rest.
    pipeline = fit_segmentation_pipeline(clean_df, n_clusters=4)
    with pytest.raises(ValueError):
        rank_segments_by_value(pipeline)


def test_assign_value_segment_orders_groups_correctly(value_ordered_groups_df):
    segment = assign_value_segment(value_ordered_groups_df)
    working = value_ordered_groups_df.assign(segment=segment)
    assert set(working.loc[working["group"] == "low", "segment"]) == {"Low"}
    assert set(working.loc[working["group"] == "mid", "segment"]) == {"Mid"}
    assert set(working.loc[working["group"] == "high", "segment"]) == {"High"}


def test_assign_value_segment_returns_ordered_categorical(clean_df):
    segment = assign_value_segment(clean_df)
    assert isinstance(segment.dtype, pd.CategoricalDtype)
    assert segment.dtype.ordered
    assert list(segment.cat.categories) == VALUE_SEGMENT_LABELS
    assert list(segment.index) == list(clean_df.index)


def test_segment_summary_order_and_customer_total(clean_df):
    summary = segment_summary(clean_df)
    assert summary["segment"].tolist() == VALUE_SEGMENT_LABELS
    assert summary["customers"].sum() == len(clean_df)


def test_segment_summary_matches_verified_counts(clean_df):
    summary = segment_summary(clean_df)
    counts = dict(zip(summary["segment"], summary["customers"]))
    assert counts == VERIFIED_SEGMENT_COUNTS


def test_segment_summary_matches_verified_churn_rates(clean_df):
    summary = segment_summary(clean_df).set_index("segment")
    for segment, expected_rate in VERIFIED_CHURN_RATES.items():
        assert summary.loc[segment, "churn_rate"] == pytest.approx(expected_rate)


def test_mid_segment_has_highest_churn_rate(clean_df):
    summary = segment_summary(clean_df).set_index("segment")
    mid_rate = summary.loc["Mid", "churn_rate"]
    assert mid_rate > summary.loc["Low", "churn_rate"]
    assert mid_rate > summary.loc["High", "churn_rate"]


def test_silhouette_diagnostic_shape(clean_df):
    result = silhouette_diagnostic(clean_df, k_values=(2, 3, 4))
    assert result["k"].tolist() == [2, 3, 4]
    assert result["silhouette_score"].between(-1, 1).all()
