import numpy as np
import pandas as pd
import pytest

from src.data.cohorts import (
    TENURE_COHORT_LABELS,
    assign_tenure_cohort,
    cohort_summary,
    empirical_retention_curve,
)
from src.data.load_data import TARGET_COLUMN

VERIFIED_COHORT_COUNTS = {"0-12": 2186, "12-24": 1024, "24-48": 1594, "48-72": 2239}


@pytest.fixture
def boundary_df() -> pd.DataFrame:
    # One customer at each cohort boundary, plus one just outside the fixed
    # edges (tenure=73) to exercise the NaN/out-of-range path.
    return pd.DataFrame({"tenure": [0, 12, 13, 24, 25, 48, 49, 72, 73]})


@pytest.fixture
def synthetic_cohort_df() -> pd.DataFrame:
    # Hand-computed: "0-12" cohort is 1/2 churned (50%), "12-24" is 1/1 (100%).
    return pd.DataFrame({
        "tenure": [0, 6, 15],
        TARGET_COLUMN: [1, 0, 1],
    })


def test_assign_tenure_cohort_matches_verified_counts(clean_df):
    cohort = assign_tenure_cohort(clean_df)
    assert cohort.value_counts().to_dict() == VERIFIED_COHORT_COUNTS
    assert cohort.isna().sum() == 0


def test_assign_tenure_cohort_boundary_months(boundary_df):
    cohort = assign_tenure_cohort(boundary_df)
    # Right-closed bins: a boundary month belongs to the *lower* band.
    assert cohort.tolist()[:8] == [
        "0-12", "0-12", "12-24", "12-24", "24-48", "24-48", "48-72", "48-72",
    ]


def test_assign_tenure_cohort_beyond_max_is_nan(boundary_df):
    cohort = assign_tenure_cohort(boundary_df)
    assert pd.isna(cohort.iloc[-1])  # tenure == 73, outside the fixed edges


def test_cohort_summary_order_and_customer_total(clean_df):
    summary = cohort_summary(clean_df)
    assert summary["cohort"].tolist() == TENURE_COHORT_LABELS
    assert summary["customers"].sum() == len(clean_df)


def test_cohort_summary_matches_hand_computed_rates(synthetic_cohort_df):
    summary = cohort_summary(synthetic_cohort_df)
    row_0_12 = summary[summary["cohort"] == "0-12"].iloc[0]
    row_12_24 = summary[summary["cohort"] == "12-24"].iloc[0]
    assert row_0_12["customers"] == 2
    assert row_0_12["churn_rate"] == pytest.approx(50.0)
    assert row_0_12["retention_rate"] == pytest.approx(50.0)
    assert row_12_24["customers"] == 1
    assert row_12_24["churn_rate"] == pytest.approx(100.0)
    assert row_12_24["retention_rate"] == pytest.approx(0.0)


def test_cohort_summary_excludes_out_of_range_rows(boundary_df):
    # 8 in-range rows + 1 out-of-range (tenure=73); churn is irrelevant here.
    boundary_df = boundary_df.assign(**{TARGET_COLUMN: 0})
    summary = cohort_summary(boundary_df)
    assert summary["customers"].sum() == 8
    assert summary["customers"].sum() < len(boundary_df)


def test_cohort_churn_rate_strictly_decreasing(clean_df):
    churn_rates = cohort_summary(clean_df)["churn_rate"].tolist()
    assert churn_rates == sorted(churn_rates, reverse=True)
    assert len(set(churn_rates)) == len(churn_rates)


def test_empirical_retention_curve_starts_at_100(clean_df):
    curve = empirical_retention_curve(clean_df)
    first_row = curve.iloc[0]
    assert first_row["month"] == 0
    assert first_row["retention_pct"] == pytest.approx(100.0)


def test_empirical_retention_curve_monotonic_nonincreasing(clean_df):
    curve = empirical_retention_curve(clean_df)
    diffs = curve["retention_pct"].diff().dropna()
    assert (diffs <= 0).all()


def test_empirical_retention_curve_beyond_max_tenure_is_zero(clean_df):
    curve = empirical_retention_curve(clean_df, max_month=80)
    beyond = curve[curve["month"] > 72]
    assert (beyond["retention_pct"] == 0.0).all()
    assert (beyond["n_at_risk"] == 0).all()


def test_empirical_retention_curve_no_nan_or_inf(clean_df):
    curve = empirical_retention_curve(clean_df)
    assert np.isfinite(curve[["retention_pct", "n_at_risk"]].to_numpy()).all()
