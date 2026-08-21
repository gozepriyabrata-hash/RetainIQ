import pandas as pd
import pytest

from src.data.cohorts import TENURE_COHORT_BOUNDARIES, assign_tenure_cohort
from src.data.load_data import TARGET_COLUMN
from src.data.lifecycle import (
    AT_RISK_SIGNAL_THRESHOLD,
    LIFECYCLE_STAGE_LABELS,
    N_RISK_SIGNALS,
    NEW_TENURE_MAX,
    assign_lifecycle_stage,
    compute_risk_signal_count,
    lifecycle_summary,
    risk_signal_diagnostic,
)

VERIFIED_STAGE_COUNTS = {"New": 1079, "Engaged": 1720, "Loyal": 1764, "At-Risk": 2480}
VERIFIED_CHURN_RATES = {"New": 27.71, "Engaged": 6.28, "Loyal": 9.18, "At-Risk": 52.42}
VERIFIED_RISK_CHURN_RATES = {0: 2.60, 1: 13.02, 2: 24.14, 3: 43.52, 4: 62.92}

# Template row with 0 risk signals; fixtures below only override what matters
# for the behavior under test.
BASE_ROW = {
    "Contract": "Two year", "PaymentMethod": "Mailed check",
    "TechSupport": "Yes", "InternetService": "DSL",
    "PhoneService": "Yes", "MultipleLines": "No",
    "OnlineSecurity": "No", "OnlineBackup": "No", "DeviceProtection": "No",
    "StreamingTV": "No", "StreamingMovies": "No",
}


def _row(**overrides) -> dict:
    return {**BASE_ROW, **overrides}


def test_compute_risk_signal_count_range_and_no_churn_or_tenure_read(clean_df):
    counts = compute_risk_signal_count(clean_df)
    assert counts.isna().sum() == 0
    assert set(counts.unique()) == set(range(N_RISK_SIGNALS + 1))
    dropped = clean_df.drop(columns=[TARGET_COLUMN, "tenure"])
    assert compute_risk_signal_count(dropped).equals(counts)


def test_compute_risk_signal_count_matches_hand_computed():
    df = pd.DataFrame([
        _row(),  # 0 signals
        _row(Contract="Month-to-month", PaymentMethod="Electronic check"),  # 2 signals
        _row(
            Contract="Month-to-month", PaymentMethod="Electronic check",
            TechSupport="No", InternetService="Fiber optic",
        ),  # 4 signals
    ])
    assert compute_risk_signal_count(df).tolist() == [0, 2, 4]


def test_assign_lifecycle_stage_at_risk_overrides_new():
    df = pd.DataFrame([{
        **_row(Contract="Month-to-month", PaymentMethod="Electronic check", TechSupport="No"),
        "tenure": 3, TARGET_COLUMN: 0,
    }])
    assert compute_risk_signal_count(df).iloc[0] >= AT_RISK_SIGNAL_THRESHOLD
    assert assign_lifecycle_stage(df).iloc[0] == "At-Risk"


def test_assign_lifecycle_stage_requires_tenure_and_service_for_loyal():
    # Long tenure (meets LOYAL_TENURE_MIN) but ServiceCount = 3 (PhoneService
    # "Yes" + TechSupport "Yes" + has-internet flag from DSL), well under
    # LOYAL_SERVICE_MIN = 5, and no risk signals.
    df = pd.DataFrame([{**_row(), "tenure": 50, TARGET_COLUMN: 0}])
    assert assign_lifecycle_stage(df).iloc[0] == "Engaged"


def test_assign_lifecycle_stage_covers_every_row_with_no_nulls(clean_df):
    stage = assign_lifecycle_stage(clean_df)
    assert stage.isna().sum() == 0
    assert isinstance(stage.dtype, pd.CategoricalDtype)
    assert stage.dtype.ordered
    assert list(stage.cat.categories) == LIFECYCLE_STAGE_LABELS


def test_assign_lifecycle_stage_independent_of_churn_column(clean_df):
    with_churn = assign_lifecycle_stage(clean_df)
    without_churn = assign_lifecycle_stage(clean_df.drop(columns=[TARGET_COLUMN]))
    assert with_churn.equals(without_churn)


def test_new_tenure_max_matches_cohorts_zero_to_twelve_boundary():
    # The notebook's cross-check (New stage churn vs. cohorts.py's "0-12"
    # cohort churn) is only a fair comparison if the two tenure boundaries
    # actually agree -- lock that coupling in explicitly.
    assert NEW_TENURE_MAX == TENURE_COHORT_BOUNDARIES[0]


def test_new_stage_is_a_subset_of_the_zero_to_twelve_cohort(clean_df):
    stage = assign_lifecycle_stage(clean_df)
    cohort = assign_tenure_cohort(clean_df)
    new_rows = stage == "New"
    assert (cohort[new_rows] == "0-12").all()


def test_lifecycle_summary_order_and_customer_total(clean_df):
    summary = lifecycle_summary(clean_df)
    assert summary["stage"].tolist() == LIFECYCLE_STAGE_LABELS
    assert summary["customers"].sum() == len(clean_df)


def test_lifecycle_summary_matches_verified_counts_and_rates(clean_df):
    summary = lifecycle_summary(clean_df).set_index("stage")
    for stage, expected_count in VERIFIED_STAGE_COUNTS.items():
        assert summary.loc[stage, "customers"] == expected_count
    for stage, expected_rate in VERIFIED_CHURN_RATES.items():
        assert summary.loc[stage, "churn_rate"] == pytest.approx(expected_rate)


def test_at_risk_has_highest_churn_rate(clean_df):
    summary = lifecycle_summary(clean_df).set_index("stage")
    at_risk_rate = summary.loc["At-Risk", "churn_rate"]
    for stage in ["New", "Engaged", "Loyal"]:
        assert at_risk_rate > summary.loc[stage, "churn_rate"]


def test_risk_signal_diagnostic_shape_and_customer_total(clean_df):
    diag = risk_signal_diagnostic(clean_df)
    assert diag["risk_count"].tolist() == list(range(N_RISK_SIGNALS + 1))
    assert diag["customers"].sum() == len(clean_df)


def test_risk_signal_diagnostic_reindexes_absent_counts_instead_of_dropping_rows():
    # A slice with no risk signals present at all must still return one row
    # per possible risk_count (0..N_RISK_SIGNALS), not just the observed one --
    # a plain groupby would otherwise silently drop the absent rows.
    df = pd.DataFrame([{**_row(), "tenure": 10, TARGET_COLUMN: 0}])
    diag = risk_signal_diagnostic(df)
    assert diag["risk_count"].tolist() == list(range(N_RISK_SIGNALS + 1))
    assert diag.set_index("risk_count").loc[0, "customers"] == 1


def test_risk_signal_diagnostic_matches_verified_and_is_strictly_increasing(clean_df):
    diag = risk_signal_diagnostic(clean_df).set_index("risk_count")
    for risk_count, expected_rate in VERIFIED_RISK_CHURN_RATES.items():
        assert diag.loc[risk_count, "churn_rate"] == pytest.approx(expected_rate)
    assert diag["churn_rate"].is_monotonic_increasing
    assert diag["churn_rate"].is_unique
