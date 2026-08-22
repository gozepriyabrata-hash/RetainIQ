import pandas as pd
import plotly.graph_objects as go
import pytest

from src.data.kpi import (
    KPI_LABELS,
    avg_tenure_kpi,
    churn_rate_kpi,
    clv_kpi,
    kpi_summary,
    mrr_loss_kpi,
    plot_mrr_breakdown,
    plot_service_attach_distribution,
    service_attach_rate_kpi,
)
from src.data.load_data import TARGET_COLUMN

VERIFIED_CHURN_RATE = 26.54
VERIFIED_MRR = {"total_mrr": 456116.60, "lost_mrr": 139130.85, "lost_mrr_pct": 30.50}
VERIFIED_CLV = 244.04
VERIFIED_AVG_TENURE = 32.37
VERIFIED_ATTACH = {"avg_services": 4.15, "pct_of_max": 46.07}


@pytest.fixture
def zero_churn_df() -> pd.DataFrame:
    return pd.DataFrame({
        TARGET_COLUMN: [0, 0, 0],
        "MonthlyCharges": [50.0, 60.0, 70.0],
    })


@pytest.fixture
def empty_df() -> pd.DataFrame:
    return pd.DataFrame({TARGET_COLUMN: pd.Series(dtype=int), "MonthlyCharges": pd.Series(dtype=float)})


@pytest.fixture
def zero_mrr_df() -> pd.DataFrame:
    return pd.DataFrame({TARGET_COLUMN: [0, 1], "MonthlyCharges": [0.0, 0.0]})


def test_churn_rate_kpi_matches_verified_value(clean_df):
    assert churn_rate_kpi(clean_df) == pytest.approx(VERIFIED_CHURN_RATE)


def test_mrr_loss_kpi_matches_verified_values(clean_df):
    mrr = mrr_loss_kpi(clean_df)
    assert mrr["total_mrr"] == pytest.approx(VERIFIED_MRR["total_mrr"])
    assert mrr["lost_mrr"] == pytest.approx(VERIFIED_MRR["lost_mrr"])
    assert mrr["lost_mrr_pct"] == pytest.approx(VERIFIED_MRR["lost_mrr_pct"])
    assert mrr["retained_mrr"] == pytest.approx(mrr["total_mrr"] - mrr["lost_mrr"])


def test_mrr_loss_kpi_handles_zero_total_mrr(zero_mrr_df):
    mrr = mrr_loss_kpi(zero_mrr_df)
    assert mrr["total_mrr"] == 0
    assert pd.isna(mrr["lost_mrr_pct"])


def test_clv_kpi_matches_verified_value(clean_df):
    assert clv_kpi(clean_df) == pytest.approx(VERIFIED_CLV)


def test_clv_kpi_raises_on_zero_churn_rate(zero_churn_df):
    with pytest.raises(ValueError):
        clv_kpi(zero_churn_df)


def test_clv_kpi_raises_on_nan_churn_rate(empty_df):
    with pytest.raises(ValueError):
        clv_kpi(empty_df)


def test_avg_tenure_kpi_matches_verified_value(clean_df):
    assert avg_tenure_kpi(clean_df) == pytest.approx(VERIFIED_AVG_TENURE)


def test_service_attach_rate_kpi_matches_verified_values(clean_df):
    attach = service_attach_rate_kpi(clean_df)
    assert attach["avg_services"] == pytest.approx(VERIFIED_ATTACH["avg_services"])
    assert attach["pct_of_max"] == pytest.approx(VERIFIED_ATTACH["pct_of_max"])


def test_service_attach_rate_kpi_independent_of_churn_column(clean_df):
    with_churn = service_attach_rate_kpi(clean_df)
    without_churn = service_attach_rate_kpi(clean_df.drop(columns=[TARGET_COLUMN]))
    assert with_churn == without_churn


def test_kpi_summary_order_columns_and_no_nulls(clean_df):
    summary = kpi_summary(clean_df)
    assert summary["kpi"].tolist() == KPI_LABELS
    assert list(summary.columns) == ["kpi", "value", "unit"]
    assert summary["value"].isna().sum() == 0


def test_plot_functions_return_nonempty_figures(clean_df):
    mrr_fig = plot_mrr_breakdown(clean_df)
    attach_fig = plot_service_attach_distribution(clean_df)
    assert isinstance(mrr_fig, go.Figure)
    assert len(mrr_fig.data) >= 1
    assert isinstance(attach_fig, go.Figure)
    assert len(attach_fig.data) >= 1
