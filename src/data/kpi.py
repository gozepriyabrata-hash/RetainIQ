"""Headline retention KPIs for the live dashboard: churn rate, MRR loss, CLV,
average tenure, and service-attach rate.

Reporting-only, like cohorts.py/segmentation.py/lifecycle.py -- nothing here
is written back into load_clean_data()'s output. See
.claude/specs/07-kpi-dashboard.md for the CLV/service-attach methodology
decisions and the verified reference values.

churn_rate_kpi and clv_kpi are computed directly from Churn -- if a future
phase ever feeds either back in as a Phase 2 model *feature* (none do
today), that would need the same design-time-vs-runtime leakage discipline
lifecycle.py already documents, since both are label-derived.
"""

import pandas as pd
import plotly.graph_objects as go

from src.data.eda import CHURN_PALETTE
from src.data.load_data import TARGET_COLUMN
from src.models.segmentation import SERVICE_YESNO_COLUMNS, compute_service_count

SERVICE_ATTACH_MAX = len(SERVICE_YESNO_COLUMNS) + 1  # 8 Yes/No add-ons + an internet flag
KPI_LABELS = ["Churn Rate", "MRR Loss", "CLV", "Avg Tenure", "Service-Attach Rate"]


def churn_rate_kpi(df: pd.DataFrame) -> float:
    """Overall churn rate as a percentage (2dp)."""
    return round(df[TARGET_COLUMN].mean() * 100, 2)


def mrr_loss_kpi(df: pd.DataFrame) -> dict[str, float]:
    """Total/retained/lost monthly recurring revenue and lost-MRR share (all 2dp).

    lost_mrr is the sum of MonthlyCharges among churned customers -- the MRR
    no longer being collected because those customers left.
    """
    total_mrr = df["MonthlyCharges"].sum()
    lost_mrr = df.loc[df[TARGET_COLUMN] == 1, "MonthlyCharges"].sum()
    retained_mrr = total_mrr - lost_mrr
    lost_mrr_pct = lost_mrr / total_mrr * 100 if total_mrr else float("nan")
    return {
        "total_mrr": round(total_mrr, 2),
        "lost_mrr": round(lost_mrr, 2),
        "retained_mrr": round(retained_mrr, 2),
        "lost_mrr_pct": round(lost_mrr_pct, 2),
    }


def clv_kpi(df: pd.DataFrame) -> float:
    """Customer lifetime value = ARPU / churn rate (model-free estimate, 2dp).

    This is the classic simplified CLV formula, used here because no
    predictive/survival model exists yet to base a forward-looking CLV on
    (.claude/specs/07-kpi-dashboard.md's scope decision 2). Caveat: churn
    rate here is the cross-sectional share of the customer base that has
    ever churned, not a true per-period (monthly) hazard rate, so the
    result should be read as a rough order-of-magnitude estimate -- not a
    monthly-cohort-based lifetime value, and not necessarily consistent
    with avg_tenure_kpi's actual average tenure. Surfaced to the dashboard
    via an st.caption, not silently presented as precise.

    Divides by the unrounded churn fraction (not the 2dp-rounded
    churn_rate_kpi output) to avoid compounding rounding error. Raises
    ValueError if churn rate is 0 or NaN (division by zero/undefined)
    instead of silently returning inf/NaN -- unreachable on real data
    today, but a real hazard on a filtered, synthetic, or empty frame.
    """
    churn_fraction = df[TARGET_COLUMN].mean()
    if pd.isna(churn_fraction) or churn_fraction == 0:
        raise ValueError("Cannot compute CLV: churn rate is 0% or undefined (division by zero).")
    arpu = df["MonthlyCharges"].mean()
    return round(arpu / churn_fraction, 2)


def avg_tenure_kpi(df: pd.DataFrame) -> float:
    """Average tenure in months (2dp)."""
    return round(df["tenure"].mean(), 2)


def service_attach_rate_kpi(df: pd.DataFrame) -> dict[str, float]:
    """Average services attached per customer, and as a % of SERVICE_ATTACH_MAX (2dp).

    Must not read Churn (leakage/independence guard) -- delegates entirely to
    compute_service_count, which itself never reads Churn.
    """
    avg_services = compute_service_count(df).mean()
    return {
        "avg_services": round(avg_services, 2),
        "pct_of_max": round(avg_services / SERVICE_ATTACH_MAX * 100, 2),
    }


def kpi_summary(df: pd.DataFrame) -> pd.DataFrame:
    """One row per headline KPI, in KPI_LABELS order, columns kpi/value/unit."""
    mrr = mrr_loss_kpi(df)
    attach = service_attach_rate_kpi(df)
    values = {
        "Churn Rate": (churn_rate_kpi(df), "%"),
        "MRR Loss": (mrr["lost_mrr"], "$/mo"),
        "CLV": (clv_kpi(df), "$"),
        "Avg Tenure": (avg_tenure_kpi(df), "months"),
        "Service-Attach Rate": (attach["avg_services"], f"of {SERVICE_ATTACH_MAX} services"),
    }
    rows = [{"kpi": label, "value": values[label][0], "unit": values[label][1]} for label in KPI_LABELS]
    return pd.DataFrame(rows, columns=["kpi", "value", "unit"])


def plot_mrr_breakdown(df: pd.DataFrame) -> go.Figure:
    """Bar chart of retained vs. lost MRR. Not saved to reports/figures/ --
    this is a live dashboard widget, not a notebook figure."""
    mrr = mrr_loss_kpi(df)
    fig = go.Figure(
        go.Bar(
            x=["Retained MRR", "Lost MRR"],
            y=[mrr["retained_mrr"], mrr["lost_mrr"]],
            marker_color=[CHURN_PALETTE[0], CHURN_PALETTE[1]],
        )
    )
    fig.update_layout(title="Monthly Recurring Revenue: Retained vs. Lost", yaxis_title="$ / month")
    return fig


def plot_service_attach_distribution(df: pd.DataFrame) -> go.Figure:
    """Histogram of services attached per customer, 0..SERVICE_ATTACH_MAX."""
    counts = compute_service_count(df)
    fig = go.Figure(
        go.Histogram(
            x=counts,
            xbins={"start": -0.5, "end": SERVICE_ATTACH_MAX + 0.5, "size": 1},
            marker_color=CHURN_PALETTE[0],
        )
    )
    fig.update_layout(
        title="Service-Attach Distribution",
        xaxis_title="Services attached",
        yaxis_title="Number of customers",
    )
    return fig
