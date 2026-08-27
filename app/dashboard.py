"""RetainIQ KPI Dashboard.

Thin Streamlit layer -- all KPI math lives in src.data.kpi; this file only
loads data and renders it (CLAUDE.md Section 4: keep the UI thin).
"""

import logging
import sys
from pathlib import Path

# `streamlit run app/dashboard.py` puts app/ (not the project root) on
# sys.path, so the `src` package wouldn't otherwise be importable.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd  # noqa: E402 -- must follow the sys.path fixup above
import streamlit as st  # noqa: E402

from src.data.kpi import kpi_summary, plot_mrr_breakdown, plot_service_attach_distribution  # noqa: E402
from src.data.load_data import load_clean_data  # noqa: E402
from src.models import train  # noqa: E402
from src.models.evaluation import TARGET_AUC  # noqa: E402
from src.models.leaderboard import (  # noqa: E402
    best_model_row,
    format_leaderboard_table,
    load_leaderboard,
    plot_leaderboard_metrics,
)

logger = logging.getLogger(__name__)

st.set_page_config(page_title="RetainIQ — KPI Dashboard", layout="wide")


@st.cache_data
def _load_data() -> pd.DataFrame:
    """Cached wrapper around load_clean_data() for the normal (non-refresh) load path."""
    return load_clean_data()


def _render_kpi_overview() -> None:
    refresh_clicked = st.button("Refresh data")
    if refresh_clicked:
        # Scoped clear (not st.cache_data.clear()) so this only invalidates
        # this page's own cached load, not every cached function process-wide
        # -- matters once a future multi-page dashboard shares the cache.
        _load_data.clear()

    try:
        # load_clean_data(rebuild=False) reads its own on-disk parquet cache,
        # independent of st.cache_data -- so a genuine refresh must call
        # rebuild=True directly, not just clear the Streamlit cache.
        df = load_clean_data(rebuild=True) if refresh_clicked else _load_data()
    except (FileNotFoundError, OSError, ValueError, KeyError):
        # Broad on purpose: this is a system boundary (raw CSV on disk, which
        # can be missing, unreadable, or shaped unexpectedly after a
        # re-download) -- st.error keeps the message generic so no absolute
        # path or traceback reaches the browser.
        st.error(
            "Could not load the Telco Customer Churn dataset. Make sure "
            "data/raw/telco.csv exists and matches the expected schema, "
            "then reload this page."
        )
        return

    summary = kpi_summary(df)
    cols = st.columns(len(summary))
    for col, (_, row) in zip(cols, summary.iterrows()):
        col.metric(row["kpi"], f"{row['value']} {row['unit']}")

    # Name the actual numeric contradiction, not just the formula limitation:
    # CLV / ARPU implies an average customer lifetime in months (since
    # CLV = ARPU / churn_fraction = ARPU / (churn_rate_pct / 100)), which
    # visibly disagrees with the separately-computed Avg Tenure tile above --
    # see clv_kpi's docstring and .claude/specs/07-kpi-dashboard.md for why.
    churn_rate_pct = summary.loc[summary["kpi"] == "Churn Rate", "value"].iloc[0]
    avg_tenure_months = summary.loc[summary["kpi"] == "Avg Tenure", "value"].iloc[0]
    implied_lifetime_months = round(100 / churn_rate_pct, 1)
    st.caption(
        "CLV is a simplified estimate (avg. monthly revenue ÷ overall churn rate) -- "
        f"it implies an average customer lifetime of ~{implied_lifetime_months} months, "
        f"which disagrees with the {avg_tenure_months}-month Avg Tenure tile above because "
        "the two use different denominators (a cross-sectional churn share vs. actual "
        "observed tenure), not a monthly-cohort-based lifetime value. See src/data/kpi.py's "
        "clv_kpi docstring for the full caveat."
    )

    left, right = st.columns(2)
    left.plotly_chart(plot_mrr_breakdown(df), width="stretch")
    right.plotly_chart(plot_service_attach_distribution(df), width="stretch")


def _render_model_leaderboard() -> None:
    try:
        # COMPARISON_TABLE_PATH is read from the `train` module at call time
        # (not via load_leaderboard's own default argument, which is bound
        # once at import time) so a test can monkeypatch
        # train.COMPARISON_TABLE_PATH and have the redirect actually take
        # effect -- same reasoning as train.py's own DEFAULT_MODEL_PATH/
        # FIGURES_DIR precedent.
        df = load_leaderboard(train.COMPARISON_TABLE_PATH)
        # Resolved before any rendering starts (rather than after the table/
        # chart) so a bad row -- e.g. an empty comparison, ValueError from
        # idxmax -- falls straight to the except block below with nothing
        # already drawn, instead of leaving a half-rendered empty table and
        # chart above the error message.
        best_row = best_model_row(df)
        # Derived fresh from this row's own test_auc rather than trusting the
        # CSV's meets_target_auc column directly, so the badge color can never
        # disagree with the numeric value stated in the same message.
        meets_target = bool(best_row["test_auc"] >= TARGET_AUC)

        st.dataframe(format_leaderboard_table(df), width="stretch")
        st.plotly_chart(plot_leaderboard_metrics(df), width="stretch")
        st.caption(
            "Brier Score (lower is better) is shown in the table only -- it's "
            "excluded from the chart above, which groups metrics where higher "
            "is better."
        )

        message = (
            f"Best model by cross-validated AUC-ROC: **{best_row['name']}** "
            f"(test AUC-ROC {best_row['test_auc']:.4f} vs. the {TARGET_AUC} target)."
        )
        if meets_target:
            st.success(message)
        else:
            st.warning(message)
    except FileNotFoundError:
        st.info(
            "No model comparison found yet. Run `python -m src.models.train` "
            "to generate reports/model_comparison.csv."
        )
    except ValueError:
        # Broad on purpose, like _render_kpi_overview's dataset-load handler:
        # this is a system boundary (a comparison CSV that's missing columns,
        # empty, or otherwise malformed/stale). Log the real exception (which
        # names the absolute file path) server-side only -- st.error stays
        # generic so no local path reaches the browser.
        logger.exception("Could not read the model comparison file at %s", train.COMPARISON_TABLE_PATH)
        st.error(
            "The model comparison file is malformed or from an older version. "
            "Re-run `python -m src.models.train` to regenerate it."
        )


def main() -> None:
    st.title("RetainIQ — KPI Dashboard")

    kpi_tab, leaderboard_tab = st.tabs(["KPI Overview", "Model Leaderboard"])
    with kpi_tab:
        _render_kpi_overview()
    with leaderboard_tab:
        _render_model_leaderboard()


main()
