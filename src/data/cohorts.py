"""Tenure-cohort analysis: churn/retention rate by band, empirical retention curve.

Kept separate from src/data/eda.py: a distinct analytical concern (binning +
cumulative retention) with its own runnable entry point, per CLAUDE.md §5.

Reporting-only: nothing here is written back into `load_clean_data()`'s
output. Binning `tenure` into `TenureCohort` is leakage-safe as a future
Phase 2 model feature (the bin edges are fixed, target-independent
constants), but if a future phase target-encodes a cohort using the churn
rates computed here, that encoding must be fit inside CV folds like any
other target encoding -- this module does not do that fitting itself.
"""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data.config import FIGURES_DIR
from src.data.eda import CHURN_PALETTE, save_fig
from src.data.load_data import TARGET_COLUMN

logger = logging.getLogger(__name__)

# -1 so tenure == 0 lands in "0-12". Bins are right-closed (pandas default),
# so a boundary month belongs to the *lower* band: tenure == 12 falls in
# "0-12", not "12-24"; tenure == 24 falls in "12-24", not "24-48"; etc.
TENURE_BIN_EDGES = [-1, 12, 24, 48, 72]
TENURE_COHORT_LABELS = ["0-12", "12-24", "24-48", "48-72"]
TENURE_COHORT_BOUNDARIES = TENURE_BIN_EDGES[1:-1]  # interior edges: [12, 24, 48]
MAX_OBSERVED_TENURE = TENURE_BIN_EDGES[-1]  # 72


def assign_tenure_cohort(df: pd.DataFrame) -> pd.Series:
    """Bin `tenure` into ordered cohort labels; NaN for tenure outside the fixed edges.

    A row with `tenure` above `MAX_OBSERVED_TENURE` (or below 0) is left as
    NaN rather than clipped into the nearest cohort -- see module docstring
    and `.claude/specs/03-cohort-analysis.md` for why this must stay visible
    rather than being silently absorbed.
    """
    return pd.cut(df["tenure"], bins=TENURE_BIN_EDGES, labels=TENURE_COHORT_LABELS, ordered=True)


def cohort_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Per-cohort customer count, churn rate (%), and retention rate (%), in cohort order.

    Logs a warning if any row's `tenure` falls outside the fixed bin edges
    (cohort is NaN) -- such rows are excluded from the per-cohort counts by
    `groupby`, so `customers.sum()` would otherwise silently undercount
    `len(df)` with no visible signal.
    """
    cohort = assign_tenure_cohort(df)
    n_unassigned = cohort.isna().sum()
    if n_unassigned:
        logger.warning(
            "%d row(s) have tenure outside the fixed cohort bin edges %s and are "
            "excluded from cohort_summary; investigate before trusting totals.",
            n_unassigned, TENURE_BIN_EDGES,
        )
    grouped = df.groupby(cohort, observed=False)[TARGET_COLUMN].agg(churn_rate="mean", customers="size")
    grouped["churn_rate"] = (grouped["churn_rate"] * 100).round(2)
    grouped["retention_rate"] = (100 - grouped["churn_rate"]).round(2)
    return grouped.reset_index(names="cohort")[["cohort", "customers", "churn_rate", "retention_rate"]]


def empirical_retention_curve(df: pd.DataFrame, max_month: int = MAX_OBSERVED_TENURE) -> pd.DataFrame:
    """1 - ECDF(tenure): % of customers with tenure >= m, for m in 0..max_month.

    NOT censoring-corrected (not Kaplan-Meier) -- see .claude/specs/03-cohort-analysis.md
    for the methodology decision behind this simplification. `n_at_risk` here
    means "customers whose observed tenure reached at least month m,
    regardless of churn status" -- not a survival-analysis risk set.
    """
    months = np.arange(0, max_month + 1)
    tenure = df["tenure"].to_numpy()
    n = len(df)
    n_at_risk = [(tenure >= m).sum() for m in months]
    retention_pct = [count / n * 100 if n else 0.0 for count in n_at_risk]
    return pd.DataFrame({"month": months, "retention_pct": retention_pct, "n_at_risk": n_at_risk})


def plot_cohort_churn_trend(df: pd.DataFrame, out_dir: Path = FIGURES_DIR) -> Path:
    """Paired bar chart: churn_rate & retention_rate per cohort, cohort order preserved."""
    summary = cohort_summary(df)
    x = np.arange(len(summary))
    width = 0.35
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - width / 2, summary["churn_rate"], width, label="Churn rate", color=CHURN_PALETTE[1])
    ax.bar(x + width / 2, summary["retention_rate"], width, label="Retention rate", color=CHURN_PALETTE[0])
    ax.set_xticks(x)
    ax.set_xticklabels(summary["cohort"])
    ax.set_xlabel("Tenure cohort (months)")
    ax.set_ylabel("Rate (%)")
    ax.set_title("Churn & Retention Rate by Tenure Cohort")
    ax.legend()
    return save_fig(fig, "cohort_churn_trend.png", out_dir)


def plot_retention_curve(
    df: pd.DataFrame, max_month: int = MAX_OBSERVED_TENURE, out_dir: Path = FIGURES_DIR
) -> Path:
    """Line chart of the empirical (non-censoring-corrected) retention curve."""
    curve = empirical_retention_curve(df, max_month)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(curve["month"], curve["retention_pct"], color=CHURN_PALETTE[1])
    for boundary in TENURE_COHORT_BOUNDARIES:
        ax.axvline(boundary, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Tenure (months)")
    ax.set_ylabel("Retention (%)")
    ax.set_title("Empirical Retention Curve (not censoring-corrected)")
    ax.set_ylim(0, 105)
    return save_fig(fig, "retention_curve.png", out_dir)


def generate_cohort_figures(df: pd.DataFrame, out_dir: Path = FIGURES_DIR) -> list[Path]:
    """Generate and save every cohort-analysis chart."""
    return [plot_cohort_churn_trend(df, out_dir), plot_retention_curve(df, out_dir=out_dir)]


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    from src.data.load_data import load_clean_data

    df = load_clean_data()
    paths = generate_cohort_figures(df)
    logger.info("Saved %d cohort figures to reports/figures/", len(paths))


if __name__ == "__main__":
    main()
