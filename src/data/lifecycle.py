"""Rule-based customer lifecycle stages: New / Engaged / Loyal / At-Risk.

At-Risk is an override, not a fourth tenure band: a customer showing enough
CLAUDE.md §6 risk signals is flagged At-Risk regardless of tenure, checked
before the tenure/service split. See .claude/specs/05-retention-funnel.md
for why (behavioral risk separates churn better than tenure alone).

Reporting-only, like cohorts.py/segmentation.py: nothing here is written
back into load_clean_data()'s output.

The four threshold constants below were selected by inspecting churn rates
across the *full* dataset at design time -- see the spec's ML guardrails
section. That is harmless today (nothing here fits Churn at runtime), but if
LifecycleStage is ever used as a Phase 2 model feature, that full-dataset
selection carries a bias into any evaluation unless the constants are
re-derived on the training split only, not reused as fixed values chosen
against data that includes the test fold.
"""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.data.config import FIGURES_DIR
from src.data.eda import CHURN_PALETTE, save_fig
from src.data.load_data import TARGET_COLUMN
from src.models.segmentation import compute_service_count

logger = logging.getLogger(__name__)

CONTRACT_RISK_VALUE = "Month-to-month"
PAYMENT_RISK_VALUE = "Electronic check"
TECHSUPPORT_RISK_VALUE = "No"
INTERNET_RISK_VALUE = "Fiber optic"
N_RISK_SIGNALS = 4
AT_RISK_SIGNAL_THRESHOLD = 3
NEW_TENURE_MAX = 12
LOYAL_TENURE_MIN = 24
LOYAL_SERVICE_MIN = 5

STAGE_NEW = "New"
STAGE_ENGAGED = "Engaged"
STAGE_LOYAL = "Loyal"
STAGE_AT_RISK = "At-Risk"
LIFECYCLE_STAGE_LABELS = [STAGE_NEW, STAGE_ENGAGED, STAGE_LOYAL, STAGE_AT_RISK]


def compute_risk_signal_count(df: pd.DataFrame) -> pd.Series:
    """Count of 4 CLAUDE.md §6 risk indicators present per row (0-4).

    Must not read Churn or tenure -- tenure drives the base stage split
    separately (see assign_lifecycle_stage), and including it here would
    double-count the same signal.
    """
    signals = pd.DataFrame({
        "contract": df["Contract"] == CONTRACT_RISK_VALUE,
        "payment": df["PaymentMethod"] == PAYMENT_RISK_VALUE,
        "tech_support": df["TechSupport"] == TECHSUPPORT_RISK_VALUE,
        "internet": df["InternetService"] == INTERNET_RISK_VALUE,
    })
    return signals.sum(axis=1).rename("RiskSignalCount")


def assign_lifecycle_stage(df: pd.DataFrame) -> pd.Series:
    """Priority-ordered stage: At-Risk (risk override) > New > Loyal > Engaged.

    Every row gets exactly one stage -- STAGE_ENGAGED is a catch-all, so
    unlike cohorts.py's bin edges there is no NaN path. Priority is enforced
    structurally, not by write order: is_new and is_loyal both carry
    ~is_at_risk, and is_loyal additionally carries ~is_new, so the three
    masks are disjoint by construction -- the order the three `stage[mask] =
    ...` assignments run in below cannot change the result.

    A row with NaN/out-of-range tenure (not expected from load_clean_data(),
    but possible from a hand-built caller frame) compares False on both the
    New and Loyal tenure checks and falls into Engaged; logged so it isn't
    silently invisible, mirroring cohorts.py's out-of-range-tenure handling.
    """
    if df["tenure"].isna().any():
        logger.warning(
            "%d row(s) have NaN tenure; they will fall into the Engaged catch-all "
            "rather than New or Loyal -- investigate before trusting stage counts.",
            df["tenure"].isna().sum(),
        )

    risk_count = compute_risk_signal_count(df)
    service_count = compute_service_count(df)

    stage = pd.Series(STAGE_ENGAGED, index=df.index)
    is_at_risk = risk_count >= AT_RISK_SIGNAL_THRESHOLD
    is_new = (~is_at_risk) & (df["tenure"] <= NEW_TENURE_MAX)
    is_loyal = (
        (~is_at_risk) & (~is_new)
        & (df["tenure"] >= LOYAL_TENURE_MIN)
        & (service_count >= LOYAL_SERVICE_MIN)
    )
    stage[is_new] = STAGE_NEW
    stage[is_loyal] = STAGE_LOYAL
    stage[is_at_risk] = STAGE_AT_RISK

    return pd.Series(
        pd.Categorical(stage, categories=LIFECYCLE_STAGE_LABELS, ordered=True),
        index=df.index, name="LifecycleStage",
    )


def lifecycle_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Per-stage customer count, avg tenure/ServiceCount, churn_rate (%), in stage order."""
    stage = assign_lifecycle_stage(df)
    working = pd.DataFrame({
        "tenure": df["tenure"],
        "ServiceCount": compute_service_count(df),
        "stage": stage,
        TARGET_COLUMN: df[TARGET_COLUMN],
    })
    grouped = working.groupby("stage", observed=False).agg(
        customers=("stage", "size"),
        avg_tenure=("tenure", "mean"),
        avg_service_count=("ServiceCount", "mean"),
        churn_rate=(TARGET_COLUMN, "mean"),
    )
    grouped = grouped.reindex(LIFECYCLE_STAGE_LABELS)
    if grouped["customers"].sum() != len(df):
        logger.warning(
            "lifecycle_summary customers (%d) does not match len(df) (%d); "
            "some rows have an unassigned stage -- investigate before trusting totals.",
            grouped["customers"].sum(), len(df),
        )
    grouped["avg_tenure"] = grouped["avg_tenure"].round(2)
    grouped["avg_service_count"] = grouped["avg_service_count"].round(2)
    grouped["churn_rate"] = (grouped["churn_rate"] * 100).round(2)
    return grouped.reset_index(names="stage")


def risk_signal_diagnostic(df: pd.DataFrame) -> pd.DataFrame:
    """churn_rate (%) per raw risk_count 0..N_RISK_SIGNALS -- disclosure table, not decisive at call time.

    Reindexed over the full 0..N_RISK_SIGNALS range (not just observed
    values) so a filtered/small input frame still returns one row per
    possible count -- a plain groupby would silently drop any count value
    absent from that particular slice, the same failure mode
    lifecycle_summary/cohort_summary/segment_summary all guard against.
    """
    risk_count = compute_risk_signal_count(df)
    working = pd.DataFrame({"risk_count": risk_count, TARGET_COLUMN: df[TARGET_COLUMN]})
    grouped = working.groupby("risk_count")[TARGET_COLUMN].agg(churn_rate="mean", customers="size")
    grouped = grouped.reindex(range(N_RISK_SIGNALS + 1))
    grouped["customers"] = grouped["customers"].fillna(0).astype(int)
    grouped["churn_rate"] = (grouped["churn_rate"] * 100).round(2)
    return grouped.reset_index(names="risk_count")[["risk_count", "customers", "churn_rate"]]


def plot_lifecycle_stage_distribution(df: pd.DataFrame, out_dir: Path = FIGURES_DIR) -> Path:
    """Bar chart of customer count per stage. NOT a monotonically-decreasing funnel -- see spec."""
    summary = lifecycle_summary(df)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(summary["stage"], summary["customers"], color=CHURN_PALETTE[0])
    ax.set_xlabel("Lifecycle stage")
    ax.set_ylabel("Number of customers")
    ax.set_title("Customer Lifecycle Stage Distribution (snapshot, not a conversion funnel)")
    for i, v in enumerate(summary["customers"]):
        ax.text(i, v + 30, f"{v:,}", ha="center")
    return save_fig(fig, "lifecycle_stage_distribution.png", out_dir)


def plot_lifecycle_churn_rate(df: pd.DataFrame, out_dir: Path = FIGURES_DIR) -> Path:
    """Bar chart of churn_rate per stage, stage order."""
    summary = lifecycle_summary(df)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(summary["stage"], summary["churn_rate"], color=CHURN_PALETTE[1])
    ax.set_xlabel("Lifecycle stage")
    ax.set_ylabel("Churn rate (%)")
    ax.set_title("Churn Rate by Lifecycle Stage")
    for i, v in enumerate(summary["churn_rate"]):
        ax.text(i, v + 1, f"{v:.2f}%", ha="center")
    return save_fig(fig, "lifecycle_churn_rate.png", out_dir)


def generate_lifecycle_figures(df: pd.DataFrame, out_dir: Path = FIGURES_DIR) -> list[Path]:
    """Generate and save every lifecycle chart."""
    return [plot_lifecycle_stage_distribution(df, out_dir), plot_lifecycle_churn_rate(df, out_dir)]


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    from src.data.load_data import load_clean_data

    df = load_clean_data()
    paths = generate_lifecycle_figures(df)
    logger.info("Saved %d lifecycle figures to reports/figures/", len(paths))


if __name__ == "__main__":
    main()
