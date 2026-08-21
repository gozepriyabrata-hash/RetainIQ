"""Reusable EDA analysis and chart-generation helpers for the Telco churn data.

Kept out of the notebook so notebooks/01_eda.ipynb only calls into this module
instead of embedding analysis or plotting logic inline.
"""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.data.config import FIGURES_DIR
from src.data.load_data import TARGET_COLUMN

logger = logging.getLogger(__name__)

SEGMENT_COLUMNS = ["Contract", "InternetService", "PaymentMethod", "TechSupport", "SeniorCitizen"]
NUMERIC_COLUMNS = ["tenure", "MonthlyCharges", "TotalCharges"]
CHURN_PALETTE = {0: "#4C72B0", 1: "#C44E52"}
IQR_MULTIPLIER = 1.5

sns.set_theme(style="whitegrid")


def churn_rate_by_segment(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Return churn rate and customer count per category of `column`."""
    grouped = df.groupby(column)[TARGET_COLUMN].agg(churn_rate="mean", customers="size")
    grouped["churn_rate"] = (grouped["churn_rate"] * 100).round(2)
    return grouped.sort_values("churn_rate", ascending=False).reset_index()


def class_balance(df: pd.DataFrame) -> pd.DataFrame:
    """Return the Churn class counts and percentages."""
    counts = df[TARGET_COLUMN].value_counts().rename({0: "No", 1: "Yes"})
    pct = (counts / counts.sum() * 100).round(2)
    return pd.DataFrame({"count": counts, "pct": pct})


def _save_fig(fig: plt.Figure, filename: str, out_dir: Path = FIGURES_DIR) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / filename
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_class_balance(df: pd.DataFrame, out_dir: Path = FIGURES_DIR) -> Path:
    fig, ax = plt.subplots(figsize=(5, 4))
    counts = df[TARGET_COLUMN].value_counts().rename({0: "No", 1: "Yes"})
    sns.barplot(x=counts.index, y=counts.values, ax=ax, hue=counts.index, palette="Set2", legend=False)
    ax.set_title("Churn Class Balance")
    ax.set_xlabel("Churn")
    ax.set_ylabel("Number of Customers")
    for i, v in enumerate(counts.values):
        ax.text(i, v + 30, f"{v} ({v / counts.sum():.1%})", ha="center")
    return _save_fig(fig, "class_balance.png", out_dir)


def plot_churn_rate_by_segment(df: pd.DataFrame, column: str, out_dir: Path = FIGURES_DIR) -> Path:
    rates = churn_rate_by_segment(df, column)
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(data=rates, x=column, y="churn_rate", ax=ax, hue=column, palette="Reds_r", legend=False)
    ax.set_title(f"Churn Rate by {column}")
    ax.set_xlabel(column)
    ax.set_ylabel("Churn Rate (%)")
    ax.tick_params(axis="x", rotation=30)
    for tick in ax.get_xticklabels():
        tick.set_ha("right")
    return _save_fig(fig, f"churn_rate_by_{column}.png", out_dir)


def plot_tenure_distribution(df: pd.DataFrame, out_dir: Path = FIGURES_DIR) -> Path:
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.histplot(
        data=df, x="tenure", hue=TARGET_COLUMN, bins=30, multiple="stack",
        palette=CHURN_PALETTE, ax=ax,
    )
    ax.set_title("Tenure Distribution by Churn")
    ax.set_xlabel("Tenure (months)")
    ax.set_ylabel("Number of Customers")
    ax.legend(title="Churn", labels=["Yes", "No"])
    return _save_fig(fig, "tenure_distribution.png", out_dir)


def plot_charges_distribution(df: pd.DataFrame, column: str, out_dir: Path = FIGURES_DIR) -> Path:
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.kdeplot(
        data=df, x=column, hue=TARGET_COLUMN, fill=True, common_norm=False,
        palette=CHURN_PALETTE, ax=ax,
    )
    ax.set_title(f"{column} Distribution by Churn")
    ax.set_xlabel(column)
    ax.set_ylabel("Density")
    return _save_fig(fig, f"{column.lower()}_distribution.png", out_dir)


def plot_correlation_heatmap(df: pd.DataFrame, out_dir: Path = FIGURES_DIR) -> Path:
    numeric_df = df[NUMERIC_COLUMNS + [TARGET_COLUMN]]
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(numeric_df.corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Correlation Between Numeric Features and Churn")
    return _save_fig(fig, "correlation_heatmap.png", out_dir)


def _iqr_fence(s: pd.Series, k: float = IQR_MULTIPLIER) -> tuple[float, float, float, float, float]:
    """Return (q1, q3, iqr, lower_bound, upper_bound) for a Tukey IQR fence."""
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    return q1, q3, iqr, q1 - k * iqr, q3 + k * iqr


def detect_outliers_iqr(df: pd.DataFrame, column: str, k: float = IQR_MULTIPLIER) -> pd.DataFrame:
    """Return rows of `df` where `column` falls outside its Tukey IQR fence."""
    _, _, _, lower, upper = _iqr_fence(df[column], k)
    return df[(df[column] < lower) | (df[column] > upper)]


def summarize_outliers(
    df: pd.DataFrame, columns: list[str] | None = None, k: float = IQR_MULTIPLIER
) -> pd.DataFrame:
    """Return per-column IQR fence bounds and outlier counts."""
    columns = NUMERIC_COLUMNS if columns is None else columns
    rows = []
    for col in columns:
        q1, q3, iqr, lower, upper = _iqr_fence(df[col], k)
        n_outliers = len(detect_outliers_iqr(df, col, k))
        rows.append({
            "column": col, "q1": q1, "q3": q3, "iqr": iqr,
            "lower_bound": lower, "upper_bound": upper,
            "n_outliers": n_outliers,
            "pct_outliers": round(n_outliers / len(df) * 100, 2) if len(df) else 0.0,
        })
    return pd.DataFrame(rows, columns=[
        "column", "q1", "q3", "iqr", "lower_bound", "upper_bound", "n_outliers", "pct_outliers",
    ])


def distribution_stats(df: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    """Return mean/median/std/skew/kurtosis/min/max per numeric column."""
    columns = NUMERIC_COLUMNS if columns is None else columns
    rows = []
    for col in columns:
        s = df[col]
        rows.append({
            "column": col, "mean": s.mean(), "median": s.median(), "std": s.std(),
            "skew": s.skew(), "kurtosis": s.kurt(), "min": s.min(), "max": s.max(),
        })
    return pd.DataFrame(rows, columns=[
        "column", "mean", "median", "std", "skew", "kurtosis", "min", "max",
    ])


def plot_outlier_boxplot(df: pd.DataFrame, column: str, out_dir: Path = FIGURES_DIR) -> Path:
    """Boxplot of `column` split by churn status, for visual outlier inspection."""
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.boxplot(
        data=df, x=TARGET_COLUMN, y=column, hue=TARGET_COLUMN,
        palette=CHURN_PALETTE, legend=False, ax=ax,
    )
    ax.set_title(f"{column} by Churn (outlier check)")
    ax.set_xlabel("Churn")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["No", "Yes"])
    ax.set_ylabel(column)
    return _save_fig(fig, f"{column.lower()}_boxplot.png", out_dir)


def generate_all_figures(df: pd.DataFrame, out_dir: Path = FIGURES_DIR) -> list[Path]:
    """Generate and save every KPI chart used in the EDA notebooks."""
    paths = [plot_class_balance(df, out_dir)]
    for col in SEGMENT_COLUMNS:
        paths.append(plot_churn_rate_by_segment(df, col, out_dir))
    paths.append(plot_tenure_distribution(df, out_dir))
    paths.append(plot_charges_distribution(df, "MonthlyCharges", out_dir))
    paths.append(plot_charges_distribution(df, "TotalCharges", out_dir))
    paths.append(plot_correlation_heatmap(df, out_dir))
    for col in NUMERIC_COLUMNS:
        paths.append(plot_outlier_boxplot(df, col, out_dir))
    return paths


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    from src.data.load_data import load_clean_data

    df = load_clean_data()
    paths = generate_all_figures(df)
    logger.info("Saved %d figures to reports/figures/", len(paths))


if __name__ == "__main__":
    main()
