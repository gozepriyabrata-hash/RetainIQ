"""Presentation layer over Phase 2's model comparison output for the live dashboard.

Reads `reports/model_comparison.csv` (written by `python -m src.models.train`,
see `src/models/train.py`) and turns it into a human-readable table and a
Plotly comparison chart. Pure functions only -- no Streamlit import -- so this
module is independently testable and keeps `app/dashboard.py` thin (CLAUDE.md
Section 4). See `.claude/specs/12-model-comparison-leaderboard.md`.
"""

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

from src.models import train
from src.models.evaluation import TARGET_AUC

REQUIRED_COLUMNS = {
    "name", "cv_auc_mean", "cv_auc_std", "test_auc", "test_pr_auc",
    "test_precision", "test_recall", "test_f1", "test_brier",
    "tuned_threshold", "meets_target_auc",
}

# The five "higher is better" metrics shown on the comparison chart.
# test_brier is deliberately excluded -- it's a lower-is-better error metric
# on different scale/semantics, and mixing it into this grouped bar chart
# would visually misrepresent it as "higher is better" (see
# plot_leaderboard_metrics's docstring).
CHART_METRICS = ("test_auc", "test_pr_auc", "test_precision", "test_recall", "test_f1")
CHART_METRIC_LABELS = {
    "test_auc": "Test AUC-ROC",
    "test_pr_auc": "Test PR-AUC",
    "test_precision": "Precision",
    "test_recall": "Recall",
    "test_f1": "F1",
}

# Column order and human-readable labels for the display table -- also acts
# as the column *selection* for format_leaderboard_table (any raw CSV column
# not listed here, e.g. test_f1_at_tuned_threshold, is intentionally omitted
# from the display for readability).
DISPLAY_COLUMNS = {
    "name": "Model",
    "cv_auc_mean": "CV AUC-ROC",
    "cv_auc_std": "CV AUC-ROC Std",
    "test_auc": "Test AUC-ROC",
    "test_pr_auc": "Test PR-AUC",
    "test_precision": "Precision",
    "test_recall": "Recall",
    "test_f1": "F1",
    "test_brier": "Brier Score",
    "tuned_threshold": "Tuned Threshold",
    "meets_target_auc": f"Meets {TARGET_AUC} Target",
}


def load_leaderboard(path: Path | None = None) -> pd.DataFrame:
    """Read + validate reports/model_comparison.csv, sorted by cv_auc_mean descending.

    Raises FileNotFoundError (actionable, names the fix) if `path` doesn't
    exist, mirroring train.load_trained_model's error pattern. Raises
    ValueError naming any missing required column(s) if the CSV is malformed
    or stale -- an actionable failure here instead of a KeyError deep inside
    a formatting or plotting call.

    `path` defaults to `train.COMPARISON_TABLE_PATH` read at call time (not
    bound as a literal default-argument value at module-import time), so
    monkeypatching `train.COMPARISON_TABLE_PATH` in a test -- or in a caller
    like the dashboard -- always takes effect, including through this
    default. See train.py's own DEFAULT_MODEL_PATH/FIGURES_DIR precedent.
    """
    if path is None:
        path = train.COMPARISON_TABLE_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"No model comparison found at {path}. Run "
            "`python -m src.models.train` first to produce it."
        )
    df = pd.read_csv(path)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"{path} is missing required column(s): {sorted(missing)}. "
            "Re-run `python -m src.models.train` to regenerate it."
        )
    return df.sort_values("cv_auc_mean", ascending=False).reset_index(drop=True)


def best_model_row(df: pd.DataFrame) -> pd.Series:
    """The row with the highest cv_auc_mean.

    Matches train.select_best_model's selection metric exactly (never
    test_auc), so this can never name a different model than the one
    `python -m src.models.train` actually persisted. Raises ValueError
    (pandas' native idxmax-on-empty behavior) if `df` has no rows -- an
    empty reports/model_comparison.csv is a malformed-input case, the same
    class of failure load_leaderboard's own checks guard against.
    """
    return df.loc[df["cv_auc_mean"].idxmax()]


def best_model_name(df: pd.DataFrame) -> str:
    """Name of the row with the highest cv_auc_mean -- see best_model_row."""
    return str(best_model_row(df)["name"])


def format_leaderboard_table(df: pd.DataFrame) -> pd.DataFrame:
    """Human-readable display copy: renamed columns, 4dp rounding, Yes/No target column.

    Returns a new DataFrame -- `df` itself is never mutated, so callers that
    also need the raw numeric precision (e.g. plot_leaderboard_metrics) can
    keep using it.
    """
    display = df.copy()
    numeric_columns = [
        c for c in DISPLAY_COLUMNS if c not in ("name", "meets_target_auc")
    ]
    display[numeric_columns] = display[numeric_columns].round(4)
    display["meets_target_auc"] = display["meets_target_auc"].map({True: "Yes", False: "No"})
    return display.rename(columns=DISPLAY_COLUMNS)[list(DISPLAY_COLUMNS.values())]


def plot_leaderboard_metrics(df: pd.DataFrame) -> go.Figure:
    """Grouped bar chart, one group per model, one bar per higher-is-better metric.

    test_brier is deliberately excluded from this chart -- see CHART_METRICS.
    """
    fig = go.Figure()
    for metric in CHART_METRICS:
        fig.add_trace(go.Bar(name=CHART_METRIC_LABELS[metric], x=df["name"], y=df[metric]))
    fig.update_layout(
        title="Model Comparison: Held-Out Test Metrics",
        barmode="group",
        xaxis_title="Model",
        yaxis_title="Score",
    )
    return fig
