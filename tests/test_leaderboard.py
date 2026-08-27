from pathlib import Path

import pandas as pd
import pytest

from src.models.leaderboard import (
    CHART_METRICS,
    best_model_name,
    best_model_row,
    format_leaderboard_table,
    load_leaderboard,
    plot_leaderboard_metrics,
)

REQUIRED_ROW_KEYS = [
    "name", "cv_auc_mean", "cv_auc_std", "test_auc", "test_pr_auc",
    "test_precision", "test_recall", "test_f1", "test_brier",
    "tuned_threshold", "meets_target_auc",
]


def _row(name: str, cv_auc_mean: float, test_auc: float, meets_target: bool = False) -> dict:
    """A fully-populated synthetic comparison row, cheap to vary per test."""
    return {
        "name": name,
        "cv_auc_mean": cv_auc_mean,
        "cv_auc_std": 0.01,
        "test_auc": test_auc,
        "test_pr_auc": 0.60,
        "test_precision": 0.55,
        "test_recall": 0.65,
        "test_f1": 0.60,
        "test_brier": 0.15,
        "tuned_threshold": 0.45,
        "meets_target_auc": meets_target,
    }


def _write_csv(path: Path, rows: list[dict]) -> Path:
    pd.DataFrame(rows, columns=REQUIRED_ROW_KEYS).to_csv(path, index=False)
    return path


def test_load_leaderboard_returns_sorted_by_cv_auc_desc(tmp_path):
    csv_path = _write_csv(
        tmp_path / "model_comparison.csv",
        [
            _row("RandomForest", 0.80, 0.79),
            _row("XGBoost", 0.85, 0.84),
            _row("LogisticRegression", 0.82, 0.83),
        ],
    )
    df = load_leaderboard(csv_path)
    assert df["cv_auc_mean"].is_monotonic_decreasing
    assert df.iloc[0]["name"] == "XGBoost"


def test_load_leaderboard_raises_actionable_error_when_missing(tmp_path):
    with pytest.raises(FileNotFoundError, match="python -m src.models.train"):
        load_leaderboard(tmp_path / "nonexistent.csv")


def test_load_leaderboard_raises_on_missing_column(tmp_path):
    rows = [_row("XGBoost", 0.85, 0.84)]
    df = pd.DataFrame(rows).drop(columns=["test_brier"])
    csv_path = tmp_path / "model_comparison.csv"
    df.to_csv(csv_path, index=False)

    with pytest.raises(ValueError, match="test_brier"):
        load_leaderboard(csv_path)


def test_load_leaderboard_default_path_reads_real_comparison_csv():
    # Exercises load_leaderboard()'s own default (train.COMPARISON_TABLE_PATH
    # resolved at call time, not bound at import time) against the real,
    # repo-tracked reports/model_comparison.csv -- see leaderboard.py's
    # docstring on why this must be call-time, not a literal default value.
    df = load_leaderboard()
    assert set(df["name"]) == {"LogisticRegression", "RandomForest", "XGBoost", "LightGBM"}
    assert df["cv_auc_mean"].is_monotonic_decreasing


def test_best_model_name_matches_max_cv_auc_mean():
    # Deliberately disagreeing winners: LightGBM leads on test_auc,
    # XGBoost leads on cv_auc_mean -- best_model_name must follow cv_auc_mean.
    df = pd.DataFrame([
        _row("XGBoost", cv_auc_mean=0.8466, test_auc=0.8434),
        _row("LightGBM", cv_auc_mean=0.8463, test_auc=0.8437),
    ])
    assert best_model_name(df) == "XGBoost"


def test_best_model_row_returns_full_row_for_max_cv_auc_mean():
    df = pd.DataFrame([
        _row("XGBoost", cv_auc_mean=0.8466, test_auc=0.8434),
        _row("LightGBM", cv_auc_mean=0.8463, test_auc=0.8437),
    ])
    row = best_model_row(df)
    assert row["name"] == "XGBoost"
    assert row["test_auc"] == pytest.approx(0.8434)


def test_best_model_row_raises_on_empty_dataframe():
    df = pd.DataFrame(columns=REQUIRED_ROW_KEYS)
    with pytest.raises(ValueError):
        best_model_row(df)


def test_format_leaderboard_table_columns_are_readable_and_input_unmutated():
    df = pd.DataFrame([
        _row("XGBoost", 0.846555, 0.843411, meets_target=False),
        _row("LogisticRegression", 0.844901, 0.839087, meets_target=True),
    ])
    original_meets_dtype = df["meets_target_auc"].dtype

    display = format_leaderboard_table(df)

    assert list(display.columns) == [
        "Model", "CV AUC-ROC", "CV AUC-ROC Std", "Test AUC-ROC", "Test PR-AUC",
        "Precision", "Recall", "F1", "Brier Score", "Tuned Threshold",
        "Meets 0.85 Target",
    ]
    assert set(display["Meets 0.85 Target"]) <= {"Yes", "No"}
    assert display.loc[display["Model"] == "LogisticRegression", "Meets 0.85 Target"].iloc[0] == "Yes"
    assert display.loc[display["Model"] == "XGBoost", "CV AUC-ROC"].iloc[0] == 0.8466
    # Input must not be mutated -- same dtype and still boolean, not "Yes"/"No".
    assert df["meets_target_auc"].dtype == original_meets_dtype
    assert set(df["meets_target_auc"].unique()) <= {True, False}


def test_plot_leaderboard_metrics_has_one_trace_per_metric_and_excludes_brier():
    df = pd.DataFrame([
        _row("XGBoost", 0.8466, 0.8434),
        _row("LightGBM", 0.8463, 0.8437),
    ])
    fig = plot_leaderboard_metrics(df)

    assert len(fig.data) == len(CHART_METRICS)
    trace_names = {trace.name for trace in fig.data}
    assert not any("brier" in name.lower() for name in trace_names)


def test_plot_leaderboard_metrics_covers_all_models():
    df = pd.DataFrame([
        _row("XGBoost", 0.8466, 0.8434),
        _row("LightGBM", 0.8463, 0.8437),
        _row("LogisticRegression", 0.8449, 0.8391),
        _row("RandomForest", 0.8195, 0.8189),
    ])
    fig = plot_leaderboard_metrics(df)

    for trace in fig.data:
        assert set(trace.x) == set(df["name"])
