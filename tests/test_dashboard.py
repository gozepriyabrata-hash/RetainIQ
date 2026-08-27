from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from src.data.kpi import kpi_summary
from src.data.load_data import load_clean_data
from src.models import train

DASHBOARD_PATH = str(Path(__file__).resolve().parents[1] / "app" / "dashboard.py")

LEADERBOARD_COLUMNS = [
    "name", "cv_auc_mean", "cv_auc_std", "test_auc", "test_pr_auc",
    "test_precision", "test_recall", "test_f1", "test_brier",
    "tuned_threshold", "meets_target_auc",
]


def _write_leaderboard_csv(tmp_path: Path, rows: list[dict]) -> Path:
    csv_path = tmp_path / "model_comparison.csv"
    pd.DataFrame(rows, columns=LEADERBOARD_COLUMNS).to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture(autouse=True)
def _clear_streamlit_cache():
    """st.cache_data is a process-global cache, so a cached _load_data() result
    from one test would otherwise leak into the next (e.g. masking the patched
    load_clean_data in the missing-dataset test below)."""
    st.cache_data.clear()
    yield
    st.cache_data.clear()


def test_dashboard_renders_five_metrics_without_exception():
    at = AppTest.from_file(DASHBOARD_PATH, default_timeout=10).run()
    assert at.exception == []
    assert len(at.metric) == 5

    expected_labels = kpi_summary(load_clean_data())["kpi"].tolist()
    assert [m.label for m in at.metric] == expected_labels


def test_dashboard_shows_friendly_error_when_dataset_missing():
    with patch("src.data.load_data.load_clean_data", side_effect=FileNotFoundError):
        at = AppTest.from_file(DASHBOARD_PATH, default_timeout=10).run()
    assert at.exception == []
    assert len(at.error) == 1
    assert len(at.metric) == 0


def test_dashboard_refresh_button_reloads_without_exception():
    at = AppTest.from_file(DASHBOARD_PATH, default_timeout=10).run()
    at.button[0].click().run()
    assert at.exception == []
    assert len(at.metric) == 5


def test_dashboard_leaderboard_tab_renders_table_chart_and_best_model_callout(
    tmp_path, monkeypatch
):
    csv_path = _write_leaderboard_csv(
        tmp_path,
        [
            {
                "name": "XGBoost", "cv_auc_mean": 0.8466, "cv_auc_std": 0.0105,
                "test_auc": 0.8434, "test_pr_auc": 0.6466, "test_precision": 0.5616,
                "test_recall": 0.6952, "test_f1": 0.6213, "test_brier": 0.1466,
                "tuned_threshold": 0.47, "meets_target_auc": False,
            },
            {
                "name": "RandomForest", "cv_auc_mean": 0.8195, "cv_auc_std": 0.0110,
                "test_auc": 0.8189, "test_pr_auc": 0.5907, "test_precision": 0.5788,
                "test_recall": 0.5695, "test_f1": 0.5741, "test_brier": 0.1537,
                "tuned_threshold": 0.37, "meets_target_auc": False,
            },
        ],
    )
    monkeypatch.setattr(train, "COMPARISON_TABLE_PATH", csv_path)

    at = AppTest.from_file(DASHBOARD_PATH, default_timeout=10).run()
    leaderboard_tab = at.tabs[1]

    assert at.exception == []
    assert len(leaderboard_tab.dataframe) == 1
    assert len(leaderboard_tab.get("plotly_chart")) == 1
    # Neither fixture row meets the 0.85 target -- expect a warning, not success.
    assert len(leaderboard_tab.warning) == 1
    assert len(leaderboard_tab.success) == 0
    assert "XGBoost" in leaderboard_tab.warning[0].value


def test_dashboard_leaderboard_tab_shows_friendly_message_when_csv_missing(
    tmp_path, monkeypatch
):
    missing_path = tmp_path / "nonexistent.csv"
    monkeypatch.setattr(train, "COMPARISON_TABLE_PATH", missing_path)

    at = AppTest.from_file(DASHBOARD_PATH, default_timeout=10).run()
    leaderboard_tab = at.tabs[1]

    assert at.exception == []
    assert len(leaderboard_tab.info) == 1
    assert "python -m src.models.train" in leaderboard_tab.info[0].value


def test_dashboard_leaderboard_tab_shows_error_when_csv_malformed(tmp_path, monkeypatch):
    csv_path = tmp_path / "model_comparison.csv"
    pd.DataFrame([{"name": "XGBoost", "cv_auc_mean": 0.85}]).to_csv(csv_path, index=False)
    monkeypatch.setattr(train, "COMPARISON_TABLE_PATH", csv_path)

    at = AppTest.from_file(DASHBOARD_PATH, default_timeout=10).run()
    leaderboard_tab = at.tabs[1]

    assert at.exception == []
    assert len(leaderboard_tab.error) == 1
    # The malformed-CSV error must never leak the absolute file path to the
    # browser (see .claude/specs/12-model-comparison-leaderboard.md's
    # Security notes) -- only a generic, actionable message.
    assert str(csv_path) not in leaderboard_tab.error[0].value


def test_dashboard_leaderboard_tab_shows_error_when_csv_empty(tmp_path, monkeypatch):
    # All required columns present, zero data rows -- best_model_row's
    # idxmax() raises ValueError on this, which must be caught by the same
    # handler as the malformed-column case, not crash the page.
    csv_path = _write_leaderboard_csv(tmp_path, rows=[])
    monkeypatch.setattr(train, "COMPARISON_TABLE_PATH", csv_path)

    at = AppTest.from_file(DASHBOARD_PATH, default_timeout=10).run()
    leaderboard_tab = at.tabs[1]

    assert at.exception == []
    assert len(leaderboard_tab.error) == 1
    # No half-rendered empty table/chart alongside the error -- the
    # best-model lookup is resolved before anything is drawn (see
    # app/dashboard.py's _render_model_leaderboard).
    assert len(leaderboard_tab.dataframe) == 0
