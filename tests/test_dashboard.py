from pathlib import Path
from unittest.mock import patch

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from src.data.kpi import kpi_summary
from src.data.load_data import load_clean_data

DASHBOARD_PATH = str(Path(__file__).resolve().parents[1] / "app" / "dashboard.py")


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
