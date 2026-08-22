import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

import src.recommend.risk_tiers as risk_tiers
from src.data.load_data import TARGET_COLUMN, load_raw_data
from src.models import scoring


@pytest.fixture
def sample_raw_customers():
    """A handful of real raw customer rows (unlabeled -- Churn stripped)."""
    raw = load_raw_data().head(5).copy()
    return raw.drop(columns=[TARGET_COLUMN])


# --- classify_risk_tier -----------------------------------------------------


def test_classify_risk_tier_interior_values():
    assert risk_tiers.classify_risk_tier(0.10) == "Low"
    assert risk_tiers.classify_risk_tier(0.40) == "Medium"
    assert risk_tiers.classify_risk_tier(0.60) == "High"
    assert risk_tiers.classify_risk_tier(0.90) == "Critical"


def test_classify_risk_tier_boundaries():
    assert risk_tiers.classify_risk_tier(0.30) == "Medium"
    assert risk_tiers.classify_risk_tier(0.50) == "High"
    assert risk_tiers.classify_risk_tier(0.70) == "High"

    assert risk_tiers.classify_risk_tier(0.2999) == "Low"
    assert risk_tiers.classify_risk_tier(0.3001) == "Medium"
    assert risk_tiers.classify_risk_tier(0.4999) == "Medium"
    assert risk_tiers.classify_risk_tier(0.5001) == "High"
    assert risk_tiers.classify_risk_tier(0.6999) == "High"
    assert risk_tiers.classify_risk_tier(0.7001) == "Critical"


def test_classify_risk_tier_extremes():
    assert risk_tiers.classify_risk_tier(0.0) == "Low"
    assert risk_tiers.classify_risk_tier(1.0) == "Critical"


def test_classify_risk_tier_raises_out_of_range():
    with pytest.raises(ValueError):
        risk_tiers.classify_risk_tier(-0.01)
    with pytest.raises(ValueError):
        risk_tiers.classify_risk_tier(1.01)


def test_classify_risk_tier_raises_on_nan():
    with pytest.raises(ValueError):
        risk_tiers.classify_risk_tier(float("nan"))


# --- assign_risk_tiers -------------------------------------------------------


def test_assign_risk_tiers_matches_scalar_classifier():
    rng = np.random.default_rng(42)
    # Random interior draws plus the exact boundaries and their nearest
    # float neighbours either side -- the batch/scalar agreement claim is
    # only meaningful if it's actually tested at the boundaries the whole
    # spec exists to pin down, not just tier interiors.
    boundary_values = [
        0.0, 0.2999, 0.30, 0.3001, 0.4999, 0.50, 0.5001, 0.6999, 0.70, 0.7001, 1.0,
    ]
    probabilities = np.concatenate([rng.uniform(0.0, 1.0, size=50), boundary_values])
    df = pd.DataFrame({"churn_probability": probabilities})

    result = risk_tiers.assign_risk_tiers(df)

    for probability, tier in zip(df["churn_probability"], result["risk_tier"]):
        assert tier == risk_tiers.classify_risk_tier(probability)


def test_assign_risk_tiers_raises_on_nan():
    df = pd.DataFrame({"churn_probability": [0.1, float("nan"), 0.5]})

    with pytest.raises(ValueError, match="churn_probability"):
        risk_tiers.assign_risk_tiers(df)


def test_assign_risk_tiers_raises_on_out_of_range():
    df = pd.DataFrame({"churn_probability": [0.1, -0.01, 0.5]})
    with pytest.raises(ValueError, match="churn_probability"):
        risk_tiers.assign_risk_tiers(df)

    df = pd.DataFrame({"churn_probability": [0.1, 1.01, 0.5]})
    with pytest.raises(ValueError, match="churn_probability"):
        risk_tiers.assign_risk_tiers(df)


def test_assign_risk_tiers_raises_on_non_numeric_column():
    df = pd.DataFrame({"churn_probability": ["low", "medium", "high"]})

    with pytest.raises(ValueError, match="churn_probability"):
        risk_tiers.assign_risk_tiers(df)


def test_assign_risk_tiers_is_ordered_categorical():
    df = pd.DataFrame({"churn_probability": [0.1, 0.4, 0.6, 0.9]})

    result = risk_tiers.assign_risk_tiers(df)

    dtype = result["risk_tier"].dtype
    assert isinstance(dtype, pd.CategoricalDtype)
    assert dtype.ordered
    assert list(dtype.categories) == risk_tiers.RISK_TIER_LABELS


def test_assign_risk_tiers_raises_on_missing_column():
    df = pd.DataFrame({"not_probability": [0.1, 0.2]})

    with pytest.raises(ValueError, match="churn_probability"):
        risk_tiers.assign_risk_tiers(df)


def test_assign_risk_tiers_empty_input_returns_empty_output():
    df = pd.DataFrame({"churn_probability": pd.Series(dtype=float)})

    result = risk_tiers.assign_risk_tiers(df)

    assert len(result) == 0
    assert "risk_tier" in result.columns


# --- risk_tier_summary --------------------------------------------------------


def test_risk_tier_summary_counts_and_percentages():
    df = pd.DataFrame({"churn_probability": [0.1, 0.1, 0.4, 0.6, 0.6, 0.6, 0.9, 0.9, 0.9, 0.9]})

    summary = risk_tiers.risk_tier_summary(df)

    assert list(summary["risk_tier"]) == risk_tiers.RISK_TIER_ORDER_DESC
    counts = dict(zip(summary["risk_tier"], summary["count"]))
    assert counts == {"Critical": 4, "High": 3, "Medium": 1, "Low": 2}
    pcts = dict(zip(summary["risk_tier"], summary["pct"]))
    assert pcts == {"Critical": 40.0, "High": 30.0, "Medium": 10.0, "Low": 20.0}


def test_risk_tier_summary_empty_input_all_zero():
    df = pd.DataFrame({"churn_probability": pd.Series(dtype=float)})

    summary = risk_tiers.risk_tier_summary(df)

    assert list(summary["risk_tier"]) == risk_tiers.RISK_TIER_ORDER_DESC
    assert (summary["count"] == 0).all()
    assert (summary["pct"] == 0.0).all()


def test_risk_tier_summary_matches_verified_real_distribution():
    # Verified during spec research (.claude/specs/10-risk-classification.md):
    # scoring all 7,043 raw customers with the real calibrated model and
    # tallying tiers with this exact boundary rule. Intentionally brittle,
    # matching 08/09's precedent for real-data regression guards.
    raw = load_raw_data().drop(columns=[TARGET_COLUMN])
    scored = scoring.score_customers(raw)

    summary = risk_tiers.risk_tier_summary(scored)

    counts = dict(zip(summary["risk_tier"], summary["count"]))
    assert counts == {"Critical": 479, "High": 1055, "Medium": 1063, "Low": 4446}


# --- classify_scored_customers ------------------------------------------------


def test_classify_scored_customers_returns_probability_and_tier(sample_raw_customers):
    result = risk_tiers.classify_scored_customers(sample_raw_customers)

    assert "churn_probability" in result.columns
    assert "risk_tier" in result.columns
    for probability, tier in zip(result["churn_probability"], result["risk_tier"]):
        assert tier == risk_tiers.classify_risk_tier(probability)


def test_classify_scored_customers_propagates_scoring_errors(sample_raw_customers):
    broken = sample_raw_customers.drop(columns=["Contract"])

    with pytest.raises(ValueError, match="Contract"):
        risk_tiers.classify_scored_customers(broken)


# --- plot_risk_tier_distribution ----------------------------------------------


def test_plot_risk_tier_distribution_returns_figure():
    df = pd.DataFrame({"churn_probability": [0.1, 0.4, 0.6, 0.9]})

    fig = risk_tiers.plot_risk_tier_distribution(df)

    assert isinstance(fig, go.Figure)
    assert len(fig.data[0].x) == 4
