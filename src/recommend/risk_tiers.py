"""Risk-tier classification: turns a calibrated churn probability
(src.models.scoring) into one of four named tiers, per CLAUDE.md Sec 7
(Critical > 70%, High 50-70%, Medium 30-50%, Low < 30%).

`risk_tier` is a derived, post-hoc label computed purely from
`churn_probability` -- a model *output*, not an input. Nothing here writes
`risk_tier` (or `churn_probability`) back into load_clean_data()'s output
or any DataFrame src.features.preprocessing/src.models.train consume; the
data flow in this module is one-way, toward reporting/serving, never back
toward training. See .claude/specs/10-risk-classification.md for the full
methodology, the boundary-inclusivity resolution, and the verified
real-dataset distribution.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from src.models import scoring

MEDIUM_THRESHOLD = 0.30
HIGH_THRESHOLD = 0.50
CRITICAL_THRESHOLD = 0.70

RISK_TIER_LABELS = ["Low", "Medium", "High", "Critical"]
RISK_TIER_ORDER_DESC = list(reversed(RISK_TIER_LABELS))
RISK_TIER_PALETTE = {
    "Low": "#4C956C", "Medium": "#F2C14E", "High": "#F28C28", "Critical": "#C44E52",
}


def classify_risk_tier(probability: float) -> str:
    """Classify a single 0-1 churn probability into Low/Medium/High/Critical.

    Boundary rule (see spec's Interpretation note): Critical is strictly
    > CRITICAL_THRESHOLD, Low is strictly < MEDIUM_THRESHOLD -- the two
    ranges CLAUDE.md Sec 7 states with a strict inequality -- which forces
    High/Medium's shared boundaries to be inclusive on their lower end,
    giving four disjoint, gap-free ranges. Raises ValueError on NaN or a
    value outside [0, 1] rather than silently clipping or coercing it.
    """
    if pd.isna(probability) or not (0.0 <= probability <= 1.0):
        raise ValueError(f"probability must be in [0, 1], got {probability!r}")
    if probability > CRITICAL_THRESHOLD:
        return "Critical"
    if probability >= HIGH_THRESHOLD:
        return "High"
    if probability >= MEDIUM_THRESHOLD:
        return "Medium"
    return "Low"


def assign_risk_tiers(df: pd.DataFrame, probability_column: str = "churn_probability") -> pd.DataFrame:
    """Return a copy of df with a new ordered-Categorical "risk_tier" column.

    Vectorized (np.select) reproduction of classify_risk_tier's boundary
    rule -- np.select evaluates conditions top-down and takes the first
    match, so listing Critical -> High -> Medium -> default Low reproduces
    the scalar function's if/elif chain exactly (directly tested for
    row-by-row agreement between the two). Raises ValueError naming
    probability_column if it's absent, and ValueError naming the count of
    bad rows if any value is non-numeric, NaN, or outside [0, 1] -- the
    same guard classify_risk_tier applies per-value, so this function can
    never fail open into a "Low" tier for an unscoreable customer (NaN
    compares False against every np.select condition, so without this
    check it would otherwise fall through to the default). An empty df
    returns an empty df with the risk_tier column present, not an error.
    """
    if probability_column not in df.columns:
        raise ValueError(f"assign_risk_tiers: df is missing required column {probability_column!r}")

    raw_values = df[probability_column].to_numpy()
    p = pd.to_numeric(df[probability_column], errors="coerce").to_numpy(dtype=float)
    invalid = np.isnan(p) | (p < 0.0) | (p > 1.0)
    if invalid.any():
        bad_values = raw_values[invalid]
        raise ValueError(
            f"{probability_column!r} must contain only numeric values in [0, 1]; "
            f"found {invalid.sum()} invalid value(s), e.g. {bad_values[0]!r}"
        )

    labels = np.select(
        [p > CRITICAL_THRESHOLD, p >= HIGH_THRESHOLD, p >= MEDIUM_THRESHOLD],
        ["Critical", "High", "Medium"],
        default="Low",
    )

    result = df.copy()
    result["risk_tier"] = pd.Categorical(labels, categories=RISK_TIER_LABELS, ordered=True)
    return result


def classify_scored_customers(raw_df: pd.DataFrame, pipeline=None, use_calibrated: bool = True) -> pd.DataFrame:
    """Raw customer rows in, (churn_probability, churn_probability_pct, risk_tier) out.

    Thin composition of scoring.score_customers -> assign_risk_tiers --
    the full path a future Phase 5 /predict endpoint calls directly. Every
    score_customers edge case (empty input, missing required column,
    missing calibrated artifact) propagates unchanged since this never
    reimplements scoring, only pipes its output onward.
    """
    scored = scoring.score_customers(raw_df, pipeline, use_calibrated)
    return assign_risk_tiers(scored)


def risk_tier_summary(df: pd.DataFrame, probability_column: str = "churn_probability") -> pd.DataFrame:
    """One row per tier (Critical->Low, most-severe first): risk_tier, count, pct.

    Reindexed to RISK_TIER_ORDER_DESC so no tier is dropped or reordered
    even if empty (count=0, pct=0.0, not NaN) -- mirrors segmentation.py's
    segment_summary reindex-then-fill pattern rather than a bare
    value_counts(), which would drop empty categories.
    """
    tiered = assign_risk_tiers(df, probability_column)
    counts = tiered["risk_tier"].value_counts().reindex(RISK_TIER_ORDER_DESC).fillna(0).astype(int)
    total = len(df)
    pct = (counts / total * 100).round(2) if total else counts.astype(float)
    return pd.DataFrame({"risk_tier": counts.index, "count": counts.values, "pct": pct.values})


def plot_risk_tier_distribution(df: pd.DataFrame, probability_column: str = "churn_probability") -> go.Figure:
    """Bar chart of risk_tier_summary's counts, Critical->Low order, RISK_TIER_PALETTE colors.

    A live dashboard-style widget returned in memory, not saved to
    reports/figures/ -- mirrors src.data.kpi.plot_mrr_breakdown's shape.
    """
    summary = risk_tier_summary(df, probability_column)
    fig = go.Figure(
        go.Bar(
            x=summary["risk_tier"],
            y=summary["count"],
            marker_color=[RISK_TIER_PALETTE[tier] for tier in summary["risk_tier"]],
        )
    )
    fig.update_layout(title="Customer Risk Tier Distribution", yaxis_title="Number of customers")
    return fig
