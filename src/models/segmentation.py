"""RFM-style customer value segmentation: derived ServiceCount + K-Means value tiers.

Substitutes tenure (loyalty/Recency), MonthlyCharges (Monetary), and a derived
ServiceCount (service breadth, standing in for Frequency) for classic RFM,
since this dataset has no transaction/order log -- see
.claude/specs/04-customer-segmentation.md for the full methodology note.

Reporting-only, like cohorts.py: nothing here is written back into
load_clean_data()'s output -- segmentation.py computes ValueSegment internally
per call. If a future phase feeds ValueSegment into the Phase 2 supervised
model, the K-Means fit is data-dependent (unlike cohorts.py's fixed tenure
bin edges) and MUST be refit inside each CV fold on the training split only
-- not precomputed once and joined in as a static column. Not implemented
here; flagged for whoever makes that call later.
"""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data.config import FIGURES_DIR
from src.data.eda import CHURN_PALETTE, save_fig
from src.data.load_data import TARGET_COLUMN

logger = logging.getLogger(__name__)

SERVICE_YESNO_COLUMNS = [
    "PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
]
N_VALUE_SEGMENTS = 3
VALUE_SEGMENT_LABELS = ["Low", "Mid", "High"]
RANDOM_STATE = 42
KMEANS_N_INIT = 10
SILHOUETTE_K_VALUES = (2, 3, 4, 5, 6)


def compute_service_count(df: pd.DataFrame) -> pd.Series:
    """Count subscribed services per customer: 8 Yes/No add-ons + a has-internet flag.

    "No phone service" / "No internet service" sentinel values never equal
    "Yes", so no special-casing is needed. Must not read Churn (leakage guard;
    directly tested via build_segmentation_features).
    """
    count = (df[SERVICE_YESNO_COLUMNS] == "Yes").sum(axis=1) + (df["InternetService"] != "No").astype(int)
    return count.rename("ServiceCount")


def build_segmentation_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return the 3-column clustering feature frame: tenure, MonthlyCharges, ServiceCount.

    Leakage guard surface: MUST NOT include Churn or any churn-derived column
    -- directly tested in tests/test_segmentation.py.
    """
    return pd.DataFrame({
        "tenure": df["tenure"],
        "MonthlyCharges": df["MonthlyCharges"],
        "ServiceCount": compute_service_count(df),
    })


def fit_segmentation_pipeline(
    df: pd.DataFrame, n_clusters: int = N_VALUE_SEGMENTS, random_state: int = RANDOM_STATE
) -> Pipeline:
    """Fit StandardScaler -> KMeans(n_init=KMEANS_N_INIT) on build_segmentation_features(df)."""
    features = build_segmentation_features(df)
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("kmeans", KMeans(n_clusters=n_clusters, random_state=random_state, n_init=KMEANS_N_INIT)),
    ])
    pipeline.fit(features)
    return pipeline


def rank_segments_by_value(pipeline: Pipeline) -> dict[int, str]:
    """Map each raw KMeans cluster id to Low/Mid/High by ascending mean standardized centroid.

    Turns arbitrary cluster IDs into a business-meaningful ordering. The
    composite score weights the 3 standardized dimensions equally, so the
    resulting order is a composite rank, not a ranking on any single metric
    -- e.g. on this dataset "Low" has a longer average tenure than "Mid"
    despite spending less, because tenure alone doesn't dominate the score.

    Raises ValueError if the pipeline wasn't fit with exactly
    len(VALUE_SEGMENT_LABELS) clusters -- silently mapping only the first 3
    cluster ids and leaving the rest unlabeled (NaN downstream) would drop
    customers from every summary without any visible signal, the same
    failure mode cohorts.py's cohort_summary explicitly guards against.
    """
    centers = pipeline.named_steps["kmeans"].cluster_centers_
    if len(centers) != len(VALUE_SEGMENT_LABELS):
        raise ValueError(
            f"rank_segments_by_value expects exactly {len(VALUE_SEGMENT_LABELS)} clusters "
            f"(one per label in {VALUE_SEGMENT_LABELS}), got {len(centers)}."
        )
    composite = centers.mean(axis=1)
    order = np.argsort(composite)  # ascending composite -> Low first
    return {int(cluster_id): label for cluster_id, label in zip(order, VALUE_SEGMENT_LABELS)}


def assign_value_segment(df: pd.DataFrame, pipeline: Pipeline | None = None) -> pd.Series:
    """Return an ordered Categorical (Low/Mid/High) Series aligned to df's index.

    Fits a new pipeline via fit_segmentation_pipeline(df) if none is given.
    """
    if pipeline is None:
        pipeline = fit_segmentation_pipeline(df)
    features = build_segmentation_features(df)
    cluster_ids = pipeline.predict(features)
    label_map = rank_segments_by_value(pipeline)
    labels = pd.Series(cluster_ids, index=df.index).map(label_map)
    segment = pd.Categorical(labels, categories=VALUE_SEGMENT_LABELS, ordered=True)
    return pd.Series(segment, index=df.index, name="ValueSegment")


def segment_summary(df: pd.DataFrame, pipeline: Pipeline | None = None) -> pd.DataFrame:
    """Per-segment customer count, avg tenure/MonthlyCharges/ServiceCount, and churn_rate (%).

    One row per segment in VALUE_SEGMENT_LABELS order (Low, Mid, High);
    reindexed so no segment is dropped or reordered even if empty. Accepts an
    already-fit `pipeline` (e.g. from fit_segmentation_pipeline) so callers
    that need both the summary and the charts don't each independently refit
    KMeans -- see generate_segmentation_figures.
    """
    segment = assign_value_segment(df, pipeline)
    features = build_segmentation_features(df)
    working = features.assign(segment=segment)
    working[TARGET_COLUMN] = df[TARGET_COLUMN]
    grouped = working.groupby("segment", observed=False).agg(
        customers=("segment", "size"),
        avg_tenure=("tenure", "mean"),
        avg_monthly_charges=("MonthlyCharges", "mean"),
        avg_service_count=("ServiceCount", "mean"),
        churn_rate=(TARGET_COLUMN, "mean"),
    )
    grouped = grouped.reindex(VALUE_SEGMENT_LABELS)
    if grouped["customers"].sum() != len(df):
        logger.warning(
            "segment_summary customers (%d) does not match len(df) (%d); "
            "some rows have an unassigned segment -- investigate before trusting totals.",
            grouped["customers"].sum(), len(df),
        )
    for col in ["avg_tenure", "avg_monthly_charges", "avg_service_count"]:
        grouped[col] = grouped[col].round(2)
    grouped["churn_rate"] = (grouped["churn_rate"] * 100).round(2)
    return grouped.reset_index(names="segment")


def silhouette_diagnostic(df: pd.DataFrame, k_values: tuple[int, ...] = SILHOUETTE_K_VALUES) -> pd.DataFrame:
    """Silhouette score per k, each k fit via fit_segmentation_pipeline on the same features.

    Disclosure table only -- k=3 (N_VALUE_SEGMENTS) is a fixed business
    choice, never overridden by this diagnostic. Reuses
    fit_segmentation_pipeline (rather than hand-building StandardScaler +
    KMeans again) so this table always describes the exact fit the module
    ships, not a parallel construction that could silently drift from it.
    """
    rows = []
    for k in k_values:
        pipeline = fit_segmentation_pipeline(df, n_clusters=k)
        scaled_features = pipeline.named_steps["scaler"].transform(build_segmentation_features(df))
        labels = pipeline.named_steps["kmeans"].labels_
        rows.append({"k": k, "silhouette_score": round(silhouette_score(scaled_features, labels), 4)})
    return pd.DataFrame(rows, columns=["k", "silhouette_score"])


def plot_segment_churn_rate(df: pd.DataFrame, out_dir: Path = FIGURES_DIR, pipeline: Pipeline | None = None) -> Path:
    """Bar chart of churn_rate per segment, Low/Mid/High order."""
    summary = segment_summary(df, pipeline)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(summary["segment"], summary["churn_rate"], color=CHURN_PALETTE[1])
    ax.set_xlabel("Value segment")
    ax.set_ylabel("Churn rate (%)")
    ax.set_title("Churn Rate by Customer Value Segment")
    for i, v in enumerate(summary["churn_rate"]):
        ax.text(i, v + 1, f"{v:.2f}%", ha="center")
    return save_fig(fig, "segment_churn_rate.png", out_dir)


def plot_segment_profile(df: pd.DataFrame, out_dir: Path = FIGURES_DIR, pipeline: Pipeline | None = None) -> Path:
    """Grouped bar chart of avg_tenure/avg_monthly_charges/avg_service_count by segment.

    Each metric is normalized to its own max across the 3 segments
    (metric / metric.max()) before plotting, since the three metrics live on
    very different scales (months vs dollars vs a 0-9 count).
    """
    summary = segment_summary(df, pipeline)
    metrics = ["avg_tenure", "avg_monthly_charges", "avg_service_count"]
    normalized = summary[metrics] / summary[metrics].max()
    x = np.arange(len(summary))
    width = 0.25
    fig, ax = plt.subplots(figsize=(7, 4))
    for i, metric in enumerate(metrics):
        offset = (i - (len(metrics) - 1) / 2) * width
        ax.bar(x + offset, normalized[metric], width, label=metric)
    ax.set_xticks(x)
    ax.set_xticklabels(summary["segment"])
    ax.set_xlabel("Value segment")
    ax.set_ylabel("Value normalized to segment max (0-1)")
    ax.set_title("Segment Profile: Tenure, Monthly Charges, Service Count (normalized)")
    ax.legend()
    return save_fig(fig, "segment_profile.png", out_dir)


def generate_segmentation_figures(df: pd.DataFrame, out_dir: Path = FIGURES_DIR) -> list[Path]:
    """Generate and save every segmentation chart, fitting the K-Means pipeline once and reusing it."""
    pipeline = fit_segmentation_pipeline(df)
    return [
        plot_segment_churn_rate(df, out_dir, pipeline),
        plot_segment_profile(df, out_dir, pipeline),
    ]


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    from src.data.load_data import load_clean_data

    df = load_clean_data()
    paths = generate_segmentation_figures(df)
    logger.info("Saved %d segmentation figures to reports/figures/", len(paths))


if __name__ == "__main__":
    main()
