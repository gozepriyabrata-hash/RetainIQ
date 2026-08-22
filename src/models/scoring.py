"""Per-customer churn probability scoring: raw customer attributes in, a
calibrated 0-100% churn probability out.

This is the scoring layer a future Phase 5 FastAPI `/predict` endpoint will
call directly -- no route or dashboard view is wired up here (see
.claude/specs/09-probability-scoring.md's Non-goals). Uses the calibrated
model (src.models.calibration) by default, since its probabilities are
verified to be meaningfully closer to observed frequencies than the raw
Phase 2 model's.
"""

import json

import numpy as np
import pandas as pd

from src.data.load_data import prepare_scoring_input
from src.models import calibration, train

PROBABILITY_DECIMALS = 4
PERCENTAGE_DECIMALS = 1


def score_customers(
    raw_df: pd.DataFrame, pipeline=None, use_calibrated: bool = True
) -> pd.DataFrame:
    """Score every row of raw_df; returns customerID (if present), churn_probability, churn_probability_pct.

    raw_df may be raw/unlabeled customer data (no Churn column required) --
    prepare_scoring_input() applies the same TotalCharges/whitespace/
    SeniorCitizen normalization clean_data() uses for training. Missing
    required feature columns raise one ValueError naming all of them, not
    just the first. An empty raw_df returns an empty result with the same
    columns, not an error.
    """
    features_df, customer_ids = prepare_scoring_input(raw_df)

    metadata_path = (
        calibration.DEFAULT_CALIBRATED_METADATA_PATH if use_calibrated
        else train.DEFAULT_METADATA_PATH
    )
    metadata = json.loads(metadata_path.read_text())
    feature_columns = metadata["feature_columns"]

    missing = [c for c in feature_columns if c not in features_df.columns]
    if missing:
        raise ValueError(f"raw_df is missing required feature columns: {missing}")
    features_df = features_df[feature_columns]

    if pipeline is None:
        pipeline = (
            calibration.load_calibrated_model() if use_calibrated
            else train.load_trained_model()
        )

    if len(features_df) == 0:
        proba = np.array([])
    else:
        proba = pipeline.predict_proba(features_df)[:, 1]

    result = pd.DataFrame(index=raw_df.index)
    if customer_ids is not None:
        result["customerID"] = customer_ids
    result["churn_probability"] = np.round(proba, PROBABILITY_DECIMALS)
    percentage = np.clip(proba * 100, 0.0, 100.0)
    result["churn_probability_pct"] = np.round(percentage, PERCENTAGE_DECIMALS)
    return result


def score_single_customer(customer: dict, pipeline=None, use_calibrated: bool = True) -> dict:
    """Score one customer dict; returns a flat dict, the shape a future /predict endpoint returns."""
    return score_customers(pd.DataFrame([customer]), pipeline, use_calibrated).iloc[0].to_dict()
