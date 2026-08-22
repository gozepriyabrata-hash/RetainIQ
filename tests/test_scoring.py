import pandas as pd
import pytest

import src.models.calibration as calibration
import src.models.scoring as scoring
import src.models.train as train
from src.data.load_data import TARGET_COLUMN, load_raw_data


@pytest.fixture
def sample_raw_customers():
    """A handful of real raw customer rows (unlabeled -- Churn stripped)."""
    raw = load_raw_data().head(5).copy()
    return raw.drop(columns=[TARGET_COLUMN])


def test_score_customers_returns_probability_and_percentage_columns(sample_raw_customers):
    result = scoring.score_customers(sample_raw_customers)

    assert "churn_probability" in result.columns
    assert "churn_probability_pct" in result.columns
    for proba, pct in zip(result["churn_probability"], result["churn_probability_pct"]):
        assert pct == pytest.approx(round(proba * 100, scoring.PERCENTAGE_DECIMALS))


def test_score_customers_matches_pipeline_predict_proba(clean_df, sample_raw_customers):
    X_train, X_test, y_train, y_test = train.split_data(clean_df)
    pipeline = calibration.load_calibrated_model()

    result = scoring.score_customers(sample_raw_customers, pipeline=pipeline)

    from src.data.load_data import prepare_scoring_input
    features_df, _ = prepare_scoring_input(sample_raw_customers)
    import json
    metadata = json.loads(calibration.DEFAULT_CALIBRATED_METADATA_PATH.read_text())
    expected_proba = pipeline.predict_proba(features_df[metadata["feature_columns"]])[:, 1]

    for actual, expected in zip(result["churn_probability"], expected_proba):
        assert actual == pytest.approx(round(expected, scoring.PROBABILITY_DECIMALS))


def test_score_customers_empty_input_returns_empty_output(sample_raw_customers):
    empty = sample_raw_customers.iloc[0:0]

    result = scoring.score_customers(empty)

    assert len(result) == 0
    assert "churn_probability" in result.columns
    assert "churn_probability_pct" in result.columns


def test_score_customers_raises_on_missing_required_column(sample_raw_customers):
    broken = sample_raw_customers.drop(columns=["Contract"])

    with pytest.raises(ValueError, match="Contract"):
        scoring.score_customers(broken)


def test_score_customers_omits_customer_id_when_absent(sample_raw_customers):
    no_id = sample_raw_customers.drop(columns=["customerID"])

    result = scoring.score_customers(no_id)

    assert "customerID" not in result.columns


def test_score_customers_includes_customer_id_when_present():
    raw = load_raw_data().head(3).drop(columns=[TARGET_COLUMN])
    assert "customerID" in raw.columns

    result = scoring.score_customers(raw)

    assert list(result["customerID"]) == list(raw["customerID"])


def test_score_customers_uses_calibrated_by_default(sample_raw_customers):
    calibrated_result = scoring.score_customers(sample_raw_customers, use_calibrated=True)
    raw_result = scoring.score_customers(sample_raw_customers, use_calibrated=False)

    # Raw vs. calibrated probabilities are verified (spec research) to
    # differ meaningfully on real data -- at least one of these 5 sample
    # rows should show a difference.
    assert not (calibrated_result["churn_probability"] == raw_result["churn_probability"]).all()


def test_score_customers_raises_when_calibrated_model_missing(sample_raw_customers, monkeypatch, tmp_path):
    # score_customers reads feature_columns from the real metadata sidecar
    # (untouched here) then calls calibration.load_calibrated_model() with
    # no args -- redirect that call itself to a nonexistent path so the real
    # FileNotFoundError/message is exercised without relying on
    # load_calibrated_model's def-time-bound default argument picking up a
    # monkeypatched module constant (it wouldn't).
    original_load_calibrated_model = calibration.load_calibrated_model

    def load_from_missing_path():
        return original_load_calibrated_model(tmp_path / "nonexistent.pkl")

    monkeypatch.setattr(calibration, "load_calibrated_model", load_from_missing_path)

    with pytest.raises(FileNotFoundError, match="python -m src.models.calibration"):
        scoring.score_customers(sample_raw_customers, use_calibrated=True)


def test_score_single_customer_returns_flat_dict():
    raw = load_raw_data().head(1).drop(columns=[TARGET_COLUMN]).iloc[0].to_dict()

    result = scoring.score_single_customer(raw)

    assert isinstance(result, dict)
    assert "churn_probability" in result
    assert "churn_probability_pct" in result
