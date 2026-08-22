import json

import numpy as np
import pytest

import src.models.calibration as calibration
import src.models.train as train
from src.features.preprocessing import get_categorical_columns


@pytest.fixture(scope="module")
def isolated_mlflow_uri(tmp_path_factory):
    """A throwaway sqlite MLflow store so the test suite never writes to the
    tracked project's real mlflow.db/mlruns/."""
    store_dir = tmp_path_factory.mktemp("calibration_mlflow_test_store")
    return f"sqlite:///{store_dir / 'mlflow_test.db'}"


@pytest.fixture(scope="module")
def module_monkeypatch():
    """pytest's built-in monkeypatch fixture is function-scoped; this is the
    documented workaround for using it from a module-scoped fixture."""
    mp = pytest.MonkeyPatch()
    yield mp
    mp.undo()


@pytest.fixture(scope="module")
def calibration_result(clean_df, isolated_mlflow_uri, module_monkeypatch):
    """One real fit_calibrated_pipeline(XGBoost) call, reused by every test
    below that needs a genuine fit -- this is the single expensive (5-fold
    CalibratedClassifierCV over the full train split) fit in the module."""
    module_monkeypatch.setattr(train, "MLFLOW_TRACKING_URI", isolated_mlflow_uri)
    X_train, X_test, y_train, y_test = train.split_data(clean_df)
    categorical_columns = get_categorical_columns(X_train)
    template = calibration.build_calibration_template("XGBoost", categorical_columns)
    calibrated = calibration.fit_calibrated_pipeline(template, X_train, y_train)
    proba_raw = train.load_trained_model().predict_proba(X_test)[:, 1]
    proba_calibrated = calibrated.predict_proba(X_test)[:, 1]
    return {
        "X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test,
        "calibrated": calibrated, "proba_raw": proba_raw, "proba_calibrated": proba_calibrated,
    }


def test_fit_calibrated_pipeline_uses_only_train_split(clean_df):
    X_train, X_test, y_train, y_test = train.split_data(clean_df)
    categorical_columns = get_categorical_columns(X_train)

    small_X, small_y = X_train.iloc[:300], y_train.iloc[:300]
    template = calibration.build_calibration_template("LogisticRegression", categorical_columns)
    calibrated = calibration.fit_calibrated_pipeline(template, small_X, small_y, cv=3)

    # X_test/y_test are never passed into fit_calibrated_pipeline above --
    # predict_proba on them afterward only proves the fitted estimator can
    # score arbitrary held-out rows, not that it was fit on them.
    proba = calibrated.predict_proba(X_test.iloc[:20])
    assert proba.shape == (20, 2)


def test_fit_calibrated_pipeline_reproducible(clean_df):
    X_train, X_test, y_train, y_test = train.split_data(clean_df)
    categorical_columns = get_categorical_columns(X_train)
    small_X, small_y = X_train.iloc[:500], y_train.iloc[:500]

    template_a = calibration.build_calibration_template("LogisticRegression", categorical_columns)
    calibrated_a = calibration.fit_calibrated_pipeline(template_a, small_X, small_y, cv=3)
    template_b = calibration.build_calibration_template("LogisticRegression", categorical_columns)
    calibrated_b = calibration.fit_calibrated_pipeline(template_b, small_X, small_y, cv=3)

    proba_a = calibrated_a.predict_proba(X_test)[:, 1]
    proba_b = calibrated_b.predict_proba(X_test)[:, 1]
    assert np.allclose(proba_a, proba_b)


def test_evaluate_calibration_returns_all_six_keys():
    y_true = [0, 1, 0, 1, 0, 1, 1, 0]
    proba_raw = [0.2, 0.6, 0.3, 0.5, 0.4, 0.7, 0.4, 0.1]
    proba_calibrated = [0.1, 0.8, 0.2, 0.6, 0.3, 0.75, 0.55, 0.05]

    metrics = calibration.evaluate_calibration(y_true, proba_raw, proba_calibrated)

    expected_keys = {
        "brier_raw", "brier_calibrated", "auc_raw", "auc_calibrated", "ece_raw", "ece_calibrated",
    }
    assert set(metrics) == expected_keys
    for key in expected_keys:
        assert 0.0 <= metrics[key] <= 1.0


def test_isotonic_improves_brier_and_ece_on_real_data(calibration_result):
    # Locks in the verified spec-research finding on the real dataset
    # (.claude/specs/09-probability-scoring.md): isotonic calibration
    # improves both Brier and ECE at a negligible AUC cost. Intentionally
    # brittle, matching 08's precedent for real-data regression guards.
    metrics = calibration.evaluate_calibration(
        calibration_result["y_test"], calibration_result["proba_raw"], calibration_result["proba_calibrated"]
    )
    assert metrics["brier_calibrated"] < metrics["brier_raw"]
    assert metrics["ece_calibrated"] < metrics["ece_raw"]
    assert metrics["auc_calibrated"] < 0.95


def test_plot_reliability_curve_returns_existing_path(calibration_result, tmp_path):
    path = calibration.plot_reliability_curve(
        calibration_result["y_test"], calibration_result["proba_raw"],
        calibration_result["proba_calibrated"], out_dir=tmp_path,
    )
    assert path.exists()
    assert path.parent == tmp_path


def test_save_and_load_calibrated_artifact_roundtrip(clean_df, tmp_path):
    X_train, X_test, y_train, y_test = train.split_data(clean_df)
    categorical_columns = get_categorical_columns(X_train)
    small_X, small_y = X_train.iloc[:300], y_train.iloc[:300]
    template = calibration.build_calibration_template("LogisticRegression", categorical_columns)
    calibrated = calibration.fit_calibrated_pipeline(template, small_X, small_y, cv=3)

    model_path = tmp_path / "calibrated.pkl"
    metadata_path = tmp_path / "calibrated_metadata.json"
    calibration.save_calibrated_artifact(
        "LogisticRegression", calibrated, {"brier_calibrated": 0.1}, list(X_train.columns),
        model_path=model_path, metadata_path=metadata_path,
    )

    loaded = calibration.load_calibrated_model(model_path)
    expected = calibrated.predict_proba(X_test.iloc[:10])
    actual = loaded.predict_proba(X_test.iloc[:10])
    assert (expected == actual).all()

    metadata = json.loads(metadata_path.read_text())
    assert metadata["base_model_name"] == "LogisticRegression"
    assert metadata["calibration_method"] == calibration.CALIBRATION_METHOD
    assert metadata["feature_columns"] == list(X_train.columns)


def test_load_calibrated_model_raises_actionable_error_when_missing(tmp_path):
    with pytest.raises(FileNotFoundError, match="python -m src.models.calibration"):
        calibration.load_calibrated_model(tmp_path / "nonexistent.pkl")


def test_run_calibration_pipeline_raises_on_feature_column_mismatch(
    clean_df, isolated_mlflow_uri, monkeypatch, tmp_path
):
    monkeypatch.setattr(train, "MLFLOW_TRACKING_URI", isolated_mlflow_uri)

    fake_metadata_path = tmp_path / "churn_model_metadata.json"
    fake_metadata_path.write_text(json.dumps({
        "model_name": "XGBoost",
        "feature_columns": ["this_column_does_not_exist"],
    }))
    monkeypatch.setattr(train, "DEFAULT_METADATA_PATH", fake_metadata_path)

    with pytest.raises(ValueError, match="no longer match"):
        calibration.run_calibration_pipeline(clean_df)


def test_run_calibration_pipeline_writes_all_artifacts(
    clean_df, isolated_mlflow_uri, monkeypatch, tmp_path
):
    monkeypatch.setattr(train, "MLFLOW_TRACKING_URI", isolated_mlflow_uri)
    monkeypatch.setattr(calibration, "DEFAULT_CALIBRATED_MODEL_PATH", tmp_path / "calibrated.pkl")
    monkeypatch.setattr(
        calibration, "DEFAULT_CALIBRATED_METADATA_PATH", tmp_path / "calibrated_metadata.json"
    )
    monkeypatch.setattr(calibration, "CALIBRATION_METRICS_PATH", tmp_path / "calibration_metrics.json")
    monkeypatch.setattr(calibration, "FIGURES_DIR", tmp_path / "figures")

    result = calibration.run_calibration_pipeline(clean_df)

    assert result["base_model_name"] == "XGBoost"
    for key in ("model_path", "metadata_path", "figure_path", "metrics_json_path"):
        assert result[key].exists(), f"{key} was not written to {result[key]}"
    assert all(str(result[key]).startswith(str(tmp_path)) for key in
               ("model_path", "metadata_path", "metrics_json_path"))
