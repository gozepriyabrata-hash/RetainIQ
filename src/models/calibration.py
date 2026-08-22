"""Probability calibration on top of the Phase 2 persisted model.

`models/churn_model.pkl` (see .claude/specs/08-churn-prediction-model.md)
ranks customers well (test AUC-ROC 0.8434) but its raw probabilities are not
well calibrated -- verified during spec research, its 10-bin Expected
Calibration Error is 0.0803 (Brier 0.1466). This module wraps the same
model architecture in an isotonic-regression `CalibratedClassifierCV`,
cross-validated on the training split only, and persists a second artifact
(`models/churn_model_calibrated.pkl`) so `models/churn_model.pkl` itself is
never overwritten. See .claude/specs/09-probability-scoring.md for the full
methodology and the verified before/after numbers (Brier 0.1466 -> 0.1376,
ECE 0.0803 -> 0.0301, AUC-ROC 0.8434 -> 0.8425).
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.model_selection import StratifiedKFold

from src.data.config import FIGURES_DIR, MODELS_DIR, PROJECT_ROOT
from src.data.load_data import TARGET_COLUMN
from src.features.preprocessing import get_categorical_columns
from src.models import train
from src.models.evaluation import (
    ECE_BINS,
    RANDOM_STATE,
    compute_classification_metrics,
    expected_calibration_error,
)

logger = logging.getLogger(__name__)

# Verified during spec research (.claude/specs/09-probability-scoring.md):
# isotonic beats sigmoid on this dataset on both Brier and ECE, at a
# negligible AUC-ROC cost, and the ~5,634-row training fold comfortably
# clears isotonic regression's usual small-sample caution threshold.
CALIBRATION_METHOD = "isotonic"

DEFAULT_CALIBRATED_MODEL_PATH = MODELS_DIR / "churn_model_calibrated.pkl"
DEFAULT_CALIBRATED_METADATA_PATH = MODELS_DIR / "churn_model_calibrated_metadata.json"
CALIBRATION_METRICS_PATH = PROJECT_ROOT / "reports" / "calibration_metrics.json"
RELIABILITY_FIGURE_FILENAME = "calibration_reliability.png"


def build_calibration_template(model_name: str, categorical_columns: list[str]) -> train.Pipeline:
    """Unfitted preprocessing+SMOTE+estimator pipeline for `model_name`, reusing train.py's builder."""
    return train.build_model_pipeline(train.MODEL_SPECS[model_name], categorical_columns)


def fit_calibrated_pipeline(
    pipeline_template,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    method: str = CALIBRATION_METHOD,
    cv: int = train.CV_FOLDS,
    random_state: int = RANDOM_STATE,
) -> CalibratedClassifierCV:
    """Fit method-calibrated probabilities for pipeline_template via internal cv-fold CV.

    Fit on X_train/y_train only. CalibratedClassifierCV's internal CV clones
    and refits pipeline_template (preprocessing + SMOTE + estimator) on each
    fold's training portion and calibrates on that fold's held-out portion --
    X_test/y_test are never touched here, matching CLAUDE.md Sec 7's split
    discipline. Uses the `estimator=` kwarg (not the removed-in-1.4
    `base_estimator=`), matching the pinned scikit-learn==1.6.1.
    """
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    calibrated = CalibratedClassifierCV(estimator=pipeline_template, method=method, cv=skf)
    calibrated.fit(X_train, y_train)
    return calibrated


def evaluate_calibration(y_true, proba_raw, proba_calibrated) -> dict[str, float]:
    """Brier, AUC-ROC, and ECE for the raw vs. calibrated probabilities -- never a bare claim."""
    raw_metrics = compute_classification_metrics(y_true, proba_raw)
    calibrated_metrics = compute_classification_metrics(y_true, proba_calibrated)
    return {
        "brier_raw": raw_metrics["brier"],
        "brier_calibrated": calibrated_metrics["brier"],
        "auc_raw": raw_metrics["auc"],
        "auc_calibrated": calibrated_metrics["auc"],
        "ece_raw": expected_calibration_error(y_true, proba_raw),
        "ece_calibrated": expected_calibration_error(y_true, proba_calibrated),
    }


def plot_reliability_curve(
    y_true, proba_raw, proba_calibrated, out_dir: Path = FIGURES_DIR
) -> Path:
    """Reliability diagram: perfect-calibration diagonal plus raw and calibrated curves."""
    frac_raw, mean_raw = calibration_curve(y_true, proba_raw, n_bins=ECE_BINS, strategy="uniform")
    frac_cal, mean_cal = calibration_curve(
        y_true, proba_calibrated, n_bins=ECE_BINS, strategy="uniform"
    )

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfectly calibrated")
    ax.plot(mean_raw, frac_raw, marker="o", label="Raw")
    ax.plot(mean_cal, frac_cal, marker="o", label="Calibrated (isotonic)")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed positive rate")
    ax.set_title("Reliability Diagram: Raw vs. Calibrated")
    ax.legend()

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / RELIABILITY_FIGURE_FILENAME
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def save_calibrated_artifact(
    model_name: str,
    calibrated: CalibratedClassifierCV,
    calibration_metrics: dict,
    feature_columns: list[str],
    model_path: Path = DEFAULT_CALIBRATED_MODEL_PATH,
    metadata_path: Path = DEFAULT_CALIBRATED_METADATA_PATH,
) -> tuple[Path, Path]:
    """Persist the fitted calibrated pipeline (joblib) and a JSON metadata sidecar."""
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(calibrated, model_path)

    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "model_name": model_name,
        "base_model_name": model_name,
        "calibration_method": CALIBRATION_METHOD,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "metrics": calibration_metrics,
        "feature_columns": list(feature_columns),
        "target_column": TARGET_COLUMN,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2))
    return model_path, metadata_path


def load_calibrated_model(model_path: Path = DEFAULT_CALIBRATED_MODEL_PATH) -> CalibratedClassifierCV:
    """Load a previously calibrated+saved pipeline.

    Raises a clear, actionable FileNotFoundError pointing at this module's
    own entry point (not src.models.train's) if no artifact exists yet.
    """
    if not model_path.exists():
        raise FileNotFoundError(
            f"No calibrated model found at {model_path}. Run "
            "`python -m src.models.calibration` first to produce it."
        )
    return joblib.load(model_path)


def run_calibration_pipeline(df: pd.DataFrame) -> dict:
    """Orchestrate calibrate -> evaluate -> plot -> log -> persist for the currently-winning model.

    Reads the currently-winning model name and feature columns from
    train.py's own persisted metadata (never re-runs compare_models), reuses
    train.split_data(df) for the identical train/test partition train.py
    used, and asserts the feature columns still match that metadata before
    fitting anything -- a staleness guard against calibrating against a df
    that no longer matches what churn_model.pkl was actually trained on.
    """
    metadata = json.loads(train.DEFAULT_METADATA_PATH.read_text())
    model_name = metadata["model_name"]

    X_train, X_test, y_train, y_test = train.split_data(df)
    if list(X_train.columns) != metadata["feature_columns"]:
        raise ValueError(
            "X_train columns no longer match churn_model_metadata.json's "
            "feature_columns -- the persisted model is stale relative to "
            "`df`. Re-run `python -m src.models.train` before calibrating."
        )

    categorical_columns = get_categorical_columns(X_train)
    template = build_calibration_template(model_name, categorical_columns)
    calibrated = fit_calibrated_pipeline(template, X_train, y_train)

    proba_raw = train.load_trained_model().predict_proba(X_test)[:, 1]
    proba_calibrated = calibrated.predict_proba(X_test)[:, 1]

    metrics = evaluate_calibration(y_test, proba_raw, proba_calibrated)
    # FIGURES_DIR/DEFAULT_CALIBRATED_MODEL_PATH/DEFAULT_CALIBRATED_METADATA_PATH
    # are passed through explicitly below (rather than relying on
    # plot_reliability_curve's/save_calibrated_artifact's own
    # bound-at-def-time defaults) so tests can monkeypatch these
    # module-level globals and have the redirect actually take effect at
    # call time -- mirrors train.py's run_training_pipeline precedent.
    figure_path = plot_reliability_curve(y_test, proba_raw, proba_calibrated, out_dir=FIGURES_DIR)

    mlflow.set_tracking_uri(train.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(train.MLFLOW_EXPERIMENT_NAME)
    with mlflow.start_run(run_name=f"{model_name}_calibrated"):
        mlflow.log_params({
            "base_model_name": model_name,
            "calibration_method": CALIBRATION_METHOD,
            "cv_folds": train.CV_FOLDS,
        })
        mlflow.log_metrics(metrics)
        # Same skops-incompatibility workaround as train.py's _log_to_mlflow:
        # this calibrated pipeline wraps the same imblearn SMOTE/Pipeline
        # object mlflow's default skops serializer rejects. Self-produced,
        # fit and logged within this same call -- same trust boundary
        # documented in .claude/specs/08-churn-prediction-model.md's
        # Security notes and re-affirmed for this artifact in
        # .claude/specs/09-probability-scoring.md's.
        mlflow.sklearn.log_model(calibrated, name="model", serialization_format="pickle")

    model_path, metadata_path = save_calibrated_artifact(
        model_name, calibrated, metrics, list(X_train.columns),
        model_path=DEFAULT_CALIBRATED_MODEL_PATH, metadata_path=DEFAULT_CALIBRATED_METADATA_PATH,
    )

    CALIBRATION_METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    CALIBRATION_METRICS_PATH.write_text(json.dumps(
        {**metrics, "base_model_name": model_name, "calibration_method": CALIBRATION_METHOD},
        indent=2,
    ))

    return {
        "base_model_name": model_name,
        "calibration_metrics": metrics,
        "model_path": model_path,
        "metadata_path": metadata_path,
        "figure_path": figure_path,
        "metrics_json_path": CALIBRATION_METRICS_PATH,
    }


def main() -> None:
    """Entry point for `python -m src.models.calibration`: calibrate and persist once."""
    logging.basicConfig(level=logging.INFO)
    from src.data.load_data import load_clean_data

    df = load_clean_data()
    result = run_calibration_pipeline(df)
    metrics = result["calibration_metrics"]
    logger.info(
        "Calibrated %s: brier %.4f -> %.4f, auc %.4f -> %.4f, ece %.4f -> %.4f, saved to %s",
        result["base_model_name"],
        metrics["brier_raw"], metrics["brier_calibrated"],
        metrics["auc_raw"], metrics["auc_calibrated"],
        metrics["ece_raw"], metrics["ece_calibrated"],
        result["model_path"],
    )


if __name__ == "__main__":
    main()
