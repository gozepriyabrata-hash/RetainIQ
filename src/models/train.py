"""Phase 2: feature pipeline, model training + comparison, MLflow.

Fits and compares four churn classifiers -- Logistic Regression (baseline),
Random Forest, XGBoost, LightGBM -- inside a single reusable
ColumnTransformer + SMOTE + estimator pipeline, evaluates each honestly on a
held-out test split, logs every run to MLflow, and persists the
best-performing model + metadata to models/ for a future Phase 5 API to
load. See .claude/specs/08-churn-prediction-model.md for the full
methodology, verified numbers, and the disclosed honest-AUC finding.

This is the real, compared, persisted, MLflow-tracked production classifier
-- unrelated to src/explain/driver_analysis.py's throwaway diagnostic model,
which is refit fresh per call, never persisted, and never compared.
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
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from lightgbm import LGBMClassifier
from mlflow.tracking import MlflowClient
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_predict,
    cross_val_score,
    train_test_split,
)
from xgboost import XGBClassifier

from src.data.config import FIGURES_DIR, MODELS_DIR, PROJECT_ROOT
from src.data.load_data import ID_COLUMN, TARGET_COLUMN
from src.features.preprocessing import build_preprocessor, get_categorical_columns
from src.models.evaluation import (
    DEFAULT_DECISION_THRESHOLD,
    RANDOM_STATE,
    TARGET_AUC,
    check_auc_leakage_guard,
    compute_classification_metrics,
    tune_decision_threshold,
)

logger = logging.getLogger(__name__)

TEST_SIZE = 0.2
CV_FOLDS = 5

# mlflow>=3.x's legacy file-based tracking store ("file:./mlruns") is in
# maintenance mode and refuses new writes -- a SQLite backend is the
# supported local, no-server alternative (still fully local, no credentials,
# no new attack surface). The metadata DB lands at mlflow.db and the
# artifact store still defaults to mlruns/ on disk; both are gitignored.
# NOTE: `mlflow ui` alone points at the legacy ./mlruns file store and will
# show an empty experiment list against this backend -- use
# `mlflow ui --backend-store-uri sqlite:///mlflow.db` instead (CLAUDE.md Sec 5).
MLFLOW_TRACKING_URI = f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}"
MLFLOW_EXPERIMENT_NAME = "retainiq-churn-classifier"

DEFAULT_MODEL_PATH = MODELS_DIR / "churn_model.pkl"
DEFAULT_METADATA_PATH = MODELS_DIR / "churn_model_metadata.json"
COMPARISON_TABLE_PATH = PROJECT_ROOT / "reports" / "model_comparison.csv"
COMPARISON_FIGURE_FILENAME = "model_comparison_auc.png"

# Fixed, disclosed hyperparameters -- chosen via a one-time small grid search
# during spec research (.claude/specs/08-churn-prediction-model.md), then
# pinned here so `python -m src.models.train` stays fast and deterministic
# on every call rather than re-searching hyperparameters every run.
MODEL_SPECS: dict[str, BaseEstimator] = {
    "LogisticRegression": LogisticRegression(max_iter=2000, C=10, random_state=RANDOM_STATE),
    "RandomForest": RandomForestClassifier(
        n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1
    ),
    "XGBoost": XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.03,
        random_state=RANDOM_STATE,
        eval_metric="logloss",
    ),
    "LightGBM": LGBMClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.03,
        random_state=RANDOM_STATE,
        verbosity=-1,
    ),
}


def _numeric_metric_keys(result: dict) -> dict[str, float]:
    """Every int/float (non-bool) entry of an evaluate_candidate result dict.

    Derived from the result itself rather than a hand-maintained key list,
    so adding/renaming a metric in evaluate_candidate can't silently stop
    being logged to MLflow (or raise a stale-key KeyError) without anyone
    updating a second list in sync.
    """
    return {
        k: v for k, v in result.items()
        if isinstance(v, (int, float)) and not isinstance(v, bool)
    }


def build_model_pipeline(estimator: BaseEstimator, categorical_columns: list[str]) -> Pipeline:
    """ColumnTransformer + SMOTE + a clone of estimator, in one imblearn Pipeline.

    clone() so MODEL_SPECS' entries stay unfitted templates: Pipeline.fit
    does not clone its steps, so fitting the estimator object directly
    would mutate MODEL_SPECS in place, silently turning a module-level
    "template" constant into a several-hundred-MB fitted model after the
    first compare_models() call, and would make a second call in the same
    process (e.g. two notebook cells) reuse that first fit instead of
    training fresh.

    imblearn's Pipeline only invokes the sampler's fit_resample during
    .fit() -- .predict()/.predict_proba() skip it automatically, so this
    same fitted pipeline is safe to reuse for scoring and (later, Phase 5)
    serving without ever resampling real inference data.
    """
    return Pipeline([
        ("pre", build_preprocessor(categorical_columns)),
        ("smote", SMOTE(random_state=RANDOM_STATE)),
        ("clf", clone(estimator)),
    ])


def split_data(
    df: pd.DataFrame, test_size: float = TEST_SIZE, random_state: int = RANDOM_STATE
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Stratified (X_train, X_test, y_train, y_test) split; X excludes TARGET_COLUMN.

    Leakage-guard surface for this module -- TARGET_COLUMN must never appear
    in X (directly tested). load_clean_data()'s output has already dropped
    ID_COLUMN; the assertion below makes that guarantee mechanical rather
    than only true by convention, in case a future caller (e.g. a Phase 5
    API payload) ever feeds this function data that hasn't gone through
    clean_data().
    """
    assert ID_COLUMN not in df.columns, f"{ID_COLUMN} must be dropped before split_data()"
    y = df[TARGET_COLUMN]
    X = df.drop(columns=[TARGET_COLUMN])
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


def _log_to_mlflow(name: str, estimator: BaseEstimator, result: dict) -> str:
    """Log params/metrics/model for one candidate's run; return its MLflow run_id."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    with mlflow.start_run(run_name=name) as run:
        mlflow.log_params({
            "model_name": name,
            **{k: v for k, v in estimator.get_params().items()},
            "random_state": RANDOM_STATE,
            "test_size": TEST_SIZE,
            "cv_folds": CV_FOLDS,
        })
        mlflow.log_metrics(_numeric_metric_keys(result))
        # mlflow>=3.x defaults sklearn model logging to skops serialization,
        # which refuses imblearn's SMOTE/Pipeline as "untrusted types."
        # This pipeline is entirely self-produced (fit by this same call,
        # never loaded from an external source), so plain pickle -- the
        # same trust boundary already used by joblib.dump elsewhere in this
        # module -- is used instead of chasing a skops trust allowlist.
        # That boundary covers *this write*; it does not automatically cover
        # every future reader of this MLflow-logged copy under mlruns/ (as
        # opposed to models/churn_model.pkl, which only this pipeline ever
        # loads). See .claude/specs/08-churn-prediction-model.md's Security
        # notes -- if this artifact is ever served, registered, or mounted
        # from anywhere other than this training run's own output (a
        # possible Phase 6 concern), that load path must be re-reviewed as
        # untrusted deserialization.
        mlflow.sklearn.log_model(
            result["pipeline"], name="model", serialization_format="pickle"
        )
        return run.info.run_id


def evaluate_candidate(
    name: str,
    estimator: BaseEstimator,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> dict:
    """Fit one candidate, score it honestly, log it to MLflow, and return its result dict.

    5-fold stratified CV on the train split only (SMOTE refit fresh inside
    each fold via the pipeline -- CLAUDE.md Sec 7): once via cross_val_score
    for the CV AUC-ROC used to select the best model, and once via
    cross_val_predict for out-of-fold probabilities the decision threshold
    is tuned against. The pipeline is then fit once on the full train split
    and scored once on the held-out test split.

    The test split's labels/probabilities are used only to report a final,
    honest score for whichever threshold and model were already chosen from
    train-only information -- never to choose the threshold or pick the
    winner. (An earlier version of this function tuned the threshold
    against y_test directly and let compare_models sort by test_auc, which
    is optimistically biased and self-contradicted this module's own
    documented guardrail -- see .claude/specs/08-churn-prediction-model.md's
    quality-review note.) The leakage guard runs on both the CV mean and the
    test AUC, for every candidate -- not just the eventual best one.
    """
    categorical_columns = get_categorical_columns(X_train)
    pipeline = build_model_pipeline(estimator, categorical_columns)

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)
    cv_auc_mean, cv_auc_std = float(cv_scores.mean()), float(cv_scores.std())
    check_auc_leakage_guard(cv_auc_mean)

    # Out-of-fold predictions on the train split only -- reuses the same
    # CV_FOLDS/random_state partition as cv_scores above -- used solely to
    # tune the decision threshold without ever looking at y_test.
    oof_proba = cross_val_predict(
        pipeline, X_train, y_train, cv=cv, method="predict_proba", n_jobs=-1
    )[:, 1]
    tuned_threshold, _oof_tuned_f1 = tune_decision_threshold(y_train, oof_proba)

    pipeline.fit(X_train, y_train)
    proba = pipeline.predict_proba(X_test)[:, 1]
    metrics_at_default = compute_classification_metrics(
        y_test, proba, threshold=DEFAULT_DECISION_THRESHOLD
    )
    check_auc_leakage_guard(metrics_at_default["auc"])
    # Reported for the model actually being persisted, but the threshold
    # itself was chosen above without ever seeing y_test.
    test_f1_at_tuned_threshold = compute_classification_metrics(
        y_test, proba, threshold=tuned_threshold
    )["f1"]

    result = {
        "name": name,
        "pipeline": pipeline,
        "cv_auc_mean": cv_auc_mean,
        "cv_auc_std": cv_auc_std,
        "test_auc": metrics_at_default["auc"],
        "test_pr_auc": metrics_at_default["pr_auc"],
        "test_precision": metrics_at_default["precision"],
        "test_recall": metrics_at_default["recall"],
        "test_f1": metrics_at_default["f1"],
        "test_brier": metrics_at_default["brier"],
        "tuned_threshold": tuned_threshold,
        "test_f1_at_tuned_threshold": test_f1_at_tuned_threshold,
        "meets_target_auc": metrics_at_default["auc"] >= TARGET_AUC,
    }
    result["mlflow_run_id"] = _log_to_mlflow(name, estimator, result)
    return result


def compare_models(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Pipeline]]:
    """Evaluate every MODEL_SPECS candidate on one shared train/test split.

    Returns (comparison_df sorted by cv_auc_mean descending, {name: fitted_pipeline}).

    Sorted by cv_auc_mean -- computed from the train split only -- rather
    than test_auc: ranking candidates by their score on the same held-out
    split that is also used to report the final metric would make that
    "final" score optimistically biased by the model-selection process
    itself (the classic multiple-comparisons-against-one-test-set problem).
    test_auc/test_pr_auc/etc. remain in the table for every candidate as
    the one-time honest read on whichever model cv_auc_mean selects.
    """
    X_train, X_test, y_train, y_test = split_data(df)
    results = [
        evaluate_candidate(name, estimator, X_train, X_test, y_train, y_test)
        for name, estimator in MODEL_SPECS.items()
    ]
    pipelines = {r["name"]: r["pipeline"] for r in results}
    comparison = pd.DataFrame(
        [{k: v for k, v in r.items() if k not in ("pipeline",)} for r in results]
    )
    comparison = comparison.sort_values("cv_auc_mean", ascending=False).reset_index(drop=True)
    return comparison, pipelines


def select_best_model(
    comparison: pd.DataFrame, pipelines: dict[str, Pipeline]
) -> tuple[str, Pipeline, pd.Series]:
    """Return (name, pipeline, row) for the top row of `comparison`.

    `comparison` is expected already sorted by cv_auc_mean descending (what
    compare_models returns) -- this function just takes row 0, it does not
    re-sort. Logs a warning (does not raise) if the selected model's test
    AUC is below TARGET_AUC -- an honest surfaced signal, not a blocking
    gate. See the Honest-AUC finding in
    .claude/specs/08-churn-prediction-model.md.
    """
    best = comparison.iloc[0]
    if not bool(best["meets_target_auc"]):
        logger.warning(
            "Best model %s test AUC %.4f is below the %.2f target -- "
            "reporting honestly, not blocking on it.",
            best["name"], best["test_auc"], TARGET_AUC,
        )
    if "mlflow_run_id" in best and isinstance(best["mlflow_run_id"], str):
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        MlflowClient().set_tag(best["mlflow_run_id"], "selected_as_best", "true")
    return best["name"], pipelines[best["name"]], best


def save_model_artifact(
    name: str,
    pipeline: Pipeline,
    metrics: dict,
    feature_columns: list[str],
    model_path: Path = DEFAULT_MODEL_PATH,
    metadata_path: Path = DEFAULT_METADATA_PATH,
) -> tuple[Path, Path]:
    """Persist the fitted pipeline (joblib) and a JSON metadata sidecar."""
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_path)

    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "model_name": name,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "metrics": metrics,
        "feature_columns": list(feature_columns),
        "target_column": TARGET_COLUMN,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2))
    return model_path, metadata_path


def load_trained_model(model_path: Path = DEFAULT_MODEL_PATH) -> Pipeline:
    """Load a previously trained+saved pipeline.

    Raises a clear, actionable FileNotFoundError (rather than a bare
    joblib/pickle stack trace) if no artifact exists yet. This is the
    load-path a future Phase 5 API would call at startup; this module adds
    it but wires nothing to it.
    """
    if not model_path.exists():
        raise FileNotFoundError(
            f"No trained model found at {model_path}. Run "
            "`python -m src.models.train` first to produce it."
        )
    return joblib.load(model_path)


# Two neutral, metric-comparison colors -- deliberately distinct from
# CHURN_PALETTE (which encodes churn-class semantics elsewhere in the repo)
# so this chart doesn't visually imply "red = churn" for an unrelated pair
# of metric series.
_METRIC_COMPARISON_COLORS = ("#4C72B0", "#DD8452")


def plot_model_comparison(comparison: pd.DataFrame, out_dir: Path = FIGURES_DIR) -> Path:
    """Grouped bar chart of test_auc and test_pr_auc per candidate model."""
    fig, ax = plt.subplots(figsize=(8, 5))
    positions = range(len(comparison))
    width = 0.35
    ax.bar(
        [p - width / 2 for p in positions], comparison["test_auc"], width,
        label="Test AUC-ROC", color=_METRIC_COMPARISON_COLORS[0],
    )
    ax.bar(
        [p + width / 2 for p in positions], comparison["test_pr_auc"], width,
        label="Test PR-AUC", color=_METRIC_COMPARISON_COLORS[1],
    )
    ax.set_xticks(list(positions))
    ax.set_xticklabels(comparison["name"])
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison: Test AUC-ROC vs PR-AUC")
    ax.legend()

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / COMPARISON_FIGURE_FILENAME
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def run_training_pipeline(df: pd.DataFrame) -> dict:
    """Orchestrate compare -> select -> persist -> plot -> write comparison CSV."""
    comparison, pipelines = compare_models(df)
    best_name, best_pipeline, best_row = select_best_model(comparison, pipelines)

    X_train, _, _, _ = split_data(df)
    metrics_to_save = best_row.drop(labels=["name", "mlflow_run_id"], errors="ignore").to_dict()
    # DEFAULT_MODEL_PATH/DEFAULT_METADATA_PATH/FIGURES_DIR are passed through
    # explicitly (rather than relying on save_model_artifact's/
    # plot_model_comparison's own bound-at-def-time defaults) so tests can
    # monkeypatch these module-level globals and have the redirect actually
    # take effect at call time.
    model_path, metadata_path = save_model_artifact(
        best_name, best_pipeline, metrics_to_save, list(X_train.columns),
        model_path=DEFAULT_MODEL_PATH, metadata_path=DEFAULT_METADATA_PATH,
    )

    figure_path = plot_model_comparison(comparison, out_dir=FIGURES_DIR)

    COMPARISON_TABLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    comparison.drop(columns=["mlflow_run_id"], errors="ignore").to_csv(
        COMPARISON_TABLE_PATH, index=False
    )

    return {
        "best_model_name": best_name,
        "comparison": comparison,
        "model_path": model_path,
        "metadata_path": metadata_path,
        "figure_path": figure_path,
        "comparison_csv_path": COMPARISON_TABLE_PATH,
    }


def main() -> None:
    """Entry point for `python -m src.models.train`: run and log the full pipeline once."""
    logging.basicConfig(level=logging.INFO)
    from src.data.load_data import load_clean_data

    df = load_clean_data()
    result = run_training_pipeline(df)
    best_row = result["comparison"].iloc[0]
    logger.info(
        "Best model: %s (cv_auc_mean=%.4f, test_auc=%.4f) saved to %s",
        result["best_model_name"],
        best_row["cv_auc_mean"],
        best_row["test_auc"],
        result["model_path"],
    )


if __name__ == "__main__":
    main()
