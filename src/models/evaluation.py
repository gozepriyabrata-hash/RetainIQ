"""Pure classification-metric helpers for the Phase 2 model comparison.

No MLflow, no filesystem I/O, no dependency on src.models.train -- kept
cheap and dependency-light so these functions are trivial to unit-test with
synthetic arrays (see tests/test_evaluation.py), independent of fitting any
real model.
"""

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

ArrayLike = Sequence[float] | npt.NDArray[np.floating]

RANDOM_STATE = 42
LEAKAGE_AUC_THRESHOLD = 0.95
TARGET_AUC = 0.85
DEFAULT_DECISION_THRESHOLD = 0.5
THRESHOLD_GRID = np.linspace(0.05, 0.95, 91)


def check_auc_leakage_guard(auc: float, threshold: float = LEAKAGE_AUC_THRESHOLD) -> None:
    """Raise if auc exceeds threshold -- CLAUDE.md Sec 2 rule 2, mechanically enforced.

    A mechanical stand-in for "STOP and investigate leakage" rather than a
    comment a future caller could miss.
    """
    if auc > threshold:
        raise ValueError(
            f"AUC {auc:.4f} exceeds the leakage threshold {threshold}. "
            "Investigate for target/probability leakage before trusting this model."
        )


def compute_classification_metrics(
    y_true: ArrayLike, y_proba: ArrayLike, threshold: float = DEFAULT_DECISION_THRESHOLD
) -> dict[str, float]:
    """AUC-ROC, PR-AUC, precision, recall, F1, and Brier score -- never accuracy alone.

    AUC-ROC, PR-AUC, and Brier score are threshold-independent (computed
    directly on y_proba); precision, recall, and F1 are computed at the
    given decision threshold. zero_division=0 keeps this safe on degenerate
    synthetic inputs (e.g. a single-class y_true) used in unit tests.
    """
    y_proba = np.asarray(y_proba)
    pred = (y_proba >= threshold).astype(int)
    return {
        "auc": float(roc_auc_score(y_true, y_proba)),
        "pr_auc": float(average_precision_score(y_true, y_proba)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "brier": float(brier_score_loss(y_true, y_proba)),
    }


def tune_decision_threshold(
    y_true: ArrayLike, y_proba: ArrayLike, thresholds: npt.NDArray[np.floating] | None = None
) -> tuple[float, float]:
    """Return (best_threshold, best_f1) -- the threshold in `thresholds` maximizing F1.

    CLAUDE.md Sec 7: "tune the decision threshold, don't assume 0.5." Ties
    are broken by np.argmax's first-occurrence behavior (the lowest tied
    threshold). `thresholds` defaults to a fresh copy of THRESHOLD_GRID each
    call (rather than binding the shared module-level array directly as a
    default argument) so no caller can accidentally mutate it in place.

    Callers decide what (y_true, y_proba) to pass -- tuning against a
    held-out test split's own labels would bias the chosen threshold
    optimistically; src.models.train tunes against out-of-fold training
    predictions instead, never against the test split.
    """
    if thresholds is None:
        thresholds = THRESHOLD_GRID
    y_proba = np.asarray(y_proba)
    f1_scores = [
        f1_score(y_true, (y_proba >= t).astype(int), zero_division=0) for t in thresholds
    ]
    best_idx = int(np.argmax(f1_scores))
    return float(thresholds[best_idx]), float(f1_scores[best_idx])
