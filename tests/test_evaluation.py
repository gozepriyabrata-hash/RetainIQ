import numpy as np
import pytest

from src.models.evaluation import (
    LEAKAGE_AUC_THRESHOLD,
    check_auc_leakage_guard,
    compute_classification_metrics,
    expected_calibration_error,
    tune_decision_threshold,
)


def test_check_auc_leakage_guard_raises_above_threshold():
    with pytest.raises(ValueError):
        check_auc_leakage_guard(0.97)
    check_auc_leakage_guard(0.83)
    check_auc_leakage_guard(LEAKAGE_AUC_THRESHOLD)


def test_compute_classification_metrics_keys_and_ranges():
    y_true = [0, 1, 0, 1, 0, 1, 1, 0]
    y_proba = [0.1, 0.9, 0.2, 0.6, 0.4, 0.8, 0.3, 0.05]

    metrics = compute_classification_metrics(y_true, y_proba)

    assert set(metrics) == {"auc", "pr_auc", "precision", "recall", "f1", "brier"}
    for key in ("auc", "pr_auc", "precision", "recall", "f1", "brier"):
        assert 0.0 <= metrics[key] <= 1.0


def test_compute_classification_metrics_threshold_changes_precision_recall():
    y_true = [0, 1, 0, 1, 0, 1, 1, 0]
    y_proba = [0.1, 0.9, 0.2, 0.6, 0.4, 0.8, 0.3, 0.05]

    low = compute_classification_metrics(y_true, y_proba, threshold=0.2)
    high = compute_classification_metrics(y_true, y_proba, threshold=0.8)

    assert (low["precision"], low["recall"], low["f1"]) != (high["precision"], high["recall"], high["f1"])
    # Threshold-independent metrics must be unaffected.
    assert low["auc"] == pytest.approx(high["auc"])
    assert low["pr_auc"] == pytest.approx(high["pr_auc"])
    assert low["brier"] == pytest.approx(high["brier"])


def test_compute_classification_metrics_handles_degenerate_all_zero_predictions():
    # threshold=0.99 predicts the negative class for every row -- precision
    # is undefined (0/0); zero_division=0 must keep this from raising.
    y_true = [0, 1, 0, 1]
    y_proba = [0.1, 0.2, 0.15, 0.3]
    metrics = compute_classification_metrics(y_true, y_proba, threshold=0.99)
    assert metrics["precision"] == 0.0
    assert metrics["recall"] == 0.0
    assert metrics["f1"] == 0.0


def test_tune_decision_threshold_finds_known_optimum():
    # y_proba perfectly separates the two classes at 0.5: any threshold in
    # (0.4, 0.6] gives a perfect F1 of 1.0. Restrict the grid to values that
    # land exactly on that plateau's low edge so the expected optimum is
    # unambiguous.
    y_true = [0, 0, 0, 1, 1, 1]
    y_proba = [0.1, 0.2, 0.3, 0.7, 0.8, 0.9]
    thresholds = np.array([0.4, 0.45, 0.5])

    best_threshold, best_f1 = tune_decision_threshold(y_true, y_proba, thresholds=thresholds)

    assert best_threshold == pytest.approx(0.4)
    assert best_f1 == pytest.approx(1.0)


def test_tune_decision_threshold_default_grid_is_within_bounds():
    y_true = [0, 1, 0, 1, 0, 1, 1, 0]
    y_proba = [0.1, 0.9, 0.2, 0.6, 0.4, 0.8, 0.3, 0.05]

    best_threshold, best_f1 = tune_decision_threshold(y_true, y_proba)

    assert 0.05 <= best_threshold <= 0.95
    assert 0.0 <= best_f1 <= 1.0


def test_expected_calibration_error_zero_on_perfect_calibration():
    # Every prediction lands in the same bin (all 0.5) whose mean predicted
    # probability (0.5) exactly matches the bin's observed positive rate
    # (2 of 4 true labels are 1) -- ECE must be exactly 0.
    y_true = [0, 1, 0, 1]
    y_proba = [0.5, 0.5, 0.5, 0.5]

    assert expected_calibration_error(y_true, y_proba) == pytest.approx(0.0)


def test_expected_calibration_error_positive_on_miscalibration():
    # 2 equal-width bins: [0, 0.5) and [0.5, 1.0]. Bin 0 (mean proba 0.1) is
    # all churners (observed rate 1.0, gap 0.9); bin 1 (mean proba 0.9) has
    # no churners (observed rate 0.0, gap 0.9). Each bin holds half the rows,
    # so ECE = 0.5*0.9 + 0.5*0.9 = 0.9, matched by hand here.
    y_true = [1, 1, 0, 0]
    y_proba = [0.1, 0.1, 0.9, 0.9]

    assert expected_calibration_error(y_true, y_proba, n_bins=2) == pytest.approx(0.9)


def test_expected_calibration_error_handles_proba_of_exactly_one():
    # A prediction of exactly 1.0 must land in the last bin, not overflow
    # into a nonexistent n_bins-th bucket.
    y_true = [1, 1]
    y_proba = [1.0, 1.0]

    assert expected_calibration_error(y_true, y_proba) == pytest.approx(0.0)


def test_expected_calibration_error_empty_input_is_zero():
    assert expected_calibration_error([], []) == 0.0
