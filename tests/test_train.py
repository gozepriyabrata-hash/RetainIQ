import json

import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LogisticRegression
from sklearn.utils.validation import check_is_fitted

import src.models.train as train
from src.data.load_data import TARGET_COLUMN
from src.models.evaluation import LEAKAGE_AUC_THRESHOLD, TARGET_AUC


@pytest.fixture(scope="module")
def isolated_mlflow_uri(tmp_path_factory):
    """A throwaway sqlite MLflow store so the test suite never writes to the
    tracked project's real mlflow.db/mlruns/."""
    store_dir = tmp_path_factory.mktemp("mlflow_test_store")
    return f"sqlite:///{store_dir / 'mlflow_test.db'}"


@pytest.fixture(scope="module")
def comparison_result(clean_df, isolated_mlflow_uri, module_monkeypatch):
    """One real compare_models(clean_df) call, reused by every test below that
    needs it -- this is the single expensive (4 models x 5-fold CV) fit in
    the module, run exactly once."""
    module_monkeypatch.setattr(train, "MLFLOW_TRACKING_URI", isolated_mlflow_uri)
    comparison, pipelines = train.compare_models(clean_df)
    return comparison, pipelines


@pytest.fixture(scope="module")
def module_monkeypatch():
    """pytest's built-in monkeypatch fixture is function-scoped; this is the
    documented workaround for using it from a module-scoped fixture."""
    mp = pytest.MonkeyPatch()
    yield mp
    mp.undo()


def test_split_data_excludes_target_and_is_stratified(clean_df):
    X_train, X_test, y_train, y_test = train.split_data(clean_df)

    assert TARGET_COLUMN not in X_train.columns
    assert TARGET_COLUMN not in X_test.columns
    assert len(X_train) + len(X_test) == len(clean_df)

    full_rate = clean_df[TARGET_COLUMN].mean()
    assert y_train.mean() == pytest.approx(full_rate, abs=0.02)
    assert y_test.mean() == pytest.approx(full_rate, abs=0.02)


def test_build_model_pipeline_predict_proba_shape(clean_df):
    X_train, X_test, y_train, y_test = train.split_data(clean_df)
    from src.features.preprocessing import get_categorical_columns
    categorical_columns = get_categorical_columns(X_train)

    # A small slice is enough to check output shape -- no need for the full
    # training set just to verify predict_proba's contract.
    small_X, small_y = X_train.iloc[:200], y_train.iloc[:200]
    pipeline = train.build_model_pipeline(LogisticRegression(max_iter=200), categorical_columns)
    pipeline.fit(small_X, small_y)

    proba = pipeline.predict_proba(X_test.iloc[:20])
    assert proba.shape == (20, 2)


def test_build_model_pipeline_smote_is_inert_at_predict_time(clean_df):
    X_train, X_test, y_train, y_test = train.split_data(clean_df)
    from src.features.preprocessing import get_categorical_columns
    categorical_columns = get_categorical_columns(X_train)

    small_X, small_y = X_train.iloc[:300], y_train.iloc[:300]
    pipeline = train.build_model_pipeline(LogisticRegression(max_iter=200), categorical_columns)
    pipeline.fit(small_X, small_y)

    # imblearn's Pipeline must only resample during .fit() -- predict_proba
    # on an arbitrary (class-imbalanced) input must return exactly one row
    # per input row, not a SMOTE-rebalanced count.
    proba = pipeline.predict_proba(X_test)
    assert len(proba) == len(X_test)


def test_build_model_pipeline_clones_estimator_leaving_model_specs_unfitted(clean_df):
    X_train, X_test, y_train, y_test = train.split_data(clean_df)
    from src.features.preprocessing import get_categorical_columns
    categorical_columns = get_categorical_columns(X_train)

    template = train.MODEL_SPECS["LogisticRegression"]
    pipeline = train.build_model_pipeline(template, categorical_columns)
    pipeline.fit(X_train.iloc[:300], y_train.iloc[:300])

    # build_model_pipeline must clone() the estimator: fitting the returned
    # pipeline must never fit MODEL_SPECS' own template object in place,
    # or a second call in the same process would silently reuse the first
    # fit instead of training fresh.
    with pytest.raises(NotFittedError):
        check_is_fitted(template)


def test_evaluate_candidate_raises_on_injected_leakage(clean_df, isolated_mlflow_uri, monkeypatch):
    monkeypatch.setattr(train, "MLFLOW_TRACKING_URI", isolated_mlflow_uri)
    X_train, X_test, y_train, y_test = train.split_data(clean_df)

    # A feature that is a direct copy of the label -- proves the leakage
    # guard is actually wired into the real evaluate_candidate path, not
    # just exercised in isolation against a fabricated AUC number.
    X_train_leaky = X_train.copy()
    X_test_leaky = X_test.copy()
    X_train_leaky["LeakyChurnCopy"] = y_train.to_numpy()
    X_test_leaky["LeakyChurnCopy"] = y_test.to_numpy()

    with pytest.raises(ValueError, match="leakage"):
        train.evaluate_candidate(
            "LogisticRegression", LogisticRegression(max_iter=200),
            X_train_leaky, X_test_leaky, y_train, y_test,
        )


def test_evaluate_candidate_reproducible(clean_df, isolated_mlflow_uri, monkeypatch):
    monkeypatch.setattr(train, "MLFLOW_TRACKING_URI", isolated_mlflow_uri)
    X_train, X_test, y_train, y_test = train.split_data(clean_df)

    result_a = train.evaluate_candidate(
        "LogisticRegression", LogisticRegression(max_iter=2000, C=10, random_state=42),
        X_train, X_test, y_train, y_test,
    )
    result_b = train.evaluate_candidate(
        "LogisticRegression", LogisticRegression(max_iter=2000, C=10, random_state=42),
        X_train, X_test, y_train, y_test,
    )

    assert result_a["test_auc"] == pytest.approx(result_b["test_auc"])
    assert result_a["cv_auc_mean"] == pytest.approx(result_b["cv_auc_mean"])


def test_compare_models_sorted_by_cv_auc_mean_desc_and_complete(comparison_result):
    comparison, _ = comparison_result
    cv_means = comparison["cv_auc_mean"].tolist()
    assert cv_means == sorted(cv_means, reverse=True)
    assert set(comparison["name"]) == set(train.MODEL_SPECS)
    assert len(comparison) == len(train.MODEL_SPECS)


def test_evaluate_candidate_auc_is_honest_for_every_candidate(comparison_result):
    comparison, _ = comparison_result
    for _, row in comparison.iterrows():
        assert 0.75 < row["test_auc"] < LEAKAGE_AUC_THRESHOLD
        assert 0.75 < row["cv_auc_mean"] < LEAKAGE_AUC_THRESHOLD


def test_compare_models_selection_uses_cv_auc_mean_not_test_auc(comparison_result):
    # Regression guard for the exact bug an earlier version of this module
    # had: selecting/sorting by test_auc instead of cv_auc_mean makes the
    # reported "final" score optimistically biased by the selection process
    # itself. On the current data these two columns disagree on the winner
    # (XGBoost leads on cv_auc_mean 0.8466; LightGBM leads on test_auc
    # 0.8437) -- different models -- so this asserts the row actually
    # returned as row 0 is the cv_auc_mean winner, not the test_auc winner.
    comparison, _ = comparison_result
    cv_auc_winner = comparison.sort_values("cv_auc_mean", ascending=False).iloc[0]["name"]
    test_auc_winner = comparison.sort_values("test_auc", ascending=False).iloc[0]["name"]
    assert comparison.iloc[0]["name"] == cv_auc_winner
    assert cv_auc_winner != test_auc_winner, (
        "current data no longer distinguishes cv- vs test-based selection -- "
        "this regression guard needs a different fixture to stay meaningful"
    )


def test_compare_models_current_winner_is_xgboost(comparison_result):
    # Locks in the verified spec-research finding: XGBoost currently wins on
    # cv_auc_mean (0.8466 vs LightGBM's 0.8463 -- a ~0.0003 margin).
    # Intentionally brittle (see .claude/specs/08's Risks section) -- should
    # fail loudly on real distributional/library drift, not silently pass.
    comparison, _ = comparison_result
    assert comparison.iloc[0]["name"] == "XGBoost"
    assert comparison.iloc[0]["cv_auc_mean"] == pytest.approx(0.8466, abs=0.01)


def test_select_best_model_picks_top_row_on_real_comparison(comparison_result):
    comparison, pipelines = comparison_result
    name, pipeline, row = train.select_best_model(comparison, pipelines)
    assert name == comparison.iloc[0]["name"]
    assert pipeline is pipelines[name]
    assert row["name"] == name


def test_select_best_model_picks_top_row_on_fabricated_data():
    # select_best_model's contract (per .claude/specs/08-churn-prediction-model.md
    # Functional Requirement 10) is "the top row of an already-sorted-by-
    # test_auc-descending comparison" -- it does not re-sort internally, so
    # the fabricated frame here must already be sorted, matching what
    # compare_models actually hands it.
    comparison = pd.DataFrame([
        {"name": "B", "test_auc": 0.90, "test_pr_auc": 0.6, "meets_target_auc": True},
        {"name": "C", "test_auc": 0.85, "test_pr_auc": 0.55, "meets_target_auc": True},
        {"name": "A", "test_auc": 0.80, "test_pr_auc": 0.5, "meets_target_auc": False},
    ])
    pipelines = {"A": "pipeline_a", "B": "pipeline_b", "C": "pipeline_c"}

    name, pipeline, row = train.select_best_model(comparison, pipelines)

    assert name == "B"
    assert pipeline == "pipeline_b"
    assert row["test_auc"] == 0.90


def test_select_best_model_warns_when_below_target(caplog):
    comparison = pd.DataFrame([
        {"name": "Weak", "test_auc": 0.70, "test_pr_auc": 0.4, "meets_target_auc": False},
    ])
    pipelines = {"Weak": "pipeline_weak"}

    with caplog.at_level("WARNING"):
        name, pipeline, row = train.select_best_model(comparison, pipelines)

    assert name == "Weak"  # still returns a result, does not raise
    assert any("below the" in record.message for record in caplog.records)


def test_save_and_load_model_artifact_roundtrip(clean_df, tmp_path):
    X_train, X_test, y_train, y_test = train.split_data(clean_df)
    from src.features.preprocessing import get_categorical_columns
    categorical_columns = get_categorical_columns(X_train)
    pipeline = train.build_model_pipeline(
        LogisticRegression(max_iter=200), categorical_columns
    )
    pipeline.fit(X_train.iloc[:300], y_train.iloc[:300])

    model_path = tmp_path / "model.pkl"
    metadata_path = tmp_path / "metadata.json"
    train.save_model_artifact(
        "LogisticRegression", pipeline, {"test_auc": 0.83}, list(X_train.columns),
        model_path=model_path, metadata_path=metadata_path,
    )

    loaded = train.load_trained_model(model_path)
    expected = pipeline.predict_proba(X_test.iloc[:10])
    actual = loaded.predict_proba(X_test.iloc[:10])
    assert (expected == actual).all()


def test_load_trained_model_raises_actionable_error_when_missing(tmp_path):
    with pytest.raises(FileNotFoundError, match="python -m src.models.train"):
        train.load_trained_model(tmp_path / "nonexistent.pkl")


def test_model_metadata_json_has_expected_keys(clean_df, tmp_path):
    X_train, X_test, y_train, y_test = train.split_data(clean_df)
    from src.features.preprocessing import get_categorical_columns
    categorical_columns = get_categorical_columns(X_train)
    pipeline = train.build_model_pipeline(
        LogisticRegression(max_iter=200), categorical_columns
    )
    pipeline.fit(X_train.iloc[:300], y_train.iloc[:300])

    model_path = tmp_path / "model.pkl"
    metadata_path = tmp_path / "metadata.json"
    train.save_model_artifact(
        "LogisticRegression", pipeline, {"test_auc": 0.83}, list(X_train.columns),
        model_path=model_path, metadata_path=metadata_path,
    )

    metadata = json.loads(metadata_path.read_text())
    assert set(metadata) == {"model_name", "trained_at", "metrics", "feature_columns", "target_column"}
    assert metadata["model_name"] == "LogisticRegression"
    assert metadata["target_column"] == TARGET_COLUMN
    assert metadata["feature_columns"] == list(X_train.columns)


def test_plot_model_comparison_returns_existing_path(comparison_result, tmp_path):
    comparison, _ = comparison_result
    path = train.plot_model_comparison(comparison, out_dir=tmp_path)
    assert path.exists()
    assert path.parent == tmp_path


def test_run_training_pipeline_writes_all_artifacts(clean_df, isolated_mlflow_uri, monkeypatch, tmp_path):
    # Shrink MODEL_SPECS to one fast model so this end-to-end test doesn't
    # duplicate the full 4-model comparison already exercised via
    # comparison_result -- it exists to prove run_training_pipeline's
    # orchestration (persistence + plotting + CSV) writes to the paths its
    # module-level globals point at, not to re-verify model quality.
    monkeypatch.setattr(train, "MLFLOW_TRACKING_URI", isolated_mlflow_uri)
    monkeypatch.setattr(train, "MODEL_SPECS", {"LogisticRegression": LogisticRegression(
        max_iter=2000, C=10, random_state=42
    )})
    monkeypatch.setattr(train, "DEFAULT_MODEL_PATH", tmp_path / "churn_model.pkl")
    monkeypatch.setattr(train, "DEFAULT_METADATA_PATH", tmp_path / "churn_model_metadata.json")
    monkeypatch.setattr(train, "COMPARISON_TABLE_PATH", tmp_path / "model_comparison.csv")
    monkeypatch.setattr(train, "FIGURES_DIR", tmp_path / "figures")

    result = train.run_training_pipeline(clean_df)

    assert result["best_model_name"] in {"LogisticRegression"}
    for key in ("model_path", "metadata_path", "figure_path", "comparison_csv_path"):
        assert result[key].exists(), f"{key} was not written to {result[key]}"
    assert all(str(result[key]).startswith(str(tmp_path)) for key in
               ("model_path", "metadata_path", "figure_path", "comparison_csv_path"))
