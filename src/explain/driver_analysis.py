"""Global churn-driver identification: numeric correlation, chi-square/Cramer's V
on the Services columns, and SHAP TreeExplainer global importance from a
dedicated diagnostic XGBoost classifier.

This is the first module in the repo to fit a real supervised classifier, but
it is a throwaway diagnostic fit -- never persisted, never compared against
other algorithms, and unrelated to CLAUDE.md Phase 2's future production model
or Phase 3's future local/per-customer explanations. It exists only to rank
global churn drivers, mirroring segmentation.py's precedent of fitting its own
unsupervised model without touching those future phases. See
.claude/specs/06-churn-driver-id.md for the full methodology and the verified
numbers this module's tests lock in.
"""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from scipy.stats import chi2_contingency
from sklearn.compose import ColumnTransformer
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from src.data.config import FIGURES_DIR
from src.data.eda import CHURN_PALETTE, NUMERIC_COLUMNS, save_fig
from src.data.load_data import ID_COLUMN, TARGET_COLUMN

logger = logging.getLogger(__name__)

# CLAUDE.md Sec 6's "Services" column bucket -- chi-square is deliberately
# scoped to just these 9, not every categorical column (see spec Non-goals).
SERVICE_COLUMNS = [
    "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies",
]
RANDOM_STATE = 42
TEST_SIZE = 0.2
CHI_SQUARE_ALPHA = 0.05
LEAKAGE_AUC_THRESHOLD = 0.95
XGB_N_ESTIMATORS = 200
XGB_MAX_DEPTH = 4
XGB_LEARNING_RATE = 0.1
TOP_N_SHAP_FEATURES = 10


def numeric_correlation_with_churn(df: pd.DataFrame) -> pd.DataFrame:
    """Pearson correlation of each NUMERIC_COLUMNS entry with Churn, ranked by |r|."""
    corr = df[NUMERIC_COLUMNS + [TARGET_COLUMN]].corr()[TARGET_COLUMN].drop(TARGET_COLUMN)
    result = pd.DataFrame({
        "column": corr.index,
        "correlation": corr.to_numpy().round(4),
    })
    result["abs_correlation"] = result["correlation"].abs().round(4)
    return result.sort_values("abs_correlation", ascending=False).reset_index(drop=True)


def _cramers_v(chi2: float, n: int, table_shape: tuple[int, int]) -> float:
    """Cramer's V effect size for a chi-square statistic on an (r x c) table."""
    min_dim = min(table_shape) - 1
    return float(np.sqrt(chi2 / (n * min_dim))) if min_dim > 0 else float("nan")


def chi_square_service_association(df: pd.DataFrame) -> pd.DataFrame:
    """Chi-square test of independence + Cramer's V for each SERVICE_COLUMNS entry vs Churn.

    Scoped to SERVICE_COLUMNS only (CLAUDE.md Sec 6's "Services" bucket) --
    Contract/PaymentMethod/etc. are Account columns, not Services, and are
    covered instead by shap_global_importance, which sees every column.

    Uses scipy's default settings: Yates' continuity correction is applied
    automatically to the one 2x2 table (PhoneService) but not to the eight
    3x2 tables, so chi2/p_value/cramers_v are not computed identically
    across every row -- noted since the table mixes both, though it does not
    change any row's significant conclusion. No multiple-comparison
    correction is applied across the 9 tests; at CLAUDE.md's alpha=0.05,
    Bonferroni-adjusting to alpha/9=0.0056 would not change any row's
    significant flag on current data either.
    """
    rows = []
    for col in SERVICE_COLUMNS:
        table = pd.crosstab(df[col], df[TARGET_COLUMN])
        chi2, p_value, dof, _ = chi2_contingency(table)
        cramers_v = _cramers_v(chi2, int(table.to_numpy().sum()), table.shape)
        rows.append({
            "column": col,
            "chi2": round(chi2, 4),
            "p_value": p_value,
            "dof": dof,
            "cramers_v": round(cramers_v, 4),
            "significant": p_value < CHI_SQUARE_ALPHA,
        })
    result = pd.DataFrame(
        rows, columns=["column", "chi2", "p_value", "dof", "cramers_v", "significant"]
    )
    return result.sort_values("cramers_v", ascending=False).reset_index(drop=True)


def build_driver_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Split df into (X, y); X excludes TARGET_COLUMN and ID_COLUMN.

    Leakage-guard surface: MUST NOT include Churn in X -- directly tested in
    tests/test_driver_analysis.py. ID_COLUMN is dropped defensively too
    (errors="ignore", since load_clean_data() already drops it) so this
    guard holds even if a caller ever passes an uncleaned frame.
    """
    y = df[TARGET_COLUMN]
    X = df.drop(columns=[TARGET_COLUMN]).drop(columns=[ID_COLUMN], errors="ignore")
    return X, y


def check_auc_leakage_guard(auc: float, threshold: float = LEAKAGE_AUC_THRESHOLD) -> None:
    """Raise if auc exceeds threshold -- CLAUDE.md Sec 2 rule 2, mechanically enforced.

    A mechanical stand-in for "STOP and investigate leakage" rather than a
    comment a future caller could miss.
    """
    if auc > threshold:
        raise ValueError(
            f"Diagnostic model AUC {auc:.4f} exceeds the leakage threshold "
            f"{threshold}. Investigate for target/probability leakage before "
            "trusting these driver rankings."
        )


def fit_driver_diagnostic_model(
    df: pd.DataFrame, random_state: int = RANDOM_STATE, test_size: float = TEST_SIZE
) -> dict:
    """Fit a throwaway diagnostic XGBoost classifier; return its pipeline + honest test metrics.

    Stratified train/test split, ColumnTransformer (scale numeric, one-hot
    categorical) fit on the train split only, scale_pos_weight computed from
    y_train only. Not persisted, not reused by any other module -- refit
    fresh on every call, exactly like segmentation.py's KMeans.

    Reports AUC-ROC and PR-AUC (CLAUDE.md Sec 2 rule 3), not accuracy.
    Recall at a fixed threshold is deliberately not reported: this diagnostic
    model never makes a churn/no-churn decision (it exists only to rank
    global drivers via shap_global_importance), so there is no decision
    threshold to compute recall at -- CLAUDE.md Sec 7's "tune the decision
    threshold, don't assume 0.5" applies to a production classifier, not
    this throwaway ranking fit.
    """
    X, y = build_driver_features(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    categorical_columns = [c for c in X.columns if c not in NUMERIC_COLUMNS]
    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), NUMERIC_COLUMNS),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_columns),
    ])

    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    pipeline = Pipeline([
        ("pre", preprocessor),
        ("clf", XGBClassifier(
            n_estimators=XGB_N_ESTIMATORS,
            max_depth=XGB_MAX_DEPTH,
            learning_rate=XGB_LEARNING_RATE,
            random_state=random_state,
            eval_metric="logloss",
            scale_pos_weight=scale_pos_weight,
        )),
    ])
    pipeline.fit(X_train, y_train)

    proba = pipeline.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, proba)
    pr_auc = average_precision_score(y_test, proba)
    check_auc_leakage_guard(auc)

    return {
        "pipeline": pipeline,
        "auc": auc,
        "pr_auc": pr_auc,
        "X_test": X_test,
        "y_test": y_test,
    }


def _feature_group_columns(preprocessor: ColumnTransformer, categorical_columns: list[str]) -> np.ndarray:
    """Map each column of the transformed feature matrix to its original source column.

    Built directly from the fitted transformer's own structure -- NUMERIC_COLUMNS
    order for the "num" step, and the fitted OneHotEncoder's categories_ (one
    array of learned categories per input column, in the same order as
    categorical_columns) for the "cat" step -- rather than parsing generated
    "cat__<col>_<level>" feature-name strings. This is exact: there is no
    prefix-matching heuristic and no ambiguity for column names that happen to
    be string prefixes of one another (e.g. a hypothetical InternetService /
    InternetServiceType pair).
    """
    encoder = preprocessor.named_transformers_["cat"]
    groups: list[str] = list(NUMERIC_COLUMNS)
    for column, categories in zip(categorical_columns, encoder.categories_):
        groups.extend([column] * len(categories))
    return np.array(groups)


def shap_global_importance(
    df: pd.DataFrame, result: dict | None = None, top_n: int = TOP_N_SHAP_FEATURES
) -> pd.DataFrame:
    """Mean |SHAP value| per original column, from TreeExplainer on the held-out test split.

    Explains the *test* split (not the training fit), so the reported global
    importance reflects generalization behavior rather than training-set
    overfitting artifacts. `df` is ignored when `result` is supplied (the
    already-fitted model's own test split is explained instead) -- callers
    that want a different df's importances must pass result=None.

    One-hot dummy columns for the same original categorical feature are
    summed *per row* first (SHAP values are additive, so this row's signed
    total is that feature's actual contribution to that prediction), and
    only then is the mean absolute value taken across rows. Summing
    per-dummy mean(|shap|) values instead (as if each dummy level were an
    independent feature) would overcount features with more levels, since
    |a| + |b| >= |a + b| whenever a dummy level's contribution partially
    offsets another's within the same row.
    """
    if result is None:
        result = fit_driver_diagnostic_model(df)

    pipeline = result["pipeline"]
    preprocessor = pipeline.named_steps["pre"]
    model = pipeline.named_steps["clf"]

    X_test_transformed = preprocessor.transform(result["X_test"])
    categorical_columns = [c for c in result["X_test"].columns if c not in NUMERIC_COLUMNS]
    original_columns = _feature_group_columns(preprocessor, categorical_columns)

    explainer = shap.TreeExplainer(model)
    shap_values = np.asarray(explainer.shap_values(X_test_transformed))
    if shap_values.ndim != 2:
        raise ValueError(
            "Expected TreeExplainer.shap_values() to return a 2D array "
            f"(n_samples, n_features) for this binary XGBClassifier, got shape "
            f"{shap_values.shape}. The output shape can vary by shap/xgboost "
            "version for some classifier paths -- update this aggregation if "
            "that has changed rather than silently computing wrong importances."
        )
    if shap_values.shape[1] != len(original_columns):
        raise ValueError(
            f"SHAP output has {shap_values.shape[1]} columns but "
            f"{len(original_columns)} were expected from the fitted "
            "preprocessor -- the feature-group mapping is out of sync with "
            "the transformed matrix."
        )

    ranked = _aggregate_shap_by_group(shap_values, original_columns)
    return ranked.sort_values("mean_abs_shap", ascending=False).head(top_n).reset_index(drop=True)


def _aggregate_shap_by_group(shap_values: np.ndarray, original_columns: np.ndarray) -> pd.DataFrame:
    """Sum each row's signed SHAP values within a group, then take mean(|.|) across rows.

    Pulled out as its own pure function so the aggregation math (see
    shap_global_importance's docstring for why this order matters) can be
    unit-tested directly with a hand-computed matrix, independent of fitting
    a model or running TreeExplainer.
    """
    rows = []
    for column in pd.unique(original_columns):
        row_signed_sum = shap_values[:, original_columns == column].sum(axis=1)
        rows.append({"column": column, "mean_abs_shap": round(float(np.abs(row_signed_sum).mean()), 4)})
    return pd.DataFrame(rows, columns=["column", "mean_abs_shap"])


def plot_numeric_correlation(df: pd.DataFrame, out_dir: Path = FIGURES_DIR) -> Path:
    """Horizontal bar chart of signed correlation with Churn per numeric column."""
    ranked = numeric_correlation_with_churn(df)
    fig, ax = plt.subplots(figsize=(6, 4))
    colors = [CHURN_PALETTE[1] if v > 0 else CHURN_PALETTE[0] for v in ranked["correlation"]]
    ax.barh(ranked["column"], ranked["correlation"], color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Correlation with Churn")
    ax.set_title("Numeric Feature Correlation with Churn")
    ax.invert_yaxis()
    return save_fig(fig, "driver_correlation_with_churn.png", out_dir)


def plot_chi_square_service_association(df: pd.DataFrame, out_dir: Path = FIGURES_DIR) -> Path:
    """Bar chart of Cramer's V per service column; the non-significant column is greyed out."""
    ranked = chi_square_service_association(df)
    fig, ax = plt.subplots(figsize=(7, 4))
    colors = [CHURN_PALETTE[1] if sig else "#B0B0B0" for sig in ranked["significant"]]
    ax.bar(ranked["column"], ranked["cramers_v"], color=colors)
    ax.set_ylabel("Cramer's V")
    ax.set_title("Chi-Square Association: Service Columns vs Churn")
    ax.tick_params(axis="x", rotation=30)
    for tick in ax.get_xticklabels():
        tick.set_ha("right")
    return save_fig(fig, "driver_chi_square_service_association.png", out_dir)


def plot_shap_global_importance(
    df: pd.DataFrame, out_dir: Path = FIGURES_DIR, result: dict | None = None
) -> Path:
    """Horizontal bar chart of the top TOP_N_SHAP_FEATURES by mean |SHAP value|."""
    ranked = shap_global_importance(df, result)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(ranked["column"], ranked["mean_abs_shap"], color=CHURN_PALETTE[1])
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title(f"Top {len(ranked)} SHAP Global Feature Importance")
    ax.invert_yaxis()
    return save_fig(fig, "driver_shap_global_importance.png", out_dir)


def generate_driver_figures(df: pd.DataFrame, out_dir: Path = FIGURES_DIR) -> list[Path]:
    """Generate and save every driver-analysis chart, fitting the diagnostic model once."""
    diagnostic_result = fit_driver_diagnostic_model(df)
    logger.info(
        "Diagnostic model: AUC-ROC=%.4f, PR-AUC=%.4f (leakage threshold=%.2f)",
        diagnostic_result["auc"], diagnostic_result["pr_auc"], LEAKAGE_AUC_THRESHOLD,
    )
    return [
        plot_numeric_correlation(df, out_dir),
        plot_chi_square_service_association(df, out_dir),
        plot_shap_global_importance(df, out_dir, diagnostic_result),
    ]


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    from src.data.load_data import load_clean_data

    df = load_clean_data()
    paths = generate_driver_figures(df)
    logger.info("Saved %d driver-analysis figures to reports/figures/", len(paths))


if __name__ == "__main__":
    main()
