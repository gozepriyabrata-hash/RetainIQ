"""Phase 3: local (per-customer) SHAP + LIME explanations, plain-English reasons.

Explains the real, persisted, production `models/churn_model.pkl` (`08`) --
not `src.explain.driver_analysis`'s disposable diagnostic model, which ranks
global drivers from a throwaway XGBoost fit unrelated to production. Both
SHAP and LIME here explain the *raw* pipeline (`train.load_trained_model()`),
never the calibrated `CalibratedClassifierCV` wrapper `src.models.calibration`
produces: `shap.TreeExplainer` cannot open a `CalibratedClassifierCV`'s
internal 5-fold structure, so both methods are kept consistent by explaining
the same raw model rather than mixing a raw-model SHAP number with a
calibrated-model LIME number. Isotonic calibration reshapes *how confident*
a probability is, not *which features* drove the prediction or their
*direction*, so the calibrated probability shown elsewhere (`src.models.
scoring`) and this module's driver explanations are complementary, not
required to numerically reconcile.

SHAP value scale note: `shap.TreeExplainer` on this XGBoost model returns
values in the model's raw margin (log-odds) space, not probability-
percentage-point space. Only the *sign* of a SHAP value is treated as a
reliable signal here ("increases"/"decreases" risk) -- the raw magnitude is
reported for reference but never phrased as a percentage-point contribution.

See .claude/specs/11-explainable-ai.md for the full methodology and the
verified numbers this module's tests lock in.
"""

import json
import logging
import re
from collections.abc import Callable
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from lime.lime_tabular import LimeTabularExplainer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder

from src.data.config import FIGURES_DIR
from src.data.eda import CHURN_PALETTE, NUMERIC_COLUMNS, save_fig
from src.data.load_data import load_clean_data, prepare_scoring_input
from src.features.preprocessing import get_categorical_columns
from src.models import train
from src.models.evaluation import RANDOM_STATE

logger = logging.getLogger(__name__)

TOP_N_LOCAL_DRIVERS = 3
TOP_N_GLOBAL_FEATURES = 10
LIME_NUM_SAMPLES = 5000
TREE_BASED_MODELS = frozenset({"RandomForest", "XGBoost", "LightGBM"})
PRODUCTION_SHAP_FIGURE_FILENAME = "production_shap_global_importance.png"

# Natural-language phrase templates for CLAUDE.md Sec 6's 5 documented churn
# signals; any other column falls back to a generic "{column} = {value}"
# phrase in humanize_reason.
FEATURE_DISPLAY_TEMPLATES = {
    "Contract": "{value} contract",
    "tenure": "{value}-month tenure",
    "TechSupport": "tech support: {value}",
    "PaymentMethod": "{value} payment method",
    "InternetService": "{value} internet service",
}


def _feature_group_columns(preprocessor: ColumnTransformer, categorical_columns: list[str]) -> np.ndarray:
    """Map each column of the transformed feature matrix to its original source column.

    Duplicated from src.explain.driver_analysis._feature_group_columns
    (same exact-mapping mechanism, built from the fitted OneHotEncoder's own
    categories_) rather than imported -- this module explains a different
    model (the real persisted one, not driver_analysis's throwaway
    diagnostic fit) and .claude/specs/11-explainable-ai.md's Non-goals keep
    driver_analysis.py untouched.
    """
    encoder = preprocessor.named_transformers_["cat"]
    groups: list[str] = list(NUMERIC_COLUMNS)
    for column, categories in zip(categorical_columns, encoder.categories_):
        groups.extend([column] * len(categories))
    return np.array(groups)


def _aggregate_shap_by_group(shap_values: np.ndarray, original_columns: np.ndarray) -> pd.DataFrame:
    """Sum each row's signed SHAP values within a group, then take mean(|.|) across rows.

    Duplicated from src.explain.driver_analysis._aggregate_shap_by_group.
    Correct for a *global*, multi-row ranking -- for a single customer's
    signed local drivers, use the dedicated single-row sum in
    local_shap_top_drivers instead, since mean(|.|) over one row discards
    the sign a local driver needs.
    """
    rows = []
    for column in pd.unique(original_columns):
        row_signed_sum = shap_values[:, original_columns == column].sum(axis=1)
        rows.append({"column": column, "mean_abs_shap": round(float(np.abs(row_signed_sum).mean()), 4)})
    return pd.DataFrame(rows, columns=["column", "mean_abs_shap"])


def _direction(signed_value: float) -> str:
    """Returns "increases"/"decreases"/"neutral" for a signed SHAP value or LIME weight."""
    if signed_value > 0:
        return "increases"
    if signed_value < 0:
        return "decreases"
    return "neutral"


# humanize_reason's direction clause per _direction's three possible outputs
# -- kept as one mapping so the two can't drift out of sync with each other.
_DIRECTION_PHRASES = {
    "increases": "increases",
    "decreases": "decreases",
    "neutral": "has no measurable effect on",
}


def _to_python(value):
    """Unwrap a numpy scalar (np.int64, np.str_, ...) to its native Python type.

    local_shap_top_drivers' output is the payload a future Phase 5
    POST /explain route returns; json.dumps chokes on numpy scalar types,
    so every value that reaches a driver dict is coerced here.
    """
    return value.item() if hasattr(value, "item") else value


def _build_lime_explainer(
    X_train: pd.DataFrame,
    feature_columns: list[str],
    categorical_columns: list[str],
    label_encoders: dict[str, LabelEncoder],
) -> LimeTabularExplainer:
    """LimeTabularExplainer over the raw (pre-one-hot) feature space.

    Operating in raw feature space (rather than the ColumnTransformer's
    one-hot output, as SHAP must) means LIME reports one row per original
    column (e.g. "Contract=Two year"), needing no dummy-aggregation step --
    verified during spec research against .claude/specs/11-explainable-ai.md.
    """
    encoded = np.column_stack([
        label_encoders[col].transform(X_train[col]) if col in label_encoders
        else X_train[col].to_numpy(dtype=float)
        for col in feature_columns
    ]).astype(float)
    categorical_feature_idx = [feature_columns.index(c) for c in categorical_columns]
    categorical_names = {
        feature_columns.index(c): list(label_encoders[c].classes_) for c in categorical_columns
    }
    return LimeTabularExplainer(
        training_data=encoded,
        feature_names=feature_columns,
        class_names=["No Churn", "Churn"],
        categorical_features=categorical_feature_idx,
        categorical_names=categorical_names,
        mode="classification",
        random_state=RANDOM_STATE,
    )


def build_explainer_context(df: pd.DataFrame) -> dict:
    """Load the persisted production pipeline once; build a reusable SHAP + LIME context.

    Fails fast, before any expensive SHAP/LIME construction: raises
    ValueError if the persisted winning model isn't tree-based (TreeExplainer
    can only open RandomForest/XGBoost/LightGBM), then raises ValueError if
    churn_model_metadata.json's feature_columns are stale relative to `df`
    (mirrors src.models.calibration.run_calibration_pipeline's staleness
    guard). Reuse one context across many explain calls -- CLAUDE.md Sec 10:
    "load the model once, not per request."
    """
    metadata = json.loads(train.DEFAULT_METADATA_PATH.read_text())
    model_name = metadata["model_name"]
    if model_name not in TREE_BASED_MODELS:
        raise ValueError(
            f"shap.TreeExplainer requires a tree-based model; the persisted "
            f"winner is {model_name!r}. Local/global SHAP explanations are "
            "not available until a tree-based model (RandomForest/XGBoost/"
            "LightGBM) is retrained and persisted as churn_model.pkl."
        )

    X_train, X_test, _, y_test = train.split_data(df)
    if list(X_train.columns) != metadata["feature_columns"]:
        raise ValueError(
            "X_train columns no longer match churn_model_metadata.json's "
            "feature_columns -- the persisted model is stale relative to "
            "`df`. Re-run `python -m src.models.train` before building an "
            "explainer context."
        )

    pipeline = train.load_trained_model()
    preprocessor = pipeline.named_steps["pre"]
    clf = pipeline.named_steps["clf"]

    categorical_columns = get_categorical_columns(X_train)
    feature_group_map = _feature_group_columns(preprocessor, categorical_columns)

    shap_explainer = shap.TreeExplainer(clf)

    label_encoders = {col: LabelEncoder().fit(X_train[col]) for col in categorical_columns}
    lime_explainer = _build_lime_explainer(
        X_train, metadata["feature_columns"], categorical_columns, label_encoders
    )

    return {
        "pipeline": pipeline,
        "preprocessor": preprocessor,
        "clf": clf,
        "model_name": model_name,
        "feature_columns": metadata["feature_columns"],
        "categorical_columns": categorical_columns,
        "feature_group_map": feature_group_map,
        "X_train": X_train,
        "X_test": X_test,
        "y_test": y_test,
        "shap_explainer": shap_explainer,
        "lime_explainer": lime_explainer,
        "label_encoders": label_encoders,
    }


def global_shap_importance(context: dict, top_n: int = TOP_N_GLOBAL_FEATURES) -> pd.DataFrame:
    """Mean |SHAP value| per original column on the production model's held-out test split.

    A second, production-honest global ranking distinct from
    src.explain.driver_analysis's diagnostic-model one -- computed on
    context["X_test"] (never the training fit), aggregating one-hot dummies
    back to their original column exactly as driver_analysis established.
    """
    X_test_transformed = context["preprocessor"].transform(context["X_test"])
    shap_values = np.asarray(context["shap_explainer"].shap_values(X_test_transformed))
    feature_group_map = context["feature_group_map"]
    if shap_values.ndim != 2:
        raise ValueError(
            "Expected TreeExplainer.shap_values() to return a 2D array "
            f"(n_samples, n_features) for this classifier, got shape "
            f"{shap_values.shape}. The output shape can vary by shap/xgboost "
            "version for some classifier paths -- update this aggregation if "
            "that has changed rather than silently computing wrong importances."
        )
    if shap_values.shape[1] != len(feature_group_map):
        raise ValueError(
            f"SHAP output has {shap_values.shape[1]} columns but "
            f"{len(feature_group_map)} were expected from the fitted "
            "preprocessor -- the feature-group mapping is out of sync with "
            "the transformed matrix."
        )
    ranked = _aggregate_shap_by_group(shap_values, feature_group_map)
    return ranked.sort_values("mean_abs_shap", ascending=False).head(top_n).reset_index(drop=True)


def plot_global_shap_importance(
    context: dict, top_n: int = TOP_N_GLOBAL_FEATURES, out_dir: Path = FIGURES_DIR
) -> Path:
    """Horizontal bar chart of the top top_n production-model SHAP global importances."""
    ranked = global_shap_importance(context, top_n)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(ranked["column"], ranked["mean_abs_shap"], color=CHURN_PALETTE[1])
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title(f"Top {len(ranked)} SHAP Global Feature Importance (Production Model)")
    ax.invert_yaxis()
    return save_fig(fig, PRODUCTION_SHAP_FIGURE_FILENAME, out_dir)


def local_shap_top_drivers(
    context: dict, features_df: pd.DataFrame, top_n: int = TOP_N_LOCAL_DRIVERS
) -> list[dict]:
    """Top top_n SHAP drivers for one customer, ranked by |signed SHAP value|.

    features_df is exactly 1 row in raw (pre-transform) feature space,
    already reindexed to context["feature_columns"]. Sums each group's
    *signed* SHAP values for this one row (not driver_analysis's
    mean(|.|)-across-rows aggregation, which discards sign -- correct only
    for the multi-row global ranking) so direction is meaningful per driver.
    """
    row_transformed = context["preprocessor"].transform(features_df)
    shap_values = np.asarray(context["shap_explainer"].shap_values(row_transformed))
    if shap_values.shape != (1, len(context["feature_group_map"])):
        raise ValueError(
            f"Expected TreeExplainer.shap_values() to return shape "
            f"(1, {len(context['feature_group_map'])}) for one customer, "
            f"got {shap_values.shape}."
        )

    feature_group_map = context["feature_group_map"]
    signed = {
        column: float(shap_values[0, feature_group_map == column].sum())
        for column in pd.unique(feature_group_map)
    }
    ranked = sorted(signed.items(), key=lambda kv: abs(kv[1]), reverse=True)[:top_n]

    drivers = []
    for column, value in ranked:
        column = str(column)
        customer_value = _to_python(features_df.iloc[0][column])
        drivers.append({
            "feature": column,
            "customer_value": customer_value,
            "shap_value": round(value, 4),
            "direction": _direction(value),
            "reason": humanize_reason(column, customer_value, value),
        })
    return drivers


def _encode_customer_row(features_df: pd.DataFrame, context: dict) -> np.ndarray:
    """Label-encode one customer row into LIME's raw-feature-space representation.

    Raises a clear, actionable ValueError naming the column and the
    offending value for a categorical value never seen in X_train -- checked
    proactively (rather than catching LabelEncoder.transform's own
    exception), since sklearn's "y contains previously unseen labels"
    message truncates the reported value in a way that's unusable for a
    user-facing error.
    """
    row = features_df.iloc[0]
    encoded = []
    for column in context["feature_columns"]:
        if column in context["label_encoders"]:
            encoder = context["label_encoders"][column]
            value = row[column]
            if value not in encoder.classes_:
                raise ValueError(
                    f"local_lime_top_drivers: column {column!r} has value "
                    f"{value!r}, which was never seen during training and "
                    "cannot be LIME-explained."
                )
            encoded.append(float(encoder.transform([value])[0]))
        else:
            encoded.append(float(row[column]))
    return np.array(encoded)


def _make_lime_predict_fn(context: dict) -> Callable[[np.ndarray], np.ndarray]:
    """A LIME-compatible predict_fn wrapping the raw pipeline's predict_proba.

    Decodes an array of label-encoded rows back to raw dtypes, then calls
    context["pipeline"].predict_proba -- imblearn's Pipeline only invokes
    SMOTE during .fit(), so .predict_proba() skips it automatically here.
    """
    feature_columns = context["feature_columns"]
    label_encoders = context["label_encoders"]
    dtypes = context["X_train"][feature_columns].dtypes

    def predict_fn(arr: np.ndarray) -> np.ndarray:
        decoded = pd.DataFrame(arr, columns=feature_columns)
        for column, encoder in label_encoders.items():
            # Clipping an out-of-range code to a boundary category is
            # defensive (LIME never perturbs a categorical column outside
            # its known codes) -- intentionally a wrong-but-valid category
            # rather than a crash on an otherwise-unreachable path.
            idx = np.clip(np.rint(decoded[column].to_numpy()).astype(int), 0, len(encoder.classes_) - 1)
            decoded[column] = encoder.inverse_transform(idx)
        for column in NUMERIC_COLUMNS:
            target_dtype = dtypes[column]
            if np.issubdtype(target_dtype, np.integer):
                # rint before the int cast: a bare .astype(int) truncates
                # toward zero (55.7 -> 55), biasing every perturbed sample
                # down by up to a full unit; round-then-cast doesn't.
                decoded[column] = np.rint(decoded[column].to_numpy()).astype(target_dtype)
            else:
                decoded[column] = decoded[column].astype(target_dtype)
        return context["pipeline"].predict_proba(decoded[feature_columns])

    return predict_fn


def _column_from_lime_description(description: str, feature_columns: list[str]) -> str:
    """Resolve a LIME condition string (e.g. "Contract=Two year", "tenure > 55.00") to its column.

    Extracts identifier-like tokens (numeric literals like "55.00" never
    match) and intersects with feature_columns -- exact word-boundary
    matching, never a bare startswith, to avoid a prefix-collision
    misattribution. Raises ValueError if the result isn't exactly one column.
    """
    tokens = set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", description))
    matches = tokens & set(feature_columns)
    if len(matches) != 1:
        raise ValueError(
            f"LIME description {description!r} did not resolve to exactly "
            f"one known feature column (matched {sorted(matches)})."
        )
    return matches.pop()


def local_lime_top_drivers(
    context: dict, features_df: pd.DataFrame, top_n: int = TOP_N_LOCAL_DRIVERS
) -> list[dict]:
    """Top top_n LIME local drivers for one customer -- CLAUDE.md Sec 7's "alternative local view".

    features_df is exactly 1 row in raw (pre-transform) feature space,
    already reindexed to context["feature_columns"]. Explains the same raw
    pipeline SHAP explains (see module docstring), so the two methods are
    genuinely comparable. LIME weight values are not exactly reproducible
    across repeated explain_instance calls on a shared, reused explainer
    (its internal RNG advances per call) even at a fixed random_state --
    directions and description strings are stable, magnitudes are not, so
    callers should treat lime_weight as approximate.
    """
    encoded_row = _encode_customer_row(features_df, context)
    predict_fn = _make_lime_predict_fn(context)
    explanation = context["lime_explainer"].explain_instance(
        encoded_row, predict_fn, num_features=top_n, num_samples=LIME_NUM_SAMPLES, labels=(1,)
    )

    drivers = []
    for description, weight in explanation.as_list(label=1):
        weight = float(weight)
        column = _column_from_lime_description(description, context["feature_columns"])
        customer_value = _to_python(features_df.iloc[0][column])
        drivers.append({
            "feature": str(description),
            "lime_weight": round(weight, 4),
            "direction": _direction(weight),
            "reason": humanize_reason(column, customer_value, weight),
        })
    return drivers


def humanize_reason(column: str, customer_value: object, signed_value: float) -> str:
    """Plain-English sentence for one driver -- CLAUDE.md Sec 7: "must be human-readable".

    Never surfaces a raw transformed-feature name (e.g. cat__Contract_...)
    -- only the original column name or its FEATURE_DISPLAY_TEMPLATES phrase.
    """
    direction_word = _DIRECTION_PHRASES[_direction(signed_value)]
    template = FEATURE_DISPLAY_TEMPLATES.get(column, "{column} = {value}")
    phrase = template.format(value=customer_value, column=column)
    return f"{phrase} {direction_word} this customer's predicted churn risk."


def explain_customer(customer: dict, context: dict | None = None) -> dict:
    """Raw customer attributes in, SHAP + LIME top-N drivers out.

    The function a future Phase 5 POST /explain endpoint (CLAUDE.md Sec 10)
    calls directly. Building a context is expensive (fits a TreeExplainer
    and a LimeTabularExplainer) -- a caller explaining more than one
    customer should build one via build_explainer_context and pass it in,
    rather than relying on this function's default.
    """
    if context is None:
        context = build_explainer_context(load_clean_data())

    # Checked against the raw dict's own keys, before prepare_scoring_input:
    # that helper unconditionally touches TotalCharges/SeniorCitizen inside
    # _clean_common_fields, so a customer missing either would otherwise
    # raise a bare KeyError instead of this named ValueError.
    missing = [c for c in context["feature_columns"] if c not in customer]
    if missing:
        raise ValueError(f"customer is missing required feature columns: {missing}")

    features_df, customer_ids = prepare_scoring_input(pd.DataFrame([customer]))
    features_df = features_df[context["feature_columns"]]

    result = {
        "shap_top_drivers": local_shap_top_drivers(context, features_df),
        "lime_top_drivers": local_lime_top_drivers(context, features_df),
    }
    if customer_ids is not None:
        result["customerID"] = _to_python(customer_ids.iloc[0])
    return result


def generate_explainability_figures(df: pd.DataFrame, out_dir: Path = FIGURES_DIR) -> list[Path]:
    """Build one explainer context and generate every explainability chart."""
    context = build_explainer_context(df)
    return [plot_global_shap_importance(context, out_dir=out_dir)]


def main() -> None:
    """Entry point for `python -m src.explain.local_explainer`: regenerate the figures."""
    logging.basicConfig(level=logging.INFO)
    df = load_clean_data()
    paths = generate_explainability_figures(df)
    logger.info("Saved %d explainability figures to reports/figures/", len(paths))


if __name__ == "__main__":
    main()
