"""Unfitted preprocessing transformer shared by model training and (in a
future Phase 5) serving. This module never fits anything -- CLAUDE.md
Section 4 keeps feature transforms in src/features/ and anything that trains
or scores in src/models/.
"""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data.eda import NUMERIC_COLUMNS


def get_categorical_columns(X: pd.DataFrame) -> list[str]:
    """Every column of X that is not one of NUMERIC_COLUMNS."""
    return [c for c in X.columns if c not in NUMERIC_COLUMNS]


def build_preprocessor(categorical_columns: list[str]) -> ColumnTransformer:
    """Scale NUMERIC_COLUMNS and one-hot encode categorical_columns. Unfitted.

    Callers are responsible for fitting this only on training data (inside a
    Pipeline / cross-validation fold), never on the full dataset before a
    train/test split.
    """
    return ColumnTransformer([
        ("num", StandardScaler(), NUMERIC_COLUMNS),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_columns),
    ])
