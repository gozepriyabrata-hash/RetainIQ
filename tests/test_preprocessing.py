import numpy as np
import pytest

from src.data.eda import NUMERIC_COLUMNS
from src.data.load_data import TARGET_COLUMN
from src.features.preprocessing import build_preprocessor, get_categorical_columns


def test_get_categorical_columns_excludes_numeric(clean_df):
    X = clean_df.drop(columns=[TARGET_COLUMN])
    categorical = get_categorical_columns(X)
    assert set(categorical).isdisjoint(NUMERIC_COLUMNS)
    assert set(categorical) | set(NUMERIC_COLUMNS) == set(X.columns)


def test_get_categorical_columns_returns_no_duplicates(clean_df):
    X = clean_df.drop(columns=[TARGET_COLUMN])
    categorical = get_categorical_columns(X)
    assert len(categorical) == len(set(categorical))


def test_build_preprocessor_fits_and_transforms_without_nan(clean_df):
    X = clean_df.drop(columns=[TARGET_COLUMN])
    categorical_columns = get_categorical_columns(X)
    preprocessor = build_preprocessor(categorical_columns)

    transformed = preprocessor.fit_transform(X)

    assert transformed.shape[0] == len(X)
    assert not np.isnan(transformed).any()


def test_build_preprocessor_handles_unseen_category_at_transform_time(clean_df):
    X = clean_df.drop(columns=[TARGET_COLUMN])
    categorical_columns = get_categorical_columns(X)
    preprocessor = build_preprocessor(categorical_columns)
    preprocessor.fit(X)

    unseen = X.iloc[[0]].copy()
    unseen["Contract"] = "NeverSeenBeforeContractType"

    # handle_unknown="ignore" must not raise on a category absent from training data.
    transformed = preprocessor.transform(unseen)
    assert transformed.shape[0] == 1
    assert not np.isnan(transformed).any()
