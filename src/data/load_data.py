"""Load and clean the IBM Telco Customer Churn dataset.

This module only cleans and validates raw data. It does not split, scale, or
encode anything -- that belongs in the Phase 2 feature pipeline.
"""

import logging

import pandas as pd

from src.data.config import PROCESSED_CSV_PATH, PROCESSED_PARQUET_PATH, RAW_DATA_PATH

logger = logging.getLogger(__name__)

TARGET_COLUMN = "Churn"
ID_COLUMN = "customerID"


def load_raw_data(path=RAW_DATA_PATH) -> pd.DataFrame:
    """Read the raw Telco churn CSV as-is."""
    return pd.read_csv(path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the raw Telco churn frame into a modeling-ready shape.

    Steps: fix blank TotalCharges, encode the target, normalize categoricals,
    and drop the customer identifier.
    """
    df = df.copy()

    # Strip stray whitespace from every text column (values and header noise alike).
    object_cols = df.select_dtypes(include="object").columns
    for col in object_cols:
        df[col] = df[col].str.strip()

    # TotalCharges is read as text because 11 rows are "" for brand-new customers
    # (tenure == 0). Coerce to numeric, then impute those rows to 0 since a
    # customer who has been on the books for zero months has been billed nothing.
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(0.0)

    df[TARGET_COLUMN] = df[TARGET_COLUMN].map({"Yes": 1, "No": 0}).astype(int)

    # Keep SeniorCitizen on the same Yes/No vocabulary as the other categorical
    # flags instead of a bare 0/1 int, so EDA and SHAP output read consistently.
    df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"})

    df = df.drop(columns=[ID_COLUMN])

    return df


def save_clean_data(df: pd.DataFrame) -> None:
    """Write the cleaned frame to data/processed/ as CSV and Parquet."""
    PROCESSED_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_CSV_PATH, index=False)
    df.to_parquet(PROCESSED_PARQUET_PATH, index=False)


def load_clean_data(rebuild: bool = False) -> pd.DataFrame:
    """Return the cleaned Telco churn dataset.

    Loads the cached data/processed/telco_clean.parquet if present; otherwise
    (or when rebuild=True) rebuilds it from the raw CSV and caches the result.
    """
    if not rebuild and PROCESSED_PARQUET_PATH.exists():
        return pd.read_parquet(PROCESSED_PARQUET_PATH)

    raw = load_raw_data()
    clean = clean_data(raw)
    save_clean_data(clean)
    return clean


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    clean = load_clean_data(rebuild=True)
    logger.info("Cleaned data shape: %s", clean.shape)
    logger.info("Saved to %s and %s", PROCESSED_CSV_PATH, PROCESSED_PARQUET_PATH)


if __name__ == "__main__":
    main()
