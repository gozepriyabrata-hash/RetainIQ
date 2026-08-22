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


def missing_total_charges_rows(raw: pd.DataFrame) -> pd.DataFrame:
    """Return raw rows whose TotalCharges is blank (pre-imputation).

    `.astype(str)` before `.str.strip()` so this doesn't raise if a future
    CSV re-download has no blanks and pandas reads the column as numeric.
    """
    blank_mask = pd.to_numeric(raw["TotalCharges"].astype(str).str.strip(), errors="coerce").isna()
    return raw.loc[blank_mask]


def _clean_common_fields(df: pd.DataFrame) -> pd.DataFrame:
    """Dataset-quirk fixes shared by clean_data() and prepare_scoring_input().

    Strips stray whitespace from every text column, coerces TotalCharges to
    numeric (imputing the 11 blank-string new-customer rows to 0, since a
    customer on the books for zero months has been billed nothing), and
    normalizes SeniorCitizen onto the same Yes/No vocabulary as the other
    categorical flags. The SeniorCitizen mapping only applies when its dtype
    isn't already object -- a generalization clean_data() never previously
    needed (the raw CSV always encodes it numerically), added so a caller
    that already passes "Yes"/"No" (e.g. an inference payload) is left
    untouched instead of raising or nulling the column out.
    """
    df = df.copy()

    object_cols = df.select_dtypes(include="object").columns
    for col in object_cols:
        df[col] = df[col].str.strip()

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(0.0)

    if df["SeniorCitizen"].dtype != "object":
        df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"})

    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the raw Telco churn frame into a modeling-ready shape.

    Steps: fix blank TotalCharges, encode the target, normalize categoricals,
    and drop the customer identifier.
    """
    df = _clean_common_fields(df)

    df[TARGET_COLUMN] = df[TARGET_COLUMN].map({"Yes": 1, "No": 0}).astype(int)

    df = df.drop(columns=[ID_COLUMN])

    return df


def prepare_scoring_input(raw_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series | None]:
    """Clean unlabeled inference input for scoring: (features_df, customer_ids).

    Unlike clean_data(), this does not require (or touch) a Churn column --
    the input is a customer to be scored, not a labeled training row. If
    ID_COLUMN is present it's captured and returned separately (so a caller
    can re-attach it to scored output) rather than silently dropped with no
    way to recover it; if absent, customer_ids is None.
    """
    customer_ids = raw_df[ID_COLUMN].copy() if ID_COLUMN in raw_df.columns else None
    features_df = _clean_common_fields(raw_df)
    features_df = features_df.drop(columns=[ID_COLUMN, TARGET_COLUMN], errors="ignore")
    return features_df, customer_ids


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
