import pandas as pd

from src.data.load_data import (
    ID_COLUMN,
    TARGET_COLUMN,
    clean_data,
    load_raw_data,
    missing_total_charges_rows,
    prepare_scoring_input,
)

EXPECTED_COLUMNS = {
    "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
    "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
    "MonthlyCharges", "TotalCharges", "Churn",
}


def test_no_missing_values(clean_df):
    assert clean_df.isnull().sum().sum() == 0


def test_churn_is_binary(clean_df):
    assert set(clean_df[TARGET_COLUMN].unique()) == {0, 1}


def test_customer_id_dropped(clean_df):
    assert ID_COLUMN not in clean_df.columns


def test_total_charges_is_numeric(clean_df):
    assert pd.api.types.is_numeric_dtype(clean_df["TotalCharges"])


def test_tenure_zero_customers_have_zero_total_charges(clean_df):
    new_customers = clean_df[clean_df["tenure"] == 0]
    assert len(new_customers) == 11
    assert (new_customers["TotalCharges"] == 0).all()


def test_row_count_preserved(clean_df):
    assert len(clean_df) == 7043


def test_expected_columns_present(clean_df):
    assert set(clean_df.columns) == EXPECTED_COLUMNS


def test_churn_column_is_integer_dtype(clean_df):
    assert pd.api.types.is_integer_dtype(clean_df[TARGET_COLUMN])


def test_missing_total_charges_pattern():
    missing = missing_total_charges_rows(load_raw_data())
    assert len(missing) == 11
    assert (missing["tenure"] == 0).all()
    assert (missing["Churn"] == "No").all()
    assert missing["Contract"].value_counts().to_dict() == {"Two year": 10, "One year": 1}


def test_clean_data_output_unchanged_after_refactor(clean_df):
    # Regression guard for the _clean_common_fields extraction: a fresh
    # clean_data(load_raw_data()) call must still be byte-for-byte identical
    # to the session-scoped clean_df fixture's own independent computation.
    fresh = clean_data(load_raw_data())
    pd.testing.assert_frame_equal(fresh, clean_df)


def test_prepare_scoring_input_drops_id_and_returns_it_separately(clean_df):
    raw = clean_df.drop(columns=[TARGET_COLUMN]).head(5).copy()
    raw.insert(0, ID_COLUMN, [f"CUST-{i}" for i in range(5)])

    features_df, customer_ids = prepare_scoring_input(raw)

    assert ID_COLUMN not in features_df.columns
    assert TARGET_COLUMN not in features_df.columns
    assert list(customer_ids) == list(raw[ID_COLUMN])


def test_prepare_scoring_input_handles_missing_customer_id(clean_df):
    raw = clean_df.drop(columns=[TARGET_COLUMN]).head(3).copy()

    features_df, customer_ids = prepare_scoring_input(raw)

    assert customer_ids is None
    assert len(features_df) == 3


def test_prepare_scoring_input_imputes_blank_total_charges():
    raw = load_raw_data().head(1).copy()
    raw["TotalCharges"] = ""

    features_df, _ = prepare_scoring_input(raw)

    assert features_df["TotalCharges"].iloc[0] == 0.0
    assert pd.api.types.is_numeric_dtype(features_df["TotalCharges"])


def test_prepare_scoring_input_does_not_require_churn_column(clean_df):
    raw = clean_df.drop(columns=[TARGET_COLUMN]).head(2)
    assert TARGET_COLUMN not in raw.columns

    features_df, _ = prepare_scoring_input(raw)  # must not raise

    assert len(features_df) == 2
