import pandas as pd

from src.data.load_data import ID_COLUMN, TARGET_COLUMN, load_raw_data, missing_total_charges_rows

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
