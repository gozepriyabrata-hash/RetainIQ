import pandas as pd
import pytest

from src.data.load_data import clean_data, load_raw_data


@pytest.fixture(scope="session")
def clean_df() -> pd.DataFrame:
    return clean_data(load_raw_data())
