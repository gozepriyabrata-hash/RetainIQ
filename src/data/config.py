"""Paths used by the data-loading module, relative to the project root."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "telco.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PROCESSED_CSV_PATH = PROCESSED_DIR / "telco_clean.csv"
PROCESSED_PARQUET_PATH = PROCESSED_DIR / "telco_clean.parquet"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
