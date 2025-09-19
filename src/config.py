from __future__ import annotations

# Defaults (can be overridden by Streamlit secrets or environment variables)
DATA_URL_SALES = "https://github.com/hsnalmasri/RossmannStoreSales/releases/download/v0.1-data/SalesData.csv"
DATA_URL_STORE = "https://github.com/hsnalmasri/RossmannStoreSales/releases/download/v0.1-data/StoreData.csv"

# Core column names
COL_STORE = "Store"
COL_DATE = "Date"
COL_SALES = "Sales"

__all__ = [
    "DATA_URL_SALES",
    "DATA_URL_STORE",
    "COL_STORE",
    "COL_DATE",
    "COL_SALES",
]
