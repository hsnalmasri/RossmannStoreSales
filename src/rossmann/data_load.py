# src/rossmann/data_load.py
from __future__ import annotations
import pandas as pd
from .paths import RAW

def _read_csv(filename: str, parse_dates: list[str] | None = None) -> pd.DataFrame:
    p = RAW / filename
    if not p.exists():
        raise FileNotFoundError(f"Expected file not found: {p}")
    return pd.read_csv(p, parse_dates=parse_dates)

def load_sales() -> pd.DataFrame:
    # assuming SalesData.csv has a Date column
    return _read_csv("SalesData.csv", parse_dates=["Date"])

def load_store() -> pd.DataFrame:
    return _read_csv("StoreData.csv")
