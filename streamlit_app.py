from __future__ import annotations

import sys
import importlib
from pathlib import Path

import pandas as pd
import streamlit as st

# Prefer installed package (pyproject). If not installed locally, add src/ to sys.path.
try:
    DL = importlib.import_module("rossmann.data_load")
    DP = importlib.import_module("rossmann.data_prep")
    from src.rossmann.ets_stepwise import fit_and_backtest_ets   # lives in src/rossmann/
except Exception:
    sys.path.append(str(Path(__file__).resolve().parent / "src"))
    DL = importlib.import_module("rossmann.data_load")
    DP = importlib.import_module("rossmann.data_prep")
    from src.rossmann.ets_stepwise import fit_and_backtest_ets

# pick your real function names without touching your files
def _get_attr(mod, *names):
    for n in names:
        if hasattr(mod, n):
            return getattr(mod, n)
    # Show what exists to help you rename the tuple below if needed
    raise AttributeError(
        f"None of {names} found in {mod.__name__}. "
        f"Available: {', '.join(sorted(k for k in dir(mod) if not k.startswith('_')))}"
    )

# Your loader IS called load_imputed per your note
load_raw = _get_attr(DL, "load_imputed")

# Try to find a monthly converter in data_prep with common names
# If you know the exact name, replace the tuple below with it first.
try:
    to_monthly = _get_attr(
        DP,
        "to_monthly_series", "to_monthly", "make_monthly_series",
        "monthly_sum", "monthly_agg",
    )
except AttributeError:
    # Safe fallback: do monthly sum here if your data_prep lacks a helper
    def to_monthly(df: pd.DataFrame, *, date_col: str, value_col: str) -> pd.Series:
        s = (
            df.set_index(pd.to_datetime(df[date_col]))[value_col]
              .resample("MS").sum(min_count=1)
              .dropna()
        )
        s.index.name = date_col
        s.name = value_col
        return s
