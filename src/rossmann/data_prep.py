# src/data_prep.py
from __future__ import annotations
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional

# --- Constants
MONTH_MAP = {'Jan':1,'Feb':2,'Mar':3,'Apr':4,'May':5,'Jun':6,
             'Jul':7,'Aug':8,'Sep':9,'Oct':10,'Nov':11,'Dec':12}

__all__ = ["load_rossmann_data"]


def _interval_to_set(s: object) -> set[int]:
    """Convert 'Feb,May,Aug,Nov' -> {2,5,8,11}. Empty set for NaN/blank."""
    if pd.isna(s) or not str(s).strip():
        return set()
    tokens = (t.strip() for t in str(s).split(','))
    return {MONTH_MAP[t] for t in tokens if t in MONTH_MAP}


def _months_diff(d1: pd.Series, d0: pd.Series) -> pd.Series:
    """(year, month) difference ignoring days. NaT-propagating."""
    y1 = d1.dt.year; m1 = d1.dt.month
    y0 = d0.dt.year; m0 = d0.dt.month
    return (y1 - y0) * 12 + (m1 - m0)


def load_rossmann_data(
    sales_csv: str | Path,
    store_csv: str | Path,
    *,
    keep_closed_days: bool = True,
    build_flat_copy: bool = True
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """
    Read, merge, and feature-engineer Rossmann sales + store metadata.

    Parameters
    ----------
    sales_csv : path-like
        Path to SalesData.csv (original train.csv) with columns:
        ['Store','Date','Sales','Customers','Open','Promo','StateHoliday','SchoolHoliday', ...]
    store_csv : path-like
        Path to StoreData.csv (original store.csv).
    keep_closed_days : bool
        If False, drop rows where Open == 0. (Useful if you only model open days.)
    build_flat_copy : bool
        If True, return a flat copy with 'Store' and 'Date' as columns (df_ml). Otherwise None.

    Returns
    -------
    df_ts : pd.DataFrame
        TS-friendly DataFrame indexed by ['Store','Date'], sorted ascending.
    df_ml : Optional[pd.DataFrame]
        Flat copy with 'Store' and 'Date' as columns (if build_flat_copy=True).
    """
    sales_csv = Path(sales_csv)
    store_csv = Path(store_csv)

    # --- Read raw
    sales = pd.read_csv(sales_csv, parse_dates=['Date'], low_memory=False)
    store  = pd.read_csv(store_csv)

    # --- Basic guards
    required_sales = {'Store', 'Date', 'Sales', 'Open', 'Promo', 'StateHoliday', 'SchoolHoliday'}
    missing = required_sales.difference(sales.columns)
    if missing:
        raise ValueError(f"SalesData missing required columns: {sorted(missing)}")

    if 'Promo2' not in store.columns:
        raise ValueError("StoreData missing required column 'Promo2'")

    # --- PROMO2: start date (ISO week/year -> Monday)
    store['promo2_start_date'] = pd.NaT
    m = (store['Promo2'].astype('Int64').eq(1)
         & store['Promo2SinceYear'].notna()
         & store['Promo2SinceWeek'].notna())

    if m.any():
        store.loc[m, 'promo2_start_date'] = store.loc[m, ['Promo2SinceYear','Promo2SinceWeek']].apply(
            lambda r: pd.Timestamp.fromisocalendar(int(r['Promo2SinceYear']),
                                                   int(r['Promo2SinceWeek']), 1),
            axis=1
        )

    # --- PROMO2: month flags promo2_m1..promo2_m12
    # Convert PromoInterval to set-of-months then to 12 booleans
    promo_sets = store['PromoInterval'].map(_interval_to_set)
    for mnum in range(1, 13):
        store[f'promo2_m{mnum}'] = promo_sets.apply(lambda S: mnum in S)

    # --- Merge minimal promo columns into sales
    promo_cols = ['Store','Promo2','promo2_start_date'] + [f'promo2_m{i}' for i in range(1,13)]
    df = sales.merge(store[promo_cols], on='Store', how='left')

    # Vectorized month membership
    month_cols = [f'promo2_m{i}' for i in range(1,13)]
    M = df[month_cols].to_numpy(dtype=bool)
    month_idx = df['Date'].dt.month.to_numpy() - 1  # 0..11
    row_idx = np.arange(len(df))
    in_interval_month = M[row_idx, month_idx]

    # promo2_active
    df['Promo2'] = pd.to_numeric(df['Promo2'], errors='coerce').fillna(0).astype('int8')
    df['promo2_active'] = (
        df['Promo2'].eq(1)
        & df['promo2_start_date'].notna()
        & (df['Date'] >= df['promo2_start_date'])
        & in_interval_month
    ).astype('int8')

    # --- COMPETITION features
    # competition_open_date from (Year, Month)
    store['CompetitionOpenSinceMonth'] = pd.to_numeric(store['CompetitionOpenSinceMonth'], errors='coerce').astype('Int64')
    store['CompetitionOpenSinceYear']  = pd.to_numeric(store['CompetitionOpenSinceYear'],  errors='coerce').astype('Int64')

    comp_date = pd.to_datetime(
        dict(
            year  = store['CompetitionOpenSinceYear'].fillna(0).astype(int).replace(0, np.nan),
            month = store['CompetitionOpenSinceMonth'].fillna(0).astype(int).replace(0, np.nan),
            day   = 1
        ),
        errors='coerce'
    )
    store['competition_open_date'] = comp_date

    df = df.merge(
        store[['Store','CompetitionDistance','competition_open_date']],
        on='Store', how='left'
    )

    # months since competition open
    df['months_since_comp_open'] = _months_diff(df['Date'], df['competition_open_date'])
    df['months_since_comp_open'] = (
        df['months_since_comp_open']
        .where(df['months_since_comp_open'].notna(), np.nan)
        .clip(lower=0)
        .fillna(0)
        .astype('int16')
    )

    # log1p distance
    df['CompetitionDistance'] = pd.to_numeric(df['CompetitionDistance'], errors='coerce')
    df['CompetitionDistance_log1p'] = np.log1p(df['CompetitionDistance']).astype('float32')

    # --- Tidy types
    for c in ('Open','Promo','SchoolHoliday'):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0).astype('int8')
    # keep StateHoliday as string (single categorical col for trees)
    df['StateHoliday'] = df['StateHoliday'].astype(str)

    # Optionally drop closed days for modeling
    if not keep_closed_days and 'Open' in df.columns:
        df = df[df['Open'] == 1].copy()

    # --- Sort & index for TS work
    df = df.sort_values(['Store','Date'])
    df_ts = df.set_index(['Store','Date'])

    df_ml = df_ts.reset_index() if build_flat_copy else None
    return df_ts, df_ml
