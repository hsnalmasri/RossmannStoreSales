# src/rossmann/data_prep.py
from __future__ import annotations
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional

from src.rossmann.data_load import load_sales, load_store_imputed

__all__ = ["load_rossmann_data"]

# ---------- helpers ----------
def _months_diff(d1: pd.Series, d0: pd.Series) -> pd.Series:
    y1 = d1.dt.year; m1 = d1.dt.month
    y0 = d0.dt.year; m0 = d0.dt.month
    return (y1 - y0) * 12 + (m1 - m0)

def _safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
    out = a.astype("float64")
    denom = b.astype("float64")
    out = np.where(denom > 0, out / denom, np.nan)
    return pd.Series(out, index=a.index)

def _abc_labels(total_sales: pd.Series, cutA: float, cutB: float) -> pd.Series:
    # sort desc, cumulative share, map to A/B/C
    s = total_sales.sort_values(ascending=False)
    cumshare = (s.cumsum() / s.sum()).reindex(total_sales.index)  # align back
    lab = pd.Series("C", index=total_sales.index, dtype="object")
    lab.loc[cumshare <= cutB] = "B"
    lab.loc[cumshare <= cutA] = "A"
    return lab

def _xyz_labels(weekly_sales: pd.DataFrame, thresholds=(0.50, 1.00),
                zero_week_cap=0.25, min_weeks=10) -> pd.Series:
    """
    weekly_sales: index Store, columns ['mean','std','weeks_n','zero_week_share']
    """
    cv = weekly_sales['std'] / weekly_sales['mean'].replace(0, np.nan)
    cv = cv.fillna(np.inf)  # mean<=0 -> Z
    lab = pd.Series("Z", index=weekly_sales.index, dtype="object")

    # base on CV
    lab.loc[cv <= thresholds[1]] = "Y"
    lab.loc[cv <= thresholds[0]] = "X"

    # overrides
    lab.loc[(weekly_sales['zero_week_share'] > zero_week_cap) | (weekly_sales['weeks_n'] < min_weeks)] = "Z"
    return lab

# ---------- main ----------
def load_rossmann_data(
    sales_csv: str | Path,
    store_csv: str | Path,
    *,
    keep_closed_days: bool = True,
    build_flat_copy: bool = True,
    # Competition options
    force_fill_open_date_for_no_competitor: bool = False,
    no_competitor_fill_date: str | pd.Timestamp | None = None,
    # Promo2 options
    fill_missing_promo2_start: bool = False,
    promo2_start_fill_date: str | pd.Timestamp | None = None,
    # ---- Segmentation knobs (defaults are sensible and editable) ----
    lookback_months: int = 12,                 # LTM window for value/variability
    abc_cutoffs: tuple[float, float] = (0.80, 0.95),
    xyz_cv_thresholds: tuple[float, float] = (0.50, 1.00),
    new_window_months: int = 3,                # first sale within N months -> NEW
    inactive_window_months: int = 3,           # no sale in last M months -> DEAD
    xyz_min_weeks: int = 10,                   # min weeks to trust variability
    xyz_zero_week_cap: float = 0.25,           # >25% zero weeks -> Z
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:

    sales_csv = Path(sales_csv)
    store_csv = Path(store_csv)

    # --- read raw + store (imputed upstream)
    sales = load_sales(sales_csv)
    if sales['Date'].dtype != 'datetime64[ns]':
        sales['Date'] = pd.to_datetime(sales['Date'])
    max_date = pd.to_datetime(sales['Date']).max()
    earliest_sales_date = pd.to_datetime(sales["Date"]).min()

    store  = load_store_imputed(
        store_csv,
        earliest_sales_date=earliest_sales_date,
        # competition
        force_fill_open_date_for_no_competitor=force_fill_open_date_for_no_competitor,
        no_competitor_fill_date=no_competitor_fill_date,
        # promo2
        fill_missing_promo2_start=fill_missing_promo2_start,
        promo2_start_fill_date=promo2_start_fill_date,
    )

    # --- guards
    required_sales = {'Store','Date','Sales','Open','Promo','StateHoliday','SchoolHoliday'}
    missing = required_sales.difference(sales.columns)
    if missing:
        raise ValueError(f"SalesData missing required columns: {sorted(missing)}")
    if 'Promo2' not in store.columns:
        raise ValueError("StoreData missing required column 'Promo2'")

    # --- merge promo/competition into sales
    promo_cols = ['Store','Promo2','promo2_start_date','promo2_enrolled','promo2_missing_start','promo2_valid_start'] \
                 + [f'promo2_m{i}' for i in range(1,13)]
    comp_cols  = ['Store','CompetitionDistance','comp_distance_missing','comp_open_missing',
                  'competition_open_date','has_competitor']
    merge_cols = list(dict.fromkeys(promo_cols + comp_cols))
    right = store[merge_cols].copy()
    right = right.loc[:, ~right.columns.duplicated(keep='first')]

    df = sales.merge(right, on='Store', how='left')

    # promo2_active (month-in-interval + date >= start)
    month_cols = [f'promo2_m{i}' for i in range(1,13)]
    if set(month_cols).issubset(df.columns):
        M = df[month_cols].to_numpy(dtype=bool)
        month_idx = df['Date'].dt.month.to_numpy() - 1
        row_idx = np.arange(len(df))
        in_interval_month = M[row_idx, month_idx]
    else:
        in_interval_month = np.zeros(len(df), dtype=bool)

    df['Promo2'] = pd.to_numeric(df['Promo2'], errors='coerce').fillna(0).astype('int8')
    df['promo2_active'] = (
        df['Promo2'].eq(1)
        & df['promo2_start_date'].notna()
        & (df['Date'] >= df['promo2_start_date'])
        & in_interval_month
    ).astype('int8')

    # competition features
    df['months_since_comp_open'] = _months_diff(df['Date'], df['competition_open_date'])
    df['months_since_comp_open'] = (
        df['months_since_comp_open'].where(df['months_since_comp_open'].notna(), np.nan)
        .clip(lower=0).fillna(0).astype('int16')
    )
    df['CompetitionDistance'] = pd.to_numeric(df['CompetitionDistance'], errors='coerce').fillna(0)
    df['CompetitionDistance_log1p'] = np.log1p(df['CompetitionDistance']).astype('float32')

    # tidy types
    for c in ('Open','Promo','SchoolHoliday'):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0).astype('int8')
    if 'StateHoliday' in df.columns:
        df['StateHoliday'] = df['StateHoliday'].astype(str)

    # optionally drop closed days
    if not keep_closed_days and 'Open' in df.columns:
        df = df[df['Open'] == 1].copy()

    # ---------- SEGMENTATION: per-store summary over LTM ----------
    ltm_start = (max_date - pd.DateOffset(months=lookback_months)).normalize()
    df_ltm = df[df['Date'] >= ltm_start].copy()

    # core LTM aggregates
    grp = df_ltm.groupby('Store', as_index=True)
    total_sales_ltm   = grp['Sales'].sum().rename('total_sales_ltm')
    customers_ltm     = grp['Customers'].sum(min_count=1).rename('customers_ltm') if 'Customers' in df.columns else pd.Series(index=total_sales_ltm.index, dtype='float64')
    open_days_ltm     = grp.apply(lambda g: (g['Open'] == 1).sum()).rename('open_days_ltm')
    sales_per_open    = _safe_div(total_sales_ltm, open_days_ltm).rename('sales_per_open_day')
    avg_ticket        = _safe_div(total_sales_ltm, customers_ltm).rename('avg_ticket')

    # promo metrics (computed on open days)
    def _avg_by_flag(g: pd.DataFrame, flag_col: str, flag_value: int) -> float:
        m = (g['Open'] == 1) & (g[flag_col] == flag_value)
        if m.any():
            return g.loc[m, 'Sales'].mean()
        return np.nan

    promo_mean = grp.apply(lambda g: _avg_by_flag(g, 'Promo', 1)).rename('promo_mean')
    nonpromo_mean = grp.apply(lambda g: _avg_by_flag(g, 'Promo', 0)).rename('nonpromo_mean')
    promo_share = grp.apply(lambda g: ((g['Open'] == 1) & (g['Promo'] == 1)).mean()).rename('promo_share')
    promo_uplift_pct = ((promo_mean - nonpromo_mean) / nonpromo_mean.replace(0, np.nan)).rename('promo_uplift_pct')

    # weekly variability for XYZ
    weekly = (df_ltm.assign(week=df_ltm['Date'].dt.to_period('W-MON').dt.start_time)
                    .groupby(['Store','week'])['Sales'].sum()
                    .reset_index())
    wstats = (weekly.groupby('Store')['Sales']
                    .agg(mean='mean', std='std', weeks_n='count'))
    zero_week_share = weekly.assign(z=(weekly['Sales'] <= 1e-9)).groupby('Store')['z'].mean()
    wstats['zero_week_share'] = zero_week_share

    # status metrics (use full history)
    g_all = df.groupby('Store')
    first_sale = g_all.apply(lambda g: g.loc[g['Sales'] > 0, 'Date'].min()).rename('first_sale_date')
    last_sale  = g_all.apply(lambda g: g.loc[g['Sales'] > 0, 'Date'].max()).rename('last_sale_date')
    days_since_last = (max_date - last_sale).dt.days.rename('days_since_last_sale')

    # growth: last 3m vs prior 3m within LTM
    def _sum_range(g: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> float:
        m = (g['Date'] >= start) & (g['Date'] <= end)
        return g.loc[m, 'Sales'].sum()

    last_3m_start = max_date - pd.DateOffset(months=3)
    prev_3m_start = max_date - pd.DateOffset(months=6)
    sales_last3 = g_all.apply(lambda g: _sum_range(g, last_3m_start, max_date)).rename('sales_last3m')
    sales_prev3 = g_all.apply(lambda g: _sum_range(g, prev_3m_start, last_3m_start)).rename('sales_prev3m')
    growth_6m = ((sales_last3 - sales_prev3) / sales_prev3.replace(0, np.nan)).rename('growth_6m')

    # assemble per-store summary
    summary = pd.concat([
        total_sales_ltm, customers_ltm, open_days_ltm, sales_per_open, avg_ticket,
        promo_share, promo_uplift_pct, wstats, first_sale, last_sale, days_since_last,
        sales_last3, sales_prev3, growth_6m
    ], axis=1)

    # value tiers (ABC)
    summary['share_of_sales'] = total_sales_ltm / total_sales_ltm.sum()
    summary['store_abc'] = _abc_labels(total_sales_ltm, cutA=abc_cutoffs[0], cutB=abc_cutoffs[1])

    # forecastability (XYZ)
    summary['store_xyz'] = _xyz_labels(
        weekly_sales=wstats,
        thresholds=xyz_cv_thresholds,
        zero_week_cap=xyz_zero_week_cap,
        min_weeks=xyz_min_weeks
    )

    # status: NEW / RUNNING / DEAD
    new_thresh  = max_date - pd.DateOffset(months=new_window_months)
    dead_thresh = max_date - pd.DateOffset(months=inactive_window_months)

    status = pd.Series("RUNNING", index=summary.index, dtype="object")
    status.loc[first_sale >= new_thresh] = "NEW"
    status.loc[last_sale < dead_thresh]  = "DEAD"
    summary['store_status'] = status

    # combined label
    summary['store_segment'] = summary['store_abc'] + summary['store_xyz']

    # percentiles / rank helpers
    summary['sales_rank_desc'] = summary['total_sales_ltm'].rank(ascending=False, method='dense').astype('int32')
    summary['sales_decile'] = pd.qcut(summary['total_sales_ltm'].rank(method='first'), 10, labels=False) + 1
    summary['top_5pct'] = (summary['sales_rank_desc'] <= max(1, int(0.05 * len(summary)))).astype('int8')

    # ---------- broadcast back to daily table ----------
    df = df.merge(summary.reset_index(), on='Store', how='left', suffixes=('',''))

    # sort & index for TS work
    df = df.sort_values(['Store','Date'])
    df_ts = df.set_index(['Store','Date'])
    df_ml = df_ts.reset_index() if build_flat_copy else None
    return df_ts, df_ml
