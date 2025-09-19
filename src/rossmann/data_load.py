from __future__ import annotations
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Union

__all__ = [
    "load_sales",
    "load_store_raw",
    "load_store_imputed",
    "impute_store_competition",
    "impute_store_promo2",
]

# ---------- IO ----------

def _as_path(p: str | Path) -> Path:
    return Path(p)

def load_sales(sales_csv: str | Path) -> pd.DataFrame:
    """Read sales CSV with parsed dates."""
    p = _as_path(sales_csv)
    return pd.read_csv(p, parse_dates=["Date"], low_memory=False)

def load_store_raw(store_csv: str | Path) -> pd.DataFrame:
    """Read store CSV as-is (no imputations)."""
    p = _as_path(store_csv)
    return pd.read_csv(p)

# ---------- Helpers ----------

_MONTH_MAP = {'Jan':1,'Feb':2,'Mar':3,'Apr':4,'May':5,'Jun':6,
              'Jul':7,'Aug':8,'Sep':9,'Oct':10,'Nov':11,'Dec':12}

def _interval_to_set(s: object) -> set[int]:
    """Convert 'Feb,May,Aug,Nov' or 'Mar,Jun,Sept,Dec' -> {2,5,8,11}."""
    if pd.isna(s) or not str(s).strip():
        return set()
    out: set[int] = set()
    for t in str(s).split(','):
        t = t.strip()
        if not t:
            continue
        key = t.title()[:3]  # normalize: 'Sept' -> 'Sep'
        if key in _MONTH_MAP:
            out.add(_MONTH_MAP[key])
    return out

def _to_monday(ts: pd.Timestamp) -> pd.Timestamp:
    """Return the Monday of the ISO week containing ts."""
    ts = pd.Timestamp(ts)
    return ts - pd.Timedelta(days=int(ts.weekday()))  # Monday=0

# ---------- Imputation: Competition ----------

def impute_store_competition(
    store: pd.DataFrame,
    *,
    earliest_sales_date: Optional[pd.Timestamp],
    force_fill_open_date_for_no_competitor: bool = False,
    no_competitor_fill_date: Optional[Union[str, pd.Timestamp]] = None,
    big_distance_multiplier: float = 2.0,
    big_distance_fallback: float = 1e6,
) -> pd.DataFrame:
    """
    Impute Competition* fields & add helper flags.

    Always:
      - comp_distance_missing, comp_open_missing
      - If CompetitionDistance missing -> fill with a very large value (means "no competitor")
      - If open date missing but competitor exists -> impute with earliest_sales_date
      - competition_open_date built safely (NaT allowed)
      - has_competitor flag

    Options:
      - force_fill_open_date_for_no_competitor: fill date for 'no competitor' rows too
      - no_competitor_fill_date: explicit date for force-fill; else earliest_sales_date
    """
    st = store.copy()

    # Ensure columns exist
    for col in ("CompetitionDistance", "CompetitionOpenSinceMonth", "CompetitionOpenSinceYear"):
        if col not in st.columns:
            st[col] = pd.NA

    # Coerce types
    st["CompetitionDistance"] = pd.to_numeric(st["CompetitionDistance"], errors="coerce")
    st["CompetitionOpenSinceMonth"] = pd.to_numeric(st["CompetitionOpenSinceMonth"], errors="coerce").astype("Int64")
    st["CompetitionOpenSinceYear"]  = pd.to_numeric(st["CompetitionOpenSinceYear"],  errors="coerce").astype("Int64")

    # Missingness flags (pre-fill)
    st["comp_distance_missing"] = st["CompetitionDistance"].isna().astype("int8")
    st["comp_open_missing"] = (st["CompetitionOpenSinceMonth"].isna() | st["CompetitionOpenSinceYear"].isna()).astype("int8")

    # Big distance sentinel for "no competitor"
    if st["CompetitionDistance"].notna().any():
        mx = float(st["CompetitionDistance"].max())
        big_dist = mx * big_distance_multiplier if np.isfinite(mx) and mx > 0 else big_distance_fallback
    else:
        big_dist = big_distance_fallback
    st["CompetitionDistance"] = st["CompetitionDistance"].fillna(big_dist)

    # Anchor timestamp
    if earliest_sales_date is None or pd.isna(earliest_sales_date):
        anchor_year, anchor_month = 2000, 1
        anchor_ts = pd.Timestamp(year=anchor_year, month=anchor_month, day=1)
    else:
        anchor_year, anchor_month = int(earliest_sales_date.year), int(earliest_sales_date.month)
        anchor_ts = pd.Timestamp(year=anchor_year, month=anchor_month, day=1)

    # If open date missing but competitor exists -> impute with earliest sales date
    needs_open_impute = st["comp_open_missing"].eq(1) & st["CompetitionDistance"].ne(big_dist)
    st.loc[needs_open_impute, "CompetitionOpenSinceYear"]  = anchor_year
    st.loc[needs_open_impute, "CompetitionOpenSinceMonth"] = anchor_month

    # Safe date assembly
    y = st["CompetitionOpenSinceYear"]
    m = st["CompetitionOpenSinceMonth"]
    mask = y.notna() & m.notna()

    comp_date = pd.Series(pd.NaT, index=st.index, dtype="datetime64[ns]")
    if mask.any():
        comp_date.loc[mask] = pd.to_datetime(
            y.loc[mask].astype("int64").astype(str) + "-" +
            m.loc[mask].astype("int64").astype(str).str.zfill(2) + "-01",
            errors="coerce"
        )
    st["competition_open_date"] = comp_date

    # has_competitor flag
    st["has_competitor"] = (st["CompetitionDistance"] != big_dist).astype("int8")

    # Optionally force-fill date for no-competitor rows
    if force_fill_open_date_for_no_competitor:
        fill_ts = pd.Timestamp(no_competitor_fill_date) if no_competitor_fill_date is not None else anchor_ts
        no_comp_mask = st["has_competitor"].eq(0) & st["competition_open_date"].isna()
        st.loc[no_comp_mask, "competition_open_date"] = fill_ts

    return st

# ---------- Imputation: Promo2 ----------

def impute_store_promo2(
    store: pd.DataFrame,
    *,
    earliest_sales_date: Optional[pd.Timestamp],
    fill_missing_promo2_start: bool = False,
    promo2_start_fill_date: Optional[Union[str, pd.Timestamp]] = None,
) -> pd.DataFrame:
    """
    Normalize and impute Promo2 fields + build month flags.

    Always:
      - promo2_enrolled: 1 if Promo2==1 else 0
      - promo2_missing_start: 1 if enrolled but missing week/year
      - promo2_valid_start: 1 if promo2_start_date is available
      - promo2_start_date from ISO (year, week, Monday) when valid
      - promo2_m1..promo2_m12 month flags from PromoInterval (normalized)

    Options:
      - fill_missing_promo2_start: if True, impute start date for enrolled rows missing week/year
      - promo2_start_fill_date: date to use when imputing; else use earliest_sales_date (Monday)
    """
    st = store.copy()

    # Ensure columns exist
    for col in ("Promo2", "Promo2SinceWeek", "Promo2SinceYear", "PromoInterval"):
        if col not in st.columns:
            st[col] = pd.NA

    # Coerce types
    st["Promo2"] = pd.to_numeric(st["Promo2"], errors="coerce").fillna(0).astype("int8")
    st["Promo2SinceWeek"] = pd.to_numeric(st["Promo2SinceWeek"], errors="coerce").astype("Int64")
    st["Promo2SinceYear"] = pd.to_numeric(st["Promo2SinceYear"], errors="coerce").astype("Int64")

    # Clamp ISO week to 1..53 (invalid -> NA)
    wk = st["Promo2SinceWeek"]
    st.loc[~wk.between(1, 53, inclusive="both"), "Promo2SinceWeek"] = pd.NA

    # Enrolment & missing start flags
    st["promo2_enrolled"] = st["Promo2"].eq(1).astype("int8")
    st["promo2_missing_start"] = (
        st["promo2_enrolled"].eq(1) &
        (st["Promo2SinceWeek"].isna() | st["Promo2SinceYear"].isna())
    ).astype("int8")

    # Build promo2_start_date safely
    start = pd.Series(pd.NaT, index=st.index, dtype="datetime64[ns]")
    mask_valid = (
        st["promo2_enrolled"].eq(1)
        & st["Promo2SinceWeek"].notna()
        & st["Promo2SinceYear"].notna()
    )
    if mask_valid.any():
        # vectorized-safe apply (small table: one row per store)
        start.loc[mask_valid] = st.loc[mask_valid, ["Promo2SinceYear","Promo2SinceWeek"]].apply(
            lambda r: pd.Timestamp.fromisocalendar(int(r["Promo2SinceYear"]), int(r["Promo2SinceWeek"]), 1),
            axis=1
        )

    # Optional imputation for missing start (enrolled stores)
    if fill_missing_promo2_start:
        if promo2_start_fill_date is not None:
            fill_ts = pd.Timestamp(promo2_start_fill_date)
        else:
            # default to Monday of earliest sales week
            if earliest_sales_date is None or pd.isna(earliest_sales_date):
                fill_ts = pd.Timestamp("2000-01-03")  # a Monday
            else:
                fill_ts = _to_monday(pd.Timestamp(earliest_sales_date))
        need_fill = st["promo2_missing_start"].eq(1) & start.isna()
        start.loc[need_fill] = fill_ts

    st["promo2_start_date"] = start
    st["promo2_valid_start"] = st["promo2_start_date"].notna().astype("int8")

    # Month flags from PromoInterval
    promo_sets = st["PromoInterval"].map(_interval_to_set)
    for m in range(1, 13):
        st[f"promo2_m{m}"] = promo_sets.apply(lambda S: m in S).astype("int8")

    return st

# ---------- High-level loader ----------

def load_store_imputed(
    store_csv: str | Path,
    *,
    earliest_sales_date: Optional[pd.Timestamp],
    # Competition options
    force_fill_open_date_for_no_competitor: bool = False,
    no_competitor_fill_date: Optional[Union[str, pd.Timestamp]] = None,
    big_distance_multiplier: float = 2.0,
    big_distance_fallback: float = 1e6,
    # Promo2 options
    fill_missing_promo2_start: bool = False,
    promo2_start_fill_date: Optional[Union[str, pd.Timestamp]] = None,
) -> pd.DataFrame:
    """
    Read store CSV and return an imputed/enriched store table with flags for
    competition and Promo2, plus promo2 month flags.
    """
    st = load_store_raw(store_csv)

    st = impute_store_competition(
        st,
        earliest_sales_date=earliest_sales_date,
        force_fill_open_date_for_no_competitor=force_fill_open_date_for_no_competitor,
        no_competitor_fill_date=no_competitor_fill_date,
        big_distance_multiplier=big_distance_multiplier,
        big_distance_fallback=big_distance_fallback,
    )

    st = impute_store_promo2(
        st,
        earliest_sales_date=earliest_sales_date,
        fill_missing_promo2_start=fill_missing_promo2_start,
        promo2_start_fill_date=promo2_start_fill_date,
    )

    return st
