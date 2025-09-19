from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st

from src.config import COL_DATE, COL_SALES, COL_STORE
from src.data_prep import load_rossmann_data
from src.viz.ets_plots import make_backtest_fig, make_forecast_fig


@st.cache_data(show_spinner=False)
def _load_data(sales_url: str, store_url: str):
    df_ts, _ = load_rossmann_data(
        sales_csv=sales_url,
        store_csv=store_url,
        keep_closed_days=False,
        build_flat_copy=False,
        # sensible defaults for imputation
        force_fill_open_date_for_no_competitor=False,
        fill_missing_promo2_start=False,
    )
    return df_ts


def _get_store_series(df_ts: pd.DataFrame, store_id: int) -> pd.Series:
    df_store = df_ts.xs(store_id, level=COL_STORE)
    s = df_store[COL_SALES].copy()
    s.index = pd.to_datetime(s.index)
    s = s.sort_index()
    s.index.freq = s.index.inferred_freq  # may be None; resampler will handle
    return s


st.title("📈 Exponential Smoothing (ETS)")

with st.sidebar:
    st.header("Data & Filters")
    sales_url = st.text_input("Sales CSV URL", value=st.secrets.get("DATA_URL_SALES", "https://github.com/hsnalmasri/RossmannStoreSales/releases/download/v0.1-data/SalesData.csv"))
    store_url = st.text_input("Store CSV URL", value=st.secrets.get("DATA_URL_STORE", "https://github.com/hsnalmasri/RossmannStoreSales/releases/download/v0.1-data/StoreData.csv"))

    df_ts = _load_data(sales_url, store_url)

    # Store picker
    store_ids = sorted({idx for idx, _ in df_ts.index.groupby(level=COL_STORE)})
    store = st.selectbox("Store", store_ids, index=0)

    st.header("ETS Parameters")
    seasonal_periods = st.number_input("Seasonal Periods (months)", min_value=1, value=12, step=1)
    trend = st.selectbox("Trend", options=[None, "add", "mul"], index=1, format_func=lambda x: "None" if x is None else x)
    seasonal = st.selectbox("Seasonal", options=[None, "add", "mul"], index=1, format_func=lambda x: "None" if x is None else x)
    damped = st.checkbox("Damped Trend", value=False)

    st.caption("Leave smoothing values empty to auto-optimize.")
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        alpha = st.text_input("α (smoothing_level)", value="")
    with col_b:
        beta = st.text_input("β (smoothing_trend)", value="")
    with col_c:
        gamma = st.text_input("γ (smoothing_seasonal)", value="")

    def _to_opt(x: str) -> Optional[float]:
        x = x.strip()
        if not x:
            return None
        try:
            v = float(x)
            if not (0.0 <= v <= 1.0):
                st.warning("Smoothing values should be in [0, 1]. Using auto for invalid input.")
                return None
            return v
        except Exception:
            st.warning("Invalid smoothing value; using auto.")
            return None

    smoothing_level = _to_opt(alpha)
    smoothing_trend = _to_opt(beta)
    smoothing_seasonal = _to_opt(gamma)

    st.header("Testing & Forecast")
    test_h = st.number_input("Backtest Horizon (months)", min_value=1, value=6, step=1)
    fcst_h = st.number_input("Forecast Horizon (months)", min_value=1, value=6, step=1)

# Extract & plot
series_daily = _get_store_series(df_ts, store)

# Backtest guard: allow only when training has ≥ 2 seasonal cycles
monthly_len = series_daily.resample("MS").sum().shape[0]
train_len = monthly_len - int(test_h)
ok_for_test = train_len >= 2 * max(1, int(seasonal_periods))

col1, col2 = st.columns(2)

with col1:
    st.subheader("Backtest (Train vs Test)")
    if not ok_for_test:
        st.info(
            f"Need ≥ 2 seasonal cycles in training. Current training length = {train_len} months; "
            f"required ≥ {2 * max(1, int(seasonal_periods))} months. Increase history or reduce test horizon."
        )
    else:
        fig_test = make_backtest_fig(
            series_daily,
            seasonal_periods=int(seasonal_periods),
            test_horizon=int(test_h),
            trend=trend,
            seasonal=seasonal,
            damped_trend=bool(damped),
            smoothing_level=smoothing_level,
            smoothing_trend=smoothing_trend,
            smoothing_seasonal=smoothing_seasonal,
            remove_bias=False,
        )
        st.plotly_chart(fig_test, use_container_width=True)

with col2:
    st.subheader("Forward Forecast")
    try:
        fig_fc = make_forecast_fig(
            series_daily,
            seasonal_periods=int(seasonal_periods),
            forecast_horizon=int(fcst_h),
            trend=trend,
            seasonal=seasonal,
            damped_trend=bool(damped),
            smoothing_level=smoothing_level,
            smoothing_trend=smoothing_trend,
            smoothing_seasonal=smoothing_seasonal,
            remove_bias=False,
        )
        st.plotly_chart(fig_fc, use_container_width=True)
    except ValueError as e:
        st.info(str(e))
