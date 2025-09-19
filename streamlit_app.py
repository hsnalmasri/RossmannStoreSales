from __future__ import annotations

# --- Ensure project root is importable (robust on all runners) ---
import sys
from pathlib import Path

_APP_ROOT = Path(__file__).resolve().parent
if str(_APP_ROOT) not in sys.path:
    sys.path.insert(0, str(_APP_ROOT))

# --- Standard imports ---
import os
from urllib.parse import urlparse
import pandas as pd
import streamlit as st

from src.config import (
    DATA_URL_SALES,
    DATA_URL_STORE,
    COL_DATE,
    COL_SALES,
    COL_STORE,
)
from src.data_prep import load_rossmann_data
from src.viz.ets_plots import make_backtest_fig, make_forecast_fig


# Page metadata (browser tab + sharing UIs)
st.set_page_config(page_title="Overall Sales — Exponential Smoothing", page_icon="🏪", layout="wide")

# Main heading
st.title("Overall Sales — Exponential Smoothing")
st.caption("Paste local file paths or use the default GitHub URLs. Tune ETS, backtest on monthly data, and forecast ahead.")

# -------------------------
# Sidebar: Data sources & ETS controls
# -------------------------
with st.sidebar:
    st.header("Data & Filters")

    # Safe config lookup: secrets -> env -> config default
    def _cfg(name: str, default_val: str) -> str:
        try:
            return st.secrets[name]  # type: ignore[index]
        except Exception:
            return os.environ.get(name, default_val)

    def _repair_http_url(s: str) -> str:
        """
        Repair common Windows paste mistakes in http(s) URLs:
        - Backslashes -> forward slashes.
        - Insert '://' after scheme if missing.
        """
        t = (s or "").strip()
        if not t:
            return t
        low = t.lower()
        if low.startswith(("http", "https")):
            if "\\" in t:
                t = t.replace("\\", "/")
            if "://" not in t:
                parts = t.split(":", 1)
                if len(parts) == 2:
                    t = parts[0] + "://" + parts[1].lstrip("/")
        return t

    def _normalize_source(inp: str, fallback: str) -> str:
        """
        Trim, auto-repair http(s), fallback if blank, then validate.
        Accepts local paths (C:/..., \\server\share\..., file:///...).
        """
        s = _repair_http_url((inp or "").strip())
        if not s:
            s = _repair_http_url((fallback or "").strip())
        if not s:
            raise ValueError("No path/URL provided.")
        parsed = urlparse(s)
        if parsed.scheme in ("http", "https") and not parsed.netloc:
            raise ValueError(f"Malformed URL (no host): {s!r}")
        return s

    sales_default = _cfg("DATA_URL_SALES", DATA_URL_SALES)
    store_default = _cfg("DATA_URL_STORE", DATA_URL_STORE)

    sales_in = st.text_input("Sales CSV (URL or path)", value=sales_default)
    store_in = st.text_input("Store CSV (URL or path)", value=store_default)
    st.caption("Tip: local examples → `C:/data/SalesData.csv`  or  `\\\\server\\share\\StoreData.csv`")

    st.divider()
    st.header("ETS Parameters")
    seasonal_periods = st.number_input("Seasonal Periods (months)", min_value=1, value=12, step=1)
    trend = st.selectbox("Trend", options=[None, "add", "mul"], index=1, format_func=lambda x: "None" if x is None else x)
    seasonal = st.selectbox("Seasonal", options=[None, "add", "mul"], index=1, format_func=lambda x: "None" if x is None else x)
    damped = st.checkbox("Damped Trend", value=False)

    st.caption("Leave α/β/γ blank to auto-optimize.")
    col_a, col_b, col_c = st.columns(3)
    alpha_in = col_a.text_input("α", value="")
    beta_in = col_b.text_input("β", value="")
    gamma_in = col_c.text_input("γ", value="")

    def _to_opt(x: str) -> float | None:
        x = x.strip()
        if not x:
            return None
        try:
            v = float(x)
            if not (0.0 <= v <= 1.0):
                st.warning("Smoothing should be in [0, 1]. Using auto.")
                return None
            return v
        except Exception:
            st.warning("Invalid number; using auto.")
            return None

    smoothing_level = _to_opt(alpha_in)
    smoothing_trend = _to_opt(beta_in)
    smoothing_seasonal = _to_opt(gamma_in)

    st.divider()
    st.header("Backtest & Forecast")
    test_h = st.number_input("Backtest Horizon (months)", min_value=1, value=6, step=1)
    fcst_h = st.number_input("Forecast Horizon (months)", min_value=1, value=6, step=1)

    st.divider()
    if st.button("🧹 Clear data cache"):
        st.cache_data.clear()
        st.success("Cache cleared.")
        st.rerun()

# -------------------------
# Cached data loader
# -------------------------
@st.cache_data(show_spinner=True)
def _load_data(sales_csv: str, store_csv: str):
    df_ts, _ = load_rossmann_data(
        sales_csv=sales_csv,
        store_csv=store_csv,
        keep_closed_days=False,
        build_flat_copy=False,
        # Imputation defaults
        force_fill_open_date_for_no_competitor=False,
        fill_missing_promo2_start=False,
    )
    return df_ts

# -------------------------
# Load + sanity checks
# -------------------------
try:
    sales_url = _normalize_source(sales_in, DATA_URL_SALES)
    store_url = _normalize_source(store_in, DATA_URL_STORE)
except Exception as e:
    st.error(f"Input error: {e}")
    st.stop()

with st.expander("🔎 Debug: resolved data sources"):
    st.write({"sales_csv": sales_url, "store_csv": store_url})

try:
    df_ts = _load_data(sales_url, store_url)
except Exception as e:
    st.error(f"Failed to load data: {e}")
    st.stop()

# -------------------------
# Store selection
# -------------------------
stores = df_ts.index.get_level_values(COL_STORE).unique().tolist()
store_id = st.selectbox("Store", options=stores, index=0)

try:
    s_daily = df_ts.xs(store_id, level=COL_STORE)[COL_SALES].copy()
except KeyError:
    st.error(f"Store {store_id} not found.")
    st.stop()

s_daily.index = pd.to_datetime(s_daily.index)
s_daily = s_daily.sort_index()

# -------------------------
# Backtest vs Forecast
# -------------------------
# Guard: need ≥ 2 cycles in training for backtest
monthly_len = s_daily.resample("MS").sum().shape[0]
train_len = monthly_len - int(test_h)
needed = 2 * max(1, int(seasonal_periods))
ok_for_test = (seasonal is None) or (train_len >= needed)

c1, c2 = st.columns(2)

with c1:
    st.subheader("Backtest (Train vs Test)")
    if not ok_for_test:
        st.info(
            f"Need ≥ 2 cycles in training. Train months = {train_len}; required ≥ {needed}. "
            f"Reduce backtest horizon or use a longer history."
        )
    else:
        try:
            fig_bt = make_backtest_fig(
                s_daily,
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
            st.plotly_chart(fig_bt, use_container_width=True)
        except Exception as e:
            st.error(f"Backtest failed: {e}")

with c2:
    st.subheader("Forward Forecast")
    # Guard for full-series fit when seasonal is used
    full_months = s_daily.resample("MS").sum().shape[0]
    if seasonal is not None and full_months < needed:
        st.info(f"Series must contain ≥ 2 cycles for seasonal ETS. Have {full_months}, need ≥ {needed}.")
    else:
        try:
            fig_fc = make_forecast_fig(
                s_daily,
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
        except Exception as e:
            st.error(f"Forecast failed: {e}")
