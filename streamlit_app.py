# streamlit_app.py
from __future__ import annotations
import io
from pathlib import Path

import pandas as pd
import streamlit as st
import requests

# --- Your forecasting helpers ---
from monthly_forecast import (
    load_prepared as _load_prepared,   # path-based loader (uses data_prep if available)
    make_monthly_series,
    fit_arima,                         # SARIMAX wrapper
    make_fig_actual_fitted_forecast,
)

# --------------------------------------------------------------------------------------
# Page setup
# --------------------------------------------------------------------------------------
st.set_page_config(page_title="Rossmann Monthly Forecast", layout="wide")
st.title("Rossmann Monthly Forecast (SARIMAX)")

# --------------------------------------------------------------------------------------
# Data location: local if available, else download from GitHub Releases via Secrets
# --------------------------------------------------------------------------------------
SALES_LOCAL = Path("data/RAW/SalesData.csv")
STORE_LOCAL = Path("data/RAW/StoreData.csv")

# Set these in Streamlit Cloud → App → Settings → Secrets
SALES_URL = st.secrets.get("DATA_URL_SALES", "")
STORE_URL = st.secrets.get("DATA_URL_STORE", "")

@st.cache_data(show_spinner=True)
def _download_bytes(url: str) -> bytes:
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    return r.content

def resolve_sales_store_sources():
    """
    Returns (sales_src, store_src) as either:
      - str file paths (if local CSVs exist), or
      - BytesIO buffers (downloaded from Release URLs in Secrets)
    """
    if SALES_LOCAL.exists() and STORE_LOCAL.exists():
        return str(SALES_LOCAL), str(STORE_LOCAL)

    if SALES_URL and STORE_URL:
        sales_buf = io.BytesIO(_download_bytes(SALES_URL))
        store_buf = io.BytesIO(_download_bytes(STORE_URL))
        return sales_buf, store_buf

    st.error(
        "Data not found.\n\n"
        "Either place CSVs at `data/RAW/` locally **or** set `DATA_URL_SALES` and "
        "`DATA_URL_STORE` in **Settings → Secrets** to your GitHub Release asset URLs."
    )
    st.stop()

def load_prepared_flexible(sales_src, store_src) -> pd.DataFrame:
    """
    Adapter that calls your existing path-based loader for file paths,
    or reads BytesIO buffers into DataFrames and merges minimal columns.
    """
    if isinstance(sales_src, (str, Path)) and isinstance(store_src, (str, Path)):
        # Use your original loader pipeline (data_prep features if available)
        return _load_prepared(sales_src, store_src, keep_closed_days=True)

    # BytesIO case (downloaded). Minimal merge fallback:
    sales_df = pd.read_csv(sales_src, parse_dates=['Date'], low_memory=False)
    store_df = pd.read_csv(store_src, low_memory=False)
    df = sales_df.merge(store_df[['Store']], on='Store', how='left')
    return df

@st.cache_data(show_spinner=True)
def get_df() -> pd.DataFrame:
    sales_src, store_src = resolve_sales_store_sources()
    return load_prepared_flexible(sales_src, store_src)

# --------------------------------------------------------------------------------------
# Sidebar controls
# --------------------------------------------------------------------------------------
with st.sidebar:
    st.header("Controls")
    horizon = st.slider("Forecast horizon (months)", 1, 12, 3)
    seasonal = st.checkbox("Seasonal model (m=12)", value=True)
    m = 12  # monthly seasonality for SARIMAX

# --------------------------------------------------------------------------------------
# Load data and build the series
# --------------------------------------------------------------------------------------
try:
    df = get_df()
except Exception as e:
    st.error(f"Failed to load data: {e}")
    st.stop()

# Store selector
stores = sorted(pd.Series(df["Store"]).dropna().unique().tolist())
store_options = ["All Stores"] + [int(s) for s in stores]
store_choice = st.selectbox("Aggregate by:", store_options, index=0)
store_val = None if store_choice == "All Stores" else int(store_choice)

# --------------------------------------------------------------------------------------
# Fit SARIMAX and plot
# --------------------------------------------------------------------------------------
with st.spinner("Fitting SARIMAX…"):
    y = make_monthly_series(df, store=store_val, agg='sum').asfreq("M").fillna(0)
    model = fit_arima(y, seasonal=seasonal, m=m)
    fig = make_fig_actual_fitted_forecast(y, model, horizon=horizon)

# Layout: chart + quick stats
col1, col2 = st.columns([3, 1])
with col1:
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Summary")
    st.write(f"**Observations:** {len(y)}")
    st.write(f"**Last month:** {y.index.max().strftime('%Y-%m') if len(y) else '—'}")
    st.write(f"**Last value:** {y.iloc[-1]:,.0f}" if len(y) else "**Last value:** —")
    st.caption("Model: SARIMAX (1,1,1) × (1,1,1,12) when seasonal is on")

st.divider()
st.subheader("Data preview (last 24 months)")
if len(y):
    st.dataframe(y.tail(24).to_frame("Sales"))
else:
    st.info("No data to display. Check your CSVs/links.")
