# streamlit_app.py
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Prefer the installed package (pyproject). If not installed locally,
# add src/ to sys.path so we can import your existing modules without editing them.
try:
    from rossmann.data_load import load_csv  # <-- change here if your actual function name differs
    from rossmann.data_prep import to_monthly_series  # <-- change here if your actual function name differs
except Exception:
    sys.path.append(str(Path(__file__).resolve().parent / "src"))
    from rossmann.data_load import load_csv  # keep your original implementation
    from rossmann.data_prep import to_monthly_series  # keep your original implementation

from ets_stepwise import fit_and_backtest_ets  # data-only core (alias exists if you used the old name)


st.set_page_config(page_title="Monthly ETS Forecast", layout="wide")
st.title("Monthly ETS Forecast")
st.caption("Upload your CSV. We’ll use your existing cleaning pipeline and run ETS (no ARIMA).")


with st.sidebar:
    st.header("Inputs")
    file = st.file_uploader("CSV file", type=["csv"])
    date_col = st.text_input("Date column", value="Date")
    value_col = st.text_input("Value/target column", value="Sales")
    n_test = st.number_input("Holdout months (Ntest)", min_value=3, max_value=36, value=6)

    st.subheader("ETS Configuration")
    trend_opt = st.selectbox("Trend", options=["add", "mul", "none"], index=0)
    seasonal_opt = st.selectbox("Seasonal", options=["add", "mul", "none"], index=0)
    sp = st.number_input("Seasonal periods", min_value=0, max_value=24, value=12)
    damped = st.checkbox("Damped trend", value=True)

if not file:
    st.info("Upload a CSV to begin.")
    st.stop()

# Load & prepare using YOUR existing cleaners (unchanged):
raw_df = load_csv(file, date_col=date_col, value_col=value_col)
y = to_monthly_series(raw_df, date_col=date_col, value_col=value_col)

# Build ETS config from sidebar selections
cfg = {
    "trend": None if trend_opt == "none" else trend_opt,
    "seasonal": None if seasonal_opt == "none" else seasonal_opt,
    "seasonal_periods": int(sp) if sp > 0 else None,
    "damped_trend": bool(damped),
}

# Fit and backtest
res = fit_and_backtest_ets(y, n_test=n_test, cfg=cfg)

# --- UI ---
col1, col2 = st.columns([2, 1], gap="large")

with col1:
    st.subheader("Train / Holdout / Forecast")
    df_plot = pd.concat(
        [
            res["y_train"].rename("y_train"),
            res["fitted"].rename("fitted"),
            res["y_test"].rename("y_test"),
            res["forecast"].rename("forecast"),
        ],
        axis=1,
    )
    st.line_chart(df_plot)

with col2:
    st.subheader("Holdout Metrics")
    st.json(res["metrics"])

    st.subheader("Parameters Used")
    st.json(res["params_used"])

st.divider()
st.subheader("Data Preview")
st.dataframe(raw_df.head(20), use_container_width=True)
