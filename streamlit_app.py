# streamlit_app.py (snippet)
import pandas as pd
import streamlit as st
from ets_stepwise import fit_ets_holdout

# Assume df_ml has columns ['Date', 'Sales']
df_ml_monthly = (
    df_ml.copy()
        .groupby('Date', as_index=False)[['Sales']].sum()
        .set_index('Date')
        .resample('MS').sum()
        .squeeze()                 # -> Series
        .rename('Sales')
)

st.sidebar.header("ETS Controls")
trend = st.sidebar.selectbox("Trend", [None, "add", "mul"], index=1)
seasonal = st.sidebar.selectbox("Seasonality", [None, "add", "mul"], index=2)
seasonal_periods = st.sidebar.number_input("Seasonal periods", min_value=1, value=12)
damped = st.sidebar.checkbox("Damped trend", value=False)
transform = st.sidebar.selectbox("Transform", ["raw", "log1p"], index=0)
n_forecast = st.sidebar.slider("Forecast horizon (months)", 1, 24, 6)

res = fit_ets_holdout(
    y=df_ml_monthly,
    n_forecast=n_forecast,
    trend=trend,
    seasonal=seasonal,
    seasonal_periods=seasonal_periods,
    damped_trend=damped,
    optimized=True,
    transform=None if transform == "raw" else "log1p",
)

# You can now plot with your preferred lib (matplotlib, plotly, altair) in Streamlit
st.subheader("Metrics (holdout)")
st.write(res["metrics"])

st.subheader("Preview")
y_train = res["y_train"]
y_test = res["y_test"]
fitted = res["fitted"]
forecast = res["forecast"]

df_plot = pd.DataFrame({
    "actual": pd.concat([y_train, y_test]),
    "fitted": fitted.reindex(y_train.index),
    "forecast": forecast.reindex(y_test.index) if len(y_test) else None,
})
st.line_chart(df_plot)
