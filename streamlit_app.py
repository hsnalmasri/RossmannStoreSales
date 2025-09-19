# streamlit_app.py
from __future__ import annotations
import streamlit as st
import pandas as pd
from pathlib import Path

# Local modules
from monthly_forecast import (
    load_prepared,
    make_monthly_series,
    fit_arima,
    make_fig_actual_fitted_forecast,
)

# Try both import styles for ETS helper (flat file or models package)
try:
    from ets_stepwise import fit_and_backtest_ets
except Exception:
    try:
        from models.ets_stepwise import fit_and_backtest_ets  # type: ignore
    except Exception as _e:
        fit_and_backtest_ets = None  # handled later

st.set_page_config(page_title='Rossmann Monthly Forecast', layout='wide')

st.title('Rossmann Monthly Forecasts')

with st.sidebar:
    st.header('Data Inputs')
    sales_path = st.text_input('SalesData.csv path', 'data/RAW/SalesData.csv')
    store_path = st.text_input('StoreData.csv path', 'data/RAW/StoreData.csv')

    st.header('Series Build')
    agg_choice = st.radio('Monthly aggregation', ['sum', 'mean'], index=0, horizontal=True)

    st.header('Model')
    model_choice = st.radio('Choose model', ['ARIMA', 'ETS (Exponential Smoothing)'], index=1)

@st.cache_data(show_spinner=True)
def get_df(sales_path: str, store_path: str) -> pd.DataFrame:
    return load_prepared(sales_path, store_path, keep_closed_days=True)

# ---- Load data ----
try:
    df = get_df(sales_path, store_path)
except Exception as e:
    st.error(f"Failed to load data: {e}")
    st.stop()

# UI: store selection (including All Stores)
stores = sorted(df['Store'].dropna().unique().tolist())
store_options = ['All Stores'] + [int(s) for s in stores]
store_choice = st.selectbox('Store to aggregate', store_options)
store_val = None if store_choice == 'All Stores' else int(store_choice)

# Build monthly series
y = make_monthly_series(df, store=store_val, agg=agg_choice).asfreq('M').fillna(0)

st.caption(f"Observations: **{len(y)}**, last month: **{y.index.max().strftime('%Y-%m')}**, last value: **{y.iloc[-1]:,.0f}**")

# =============================
# ARIMA block (simple baseline)
# =============================
if model_choice == 'ARIMA':
    with st.spinner('Fitting ARIMA baseline...'):
        horizon = st.slider('Forecast horizon (months)', 1, 24, 6)
        model = fit_arima(y, seasonal=True, m=12)
        fig = make_fig_actual_fitted_forecast(y, model, horizon=horizon)

    st.plotly_chart(fig, use_container_width=True)
    st.divider()
    st.subheader('Data preview')
    st.dataframe(y.to_frame('Sales').tail(24))
    st.caption('Model: SARIMAX(1,1,1) × (1,1,1,12) baseline')

# ==============================================
# ETS block (step-wise diff-log pipeline + dual)
# ==============================================
else:
    if fit_and_backtest_ets is None:
        st.error('ets_stepwise.py is not found. Place it next to monthly_forecast.py or under models/.')
        st.stop()

    with st.sidebar:
        st.subheader('Backtest & Forecast steps')
        k_back = st.slider('Backtest last k steps', 1, 24, 3)
        k_fut = st.slider('Future forecast steps', 1, 24, 6)

        st.subheader('ETS Structure')
        trend = st.selectbox('Trend', ['none', 'add'], index=1)
        damped = st.toggle('Damped trend', value=True)
        phi = st.slider('Damping φ', 0.80, 0.99, 0.95, 0.01) if damped else None
        seasonal = st.selectbox('Seasonality', ['none', 'add', 'mul'], index=0)
        m = st.number_input('Seasonal period m', min_value=0, max_value=366, value=12, step=1)

        st.subheader('Smoothing parameters')
        manual = st.toggle('Manual α/β/γ (turn off to auto-optimize)', value=False)
        if manual:
            alpha = st.slider('α (level)', 0.00, 1.00, 0.30, 0.01)
            beta  = st.slider('β (trend)', 0.00, 1.00, 0.10, 0.01)
            gamma = st.slider('γ (seasonal)', 0.00, 1.00, 0.10, 0.01)
        else:
            alpha = beta = gamma = None

    # Normalize seasonal inputs
    seasonal_kw = None if seasonal == 'none' else seasonal
    m_kw = int(m) if (seasonal_kw is not None and int(m) >= 2) else None
    trend_kw = None if trend == 'none' else 'add'

    with st.spinner('Fitting ETS (diff-log pipeline) and generating plots...'):
        res = fit_and_backtest_ets(
            y,
            steps=int(k_back),
            error_type='A',  # dy often fine with additive error
            trend=trend_kw,
            damped=bool(damped),
            phi=phi,
            seasonal=seasonal_kw,
            m=m_kw,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            optimized=not manual,
        )

    # Backtest figure + metrics
    st.subheader('Backtest (holdout)')
    st.plotly_chart(res['fig_backtest'], use_container_width=True)

    # Metrics row
    c1, c2, c3 = st.columns(3)
    c1.metric('MAE', f"{res['metrics']['MAE']:.2f}")
    c2.metric('RMSE', f"{res['metrics']['RMSE']:.2f}")
    c3.metric('MAPE', f"{res['metrics']['MAPE']:.2f}%")

    with st.expander('Backtest table (Actual vs Forecast & AbsError)'):
        st.dataframe(res['tables']['test'])

    # Future forecast (use separate k_fut)
    # Re-run same config but reuse function output by changing steps
    res_future = fit_and_backtest_ets(
        y,
        steps=int(k_fut),
        error_type='A',
        trend=trend_kw,
        damped=bool(damped),
        phi=phi,
        seasonal=seasonal_kw,
        m=m_kw,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        optimized=not manual,
    )

    st.subheader('Future Forecast')
    st.plotly_chart(res_future['fig_forecast'], use_container_width=True)

    with st.expander('Future forecast table'):
        st.dataframe(res_future['tables']['future'])

    st.divider()
    st.subheader('Actuals by Month (YoY overlay)')
    st.plotly_chart(res['fig_yoy'], use_container_width=True)

    st.caption('ETS is fit on diff(log1p(y)); forecasting is iteratively reconstructed then inverse-transformed to level.')
