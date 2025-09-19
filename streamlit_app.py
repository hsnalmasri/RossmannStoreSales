# streamlit_app.py
from __future__ import annotations
import streamlit as st
import pandas as pd
from pathlib import Path

# Local modules
from monthly_forecast import (
    load_prepared as _load_prepared_csvs,  # legacy local-path loader
    make_monthly_series,
    fit_arima,
    make_fig_actual_fitted_forecast,
)
from ets_stepwise import fit_and_backtest_ets

st.set_page_config(page_title='Rossmann Monthly Forecast', layout='wide')

st.title('Rossmann Monthly Forecast Dashboard')

# -----------------------------
# Data source controls
# -----------------------------
with st.sidebar:
    st.header('Data Source')
    src = st.radio('Load data from', ['Secrets URLs', 'Local paths'], index=0)

    if src == 'Local paths':
        sales_path = st.text_input('SalesData.csv path', 'data/RAW/SalesData.csv')
        store_path = st.text_input('StoreData.csv path', 'data/RAW/StoreData.csv')
    else:
        st.caption('Reading from Streamlit secrets (see `.streamlit/secrets.toml`).')
        # Optional preview of the domain only (don’t print full URLs if you don’t want to)
        try:
            _su = st.secrets['data'].get('sales_url', '')
            _tu = st.secrets['data'].get('store_url', '')
            if _su:
                st.write('Sales URL domain:', pd.io.common.urlparse(_su).netloc)
            if _tu:
                st.write('Store URL domain:', pd.io.common.urlparse(_tu).netloc)
        except Exception:
            st.info('Add keys data.sales_url and data.store_url to secrets.')

    st.header('Series Build')
    agg_choice = st.radio('Monthly aggregation', ['sum', 'mean'], index=0, horizontal=True)

    st.header('Model')
    model_choice = st.radio('Model type', ['ARIMA', 'Exponential Smoothing (ETS)'], index=1)

    if model_choice == 'ARIMA':
        horizon = st.slider('Forecast horizon (months)', 1, 24, 6)

    if model_choice == 'Exponential Smoothing (ETS)':
        st.subheader('ETS Parameters')
        k_back = st.slider('Backtest last k steps', 1, 24, 3)
        k_fut = st.slider('Future forecast steps', 1, 24, 6)
        trend = st.selectbox('Trend', ['none', 'add'], index=1)
        seasonal = st.selectbox('Seasonal', ['none', 'add', 'mul'], index=0)
        m = st.number_input('Seasonal Period (m)', min_value=0, max_value=366, value=12)
        damped = st.checkbox('Damped Trend', value=True)
        phi = st.slider('Damping factor (phi)', 0.80, 0.99, 0.95)
        manual = st.checkbox('Manual α/β/γ (disable to auto-optimize)', value=False)
        if manual:
            alpha = st.slider('Alpha (level)', 0.0, 1.0, 0.2)
            beta = st.slider('Beta (trend)', 0.0, 1.0, 0.1)
            gamma = st.slider('Gamma (seasonal)', 0.0, 1.0, 0.1)
        else:
            alpha = beta = gamma = None

# -----------------------------
# Data loading (URLs via secrets OR local CSV paths)
# -----------------------------

@st.cache_data(show_spinner=True)
def load_from_urls(sales_url: str, store_url: str, keep_closed_days: bool = True) -> pd.DataFrame:
    sales = pd.read_csv(sales_url, parse_dates=['Date'], low_memory=False)
    store = pd.read_csv(store_url)
    df = sales.merge(store[['Store']], on='Store', how='left')
    if not keep_closed_days and 'Open' in df.columns:
        df = df[df['Open'] == 1]
    return df

@st.cache_data(show_spinner=True)
def load_from_local(sales_path: str, store_path: str) -> pd.DataFrame:
    # Use existing helper that expects file paths
    return _load_prepared_csvs(sales_path, store_path, keep_closed_days=True)

# Choose loader based on sidebar selection
try:
    if src == 'Secrets URLs':
        sales_url = st.secrets['data']['sales_url']
        store_url = st.secrets['data']['store_url']
        df = load_from_urls(sales_url, store_url, keep_closed_days=True)
    else:
        df = load_from_local(sales_path, store_path)
except Exception as e:
    st.error(f"Failed to load data: {e}")
    st.stop()

# -----------------------------
# Series selection and build
# -----------------------------
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
    # Normalize seasonal inputs
    seasonal_kw = None if seasonal == 'none' else seasonal
    m_kw = int(m) if (seasonal_kw is not None and int(m) >= 2) else None
    trend_kw = None if trend == 'none' else 'add'

    with st.spinner('Fitting ETS (diff-log pipeline) and generating plots...'):
        res = fit_and_backtest_ets(
            y,
            steps=int(k_back),
            error_type='A',
            trend=trend_kw,
            damped=bool(damped),
            phi=phi,
            seasonal=seasonal_kw,
            m=m_kw,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            optimized=(not manual),
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

    # Future forecast
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
        optimized=(not manual),
    )

    st.subheader('Future Forecast')
    st.plotly_chart(res_future['fig_forecast'], use_container_width=True)

    with st.expander('Future forecast table'):
        st.dataframe(res_future['tables']['future'])

    st.divider()
    st.subheader('Actuals by Month (YoY overlay)')
    st.plotly_chart(res['fig_yoy'], use_container_width=True)

    st.caption('ETS is fit on diff(log1p(y)); forecasting is iteratively reconstructed then inverse-transformed to level.')
