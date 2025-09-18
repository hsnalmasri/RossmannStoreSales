
# streamlit_app.py
from __future__ import annotations
import streamlit as st
import pandas as pd
from pathlib import Path

from monthly_forecast import load_prepared, make_monthly_series, fit_arima, make_fig_actual_fitted_forecast

st.set_page_config(page_title='Rossmann Monthly Forecast', layout='wide')

st.title('Rossmann Monthly Forecast (ARIMA)')

with st.sidebar:
    st.header('Data Inputs')
    sales_path = st.text_input('SalesData.csv path', 'data/RAW/SalesData.csv')
    store_path = st.text_input('StoreData.csv path', 'data/RAW/StoreData.csv')
    horizon = st.slider('Forecast horizon (months)', 1, 12, 3)

@st.cache_data(show_spinner=True)
def get_df(sales_path: str, store_path: str) -> pd.DataFrame:
    return load_prepared(sales_path, store_path, keep_closed_days=True)

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

# Build monthly series and fit ARIMA
with st.spinner('Fitting ARIMA model...'):
    y = make_monthly_series(df, store=store_val, agg='sum').asfreq('M').fillna(0)
    model = fit_arima(y, seasonal=True, m=12)
    fig = make_fig_actual_fitted_forecast(y, model, horizon=horizon)

col1, col2 = st.columns([3, 1])
with col1:
    st.plotly_chart(fig, use_container_width=True)
with col2:
    st.subheader('Summary')
    st.write(f"**Observations:** {len(y)}")
    st.write(f"**Last month:** {y.index.max().strftime('%Y-%m')}")
    st.write(f"**Last value:** {y.iloc[-1]:,.0f}")
    st.caption('Model: auto_arima (seasonal=12)')

st.divider()
st.subheader('Data preview')
st.dataframe(y.to_frame('Sales').tail(24))

st.caption('Tip: Put CSVs under data/RAW/ and adjust paths above if needed.')
