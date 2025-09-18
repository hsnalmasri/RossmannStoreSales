# monthly_forecast.py
from __future__ import annotations
import pandas as pd
import numpy as np
from pathlib import Path
import plotly.graph_objects as go
from statsmodels.tsa.statespace.sarimax import SARIMAX

# If you already have src/data_prep.load_rossmann_data, reuse it.
try:
    from src.data_prep import load_rossmann_data
except Exception:
    load_rossmann_data = None  # fallback to simple CSV read if missing


def load_prepared(sales_csv: str | Path, store_csv: str | Path,
                  keep_closed_days: bool = True) -> pd.DataFrame:
    """
    Returns a flat DataFrame with at least ['Store','Date','Sales','Open','Promo',...]
    """
    sales_csv = Path(sales_csv)
    store_csv = Path(store_csv)

    if load_rossmann_data is not None:
        _, df_ml = load_rossmann_data(
            sales_csv=sales_csv,
            store_csv=store_csv,
            keep_closed_days=keep_closed_days,
            build_flat_copy=True,
        )
        return df_ml

    # Fallback: minimal loader
    sales = pd.read_csv(sales_csv, parse_dates=['Date'], low_memory=False)
    store = pd.read_csv(store_csv)
    df = sales.merge(store[['Store']], on='Store', how='left')  # no features in fallback
    if not keep_closed_days and 'Open' in df.columns:
        df = df[df['Open'] == 1]
    return df


def make_monthly_series(df: pd.DataFrame, store: int | None = None,
                        agg: str = 'sum') -> pd.Series:
    """
    Aggregate to monthly series.
    - If store is None: aggregates all stores.
    - agg: 'sum' (default) or 'mean'.
    Returns a Series with monthly frequency (month end) named 'Sales'.
    """
    dff = df.copy()
    if store is not None:
        dff = dff[dff['Store'] == store]
    dff = dff.sort_values('Date')
    y = dff.set_index('Date')['Sales']
    y = pd.to_numeric(y, errors='coerce').fillna(0)
    y_m = y.resample('M').sum() if agg == 'sum' else y.resample('M').mean()
    y_m.name = 'Sales'
    return y_m


def fit_arima(y: pd.Series, seasonal: bool = True, m: int = 12):
    """
    Fit a SARIMAX model (ARIMA with optional seasonality).
    Default order is (1,1,1) and seasonal (1,1,1,m) if seasonal=True.
    """
    order = (1, 1, 1)
    seasonal_order = (1, 1, 1, m) if seasonal else (0, 0, 0, 0)
    model = SARIMAX(
        y,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    ).fit(disp=False)
    return model


def make_fig_actual_fitted_forecast(y: pd.Series, model, horizon: int = 3) -> go.Figure:
    """
    Return a Plotly figure with Actual, Fitted (in-sample), and Forecast (+CI).
    """
    # In-sample fitted
    fitted = pd.Series(model.fittedvalues, index=y.index, name='Fitted')

    # Forecast with confidence intervals
    fc_res = model.get_forecast(steps=horizon)
    fc = fc_res.predicted_mean.rename('Forecast')
    ci = fc_res.conf_int()

    # Grab first two columns as lower/upper (names vary by statsmodels version)
    lower = ci.iloc[:, 0]
    upper = ci.iloc[:, 1]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=y.index, y=y.values, mode='lines', name='Actual'))
    fig.add_trace(go.Scatter(x=fitted.index, y=fitted.values, mode='lines', name='Fitted'))
    fig.add_trace(go.Scatter(x=fc.index, y=fc.values, mode='lines', name='Forecast'))
    fig.add_trace(
        go.Scatter(
            x=list(lower.index) + list(upper.index[::-1]),
            y=list(upper.values) + list(lower.values[::-1]),
            fill='toself',
            name='Forecast CI',
            hoverinfo='skip',
            line=dict(width=0),
            opacity=0.2,
        )
    )
    fig.update_layout(
        title='Monthly Sales: Actual, Fitted, and Forecast',
        xaxis_title='Month',
        yaxis_title='Sales',
        legend_title='',
        margin=dict(l=40, r=20, t=60, b=40),
    )
    return fig


def run_example(
    sales_csv: str | Path = 'data/RAW/SalesData.csv',
    store_csv: str | Path = 'data/RAW/StoreData.csv',
    store: int | None = None,
    horizon: int = 3,
    seasonal: bool = True,
    m: int = 12,
):
    df = load_prepared(sales_csv, store_csv, keep_closed_days=True)
    y = make_monthly_series(df, store=store, agg='sum').asfreq('M').fillna(0)
    model = fit_arima(y, seasonal=seasonal, m=m)
    fig = make_fig_actual_fitted_forecast(y, model, horizon=horizon)
    return fig, y, model


if __name__ == '__main__':
    fig, y, model = run_example()
    print(model.summary())
