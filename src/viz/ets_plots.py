from __future__ import annotations

from typing import Optional
import pandas as pd
import plotly.graph_objs as go

from ets_plots import (  # noqa: F401
    make_backtest_fig,
    make_forecast_fig,
)

from src.models.ets import fit_ets_manual

__all__ = ["make_backtest_fig", "make_forecast_fig"]


def _monthly_sum(series: pd.Series) -> pd.Series:
    """DatetimeIndex series -> monthly sum with freq MS."""
    m = series.resample("MS").sum()
    m.index.freq = m.index.inferred_freq or "MS"
    return m


def _train_test_split(y: pd.Series, test_h: int) -> tuple[pd.Series, pd.Series]:
    if test_h <= 0 or test_h >= len(y):
        raise ValueError("viz.ets_plots:_train_test_split: invalid test horizon.")
    return y.iloc[:-test_h], y.iloc[-test_h:]


def make_backtest_fig(
    y_daily: pd.Series,
    *,
    seasonal_periods: int,
    test_horizon: int,
    trend: Optional[str],
    seasonal: Optional[str],
    damped_trend: bool,
    smoothing_level: Optional[float],
    smoothing_trend: Optional[float],
    smoothing_seasonal: Optional[float],
    remove_bias: bool = False,
) -> go.Figure:
    """Backtest plot: train on history minus last H months, forecast H, compare with actual."""
    y_m = _monthly_sum(y_daily)
    train, test = _train_test_split(y_m, test_horizon)

    # Need ≥ 2 seasonal cycles for the train segment when seasonal is used
    if seasonal is not None and len(train) < 2 * max(1, seasonal_periods):
        raise ValueError("viz.ets_plots: training must contain at least two seasonal cycles.")

    out = fit_ets_manual(
        train,
        horizon=test_horizon,
        seasonal_periods=seasonal_periods,
        trend=trend,
        seasonal=seasonal,
        damped_trend=damped_trend,
        smoothing_level=smoothing_level,
        smoothing_trend=smoothing_trend,
        smoothing_seasonal=smoothing_seasonal,
        remove_bias=remove_bias,
    )

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=train.index, y=train.values, name="Train (Monthly Sales)", mode="lines"))
    fig.add_trace(go.Scatter(x=test.index, y=test.values, name="Test Actual", mode="lines"))
    fig.add_trace(go.Scatter(x=out["forecast"].index, y=out["forecast"].values, name="ETS Forecast", mode="lines"))

    fig.update_layout(
        title="ETS Backtest: Train vs Test",
        xaxis_title="Month",
        yaxis_title="Sales",
        hovermode="x unified",
        legend=dict(orientation="h", y=-0.2),
        margin=dict(l=40, r=20, t=60, b=60),
    )
    return fig


def make_forecast_fig(
    y_daily: pd.Series,
    *,
    seasonal_periods: int,
    forecast_horizon: int,
    trend: Optional[str],
    seasonal: Optional[str],
    damped_trend: bool,
    smoothing_level: Optional[float],
    smoothing_trend: Optional[float],
    smoothing_seasonal: Optional[float],
    remove_bias: bool = False,
) -> go.Figure:
    """Fit on full monthly history and forecast the requested horizon."""
    y_m = _monthly_sum(y_daily)

    # Need ≥ 2 cycles overall if seasonal
    if seasonal is not None and len(y_m) < 2 * max(1, seasonal_periods):
        raise ValueError("viz.ets_plots: series must contain at least two seasonal cycles.")

    out = fit_ets_manual(
        y_m,
        horizon=forecast_horizon,
        seasonal_periods=seasonal_periods,
        trend=trend,
        seasonal=seasonal,
        damped_trend=damped_trend,
        smoothing_level=smoothing_level,
        smoothing_trend=smoothing_trend,
        smoothing_seasonal=smoothing_seasonal,
        remove_bias=remove_bias,
    )

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=y_m.index, y=y_m.values, name="History (Monthly Sales)", mode="lines"))
    fig.add_trace(go.Scatter(x=out["forecast"].index, y=out["forecast"].values, name="ETS Forecast", mode="lines"))

    fig.update_layout(
        title="ETS Forward Forecast",
        xaxis_title="Month",
        yaxis_title="Sales",
        hovermode="x unified",
        legend=dict(orientation="h", y=-0.2),
        margin=dict(l=40, r=20, t=60, b=60),
    )
    return fig
