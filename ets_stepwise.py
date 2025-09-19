"""
Exponential Smoothing (ETS) step-wise forecasting module
- Manual control of ETS hyperparameters from Streamlit
- Two outputs: (1) backtest plot for last k steps; (2) future forecast plot for k steps
- Transformation pipeline: log1p -> first difference -> ETS -> iterative reconstruction -> inverse log1p
- Plotting helpers return Plotly figures with required styles

Public API
----------
fit_and_backtest_ets(
    s: pd.Series,
    steps: int,
    error_type: str = "A",           # 'A' or 'M' (controls Box-Cox + multiplicative settings)
    trend: str | None = "add",        # 'add' | None
    damped: bool = True,
    phi: float | None = None,         # 0.8..0.99 typical, if damped
    seasonal: str | None = None,      # 'add' | 'mul' | None
    m: int | None = None,             # seasonal period
    alpha: float | None = None,
    beta: float | None = None,
    gamma: float | None = None,
    optimized: bool = True,
) -> dict
    returns {
      'fig_backtest': go.Figure,
      'fig_forecast': go.Figure,
      'metrics': { 'MAE': float, 'RMSE': float, 'MAPE': float },
      'tables': { 'test': pd.DataFrame, 'future': pd.DataFrame }
    }

Notes
-----
• We fit ETS on the differenced log1p series (dy). ETS can model this stationary-ish series well.
• For k-step forecasts we iterate: y_hat_t = y_prev + dy_hat_t; level = expm1(y_hat_t).
• Backtest holds out the last `steps` points of the ORIGINAL series.
"""
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Optional, Literal

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# -------------------------------
# Utilities: metrics & transforms
# -------------------------------

def _safe_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = y_true != 0
    if mask.sum() == 0:
        return float("nan")
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def to_month_str(idx: pd.DatetimeIndex) -> pd.Index:
    return pd.Index([d.strftime("%b") for d in idx])


# -------------------------------
# Transformation pipeline helpers
# -------------------------------

def log1p_diff(s: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Return (y, dy) where y = log1p(level), dy = y.diff().dropna()."""
    y = np.log1p(s.astype(float))
    dy = y.diff().dropna()
    return y, dy


def invert_stepwise_from_diff(
    last_y: float,
    dy_forecast: pd.Series,
) -> pd.Series:
    """Iteratively reconstruct y_t from dy forecasts, given last observed y.
    y_t = y_{t-1} + dy_hat_t.
    """
    vals = []
    y_prev = last_y
    for _, dy_t in dy_forecast.items():
        y_t = y_prev + float(dy_t)
        vals.append(y_t)
        y_prev = y_t
    return pd.Series(vals, index=dy_forecast.index)


def inv_log1p(y: pd.Series) -> pd.Series:
    return np.expm1(y)


# -------------------------------
# ETS fitting on differenced series
# -------------------------------
@dataclass
class ETSConfig:
    error_type: Literal["A", "M"] = "A"
    trend: Optional[Literal["add", None]] = "add"
    damped: bool = True
    phi: Optional[float] = None
    seasonal: Optional[Literal["add", "mul", None]] = None
    m: Optional[int] = None
    alpha: Optional[float] = None
    beta: Optional[float] = None
    gamma: Optional[float] = None
    optimized: bool = True


def _fit_ets_on_diff(
    dy: pd.Series,
    cfg: ETSConfig,
):
    # Map error type to use_boxcox/multiplicative decision for residual variance. For dy, variance is often stable; keep additive.
    seasonal = cfg.seasonal
    seasonal_periods = cfg.m if seasonal else None

    model = ExponentialSmoothing(
        dy,
        trend=cfg.trend,               # 'add' or None on dy
        damped_trend=cfg.damped,
        seasonal=seasonal,             # optional on dy if there is residual seasonality
        seasonal_periods=seasonal_periods,
        initialization_method="estimated",
    )

    fit = model.fit(
        smoothing_level=cfg.alpha,
        smoothing_slope=cfg.beta,
        smoothing_seasonal=cfg.gamma,
        damping_trend=cfg.phi,
        optimized=cfg.optimized,
    )
    return fit


# -------------------------------
# Backtest (last k) and future forecast
# -------------------------------

def fit_and_backtest_ets(
    s: pd.Series,
    steps: int,
    error_type: str = "A",
    trend: Optional[str] = "add",
    damped: bool = True,
    phi: Optional[float] = None,
    seasonal: Optional[str] = None,
    m: Optional[int] = None,
    alpha: Optional[float] = None,
    beta: Optional[float] = None,
    gamma: Optional[float] = None,
    optimized: bool = True,
):
    assert steps >= 1, "steps must be >= 1"
    s = s.dropna().astype(float)
    if len(s) <= steps + 2:
        raise ValueError("Not enough data for backtest with requested steps.")

    # Transform
    y, dy = log1p_diff(s)

    # Define backtest split on ORIGINAL index: last `steps` observations are test
    split_idx = s.index[-steps - 1]  # we need one extra for the last_y anchor

    s_train = s.loc[:split_idx]
    s_test = s.loc[split_idx:]
    # Align to dy: dy starts from second point, so training dy must end at split_idx (exclusive)
    dy_train = dy.loc[:split_idx].iloc[:-1]

    cfg = ETSConfig(
        error_type=error_type,
        trend=(None if trend in (None, "none", "") else "add"),
        damped=damped,
        phi=phi,
        seasonal=(None if seasonal in (None, "none", "") else seasonal),
        m=m,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        optimized=optimized,
    )

    fit = _fit_ets_on_diff(dy_train, cfg)

    # In-sample fitted dy for training period
    dy_fitted = pd.Series(fit.fittedvalues, index=dy_train.index)

    # Reconstruct fitted level for training window for visualization
    y_train = np.log1p(s_train)
    # Start from the second point of y_train to align with dy_fitted
    y_fitted = y_train.iloc[0] + dy_fitted.cumsum()
    level_fitted = inv_log1p(y_fitted)

    # Step-wise forecast for `steps` on dy, then invert
    dy_fc = fit.forecast(steps)
    # anchor y at the last observed point BEFORE the test window
    last_y = y.loc[split_idx]
    y_fc = invert_stepwise_from_diff(last_y, dy_fc)
    level_fc = inv_log1p(y_fc)

    # Build test target for comparison (exclude the anchor point)
    s_test_eval = s.loc[s.index.get_loc(split_idx) + 1:]
    s_test_eval = s_test_eval.iloc[:steps]

    # Metrics
    metrics = {
        "MAE": mae(s_test_eval.values, level_fc.values),
        "RMSE": rmse(s_test_eval.values, level_fc.values),
        "MAPE": _safe_mape(s_test_eval.values, level_fc.values),
    }

    # Future forecast from the very end
    dy_fit_full = _fit_ets_on_diff(dy, cfg)
    dy_future = dy_fit_full.forecast(steps)
    last_y_full = y.iloc[-1]
    y_future = invert_stepwise_from_diff(last_y_full, dy_future)
    level_future = inv_log1p(y_future)

    # --------------------
    # Plotly: Backtest fig
    # --------------------
    fig_back = go.Figure()

    # Actual (solid with markers)
    fig_back.add_trace(go.Scatter(
        x=s.index, y=s.values, mode="lines+markers",
        name="Actual", line=dict(dash="solid"),
    ))

    # Fitted (dashed) over train region except the very first anchor point
    fig_back.add_trace(go.Scatter(
        x=level_fitted.index, y=level_fitted.values,
        mode="lines", name="Fitted (train)", line=dict(dash="dash")
    ))

    # Test (solid with markers, different color handled by Plotly)
    fig_back.add_trace(go.Scatter(
        x=s_test_eval.index, y=s_test_eval.values,
        mode="lines+markers", name="Test (holdout)"
    ))

    # Forecast (dashed)
    fig_back.add_trace(go.Scatter(
        x=level_fc.index, y=level_fc.values,
        mode="lines", name=f"Forecast (k={steps})", line=dict(dash="dash")
    ))

    fig_back.update_layout(
        title=f"ETS Backtest: last {steps} steps",
        xaxis_title="Date",
        yaxis_title="Level",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=40, r=20, t=60, b=40),
    )

    # ----------------------------------
    # Plotly: Future forecast (k steps)
    # ----------------------------------
    fig_future = go.Figure()
    fig_future.add_trace(go.Scatter(
        x=s.index, y=s.values, mode="lines+markers",
        name="Actual", line=dict(dash="solid"),
    ))
    fig_future.add_trace(go.Scatter(
        x=level_future.index, y=level_future.values,
        mode="lines", name=f"Forecast (next {steps})", line=dict(dash="dash")
    ))
    fig_future.update_layout(
        title=f"ETS Future Forecast: next {steps} steps",
        xaxis_title="Date",
        yaxis_title="Level",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=40, r=20, t=60, b=40),
    )

    # ----------------------------
    # Optional: Year-over-year map
    # ----------------------------
    # Re-plot actuals by year (x = months Jan..Dec)
    yoy = (
        s.copy()
        .to_frame("value")
        .assign(year=lambda df: df.index.year, month=lambda df: df.index.strftime("%b"))
    )
    # Ensure month order
    month_order = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    yoy["month"] = pd.Categorical(yoy["month"], categories=month_order, ordered=True)
    yoy = yoy.sort_values(["year", "month"])  # year-over-year lines

    fig_yoy = go.Figure()
    for yr, dfy in yoy.groupby("year"):
        fig_yoy.add_trace(go.Scatter(
            x=dfy["month"], y=dfy["value"], mode="lines+markers", name=str(yr)
        ))
    fig_yoy.update_layout(
        title="Actuals by Month (YoY overlay)",
        xaxis_title="Month",
        yaxis_title="Level",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=40, r=20, t=60, b=40),
    )

    # Tables for Streamlit display if needed
    test_tbl = pd.DataFrame({
        "Actual": s_test_eval,
        "Forecast": level_fc,
        "AbsError": (s_test_eval - level_fc).abs(),
    })
    fut_tbl = pd.DataFrame({"Forecast": level_future})

    return {
        "fig_backtest": fig_back,
        "fig_forecast": fig_future,
        "fig_yoy": fig_yoy,
        "metrics": metrics,
        "tables": {"test": test_tbl, "future": fut_tbl},
    }
