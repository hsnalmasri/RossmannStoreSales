# ets_stepwise.py
import numpy as np
import pandas as pd
from typing import Optional, Dict, Any
from statsmodels.tsa.holtwinters import ExponentialSmoothing


class ETSDataError(ValueError):
    pass


def _ensure_monthly_index(y: pd.Series) -> pd.Series:
    if not isinstance(y.index, pd.DatetimeIndex):
        raise ETSDataError("Input series index must be a pandas.DatetimeIndex.")
    # Coerce to Month Start if needed
    if y.index.freq is None:
        # Try to infer; if still None, resample to monthly start
        try:
            y = y.asfreq('MS')
        except Exception:
            y = y.resample('MS').sum()
    elif y.index.freqstr not in ('MS', 'M'):
        # Normalize to Month Start
        y = y.asfreq('MS')
    return y


def _apply_transform(y: pd.Series, transform: Optional[str]) -> pd.Series:
    if transform is None:
        return y
    if transform == 'log1p':
        if (y < 0).any():
            raise ETSDataError("log1p transform requires non-negative data.")
        return np.log1p(y)
    raise ETSDataError(f"Unsupported transform: {transform}")


def _inverse_transform(x: pd.Series, transform: Optional[str]) -> pd.Series:
    if transform is None:
        return x
    if transform == 'log1p':
        return np.expm1(x)
    return x  # fallback


def fit_ets_holdout(
    y: pd.Series,
    n_forecast: int = 6,
    *,
    trend: Optional[str] = "add",              # "add" | "mul" | None
    seasonal: Optional[str] = "add",           # "add" | "mul" | None
    seasonal_periods: Optional[int] = 12,
    damped_trend: bool = False,
    initialization_method: str = "estimated",
    optimized: bool = True,
    smoothing_level: Optional[float] = None,
    smoothing_trend: Optional[float] = None,
    smoothing_seasonal: Optional[float] = None,
    damping_trend: Optional[float] = None,
    transform: Optional[str] = None,           # None | "log1p"
    remove_bias: bool = False,
) -> Dict[str, Any]:
    """
    Fit an ETS model with a simple holdout backtest on the last `n_forecast` periods.
    Returns data only (no plotting).

    Output dict keys:
      - 'y_train', 'y_test'
      - 'fitted' (in-sample one-step-ahead on train)
      - 'forecast' (out-of-sample length n_forecast)
      - 'residuals' (train residuals)
      - 'metrics' (mae/mape/rmse on test if test exists)
      - 'components' (level, trend, season on model scale)
      - 'model_info' (config, params, aic/bic, transform)
    """
    if not isinstance(y, pd.Series):
        raise ETSDataError("`y` must be a pandas Series.")

    y = y.sort_index()
    y = _ensure_monthly_index(y)

    if seasonal == "mul" or trend == "mul":
        if (y <= 0).any():
            raise ETSDataError("Multiplicative components require strictly positive data.")

    if n_forecast < 0 or n_forecast >= len(y):
        raise ETSDataError("`n_forecast` must be >= 0 and less than the length of `y`.")

    # Split
    y_train = y.iloc[:-n_forecast] if n_forecast > 0 else y.copy()
    y_test = y.iloc[-n_forecast:] if n_forecast > 0 else pd.Series(dtype=y.dtype)

    # Apply transform (if any)
    y_train_t = _apply_transform(y_train, transform)

    # Build and fit model
    model = ExponentialSmoothing(
        y_train_t,
        trend=trend,
        seasonal=seasonal,
        seasonal_periods=seasonal_periods,
        damped_trend=damped_trend,
        initialization_method=initialization_method,
    )

    fit = model.fit(
        optimized=optimized,
        smoothing_level=smoothing_level,
        smoothing_trend=smoothing_trend,
        smoothing_seasonal=smoothing_seasonal,
        damping_trend=damping_trend,
        remove_bias=remove_bias,
    )

    # In-sample fitted (on model scale), residuals
    fitted_t = fit.fittedvalues
    resid_t = fit.resid

    # Forecast on model scale
    fcst_t = fit.forecast(n_forecast) if n_forecast > 0 else pd.Series(dtype=y.dtype)

    # Inverse transform back to original scale (for user consumption)
    fitted = _inverse_transform(fitted_t, transform)
    forecast = _inverse_transform(fcst_t, transform)
    residuals = y_train - fitted.reindex(y_train.index, fill_value=np.nan)

    # Basic metrics on test (if present)
    metrics = {}
    if n_forecast > 0 and len(y_test) == n_forecast:
        # Align forecast index to y_test if needed
        forecast = forecast.set_axis(y_test.index)
        err = y_test - forecast
        mae = err.abs().mean()
        rmse = np.sqrt((err**2).mean())
        # Safe MAPE (ignore zeros)
        mask = y_test != 0
        mape = (err[mask].abs() / y_test[mask]).mean() * 100 if mask.any() else np.nan
        metrics = {"mae": float(mae), "rmse": float(rmse), "mape": float(mape)}

    # Components (note: components are on model scale; we keep them as-is)
    components = {
        "level": getattr(fit, "level", None),
        "trend": getattr(fit, "trend", None),
        "season": getattr(fit, "season", None),
    }

    model_info = {
        "aic": getattr(fit, "aic", None),
        "bic": getattr(fit, "bic", None),
        "sse": getattr(fit, "sse", None),
        "params": {
            "alpha": getattr(fit, "model", None) and fit.params.get("smoothing_level"),
            "beta": getattr(fit, "model", None) and fit.params.get("smoothing_trend"),
            "gamma": getattr(fit, "model", None) and fit.params.get("smoothing_seasonal"),
            "phi": getattr(fit, "model", None) and fit.params.get("damping_trend"),
        },
        "config": {
            "trend": trend,
            "seasonal": seasonal,
            "seasonal_periods": seasonal_periods,
            "damped_trend": damped_trend,
            "initialization_method": initialization_method,
            "optimized": optimized,
            "remove_bias": remove_bias,
            "transform": transform,
            "n_forecast": n_forecast,
        },
        "model_scale": "log1p" if transform == "log1p" else "raw",
    }

    return {
        "y_train": y_train,
        "y_test": y_test,
        "fitted": fitted,
        "forecast": forecast,
        "residuals": residuals,
        "metrics": metrics,
        "components": components,
        "model_info": model_info,
    }
