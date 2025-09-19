# ets_stepwise.py
from __future__ import annotations

import math
from typing import Dict, Any

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing


def _fit_single(y: pd.Series, cfg: Dict[str, Any]):
    """
    Internal: fit an ETS model with the given config on a Series y.
    Assumes y is monthly (MS) indexed and non-empty.
    """
    model = ExponentialSmoothing(
        y,
        trend=cfg.get("trend", "add"),
        seasonal=cfg.get("seasonal", "add"),
        seasonal_periods=cfg.get("seasonal_periods", 12),
        damped_trend=cfg.get("damped_trend", True),
        initialization_method="estimated",
    )
    return model.fit(optimized=not cfg.get("manual", False))


def _metrics(y_true: pd.Series, y_pred: pd.Series) -> Dict[str, float]:
    """
    Basic holdout metrics. (NaNs in denominators are ignored in MAPE.)
    """
    e = y_true - y_pred
    mae = float(e.abs().mean())
    rmse = float(math.sqrt((e**2).mean()))
    mape = float((e.abs() / y_true.replace(0, np.nan)).mean() * 100)
    return {"mae": mae, "rmse": rmse, "mape": mape}


def fit_and_backtest_ets(
    y: pd.Series,
    n_test: int = 6,
    cfg: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """
    Fit ETS on the training part (y[:-n_test]), forecast the next n_test months,
    and return DATA ONLY (no plotting).

    Returns dict:
        {
            "y_train": pd.Series,
            "y_test": pd.Series,
            "fitted": pd.Series,     # in-sample fitted values (train)
            "forecast": pd.Series,   # holdout predictions aligned to y_test index
            "params_used": dict,
            "metrics": {"mae":..., "rmse":..., "mape":...},
        }
    """
    if cfg is None:
        cfg = {}

    if not isinstance(y, pd.Series):
        y = pd.Series(y.squeeze(), index=getattr(y, "index", None), name="y")

    if n_test <= 0 or n_test >= len(y):
        raise ValueError("n_test must be between 1 and len(y)-1")

    y_train = y.iloc[:-n_test]
    y_test = y.iloc[-n_test:]

    fit = _fit_single(y_train, cfg)

    fitted_vals = pd.Series(fit.fittedvalues, index=y_train.index, name="fitted")
    fcst = fit.forecast(n_test)
    fcst.index = y_test.index
    fcst.name = "forecast"

    return {
        "y_train": y_train,
        "y_test": y_test,
        "fitted": fitted_vals,
        "forecast": fcst,
        "params_used": {
            "trend": cfg.get("trend", "add"),
            "seasonal": cfg.get("seasonal", "add"),
            "seasonal_periods": cfg.get("seasonal_periods", 12),
            "damped_trend": cfg.get("damped_trend", True),
        },
        "metrics": _metrics(y_test, fcst),
    }


# --- Backwards-compat alias if older code imports this name ---
def fit_ets_holdout(*args, **kwargs):
    return fit_and_backtest_ets(*args, **kwargs)
