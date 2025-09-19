import numpy as np
import pandas as pd
from src.models.ets import fit_ets_and_forecast, fit_ets_manual

# Create a synthetic monthly seasonal series
idx = pd.date_range("2018-01-01", periods=60, freq="MS")
y = pd.Series(100 + 0.8*np.arange(60) + 10*np.sin(2*np.pi*(idx.month/12)), index=idx)

# Default (auto)
out = fit_ets_and_forecast(y, horizon=6, seasonal_periods=12)
assert set(out.keys()) == {"fitted","forecast","residuals"}
assert len(out["forecast"]) == 6
assert out["fitted"].index.equals(y.index)

# Manual (add-add, damped False)
out2 = fit_ets_manual(
    y, horizon=6, seasonal_periods=12,
    trend="add", seasonal="add", damped_trend=False,
    smoothing_level=None, smoothing_trend=None, smoothing_seasonal=None
)
assert len(out2["forecast"]) == 6
print("ETS smoke tests passed.")