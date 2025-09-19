import pandas as pd
from src.models.ets import fit_ets_manual

idx = pd.date_range("2024-01-01", periods=60, freq="D")
y = pd.Series(100 + (idx.dayofweek >= 5).astype(int) * 20, index=idx).astype(float)

res = fit_ets_manual(y, horizon=7, seasonal_periods=7)
assert set(res) == {"fitted", "forecast", "residuals"}
assert len(res["forecast"]) == 7
print("OK:", res["forecast"].round(2).tolist()[:3])
