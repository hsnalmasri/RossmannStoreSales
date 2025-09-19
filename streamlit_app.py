# --- Overall Sales & Forecast (ETS) page -------------------------------------
import pandas as pd
import numpy as np
import streamlit as st
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_error
import plotly.graph_objects as go
import plotly.express as px

# ------------------------- Data loading helpers ------------------------------
def _try_attr(module, *names):
    for n in names:
        if hasattr(module, n):
            return getattr(module, n)
    return None

def load_monthly_series():
    """
    Tries your project loaders and returns a single monthly series `y`:
    DatetimeIndex @ monthly start, name='Sales', float.
    """
    y = None

    # 1) Try your internal loaders if available
    try:
        from src.rossmann import data_load as DL  # adjust if your package path differs
    except Exception:
        DL = None

    if DL:
        loader = _try_attr(
            DL,
            "load_store_imputed",  # you said this exists
            "load_store_imputed",        # earlier mention
            "load_raw",            # fallback
        )
        if loader:
            df = loader()
            # Expect columns like ['Date', 'Sales'] (adjust here if your names differ)
            # Robust monthly conversion:
            if "Date" in df.columns:
                idx = pd.to_datetime(df["Date"])
                values = df["Sales"].astype(float).values
                s = pd.Series(values, index=idx, name="Sales").sort_index()
            else:
                # If Date already is the index
                s = df["Sales"].astype(float).copy()
                s.index = pd.to_datetime(s.index)
            y = (
                s.resample("MS")  # month start
                 .sum()
                 .astype(float)
                 .rename("Sales")
            )

    # 2) If no internal loader worked, allow CSV upload as a fallback
    if y is None:
        st.info("No internal loader found. Upload a CSV with columns ['Date','Sales'].")
        up = st.file_uploader("Upload CSV", type=["csv"])
        if up is None:
            st.stop()
        df = pd.read_csv(up)
        if "Date" not in df.columns or "Sales" not in df.columns:
            st.error("CSV must contain 'Date' and 'Sales' columns.")
            st.stop()
        s = pd.Series(
            df["Sales"].astype(float).values,
            index=pd.to_datetime(df["Date"]),
            name="Sales",
        ).sort_index()
        y = s.resample("MS").sum().astype(float).rename("Sales")

    # Drop missing months if any (after resampling they are zeros; if you prefer, fillna)
    y = y.dropna()
    return y

# ---------------------------- Utility functions ------------------------------
def allowed_test_max(n_obs: int, season_length: int) -> int:
    """
    Max Ntest such that train has at least two full seasonal cycles.
    """
    return max(0, n_obs - 2 * season_length)

def smape(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    diff = np.abs(y_true - y_pred)
    with np.errstate(divide="ignore", invalid="ignore"):
        s = np.where(denom == 0, 0.0, diff / denom)
    return 100.0 * np.mean(s)

def fit_ets(y, trend, seasonal, seasonal_periods, damped_trend, use_boxcox, initialization_method):
    model = ExponentialSmoothing(
        y,
        trend=trend,                 # None, 'add', 'mul'
        seasonal=seasonal,           # None, 'add', 'mul'
        seasonal_periods=seasonal_periods if seasonal else None,
        damped_trend=damped_trend,
        initialization_method=initialization_method,  # 'estimated' | 'heuristic'
    )
    fit = model.fit(
        optimized=True,
        use_boxcox=use_boxcox,   # False | True | 'log'
        remove_bias=False,
    )
    return fit

def make_holdout_plot(y_train, y_test, y_hat):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=y_train.index, y=y_train, mode="lines", name="Train"))
    fig.add_trace(go.Scatter(x=y_test.index, y=y_test, mode="lines", name="Test"))
    fig.add_trace(go.Scatter(x=y_test.index, y=y_hat, mode="lines", name="Forecast on Test"))
    fig.update_layout(
        title="Hold-out Backtest (ETS)",
        xaxis_title="Date",
        yaxis_title="Sales",
        hovermode="x unified",
        legend=dict(orientation="h", y=1.1)
    )
    return fig

def make_forecast_plot(y, fcast, conf_low=None, conf_high=None):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=y.index, y=y, mode="lines", name="History"))
    fig.add_trace(go.Scatter(x=fcast.index, y=fcast, mode="lines", name="Forecast"))
    if conf_low is not None and conf_high is not None:
        fig.add_trace(go.Scatter(
            x=fcast.index.tolist() + fcast.index[::-1].tolist(),
            y=conf_high.tolist() + conf_low[::-1].tolist(),
            fill="toself",
            name="Confidence band",
            line=dict(width=0),
            opacity=0.2
        ))
    fig.update_layout(
        title="Forecast Ahead (ETS)",
        xaxis_title="Date",
        yaxis_title="Sales",
        hovermode="x unified",
        legend=dict(orientation="h", y=1.1)
    )
    return fig

# ------------------------------- UI / Logic ----------------------------------
st.header("📊 Management — Overall Sales & Forecast (ETS)")

# Load series
y = load_monthly_series()
n_obs = len(y)
min_dt, max_dt = y.index.min(), y.index.max()
years_span = (max_dt.year - min_dt.year) + 1

with st.sidebar:
    st.subheader("Settings")

    # Seasonality (default monthly = 12)
    season_length = st.number_input("Seasonal Periods (e.g., 12 for monthly)", min_value=1, value=12, step=1)

    # Enforce two full cycles in TRAIN
    max_ntest = allowed_test_max(n_obs, season_length)
    if max_ntest < 1:
        st.error(
            f"Not enough data for two full cycles in train. "
            f"Need ≥ {2*season_length} observations; you have {n_obs}."
        )
        st.stop()

    Ntest = st.slider(
        "Hold-out size (months)",
        min_value=1, max_value=max_ntest, value=min(6, max_ntest), step=1,
        help=f"Train will keep at least two seasonal cycles (≥ {2*season_length} months).",
    )

    st.caption(f"Data range: **{min_dt.date()}** → **{max_dt.date()}**  |  ~**{years_span}** year(s), n={n_obs}")

    st.markdown("---")
    st.subheader("ETS Parameters (Manual)")
    trend = st.selectbox("Trend", [None, "add", "mul"], format_func=lambda x: "None" if x is None else x)
    seasonal = st.selectbox("Seasonal", [None, "add", "mul"], format_func=lambda x: "None" if x is None else x)
    damped_trend = st.checkbox("Damped trend", value=False)
    use_boxcox = st.selectbox("Box-Cox", [False, True, "log"], index=0)
    init_method = st.selectbox("Initialization", ["estimated", "heuristic"], index=0)

    st.markdown("---")
    horizon = st.slider("Forecast horizon (months)", 1, 24, 12)

# -------------------------- Split & Backtest (Plot 1) ------------------------
y_train = y.iloc[:-Ntest]
y_test  = y.iloc[-Ntest:]

# Extra guard: ensure training still has ≥ 2 cycles
if len(y_train) < 2 * season_length:
    st.error(
        f"Train after split has only {len(y_train)} points; needs ≥ {2*season_length} "
        f"for two full cycles. Reduce hold-out size."
    )
    st.stop()

# Fit on TRAIN and forecast TEST window
try:
    fit_tr = fit_ets(
        y_train,
        trend=trend,
        seasonal=seasonal,
        seasonal_periods=season_length,
        damped_trend=damped_trend,
        use_boxcox=use_boxcox,
        initialization_method=init_method,
    )
    yhat_test = pd.Series(
        fit_tr.forecast(Ntest),
        index=y_test.index,
        name="ETS Forecast (Test)"
    )
except Exception as e:
    st.exception(e)
    st.stop()

# Metrics
mae  = mean_absolute_error(y_test, yhat_test)
mape = np.mean(np.abs((y_test - yhat_test) / np.clip(np.abs(y_test), 1e-12, None))) * 100
sm   = smape(y_test, yhat_test)

col1, col2, col3 = st.columns(3)
col1.metric("MAE", f"{mae:,.0f}")
col2.metric("MAPE", f"{mape:,.2f}%")
col3.metric("sMAPE", f"{sm:,.2f}%")

st.plotly_chart(make_holdout_plot(y_train, y_test, yhat_test), use_container_width=True)

# -------------------------- Forecast Ahead (Plot 2) --------------------------
# Refit on FULL history and forecast `horizon` steps
try:
    fit_full = fit_ets(
        y,
        trend=trend,
        seasonal=seasonal,
        seasonal_periods=season_length,
        damped_trend=damped_trend,
        use_boxcox=use_boxcox,
        initialization_method=init_method,
    )
    fcast = pd.Series(
        fit_full.forecast(horizon),
        index=pd.date_range(y.index[-1] + pd.offsets.MonthBegin(1), periods=horizon, freq="MS"),
        name="ETS Forecast"
    )
except Exception as e:
    st.exception(e)
    st.stop()

# (Optional) naive confidence band using in-sample residual std (not a true PI)
resid = y - fit_full.fittedvalues.reindex(y.index).dropna()
sigma = float(np.nanstd(resid)) if len(resid) else 0.0
conf_low  = fcast - 1.96 * sigma
conf_high = fcast + 1.96 * sigma

st.plotly_chart(make_forecast_plot(y, fcast, conf_low, conf_high), use_container_width=True)

# ------------------------------ Notes panel ----------------------------------
with st.expander("Notes"):
    st.write(
        """
- The hold-out slider is constrained so the training window always contains **≥ 2 seasonal cycles**.
- If your dataset is weekly or daily, set the **Seasonal Periods** accordingly (e.g., 52 for weekly, 7 for daily) —
  but for management rollups we recommend monthly (12).
- Confidence band shown is a **rough residual-std band**, not a formal prediction interval.
- If you later want an **Auto/Stepwise ETS** option, we can add a small grid search toggle without moving plotting back
  into `ets_stepwise.py`.
        """
    )
