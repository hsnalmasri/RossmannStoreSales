from __future__ import annotations
import sys
import importlib
from pathlib import Path
from typing import Dict, Any, Optional

import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# ──────────────────────────────
# Page setup & safe first paint
# ──────────────────────────────
st.set_page_config(page_title="Rossmann Forecasts (Plotly)", layout="wide")
st.title("📈 Rossmann Monthly Forecasts — Plotly Edition")

# ──────────────────────────────
# Robust import harness
# ──────────────────────────────
def import_block():
    try:
        DL = importlib.import_module("rossmann.data_load")
        DP = importlib.import_module("rossmann.data_prep")
        ETS = importlib.import_module("rossmann.ets_stepwise")
        source = "installed package"
    except Exception:
        # Fallback to local ./src as editable layout
        sys.path.append(str(Path(__file__).resolve().parent / "src"))
        DL = importlib.import_module("rossmann.data_load")
        DP = importlib.import_module("rossmann.data_prep")
        ETS = importlib.import_module("rossmann.ets_stepwise")
        source = "local ./src fallback"
    return DL, DP, ETS, source

try:
    DL, DP, ETS, source = import_block()
    st.caption(f"✅ Modules imported from **{source}**")
except Exception as e:
    st.error("Import failed. See details below.")
    st.exception(e)
    st.stop()

# Resolve required callables with explicit names or sensible fallbacks
try:
    load_raw = getattr(DL, "load_store_imputed")  # user confirmed this exists
except AttributeError:
    st.error("`load_store_imputed` not found in `rossmann.data_load`. Available:")
    st.code("\n".join(sorted(k for k in dir(DL) if not k.startswith("_"))))
    st.stop()

# Try common names for monthly converter in data_prep.py
_monthly_candidates = (
    "to_monthly_series",
    "to_monthly_sales",
    "to_monthly",
    "as_monthly_series",
)
_to_monthly = None
for name in _monthly_candidates:
    if hasattr(DP, name):
        _to_monthly = getattr(DP, name)
        st.caption(f"✅ Using monthly converter: `{name}` from `data_prep.py`")
        break
if _to_monthly is None:
    st.error(
        "Could not find a monthly converter in `data_prep.py`.\n"
        + "Tried: " + ", ".join(_monthly_candidates)
    )
    st.code("\n".join(sorted(k for k in dir(DP) if not k.startswith("_"))))
    st.stop()

# Optional: get ETS runner (data-only return expected)
if hasattr(ETS, "fit_and_backtest_ets"):
    fit_and_backtest_ets = getattr(ETS, "fit_and_backtest_ets")
else:
    st.error("`fit_and_backtest_ets` not found in `ets_stepwise.py`.")
    st.code("\n".join(sorted(k for k in dir(ETS) if not k.startswith("_"))))
    st.stop()

# ──────────────────────────────
# Data loading (cached)
# ──────────────────────────────
@st.cache_data(show_spinner=False)
def load_data_cached() -> Any:
    return load_raw()

@st.cache_data(show_spinner=False)
def to_monthly_cached(df_raw: Any) -> pd.Series:
    # Expect the helper to return a monthly pandas Series (DatetimeIndex, freq='MS' ideally)
    y = _to_monthly(df_raw)
    if isinstance(y, pd.DataFrame):
        # If user function returns a DataFrame with a single column, convert to Series
        if y.shape[1] == 1:
            y = y.iloc[:, 0]
        else:
            # Try a common column name
            for col in ("Sales", "sales", "y", "value"):
                if col in y.columns:
                    y = y[col]
                    break
    if not isinstance(y, pd.Series):
        raise TypeError("Monthly converter must return a pandas Series or a 1-col DataFrame.")
    if not isinstance(y.index, pd.DatetimeIndex):
        raise TypeError("Returned monthly series must have a DatetimeIndex.")
    return y.sort_index()

# ──────────────────────────────
# Sidebar controls
# ──────────────────────────────
with st.sidebar:
    st.header("⚙️ Controls")
    show_raw = st.checkbox("Show raw loaded data", value=False)
    show_monthly = st.checkbox("Show monthly head/tail", value=True)
    n_test = st.number_input("Test periods (months)", min_value=0, max_value=36, value=6, step=1)
    horizon = st.number_input("Forecast horizon (months)", min_value=1, max_value=36, value=6, step=1)
    season_m = st.selectbox("Seasonal period (m)", [0, 6, 12], index=2, help="0 = no seasonality")
    manual = st.checkbox("Manual tuning (if supported)", value=False)
    run_btn = st.button("Run Forecast", type="primary")

# ──────────────────────────────
# Load & prep data
# ──────────────────────────────
try:
    with st.spinner("Loading data…"):
        df_raw = load_data_cached()
    if show_raw:
        st.subheader("Raw snapshot")
        st.dataframe(df_raw.head(50), use_container_width=True)
except Exception as e:
    st.error("Failed to load raw data.")
    st.exception(e)
    st.stop()

try:
    with st.spinner("Converting to monthly series…"):
        y = to_monthly_cached(df_raw)
    if show_monthly:
        c1, c2 = st.columns(2)
        with c1:
            st.caption("Monthly — head")
            st.dataframe(y.head(12))
        with c2:
            st.caption("Monthly — tail")
            st.dataframe(y.tail(12))
    st.caption(f"Series span: {y.index.min().date()} → {y.index.max().date()} | N={len(y)}")
except Exception as e:
    st.error("Failed to build monthly series.")
    st.exception(e)
    st.stop()

# ──────────────────────────────
# Modeling wrapper
# ──────────────────────────────
@st.cache_data(show_spinner=False)
def run_ets(y: pd.Series, n_test: int, horizon: int, season_m: int, manual: bool) -> Dict[str, Any]:
    """
    Call into ets_stepwise.fit_and_backtest_ets(y, ...) and return the result dict.
    The underlying function is expected to RETURN DATA ONLY (no plotting).
    We keep the signature flexible to match user implementation.
    """
    # Try a few common calling signatures to avoid breakage.
    # Prefer explicit params if the user's function supports them; otherwise fall back.
    try:
        return fit_and_backtest_ets(
            y,
            n_test=n_test,
            horizon=horizon,
            seasonal=season_m if season_m else None,
            manual=manual,
        )
    except TypeError:
        # Older/alternate implementation, try minimal call
        return fit_and_backtest_ets(y)

# ──────────────────────────────
# Plotting with Plotly GO
# ──────────────────────────────
def make_forecast_figure(res: Dict[str, Any], title: str = "ETS Forecast") -> go.Figure:
    fig = go.Figure()

    def add_line(name: str, series: Optional[pd.Series], mode: str = "lines"):
        if isinstance(series, pd.Series) and len(series) > 0:
            fig.add_trace(
                go.Scatter(x=series.index, y=series.values, name=name, mode=mode)
            )

    # Try common keys
    y_train = res.get("y_train") or res.get("train")
    y_test = res.get("y_test") or res.get("test")
    y_fcst = res.get("y_fcst") or res.get("forecast") or res.get("yhat")
    y_lo = res.get("yhat_lo") or res.get("y_lo")
    y_hi = res.get("yhat_hi") or res.get("y_hi")

    add_line("Train", y_train)
    add_line("Test", y_test)
    add_line("Forecast", y_fcst)

    # Confidence band if available
    if isinstance(y_lo, pd.Series) and isinstance(y_hi, pd.Series):
        # Lower band as a hidden trace to anchor fill area
        fig.add_trace(
            go.Scatter(
                x=y_lo.index,
                y=y_lo.values,
                mode="lines",
                line=dict(width=0),
                showlegend=False,
                hoverinfo="skip",
                name="Lower",
            )
        )
        # Upper band filled to previous trace (the lower)
        fig.add_trace(
            go.Scatter(
                x=y_hi.index,
                y=y_hi.values,
                mode="lines",
                line=dict(width=0),
                fill="tonexty",
                fillcolor="rgba(99,110,250,0.2)",  # Plotly default blue w/ alpha
                name="Confidence band",
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Sales",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=60, r=20, b=20, l=20),
    )
    return fig

# ──────────────────────────────
# Run
# ──────────────────────────────
if run_btn:
    with st.spinner("Fitting ETS and generating forecast…"):
        try:
            res = run_ets(y, n_test=int(n_test), horizon=int(horizon), season_m=int(season_m), manual=bool(manual))
        except Exception as e:
            st.error("Model run failed.")
            st.exception(e)
            st.stop()

    # Diagnostics panel
    meta_keys = [k for k in ("best_cfg", "aic", "bic", "mape", "rmse") if k in res]
    if meta_keys:
        st.subheader("Model summary")
        st.json({k: res[k] for k in meta_keys})

    # Shapes to avoid silent empties
    shapes = {k: getattr(v, "shape", None) for k, v in res.items() if hasattr(v, "shape")}
    if shapes:
        with st.expander("Debug: result shapes"):
            st.json(shapes)

    # Plot
    fig = make_forecast_figure(res, title=f"ETS Forecast — {res.get('best_cfg', 'auto')}")
    st.plotly_chart(fig, use_container_width=True)

    # Optional: tabular outputs
    with st.expander("Forecast table"):
        if isinstance(res.get("y_fcst"), pd.Series):
            st.dataframe(res["y_fcst"].to_frame("Forecast"))
        elif isinstance(res.get("forecast"), pd.Series):
            st.dataframe(res["forecast"].to_frame("Forecast"))
        else:
            st.info("No forecast series found in result.")
else:
    st.info("👈 Configure options in the sidebar and click **Run Forecast**.")
