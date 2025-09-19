from __future__ import annotations

import streamlit as st

st.set_page_config(page_title="Rossmann Demand Planning", page_icon="🏪", layout="wide")

st.title("Rossmann Store Sales — Demand Planning")
st.write(
    "Use the sidebar to open pages. The **Exponential Smoothing (ETS)** page lets you tune "
    "parameters, run a **backtest** on monthly data, and generate a **forward forecast**."
)

st.markdown(
    """
**Pages**
- 📈 Exponential Smoothing (ETS): test vs. forecast on monthly-aggregated Sales.
- (Add more pages later: segmentation overview, promo impact, competition effects, etc.)
"""
)
