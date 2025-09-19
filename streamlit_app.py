from __future__ import annotations

import streamlit as st

st.set_page_config(page_title="Rossmann Demand Planning", page_icon="🏪", layout="wide")

st.title("Rossmann Store Sales — Demand Planning")
st.write(
    "Use the sidebar to open pages. The **Exponential Smoothing (ETS)** page lets you tune "
    "parameters, run a **backtest** on monthly data, and generate a **forward forecast**."
)

# --- Sidebar quick nav ---
with st.sidebar:
    st.header("Pages")
    # Prefer st.page_link if available (Streamlit >= 1.31), else fallback.
    if hasattr(st, "page_link"):
        st.page_link("pages/1_Exponential_Smoothing.py", label="📈 Exponential Smoothing (ETS)")
    else:
        st.markdown(
            "- 📈 [Exponential Smoothing (ETS)](pages/1_Exponential_Smoothing.py)",
            unsafe_allow_html=True,
        )

st.markdown(
    """
### Pages
Below are the available pages in this app:

- 📈 **Exponential Smoothing (ETS):** test vs. forecast on monthly-aggregated Sales.

> (More coming soon: segmentation overview, promo impact, competition effects, etc.)
"""
)

# Also show a big button / link in main content for quick access
if hasattr(st, "page_link"):
    st.page_link("pages/1_Exponential_Smoothing.py", label="➡️ Go to **Exponential Smoothing (ETS)**")
else:
    st.markdown(
        '<p><a href="pages/1_Exponential_Smoothing.py">➡️ Go to <b>Exponential Smoothing (ETS)</b></a></p>',
        unsafe_allow_html=True,
    )
