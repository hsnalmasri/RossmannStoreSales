from __future__ import annotations

import streamlit as st

st.set_page_config(page_title="Rossmann Demand Planning", page_icon="🏪", layout="wide")

st.title("Rossmann Store Sales — Demand Planning")
st.write(
    "Use the sidebar to switch pages. The **Exponential Smoothing (ETS)** view lets you tune "
    "parameters, run a **backtest** on monthly data, and generate a **forward forecast**."
)

# -------------------------
# Sidebar: radio navigation
# -------------------------
with st.sidebar:
    st.header("Navigation")
    page = st.radio(
        "Select a page",
        options=["Exponential Smoothing (ETS)"],
        index=0,
        label_visibility="collapsed",
    )

# -------------------------
# Simple router
# -------------------------


if page == "Exponential Smoothing (ETS)":
    if hasattr(st, "switch_page"):
        try:
            st.switch_page("pages/1_Exponential_Smoothing.py")
        except Exception:
            st.error("Navigation failed. Open the page from the default sidebar list on the left.")
    else:
        st.info(
            "Your Streamlit version doesn't support `switch_page`. "
            "Please open **pages/1_Exponential_Smoothing.py** from the built-in sidebar pages list."
        )
