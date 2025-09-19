from __future__ import annotations

import streamlit as st

st.set_page_config(page_title="Rossmann Demand Planning", page_icon="🏪", layout="wide")

st.title("Rossmann Store Sales — Demand Planning")
st.write(
    "Use the sidebar to open pages. The **Exponential Smoothing (ETS)** page lets you tune "
    "parameters, run a **backtest** on monthly data, and generate a **forward forecast**."
)

def _safe_page_link():
    """Try page_link; if it fails, try switch_page; else show a plain markdown item."""
    shown = False
    # 1) Try st.page_link (Streamlit ≥ 1.31) — may fail in some environments
    if hasattr(st, "page_link"):
        try:
            st.page_link("pages/1_Exponential_Smoothing.py", label="📈 Exponential Smoothing (ETS)")
            shown = True
        except Exception:
            shown = False
    # 2) Fallback: button + switch_page
    if not shown:
        col1, _ = st.columns([1, 3])
        with col1:
            if st.button("📈 Open Exponential Smoothing (ETS)"):
                if hasattr(st, "switch_page"):
                    try:
                        st.switch_page("pages/1_Exponential_Smoothing.py")
                    except Exception:
                        # Last resort: do nothing; Streamlit will still list the page in its built-in sidebar
                        pass
                # If no switch_page, we just rely on the built-in multi-page sidebar
    # 3) Always render a plain list item so something is visible
    st.markdown("- 📈 Exponential Smoothing (ETS) — also available in the left sidebar pages list.")

# --- Sidebar quick nav ---
with st.sidebar:
    st.header("Pages")
    _safe_page_link()

st.markdown(
    """
### Pages
Below are the available pages in this app:
"""
)
_safe_page_link()
