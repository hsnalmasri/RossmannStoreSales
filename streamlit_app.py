# ------------------------- Data loading helpers ------------------------------
import streamlit as st
import pandas as pd

@st.cache_data(show_spinner=False)
def load_monthly_series():
    """
    Loads data via your data loader `load_store_imputed(store_csv, sales_csv)`,
    using URLs stored in Streamlit secrets, then returns a monthly Sales series.
    Secrets required:
      - DATA_URL_SALES
      - DATA_URL_STORE
    """
    # 1) Secrets -> URLs
    try:
        sales_url = st.secrets["DATA_URL_SALES"]
        store_url = st.secrets["DATA_URL_STORE"]
    except Exception:
        st.error(
            "Missing secrets. Please add DATA_URL_SALES and DATA_URL_STORE to Streamlit secrets."
        )
        st.stop()

    # 2) Import your loader
    try:
        from src.rossmann import data_load as DL  # adjust path if needed
    except Exception as e:
        st.exception(e)
        st.error("Could not import src.rossmann.data_load. Check your module path.")
        st.stop()

    if not hasattr(DL, "load_store_imputed"):
        st.error("Expected function `load_store_imputed` not found in data_load.py.")
        st.stop()

    # 3) Call your project loader with URL arguments
    #    Signature inferred from the error: needs 'store_csv' (and likely 'sales_csv')
    try:
        df = DL.load_store_imputed(store_csv=store_url, sales_csv=sales_url)
    except TypeError as e:
        st.exception(e)
        st.error(
            "load_store_imputed signature mismatch. "
            "It must accept named params store_csv= and sales_csv=."
        )
        st.stop()
    except Exception as e:
        st.exception(e)
        st.error("Failed inside load_store_imputed. See exception above.")
        st.stop()

    # 4) Standardize to monthly Sales series
    #    Expect either ['Date','Sales'] columns or Date index + 'Sales' column.
    if "Date" in df.columns:
        s = pd.Series(
            df["Sales"].astype(float).values,
            index=pd.to_datetime(df["Date"]),
            name="Sales",
        ).sort_index()
    else:
        if "Sales" not in df.columns:
            st.error("The returned dataframe must contain a 'Sales' column.")
            st.stop()
        s = df["Sales"].astype(float).copy()
        s.index = pd.to_datetime(s.index)
        s = s.sort_index()

    y = (
        s.resample("MS")  # Month start
         .sum()
         .astype(float)
         .rename("Sales")
    ).dropna()

    if len(y) < 3:
        st.error("Not enough monthly data after resampling.")
        st.stop()

    return y
