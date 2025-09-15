# src/rossmann/run_check.py
from __future__ import annotations
from pathlib import Path
from .paths import RAW
from .data_prep import load_rossmann_data

def main():
    print(f"RAW data folder: {RAW}")

    sales_path = RAW / "SalesData.csv"
    store_path = RAW / "StoreData.csv"

    try:
        df_ts, df_ml = load_rossmann_data(
            sales_csv=sales_path,
            store_csv=store_path,
            keep_closed_days=True,
            build_flat_copy=True,
        )

        # Basic prints
        # df_ml is the flat version with columns
        if df_ml is not None and "Date" in df_ml.columns:
            dmin = df_ml["Date"].min()
            dmax = df_ml["Date"].max()
        else:
            # fall back to index if needed
            dmin = df_ts.index.get_level_values("Date").min()
            dmax = df_ts.index.get_level_values("Date").max()

        print(f"SalesData.csv exists: {sales_path.exists()} | StoreData.csv exists: {store_path.exists()}")
        print("TS (multi-index) shape:", df_ts.shape)
        print("Flat ML shape:", None if df_ml is None else df_ml.shape)
        print("Date range:", dmin, "→", dmax)

        # Light assertions (adjust if your column names differ)
        if df_ml is not None:
            expected = {"Store", "Date", "Sales"}
            missing = expected.difference(df_ml.columns)
            assert not missing, f"Flat ML df missing columns: {missing}"

        print("✓ basic checks passed")

    except Exception as e:
        print("✗ check failed:", e)

if __name__ == "__main__":
    main()
