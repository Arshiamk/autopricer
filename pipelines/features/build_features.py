import os

import numpy as np
import pandas as pd


def build_features():
    print("Building features from raw (or mart) data...")
    data_dir = os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw")
    sales_df = pd.read_csv(os.path.join(data_dir, "sales.csv"))
    enquiries_df = pd.read_csv(os.path.join(data_dir, "enquiries.csv"))
    vehicles_df = pd.read_csv(os.path.join(data_dir, "vehicles.csv"))
    regions_df = pd.read_csv(os.path.join(data_dir, "regions.csv"))

    # In production, this would read from `dbt` mart `mart_training_set`
    df = enquiries_df.merge(sales_df, on="enquiry_id", how="left")
    df = df.merge(vehicles_df, on="vehicle_id")
    df = df.merge(regions_df, on="region_id")

    # 1. Vehicle Age
    df["vehicle_age"] = 2025 - df["year"]

    # 2. Mileage Band
    bins = [0, 30000, 60000, 100000, np.inf]
    labels = ["low", "medium", "high", "very_high"]
    df["mileage_band"] = pd.cut(df["mileage"], bins=bins, labels=labels)

    # 3. Damage Severity Score
    damage_map = {"none": 0, "scratches": 1, "dents": 2, "mechanical": 3, "structural": 4}
    df["damage_severity_score"] = df["damage_type"].map(damage_map).fillna(0).astype(int)

    # 4. Seasonality (Month sin/cos)
    df["enquiry_month"] = pd.to_datetime(df["date_x"]).dt.month
    df["month_sin"] = np.sin((df["enquiry_month"] - 1) * (2.0 * np.pi / 12))
    df["month_cos"] = np.cos((df["enquiry_month"] - 1) * (2.0 * np.pi / 12))

    out_dir = os.path.join(os.path.dirname(__file__), "..", "..", "data")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "features.parquet")
    df.to_parquet(out_path, index=False)
    print(f"Features built and saved to {out_path} ({len(df)} rows)")


if __name__ == "__main__":
    build_features()
