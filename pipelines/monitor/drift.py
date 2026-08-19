import json
import os

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp


def calculate_psi(expected, actual, buckets=10):
    """Calculate Population Stability Index for a single feature."""
    # Break into quantiles based on expected distribution
    breakpoints = np.percentile(expected, np.linspace(0, 100, buckets + 1))
    breakpoints[0] = -np.inf
    breakpoints[-1] = np.inf

    expected_percents = np.histogram(expected, bins=breakpoints)[0] / len(expected)
    actual_percents = np.histogram(actual, bins=breakpoints)[0] / len(actual)

    # Replace 0s with a small value to avoid division by zero
    expected_percents = np.maximum(expected_percents, 0.0001)
    actual_percents = np.maximum(actual_percents, 0.0001)

    psi_value = np.sum(
        (actual_percents - expected_percents) * np.log(actual_percents / expected_percents)
    )
    return psi_value


def check_drift():
    print("Running Drift Detection (PSI + KS test)...")

    # In reality, compare reference dataset with last 30 days
    # We will simulate drift by loading the same feature set and adding noise
    data_dir = os.path.join(os.path.dirname(__file__), "..", "..", "data")
    features_path = os.path.join(data_dir, "features.parquet")

    if not os.path.exists(features_path):
        print("Features not found.")
        return

    df_ref = pd.read_parquet(features_path).sample(frac=0.5, random_state=42)
    df_cur = pd.read_parquet(features_path).sample(frac=0.5, random_state=123)

    # Simulate a drift in mileage
    df_cur["mileage"] = df_cur["mileage"] * 1.5

    drift_report = {"features": {}}
    numeric_features = ["vehicle_age", "mileage", "offer_price", "damage_severity_score"]

    for f in numeric_features:
        if f not in df_ref.columns:
            continue

        ref_data = df_ref[f].dropna().values
        cur_data = df_cur[f].dropna().values

        psi = calculate_psi(ref_data, cur_data)
        ks_stat, p_value = ks_2samp(ref_data, cur_data)

        drift_report["features"][f] = {
            "psi": round(psi, 3),
            "ks_p_value": round(p_value, 4),
            "drift_detected": bool(psi > 0.25 or p_value < 0.05),
        }

    reports_dir = os.path.join(os.path.dirname(__file__), "..", "..", "reports")
    os.makedirs(reports_dir, exist_ok=True)

    out_path = os.path.join(reports_dir, "drift_report.json")
    with open(out_path, "w") as f:
        json.dump(drift_report, f, indent=2)

    print(f"Drift report generated at {out_path}")


if __name__ == "__main__":
    check_drift()
