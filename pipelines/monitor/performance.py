import json
import os
import pickle

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, mean_absolute_error

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")

PRICE_FEATURES = [
    "make",
    "model",
    "fuel_type",
    "body_type",
    "channel",
    "vehicle_age",
    "mileage",
    "damage_severity_score",
    "risk_score",
]

# Retrain trigger: flag if price MAE drifts above this threshold (in GBP).
MAE_ALERT_THRESHOLD = 3000.0


def check_performance():
    """Score the current models against the latest feature snapshot.

    In production this would run against a rolling window of recent enquiries;
    here it runs against the latest generated dataset to demonstrate the
    monitoring loop end-to-end.
    """
    print("Running Performance Monitoring...")

    features_path = os.path.join(ROOT, "data", "features.parquet")
    price_model_path = os.path.join(ROOT, "models", "price_model.pkl")
    conv_model_path = os.path.join(ROOT, "models", "conversion_model.pkl")

    if not os.path.exists(features_path):
        print("Features not found.")
        return
    if not (os.path.exists(price_model_path) and os.path.exists(conv_model_path)):
        print("Models not found. Run `make train` first.")
        return

    df = pd.read_parquet(features_path)
    with open(price_model_path, "rb") as f:
        price_model = pickle.load(f)
    with open(conv_model_path, "rb") as f:
        conversion_model = pickle.load(f)

    # Price model error on observed sales
    won_df = df[df["won"] == 1].copy()
    price_pred = price_model.predict(won_df[PRICE_FEATURES])
    mae = float(mean_absolute_error(won_df["sale_price"], price_pred))

    # Conversion model calibration on all enquiries
    df["won"] = df["won"].fillna(0).astype(int)
    df["offer_to_value_ratio"] = df["offer_price"] / np.maximum(
        price_model.predict(df[PRICE_FEATURES]), 1.0
    )
    conv_features = PRICE_FEATURES + [
        "offer_price",
        "offer_to_value_ratio",
        "month_sin",
        "month_cos",
    ]
    win_prob = conversion_model.predict_proba(df[conv_features])[:, 1]
    brier = float(brier_score_loss(df["won"], win_prob))

    total_profit_actual = float(won_df["gross_margin"].sum())

    report = {
        "price_mae": round(mae, 2),
        "conversion_brier": round(brier, 4),
        "total_profit_actual": total_profit_actual,
        "win_rate": round(float(df["won"].mean()), 3),
        "performance_degraded": bool(mae > MAE_ALERT_THRESHOLD),
    }

    reports_dir = os.path.join(ROOT, "reports")
    os.makedirs(reports_dir, exist_ok=True)

    out_path = os.path.join(reports_dir, "performance_report.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"Performance report generated at {out_path}")


if __name__ == "__main__":
    check_performance()
