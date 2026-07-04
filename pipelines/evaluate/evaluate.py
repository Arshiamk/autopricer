import json
import os
import pickle

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, mean_absolute_error, roc_auc_score
from sklearn.model_selection import train_test_split

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


def evaluate():
    """Evaluate the trained models on held-out data and write a metrics report.

    Reproduces the train/test splits used by the training scripts (same
    random_state) so metrics are computed on data the models never saw.
    """
    print("Evaluating models on held-out data...")

    features_path = os.path.join(ROOT, "data", "features.parquet")
    if not os.path.exists(features_path):
        print("Features not built, run build_features.py first!")
        return

    df = pd.read_parquet(features_path)

    with open(os.path.join(ROOT, "models", "price_model.pkl"), "rb") as f:
        price_model = pickle.load(f)
    with open(os.path.join(ROOT, "models", "conversion_model.pkl"), "rb") as f:
        conversion_model = pickle.load(f)
    with open(os.path.join(ROOT, "models", "quantile_calibration.json")) as f:
        qcal = json.load(f)

    # --- Price model (same three-way split as train_price_model.py) ---
    won_df = df[df["won"] == 1.0].copy()
    X = won_df[PRICE_FEATURES]
    y = won_df["sale_price"]
    X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, test_size=0.4, random_state=42)
    _, X_test, _, y_test = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=42)

    y_pred = price_model.predict(X_test)
    price_mae = float(mean_absolute_error(y_test, y_pred))
    q10_cov = float((y_test.to_numpy() < qcal["q10_ratio"] * y_pred).mean())

    # --- Conversion model (same split as train_conversion_model.py) ---
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
    Xc = df[conv_features]
    yc = df["won"]
    _, Xc_test, _, yc_test = train_test_split(Xc, yc, test_size=0.2, random_state=42)
    yc_prob = conversion_model.predict_proba(Xc_test)[:, 1]
    auc = float(roc_auc_score(yc_test, yc_prob))
    brier = float(brier_score_loss(yc_test, yc_prob))

    report = {
        "price_mae": round(price_mae, 2),
        "q10_empirical_coverage": round(q10_cov, 3),
        "conversion_roc_auc": round(auc, 3),
        "conversion_brier": round(brier, 3),
        "test_rows_price": int(len(X_test)),
        "test_rows_conversion": int(len(Xc_test)),
    }

    reports_dir = os.path.join(ROOT, "reports")
    os.makedirs(reports_dir, exist_ok=True)
    with open(os.path.join(reports_dir, "model_evaluation.json"), "w") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))
    print("Wrote reports/model_evaluation.json")


if __name__ == "__main__":
    evaluate()
