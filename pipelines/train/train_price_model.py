import json
import os
import pickle

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor


def train_price_models():
    print("Loading data for price models...")
    features_path = os.path.join(os.path.dirname(__file__), "..", "..", "data", "features.parquet")
    if os.path.exists(features_path):
        df = pd.read_parquet(features_path)
    else:
        print("Features not built, run build_features.py first!")
        return

    # Sale price is only observed for won enquiries
    train_df = df[df["won"] == 1.0].copy()

    features = [
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
    target = "sale_price"

    X = train_df[features]
    y = train_df[target]

    # Three-way split: fit / conformal calibration / test
    X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, test_size=0.4, random_state=42)
    X_calib, X_test, y_calib, y_test = train_test_split(
        X_tmp, y_tmp, test_size=0.5, random_state=42
    )

    categorical_features = ["make", "model", "fuel_type", "body_type", "channel"]
    numeric_features = ["vehicle_age", "mileage", "damage_severity_score", "risk_score"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical_features,
            ),
        ]
    )

    print("Training XGBRegressor point-estimate model...")
    price_model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                XGBRegressor(n_estimators=400, learning_rate=0.08, max_depth=6, random_state=42),
            ),
        ]
    )
    price_model.fit(X_train, y_train)

    y_pred = price_model.predict(X_test)
    print(f"Point Estimate MAE: {mean_absolute_error(y_test, y_pred):.2f}")

    # Quantile bounds via split-conformal calibration.
    # Sale-price noise in this domain is multiplicative (a car worth ~10k varies by
    # hundreds, a car worth ~40k by thousands), so we calibrate the ratio
    # y / prediction on a held-out calibration set and derive q10/q90 bounds by
    # scaling the point prediction. This gives per-vehicle bounds with guaranteed
    # marginal coverage, without fitting separate (and easily underfit) quantile
    # regressors over a wide one-hot feature space.
    print("Calibrating conformal quantile bounds (q10/q90)...")
    calib_ratios = y_calib.to_numpy() / np.maximum(price_model.predict(X_calib), 1.0)
    q10_ratio = float(np.quantile(calib_ratios, 0.10))
    q90_ratio = float(np.quantile(calib_ratios, 0.90))

    q10_coverage = float((y_test.to_numpy() < q10_ratio * y_pred).mean())
    q90_coverage = float((y_test.to_numpy() < q90_ratio * y_pred).mean())
    print(f"q10 ratio: {q10_ratio:.3f} (empirical test coverage {q10_coverage:.3f}, target 0.10)")
    print(f"q90 ratio: {q90_ratio:.3f} (empirical test coverage {q90_coverage:.3f}, target 0.90)")

    model_dir = os.path.join(os.path.dirname(__file__), "..", "..", "models")
    os.makedirs(model_dir, exist_ok=True)

    with open(os.path.join(model_dir, "price_model.pkl"), "wb") as f:
        pickle.dump(price_model, f)
    with open(os.path.join(model_dir, "quantile_calibration.json"), "w") as f:
        json.dump(
            {
                "method": "split-conformal calibration of y/prediction ratios",
                "q10_ratio": q10_ratio,
                "q90_ratio": q90_ratio,
                "q10_test_coverage": q10_coverage,
                "q90_test_coverage": q90_coverage,
                "calibration_rows": int(len(X_calib)),
            },
            f,
            indent=2,
        )

    print(f"Saved price model and quantile calibration to {os.path.abspath(model_dir)}")


if __name__ == "__main__":
    train_price_models()
