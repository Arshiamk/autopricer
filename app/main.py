import hashlib
import json
import os
import pickle
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, Dict

import numpy as np
import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security.api_key import APIKeyHeader

from app.dvla import fetch_dvla_data
from app.optimiser import compute_expected_costs, optimise_offer
from app.schemas import QuoteRequest, QuoteResponse

# The synthetic dataset (data/seed/generate.py) and the feature pipeline both anchor
# vehicle age to 2025. Serving must use the same anchor so `vehicle_age` means the
# same thing at train and inference time.
VEHICLE_AGE_REFERENCE_YEAR = 2025

# Model registry
models: Dict[str, Any] = {}


def get_model_path(filename: str) -> str:
    return os.path.join(os.path.dirname(__file__), "..", "models", filename)


def get_file_hash(filepath: str) -> str:
    if not os.path.exists(filepath):
        return "missing"
    hash_md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def load_models() -> None:
    model_source = os.getenv("MODEL_SOURCE", "local")
    if model_source == "mock":
        models["price_model"] = {
            "version_hash": "mock-123",
            "trained_at": "2026-02-20",
            "training_rows": 0,
        }
        models["conversion_model"] = {
            "version_hash": "mock-456",
            "trained_at": "2026-02-20",
            "training_rows": 0,
        }
        return

    try:
        with open(get_model_path("price_model.pkl"), "rb") as f:
            models["price_model"] = pickle.load(f)
        with open(get_model_path("conversion_model.pkl"), "rb") as f:
            models["conversion_model"] = pickle.load(f)
        # Conformal quantile bounds, expressed as ratios of the point prediction
        # (see pipelines/train/train_price_model.py)
        with open(get_model_path("quantile_calibration.json")) as f:
            models["quantile_calibration"] = json.load(f)

        models["meta"] = {
            "price_model": {
                "file_hash": get_file_hash(get_model_path("price_model.pkl")),
                "loaded_at": datetime.now().isoformat(),
            },
            "conversion_model": {
                "file_hash": get_file_hash(get_model_path("conversion_model.pkl")),
                "loaded_at": datetime.now().isoformat(),
            },
        }
    except FileNotFoundError:
        print("Warning: Models not found on disk. Run `make train` first.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_models()
    yield
    models.clear()


app = FastAPI(title="AutoPricer API", version="0.1.0", lifespan=lifespan)

# CORS Middleware
origins = [
    os.getenv("DASHBOARD_ORIGIN", "*"),
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API Key Auth
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=True)


def get_api_key(api_key: str = Security(api_key_header)):
    expected_key = os.getenv("API_KEY", "default-dev-key")
    if api_key != expected_key:
        raise HTTPException(status_code=403, detail="Could not validate credentials")
    return api_key


@app.get("/health")
def health():
    if os.getenv("MODEL_SOURCE", "local") == "mock":
        return {"status": "ok", "models": models}

    return {"status": "ok", "models": models.get("meta", "Not loaded")}


@app.get("/lookup")
async def dvla_lookup(reg: str, api_key: str = Depends(get_api_key)):
    """
    Look up vehicle details by UK registration number.
    Uses the official DVLA/DVSA APIs when keys are configured, otherwise a
    deterministic synthetic mock so the demo works without credentials.
    """
    result = await fetch_dvla_data(reg)
    if result.get("status") == "error":
        raise HTTPException(status_code=404, detail=result.get("message"))
    return result


@app.post("/quote", response_model=QuoteResponse)
def get_quote(req: QuoteRequest, api_key: str = Depends(get_api_key)):
    model_source = os.getenv("MODEL_SOURCE", "local")

    region_risk_score = 0.5
    e_costs = compute_expected_costs(req.damage_flag, req.channel, region_risk_score)

    if model_source == "mock":
        e_sale = 10000.0
        price_q10 = 9000.0

        def mock_p_win(offer: float) -> float:
            return 1.0 / (1.0 + np.exp(-(offer - 9000) / 500.0))

        result = optimise_offer(e_sale, price_q10, e_costs, mock_p_win)
        return QuoteResponse(**result)

    if "price_model" not in models:
        raise HTTPException(status_code=503, detail="Models are not loaded.")

    # Normalise free-text inputs to the casing the models were trained on
    req_make = (req.make or "Unknown").title()
    req_model = req.model or "Unknown"
    req_fuel = (req.fuel_type or "petrol").lower()
    # body_type is not part of the public request schema; default to the most
    # common category in the training data
    req_body = "hatchback"

    damage_map = {"none": 0, "scratches": 1, "dents": 2, "mechanical": 3, "structural": 4}
    damage_severity_score = damage_map.get(req.damage_type or "none", 0)
    month = datetime.now().month
    month_sin = np.sin((month - 1) * (2.0 * np.pi / 12))
    month_cos = np.cos((month - 1) * (2.0 * np.pi / 12))

    features_dict = {
        "make": req_make,
        "model": req_model,
        "fuel_type": req_fuel,
        "body_type": req_body,
        "channel": req.channel,
        "vehicle_age": max(0, VEHICLE_AGE_REFERENCE_YEAR - req.year),
        "mileage": req.mileage,
        "damage_severity_score": damage_severity_score,
        "risk_score": region_risk_score,
        "month_sin": month_sin,
        "month_cos": month_cos,
    }

    df_features = pd.DataFrame([features_dict])

    e_sale = float(models["price_model"].predict(df_features)[0])
    price_q10 = e_sale * models["quantile_calibration"]["q10_ratio"]

    def predict_p_win(offer: float) -> float:
        df_conv = df_features.copy()
        df_conv["offer_price"] = offer
        # Mirrors the engineered feature used in training (train_conversion_model.py):
        # the offer expressed relative to the model's estimated sale value.
        df_conv["offer_to_value_ratio"] = offer / max(e_sale, 1.0)
        return float(models["conversion_model"].predict_proba(df_conv)[0][1])

    result = optimise_offer(e_sale, price_q10, e_costs, predict_p_win)
    return QuoteResponse(**result)
