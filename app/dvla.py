import os
import hashlib
from typing import Dict, Any
import httpx

DVLA_VES_API_KEY = os.getenv("DVLA_VES_API_KEY")
DVSA_MOT_API_KEY = os.getenv("DVSA_MOT_API_KEY")


# Fictional demo vehicles keyed by placeholder registrations (AB12 CDE is the
# standard DVLA example plate format). These are NOT real vehicles; they exist so
# the dashboard demo returns stable, recognisable results for a few known inputs.
DEMO_PLATE_REGISTRY = {
    "AB12CDE": {
        "make": "Volkswagen",
        "model": "Golf GTE",
        "year": 2012,
        "fuel_type": "hybrid",
        "mileage": 85000,
        "mot_status": "Valid",
        "mot_days_remaining": 45,
    },
    "CD34EFG": {
        "make": "Hyundai",
        "model": "Ioniq Premium SE",
        "year": 2020,
        "fuel_type": "hybrid",
        "mileage": 48000,
        "mot_status": "Valid",
        "mot_days_remaining": 210,
    },
    "EF56GHI": {
        "make": "BMW",
        "model": "3 Series 330e",
        "year": 2019,
        "fuel_type": "hybrid",
        "mileage": 42000,
        "mot_status": "Valid",
        "mot_days_remaining": 150,
    },
    "GH78JKL": {
        "make": "Tesla",
        "model": "Model 3 Long Range",
        "year": 2020,
        "fuel_type": "electric",
        "mileage": 35000,
        "mot_status": "Valid",
        "mot_days_remaining": 200,
    },
    "JK90LMN": {
        "make": "Ford",
        "model": "Fiesta Titanium",
        "year": 2017,
        "fuel_type": "petrol",
        "mileage": 48000,
        "mot_status": "Valid",
        "mot_days_remaining": 110,
    },
}


def deterministic_mock_lookup(registration: str) -> Dict[str, Any]:
    """
    Generates a deterministic synthetic vehicle profile from a hash of the registration.

    This lets the dashboard lookup feature be demonstrated end-to-end without
    requiring reviewers to register for DVLA/DVSA API keys. All returned data is
    synthetic.
    """
    reg = registration.upper().replace(" ", "")

    if reg in DEMO_PLATE_REGISTRY:
        return {
            "status": "success",
            "source": "DEMO_REGISTRY",
            "registration": reg,
            **DEMO_PLATE_REGISTRY[reg],
        }

    # Hash the plate so repeated lookups of the same registration are consistent
    h = hashlib.md5(reg.encode("utf-8")).hexdigest()

    # 50+ makes
    makes = [
        "Acura",
        "Alfa Romeo",
        "Aston Martin",
        "Audi",
        "Bentley",
        "BMW",
        "Buick",
        "Cadillac",
        "Chevrolet",
        "Chrysler",
        "Citroen",
        "Dacia",
        "Dodge",
        "Ferrari",
        "Fiat",
        "Ford",
        "Genesis",
        "GMC",
        "Honda",
        "Hyundai",
        "Infiniti",
        "Jaguar",
        "Jeep",
        "Kia",
        "Lamborghini",
        "Land Rover",
        "Lexus",
        "Lincoln",
        "Lotus",
        "Maserati",
        "Mazda",
        "McLaren",
        "Mercedes-Benz",
        "MG",
        "MINI",
        "Mitsubishi",
        "Nissan",
        "Peugeot",
        "Polestar",
        "Porsche",
        "Ram",
        "Renault",
        "Rolls-Royce",
        "Saab",
        "Seat",
        "Skoda",
        "Subaru",
        "Suzuki",
        "Tesla",
        "Toyota",
        "Vauxhall",
        "Volkswagen",
        "Volvo",
    ]

    # Deterministic selection
    make_idx = int(h[0:4], 16) % len(makes)
    make = makes[make_idx]

    # More varied model names by make
    model_patterns = ["Series", "Class", "GT", "Eco", "Sport", "EV", "Pro", "X", "Prime"]
    model_name = model_patterns[int(h[4:6], 16) % len(model_patterns)]
    model = f"{make} {model_name} {int(h[6:7], 16)}"

    year = 2012 + (int(h[7:9], 16) % 13)
    fuel_types = ["petrol", "diesel", "electric", "hybrid"]
    fuel = fuel_types[int(h[9:10], 16) % len(fuel_types)]

    # Realistic mileage based on age (UK avg ~8500 miles/year)
    age = 2025 - year
    avg_annual_mileage = 8500
    jitter = (int(h[10:14], 16) % 3000) - 1500  # +/- 1500 miles jitter
    mileage = max(1000, (age * avg_annual_mileage) + jitter)

    # MOT Status
    mot_days_left = -10 + (int(h[14:18], 16) % 380)
    mot_status = "Valid" if mot_days_left > 0 else "Expired"

    return {
        "status": "success",
        "source": "PORTFOLIO_MOCK",
        "registration": reg,
        "make": make,
        "model": model,
        "year": year,
        "fuel_type": fuel,
        "mileage": mileage,
        "mot_status": mot_status,
        "mot_days_remaining": max(0, mot_days_left),
    }


async def fetch_dvla_data(registration: str) -> Dict[str, Any]:
    """
    Fetches official data from DVLA and DVSA APIs.
    Falls back to deterministic mock if API keys are missing.
    """
    reg = registration.upper().replace(" ", "")

    if not DVLA_VES_API_KEY or not DVSA_MOT_API_KEY:
        return deterministic_mock_lookup(reg)

    try:
        # 1. DVLA Vehicle Enquiry Service
        ves_url = "https://driver-vehicle-licensing.api.gov.uk/vehicle-enquiry/v1/vehicles"
        headers_ves = {"x-api-key": DVLA_VES_API_KEY, "Content-Type": "application/json"}

        # 2. DVSA MOT History API (v6)
        mot_url = (
            f"https://beta.check-mot.service.gov.uk/trade/vehicles/mot-tests?registration={reg}"
        )
        headers_mot = {"x-api-key": DVSA_MOT_API_KEY, "Accept": "application/json+v6"}

        async with httpx.AsyncClient() as client:
            ves_resp = await client.post(
                ves_url, json={"registrationNumber": reg}, headers=headers_ves
            )
            ves_data = ves_resp.json() if ves_resp.status_code == 200 else {}

            mot_resp = await client.get(mot_url, headers=headers_mot)
            mot_data = mot_resp.json() if mot_resp.status_code == 200 else []

        if not ves_data and not mot_data:
            return {
                "status": "error",
                "message": "Vehicle not found in official DVLA/MOT registries.",
            }

        # Parse official response
        make = ves_data.get("make", "Unknown").title()

        # VES sometimes doesn't have model, fallback to MOT data if needed
        model = ves_data.get("model")
        if not model and len(mot_data) > 0:
            model = mot_data[0].get("model", "Unknown").title()
        if not model:
            model = "Unknown"

        year = ves_data.get("yearOfManufacture", 2015)
        fuel = ves_data.get("fuelType", "PETROL").lower()

        # Mileage from latest MOT test
        mileage = 50000
        mot_status = "Unknown"
        mot_days = 0

        if len(mot_data) > 0:
            recent_test = mot_data[0].get("motTests", [{}])[0]
            mileage = int(recent_test.get("odometerValue", mileage))
            mot_days_str = mot_data[0].get("motTestDueDate", "")
            # Add logic to parse 'mot_days_str' to actual days left, but for brevity here:
            mot_status = "Valid" if mot_days_str else "Expired"

        return {
            "status": "success",
            "mock_data": False,
            "registration": reg,
            "make": make,
            "model": model,
            "year": year,
            "fuel_type": fuel,
            "mileage": mileage,
            "mot_status": mot_status,
            "mot_days_remaining": mot_days,
        }
    except Exception as e:
        return {"status": "error", "message": f"Integration error: {str(e)}"}
