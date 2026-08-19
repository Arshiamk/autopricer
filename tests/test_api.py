from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert "models" in response.json()


def test_quote_valid():
    payload = {
        "vehicle_id": "V1",
        "make": "Ford",
        "model": "Focus",
        "year": 2019,
        "mileage": 45000,
        "fuel_type": "petrol",
        "channel": "dealer",
        "damage_flag": False,
        "region_id": "R1",
    }
    response = client.post("/quote", json=payload, headers={"X-API-Key": "default-dev-key"})
    assert response.status_code == 200
    data = response.json()
    assert "recommended_offer" in data
    assert "expected_value" in data
    assert "p_win" in data
    assert "risk_band" in data
    assert "explanation" in data


def test_quote_invalid():
    payload = {"make": "Ford"}  # Missing lots of fields
    response = client.post("/quote", json=payload, headers={"X-API-Key": "default-dev-key"})
    assert response.status_code == 422
