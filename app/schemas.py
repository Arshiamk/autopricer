from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class QuoteRequest(BaseModel):
    vehicle_id: Optional[str] = None
    make: str
    model: str
    year: int = Field(..., ge=1990, le=2026)
    mileage: int = Field(..., ge=0)
    fuel_type: str
    channel: str
    damage_flag: bool
    damage_type: Optional[str] = None
    region_id: Optional[str] = None


class QuoteResponse(BaseModel):
    recommended_offer: float
    expected_value: float
    p_win: float
    risk_band: str
    explanation: Dict[str, Any]
