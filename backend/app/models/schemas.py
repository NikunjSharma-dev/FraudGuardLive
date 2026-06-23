"""Pydantic request/response schemas."""
from datetime import datetime
from typing import Optional, List, Dict, Any
from uuid import UUID

from pydantic import BaseModel, Field, field_validator, ConfigDict


# ─────────────────────────────────────────────────────────────────────────────
# REQUEST SCHEMAS
# ─────────────────────────────────────────────────────────────────────────────

class TransactionSubmitRequest(BaseModel):
    """Payload for submitting a new transaction."""
    account_id: str   = Field(..., description="Unique account identifier")
    amount:     float = Field(..., gt=0, description="Transaction amount in INR")
    lat:        float = Field(..., ge=-90, le=90,     description="Merchant latitude")
    lon:        float = Field(..., ge=-180, le=180,   description="Merchant longitude")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "account_id": "ACC10294",
                "amount": 5000.00,
                "lat": 19.0760,
                "lon": 72.8777,
            }
        }
    )

    @field_validator("amount")
    @classmethod
    def amount_must_be_positive_and_rounded(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("Amount must be strictly positive")
        return round(v, 2)


class OTPVerifyRequest(BaseModel):
    """Payload for Step-Up MFA verification."""
    otp: str = Field(
        ...,
        min_length=6,
        max_length=6,
        pattern=r"^\d{6}$",
        description="Strictly 6-digit One Time Password",
    )

    model_config = ConfigDict(
        json_schema_extra={"example": {"otp": "482910"}}
    )


# ─────────────────────────────────────────────────────────────────────────────
# RESPONSE SCHEMAS
# ─────────────────────────────────────────────────────────────────────────────

class TransactionResponse(BaseModel):
    """Synchronous response returned immediately after submission."""
    transaction_id: str
    status:         str
    risk_score:     Optional[float]            = None
    message:        Optional[str]              = None
    explanation:    Optional[Dict[str, float]] = None  # SHAP feature attributions


class TransactionDetail(BaseModel):
    """Full transaction view — maps directly to SQLAlchemy ORM output."""
    id:           UUID
    account_id:   str
    amount:       float
    status:       str
    is_fraudulent: bool
    risk_score:   Optional[float] = None
    latitude:     Optional[float] = None
    longitude:    Optional[float] = None
    created_at:   datetime

    model_config = ConfigDict(from_attributes=True)


class LedgerSummaryResponse(BaseModel):
    """Aggregated statistics for the Admin Dashboard."""
    total_volume:     float
    fraud_count:      int
    throughput:       float
    transactions:     List[Dict[str, Any]]
    status_breakdown: Dict[str, int]


class VolumeTrendPoint(BaseModel):
    """Single hourly data point for the volume trend chart."""
    hour:        str
    tx_count:    int
    volume:      float
    fraud_count: int


class HealthResponse(BaseModel):
    status:           str
    service:          str
    version:          str
    ml_engine_active: bool
