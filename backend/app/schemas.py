"""Pydantic request/response schemas for the API."""
from __future__ import annotations

from pydantic import BaseModel, Field


class TextRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Free-text news headline or body")


class SentimentResponse(BaseModel):
    score: float
    label: str
    hits: int


class TickerResponse(BaseModel):
    tickers: list[str]


class EventStudyRequest(BaseModel):
    asset_prices: list[float] = Field(..., min_length=3)
    market_prices: list[float] = Field(..., min_length=3)
    event_index: int = Field(..., ge=0, description="Event index in returns space")
    est_len: int = Field(120, gt=2)
    est_gap: int = Field(10, ge=0)
    pre: int = Field(1, ge=0)
    post: int = Field(5, ge=0)
    ticker: str | None = Field(None, description="Optional label used when persisting")


class EventStudyResponse(BaseModel):
    relative_days: list[int]
    ar: list[float]
    car_path: list[float]
    car: float
    t_stat: float
    p_value: float
    alpha: float
    beta: float
    resid_std: float


class CarValuesRequest(BaseModel):
    car_values: list[float] = Field(..., min_length=1, description="Per-event CARs")


class StandardizedTestRequest(BaseModel):
    cars: list[float] = Field(..., min_length=1, description="Per-event CARs")
    resid_stds: list[float] = Field(..., min_length=1, description="Per-event estimation resid std")
    window_lens: list[int] = Field(..., min_length=1, description="Per-event event-window length")
    est_lens: list[int] = Field(..., min_length=1, description="Per-event estimation-window length")
