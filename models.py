from pydantic import BaseModel, Field
from typing import Dict, Optional

class PredictionResult(BaseModel):
    symbol: str
    signal: str
    confidence: float
    precision_score: float
    last_price: float
    last_date: str
    rows_trained: int
    importance: Optional[Dict[str, float]] = None
    options_context: Optional[Dict[str, Optional[float]]] = None
    importance_descriptions: Optional[Dict[str, str]] = None
    options_filtered: bool = False

class ScanRequest(BaseModel):
    # Clamps restored Jul 5 2026 (May-2026 hardening, lost in the Jun 7 refactor)
    market_id: str
    min_confidence: float = Field(0.65, ge=0.0, le=1.0)
    top_n: int = Field(10, ge=1, le=500)
    task_id: Optional[str] = None
    force_refresh: bool = False
    premium_only: bool = False


class PositionOpenRequest(BaseModel):
    # Positions layer (Aug 13 2026) — user-held trades, reads signals only
    symbol: str = Field(..., min_length=1, max_length=12, pattern=r"^[A-Za-z0-9.\-]+$")
    entry_price: float = Field(..., gt=0)
    entry_date: Optional[str] = Field(None, pattern=r"^\d{4}-\d{2}-\d{2}$")
    stop_pct: Optional[float] = Field(None, gt=0, le=50)
    notes: Optional[str] = Field(None, max_length=300)


class PositionCloseRequest(BaseModel):
    exit_price: float = Field(..., gt=0)


class PositionEditRequest(BaseModel):
    # Partial edit of an OPEN position. Every field optional; the endpoint reads
    # model_fields_set so an omitted stop_pct means "leave it" while an explicit
    # null means "clear it".
    entry_price: Optional[float] = Field(None, gt=0)
    entry_date: Optional[str] = Field(None, pattern=r"^\d{4}-\d{2}-\d{2}$")
    stop_pct: Optional[float] = Field(None, gt=0, le=50)
    notes: Optional[str] = Field(None, max_length=300)
