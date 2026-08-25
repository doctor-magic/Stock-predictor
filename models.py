from pydantic import BaseModel, Field
from typing import Any, Dict, Optional

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
    # Minervini Trend Template (Aug 2026) — display only, no gate, no
    # setup_log column. None when the symbol lacks 260 sessions of history.
    trend_template: Optional[Dict[str, Any]] = None
    # Precision alongside its base rate and how often the model fires at all.
    # Display only, like trend_template — no gate reads it.
    signal_quality: Optional[Dict[str, Any]] = None

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
    shares: Optional[float] = Field(None, gt=0)


class PositionCloseRequest(BaseModel):
    exit_price: float = Field(..., gt=0)


class BankScenarioRequest(BaseModel):
    # Yield curve → banks scenario (Aug 2026). Every bound is a clamp, not a
    # suggestion: the impulse response is linear in the shock, so an unclamped
    # shock would render a chart with an absurd axis rather than fail.
    d_r3m_bp: float = Field(0.0, ge=-500, le=500)
    d_slope_bp: float = Field(0.0, ge=-500, le=500)
    horizon: int = Field(12, ge=1, le=40)
    # Quarterly AR decay of the shock itself. 1.0 = permanent; the paper's VAR
    # (their Chart 4) implies roughly 0.66 for the 3m rate.
    persistence: float = Field(1.0, ge=0.0, le=1.0)
    slope_persistence: Optional[float] = Field(None, ge=0.0, le=1.0)
    # "unanticipated" is the paper's own convention and decides the sign on
    # impact — see the timing note in bank_rates.impulse_response.
    timing: str = Field("unanticipated", pattern=r"^(unanticipated|anticipated)$")


class PositionEditRequest(BaseModel):
    # Partial edit of an OPEN position. Every field optional; the endpoint reads
    # model_fields_set so an omitted stop_pct means "leave it" while an explicit
    # null means "clear it".
    entry_price: Optional[float] = Field(None, gt=0)
    entry_date: Optional[str] = Field(None, pattern=r"^\d{4}-\d{2}-\d{2}$")
    stop_pct: Optional[float] = Field(None, gt=0, le=50)
    notes: Optional[str] = Field(None, max_length=300)
    shares: Optional[float] = Field(None, gt=0)
