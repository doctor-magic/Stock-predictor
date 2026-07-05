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
