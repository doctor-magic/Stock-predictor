from pydantic import BaseModel
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

class ScanRequest(BaseModel):
    market_id: str
    min_confidence: float = 0.65
    top_n: int = 10
