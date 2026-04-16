from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict

import core_logic
import db
from models import PredictionResult, ScanRequest

app = FastAPI(title="AZRG Predictor API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Database on startup
@app.on_event("startup")
def startup_event():
    db.init_db()

@app.get("/api/predict/{ticker}", response_model=PredictionResult)
def predict_symbol(ticker: str):
    result = core_logic.get_prediction(ticker.upper())
    if not result:
        raise HTTPException(status_code=404, detail=f"Ticker {ticker} not found or no data available.")
    return result

@app.post("/api/scan", response_model=List[Dict])
def scan_market(req: ScanRequest):
    """
    In a real scenario, this would trigger an async background job downloading all stocks.
    For demonstration, we check the DB or we can run a loop (which would be slow for HTTP request).
    Ideally, we return DB cache. Check `fetch_24h.py` mechanism.
    """
    results = db.get_latest_scan(
        market_id=req.market_id,
        min_confidence=req.min_confidence,
        top_n=req.top_n
    )
    return results

@app.get("/health")
def health_check():
    return {"status": "ok"}
