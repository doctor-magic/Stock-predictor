from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict
import os
import glob

import core_logic
import db
from models import PredictionResult, ScanRequest

# Master list — single source of truth. Frontend should always prefer this over its local copy.
IMPORTANCE_DESCRIPTIONS: dict[str, str] = {
    "sma200_dist":  "Distance from 200-day SMA — measures long-term trend positioning.",
    "rsi":          "RSI(14) — momentum oscillator; >70 overbought, <30 oversold.",
    "bb_pos":       "Bollinger Band position — 0 = lower band, 1 = upper band.",
    "ema9":         "9-day EMA — short-term price trend anchor.",
    "ema21":        "21-day EMA — crossover with EMA9 is a momentum signal.",
    "ema50":        "50-day EMA — intermediate trend used by institutions.",
    "ema_cross":    "EMA9 > EMA21 flag — 1 = bullish momentum, 0 = bearish.",
    "macd_gap":     "MACD histogram — gap between MACD line and its signal line.",
    "vol_ratio":    "Volume ratio — today's volume vs 20-day avg; >1.5 signals accumulation.",
    "ret_3d":       "3-day return — short-term price momentum.",
    "ret_5d":       "5-day return — weekly price momentum.",
    "ret_10d":      "10-day return — two-week price drift.",
    # Options features
    "pc_ratio":     "ATM put/call OI ratio (3-strike weighted) — >1 signals hedging pressure.",
    "iv_skew":      "IV skew — 5% OTM put IV minus 5% OTM call IV; positive = fear premium on downside.",
    "volume_shock": "Option turnover ratio — today's option volume / total OI; spike = unusual positioning.",
}

app = FastAPI(title="Stock Predictor Pro API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def startup_event():
    db.init_db()

@app.get("/api/predict/{ticker}", response_model=PredictionResult)
def predict_symbol(ticker: str):
    result = core_logic.get_prediction(ticker.upper())
    if not result:
        raise HTTPException(status_code=404, detail=f"Ticker {ticker} not found.")
    # Attach only descriptions for features actually present in this prediction's importance dict.
    # Key names must match exactly — if the model ever renames a feature, the tooltip silently
    # degrades to "Technical Indicator" rather than crashing.
    result["importance_descriptions"] = {
        k: IMPORTANCE_DESCRIPTIONS.get(k, "Technical Indicator")
        for k in (result.get("importance") or {})
    }
    return result

GLOBAL_PROGRESS = {}
GLOBAL_RESULTS = {}

@app.get("/api/scan/progress/{task_id}")
async def get_scan_progress(task_id: str):
    progress = GLOBAL_PROGRESS.get(task_id, {"current": 0, "total": 1, "message": "Initiating connection..."})
    # If results are ready, include them
    if task_id in GLOBAL_RESULTS:
        progress = {**progress, "done": True, "results": GLOBAL_RESULTS.pop(task_id)}
    return progress

@app.post("/api/scan")
async def scan_market(req: ScanRequest):
    import threading

    # ── Cache check: return today's results if available ──
    if not req.force_refresh:
        cached = db.get_latest_scan(req.market_id)
        if cached:
            filtered = [r for r in cached if r['confidence'] >= req.min_confidence][:req.top_n]
            return {"status": "done", "results": filtered}

    # ── Run scan in a background thread so progress polling isn't blocked ──
    task_id = req.task_id or str(id(req))
    GLOBAL_PROGRESS[task_id] = {"current": 0, "total": 1, "message": "Starting scan..."}

    def background_scan():
        def update_progress(current, total, message):
            GLOBAL_PROGRESS[task_id] = {"current": current, "total": total, "message": message}
        try:
            results = core_logic.run_market_scan(req.market_id, progress_callback=update_progress)
            db.save_scan_results(req.market_id, results)
            filtered = [r for r in results if r['confidence'] >= req.min_confidence][:req.top_n]
            GLOBAL_RESULTS[task_id] = filtered
            GLOBAL_PROGRESS[task_id] = {"current": 1, "total": 1, "message": "Done!", "done": True}
        except Exception as e:
            GLOBAL_PROGRESS[task_id] = {"current": 0, "total": 1, "message": f"Error: {e}", "error": True}

    threading.Thread(target=background_scan, daemon=True).start()
    return {"status": "started", "task_id": task_id}

class SaveScanRequest(BaseModel):
    market_id: str
    results: List[Dict]

@app.post("/api/scan/save")
def save_scan(req: SaveScanRequest):
    db.update_scan_results(req.market_id, req.results)
    return {"status": "success"}


@app.get("/api/recommendations")
def get_recommendations():
    # Folder is where the python script runs (Desktop)
    directory = os.path.dirname(os.path.abspath(__file__))
    files = sorted(glob.glob(os.path.join(directory, "stock_recommendations_*.txt")), reverse=True)
    
    reports = []
    # Month mapping for nice display if needed
    months = {
        "01": "ינואר", "02": "פברואר", "03": "מרץ",    "04": "אפריל",
        "05": "מאי",   "06": "יוני",   "07": "יולי",   "08": "אוגוסט",
        "09": "ספטמבר","10": "אוקטובר","11": "נובמבר", "12": "דצמבר",
    }
    
    for f in files:
        basename = os.path.basename(f)
        # Try to parse date from 'stock_recommendations_15_04_2026.txt'
        parts = basename.replace("stock_recommendations_", "").replace(".txt", "").split("_")
        if len(parts) == 3:
            d, m, y = parts
            friendly_date = f"{d} ב{months.get(m, m)} {y}"
        else:
            friendly_date = basename
            
        with open(f, "r", encoding="utf-8") as file:
            content = file.read()
            
        reports.append({
            "id": basename,
            "date": friendly_date,
            "content": content
        })
        
    return reports


from fastapi.staticfiles import StaticFiles

# ── SERVE FRONTEND (If built) ──
frontend_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "frontend", "dist")
if os.path.exists(frontend_dir):
    app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")
