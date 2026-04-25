from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from typing import List, Dict
import os
import glob
import re
import time

import core_logic
import db
from models import PredictionResult, ScanRequest

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
    "pc_ratio":     "ATM put/call OI ratio (3-strike weighted) — >1 signals hedging pressure.",
    "iv_skew":      "IV skew — 5% OTM put IV minus 5% OTM call IV; positive = fear premium on downside.",
    "volume_shock": "Option turnover ratio — today's option volume / total OI; spike = unusual positioning.",
}

app = FastAPI(title="Stock Predictor Pro API")

# No CORS middleware — frontend is co-hosted on the same origin

TICKER_RE = re.compile(r'^[A-Z0-9.\-]{1,15}$')

_scan_rate_limit: dict[str, float] = {}
SCAN_COOLDOWN = 300  # seconds between force-refresh per IP

@app.on_event("startup")
def startup_event():
    db.init_db()

@app.get("/api/predict/{ticker}", response_model=PredictionResult)
def predict_symbol(ticker: str):
    ticker = ticker.upper().strip()
    if not TICKER_RE.match(ticker):
        raise HTTPException(status_code=400, detail="Invalid ticker format.")
    result = core_logic.get_prediction(ticker)
    if not result:
        raise HTTPException(status_code=404, detail=f"Ticker {ticker} not found.")
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
    if task_id in GLOBAL_RESULTS:
        progress = {**progress, "done": True, "results": GLOBAL_RESULTS.pop(task_id)}
    return progress

@app.post("/api/scan")
async def scan_market(req: ScanRequest, request: Request):
    import threading

    client_ip = request.client.host

    if req.force_refresh:
        now = time.time()
        last = _scan_rate_limit.get(client_ip, 0)
        if now - last < SCAN_COOLDOWN:
            remaining = int(SCAN_COOLDOWN - (now - last))
            raise HTTPException(status_code=429, detail=f"Too many requests. Wait {remaining} seconds before refreshing again.")
        _scan_rate_limit[client_ip] = now

    if not req.force_refresh:
        cached = db.get_latest_scan(req.market_id)
        if cached:
            filtered = [r for r in cached if r['confidence'] >= req.min_confidence][:req.top_n]
            return {"status": "done", "results": filtered}

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
        except Exception:
            GLOBAL_PROGRESS[task_id] = {"current": 0, "total": 1, "message": "Scan failed. Please try again.", "error": True}

    threading.Thread(target=background_scan, daemon=True).start()
    return {"status": "started", "task_id": task_id}


@app.get("/api/recommendations")
def get_recommendations():
    directory = os.path.dirname(os.path.abspath(__file__))
    files = sorted(glob.glob(os.path.join(directory, "stock_recommendations_*.txt")), reverse=True)

    months = {
        "01": "ינואר", "02": "פברואר", "03": "מרץ",    "04": "אפריל",
        "05": "מאי",   "06": "יוני",   "07": "יולי",   "08": "אוגוסט",
        "09": "ספטמבר","10": "אוקטובר","11": "נובמבר", "12": "דצמבר",
    }

    reports = []
    for f in files:
        basename = os.path.basename(f)
        parts = basename.replace("stock_recommendations_", "").replace(".txt", "").split("_")
        if len(parts) == 3:
            d, m, y = parts
            friendly_date = f"{d} ב{months.get(m, m)} {y}"
        else:
            friendly_date = basename

        with open(f, "r", encoding="utf-8") as file:
            content = file.read()

        reports.append({"id": basename, "date": friendly_date, "content": content})

    return reports


from fastapi.staticfiles import StaticFiles

frontend_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "frontend", "dist")
if os.path.exists(frontend_dir):
    app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")
