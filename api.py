from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from typing import List, Dict
import os
import glob
import re
import time
import json as _json
import urllib.request as _req
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor

# self-load api_data.env so FRED_API_KEY is available without EnvironmentFile in systemd
_env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "api_data.env")
if os.path.exists(_env_path):
    with open(_env_path) as _ef:
        for _line in _ef:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _, _v = _line.partition("=")
                os.environ.setdefault(_k.strip(), _v.strip())

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
    "vix":          "CBOE VIX — implied volatility of S&P 500; >30 = high fear, model becomes more conservative on BUY signals.",
    "dgs10":        "10-Year Treasury yield — rising yield tightens financial conditions and pressures growth stocks.",
    "t10y2y":       "Yield curve (10Y minus 2Y) — negative = inverted curve, historically precedes slowdowns.",
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


# ── Macro / FRED endpoints ──────────────────────────────────────────────────

FRED_API_KEY = os.getenv("FRED_API_KEY", "")

_MACRO_TTL      = 3600    # 1 h  – MacroPulse strip
_MACRO_DASH_TTL = 21600   # 6 h  – FRED Dashboard

_macro_cache:      dict = {"ts": 0, "data": None}
_macro_dash_cache: dict = {"ts": 0, "data": None}

FRED_INDICATOR_META = [
    {"id": "CPI",       "series": "CPIAUCSL", "label": "CPI Inflation",             "unit": "%",  "transform": "yoy", "good": "down"},
    {"id": "CORE_CPI",  "series": "CPILFESL", "label": "Core CPI (ex Food/Energy)", "unit": "%",  "transform": "yoy", "good": "down"},
    {"id": "PCE",       "series": "PCEPI",    "label": "PCE Inflation",             "unit": "%",  "transform": "yoy", "good": "down"},
    {"id": "UNRATE",    "series": "UNRATE",   "label": "Unemployment Rate",         "unit": "%",  "transform": "raw", "good": "down"},
    {"id": "PAYROLLS",  "series": "PAYEMS",   "label": "Non-Farm Payrolls",         "unit": "K",  "transform": "mom", "good": "up"},
    {"id": "10Y",       "series": "DGS10",    "label": "10Y Treasury Yield",        "unit": "%",  "transform": "raw", "good": "neutral"},
    {"id": "2Y",        "series": "DGS2",     "label": "2Y Treasury Yield",         "unit": "%",  "transform": "raw", "good": "neutral"},
    {"id": "M2",        "series": "M2SL",     "label": "M2 Money Supply",           "unit": "%",  "transform": "yoy", "good": "neutral"},
    {"id": "FED_FUNDS", "series": "FEDFUNDS", "label": "Fed Funds Rate",            "unit": "%",  "transform": "raw", "good": "neutral"},
    {"id": "CFNAI",     "series": "CFNAI",    "label": "Chicago Fed CFNAI",         "unit": "",   "transform": "raw", "good": "up"},
    {"id": "SENTIMENT", "series": "UMCSENT",  "label": "Consumer Sentiment",        "unit": "idx","transform": "raw", "good": "up"},
]


def _fred_obs(series_id: str, limit: int = 14) -> list:
    url = (
        f"https://api.stlouisfed.org/fred/series/observations"
        f"?series_id={series_id}&frequency=m&aggregation_method=eop"
        f"&limit={limit}&sort_order=desc&api_key={FRED_API_KEY}&file_type=json"
    )
    with _req.urlopen(url, timeout=10) as r:
        obs = _json.loads(r.read())["observations"]
    return [o for o in reversed(obs) if o["value"] != "."]


def _build_indicator(meta: dict, obs: list) -> dict:
    base = {k: meta[k] for k in ("id", "label", "unit", "good")}
    if not obs:
        return {**base, "current": None, "delta": None, "trend": "flat", "series": []}

    values = [float(o["value"]) for o in obs]
    transform = meta["transform"]

    if transform == "yoy":
        yoy_vals = [(values[i] / values[i - 12] - 1) * 100 for i in range(12, len(values))]
        series = [{"value": v} for v in yoy_vals[-12:]]
        current = yoy_vals[-1] if yoy_vals else None
        delta = (yoy_vals[-1] - yoy_vals[-2]) if len(yoy_vals) >= 2 else None
    elif transform == "mom":
        mom_vals = [values[i] - values[i - 1] for i in range(1, len(values))]
        series = [{"value": v} for v in mom_vals[-12:]]
        current = mom_vals[-1] if mom_vals else None
        delta = (mom_vals[-1] - mom_vals[-2]) if len(mom_vals) >= 2 else None
    else:
        series = [{"value": v} for v in values[-12:]]
        current = values[-1] if values else None
        delta = (values[-1] - values[-2]) if len(values) >= 2 else None

    if len(series) >= 4:
        last2 = sum(s["value"] for s in series[-2:]) / 2
        prev2 = sum(s["value"] for s in series[-4:-2]) / 2
        thresh = max(abs(prev2) * 0.005, 0.01)
        trend = "up" if last2 - prev2 > thresh else "down" if prev2 - last2 > thresh else "flat"
    else:
        trend = "flat"

    return {**base, "current": current, "delta": delta, "trend": trend, "series": series}


@app.get("/api/macro")
def get_macro():
    now = time.time()
    if _macro_cache["data"] and now - _macro_cache["ts"] < _MACRO_TTL:
        return _macro_cache["data"]

    import yfinance as yf
    try:
        tickers = yf.download("^VIX ^TNX SPY", period="5d", progress=False, auto_adjust=True)
        close = tickers["Close"]

        vix      = float(close["^VIX"].dropna().iloc[-1])
        rate_10y = float(close["^TNX"].dropna().iloc[-1])
        spy_s    = close["SPY"].dropna()
        spy_change = float((spy_s.iloc[-1] / spy_s.iloc[-2] - 1) * 100)

        yield_curve = None
        if FRED_API_KEY:
            try:
                obs2 = _fred_obs("DGS2", limit=2)
                rate_2y = float(obs2[-1]["value"]) if obs2 else None
                yield_curve = round(rate_10y - rate_2y, 2) if rate_2y else None
            except Exception:
                pass

        if vix < 20 and (yield_curve is None or yield_curve > 0):
            regime, label, desc = "risk-on",  "Risk On",  "Low volatility, positive yield curve — favorable conditions."
        elif vix > 30 or (yield_curve is not None and yield_curve < -0.5):
            regime, label, desc = "risk-off", "Risk Off", "High volatility or inverted yield curve — defensive posture."
        else:
            regime, label, desc = "caution",  "Caution",  "Mixed signals — elevated uncertainty, proceed selectively."

        data = {
            "vix": round(vix, 2), "yield_curve": yield_curve,
            "rate_10y": round(rate_10y, 2), "spy_change": round(spy_change, 2),
            "regime": regime, "regime_label": label, "regime_desc": desc,
        }
        _macro_cache["ts"] = now
        _macro_cache["data"] = data
        return data
    except Exception:
        if _macro_cache["data"]:
            return _macro_cache["data"]
        raise HTTPException(status_code=503, detail="Macro data temporarily unavailable.")


@app.get("/api/macro-dashboard")
def get_macro_dashboard():
    if not FRED_API_KEY:
        raise HTTPException(status_code=503, detail="FRED_API_KEY not configured on server.")

    now = time.time()
    if _macro_dash_cache["data"] and now - _macro_dash_cache["ts"] < _MACRO_DASH_TTL:
        return _macro_dash_cache["data"]

    def fetch_one(meta):
        try:
            return _build_indicator(meta, _fred_obs(meta["series"], limit=26))
        except Exception:
            return _build_indicator(meta, [])

    with ThreadPoolExecutor(max_workers=6) as pool:
        indicators = list(pool.map(fetch_one, FRED_INDICATOR_META))

    data = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "indicators": indicators,
    }
    _macro_dash_cache["ts"] = now
    _macro_dash_cache["data"] = data
    return data


# ── Static frontend ──────────────────────────────────────────────────────────

from fastapi.staticfiles import StaticFiles

frontend_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "frontend", "dist")
if os.path.exists(frontend_dir):
    app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")
