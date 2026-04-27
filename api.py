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
    "low52w_dist":  "Distance from 52-week low — 0 = at annual low, 0.5 = 50% above it; low values flag potential bounce setups.",
    "atr_pct":      "ATR(14) as % of price — measures stock volatility regime; high = stressed/hyped market.",
    "ret_3d_atr":   "3-day return in ATR units — how many daily ranges the stock moved in 3 days.",
    "ret_5d_atr":   "5-day return in ATR units — weekly momentum normalised by the stock's own volatility.",
    "ret_10d_atr":  "10-day return in ATR units — two-week drift relative to typical daily range.",
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


# ── Macro Bull Score ─────────────────────────────────────────────────────────

_macro_score_cache: dict = {"ts": 0, "data": None}
_MACRO_SCORE_TTL = 7200  # 2 h


def _bull_score_item(iid, val):
    if val is None:
        return None
    tiers = {
        "VIX":        [(12,80),(16,70),(18,50),(22,20),(25,0),(30,-40),(38,-70),(999,-100)],
        "YIELD_CURVE":[(-1.0,-100),(-0.5,-60),(-0.25,-30),(0,-10),(0.25,20),(0.5,50),(1.0,80),(99,100)],
        "CPI":        [(2.0,80),(2.5,60),(3.0,20),(3.5,0),(4.5,-40),(6.0,-70),(99,-100)],
        "PCE":        [(2.0,80),(2.5,60),(2.8,20),(3.2,-20),(3.8,-60),(99,-90)],
        "FED_FUNDS":  [(2.0,70),(3.0,50),(4.0,20),(4.5,0),(5.0,-20),(5.5,-50),(99,-80)],
        "CORE_CPI":   [(2.0,70),(2.5,50),(3.0,10),(3.5,-20),(4.0,-60),(99,-90)],
        "UNRATE":     [(3.5,30),(4.0,70),(4.5,50),(5.0,10),(5.5,-30),(6.5,-70),(99,-100)],
        "10Y":        [(3.0,60),(3.5,40),(4.0,20),(4.5,0),(5.0,-30),(5.5,-60),(99,-90)],
        "CFNAI":      [(-0.7,-80),(-0.35,-40),(0,-10),(0.2,30),(0.5,70),(99,100)],
        "PAYROLLS":   [(-200,-100),(-50,-60),(0,-20),(75,10),(150,50),(250,80),(999,100)],
        "SENTIMENT":  [(50,-70),(55,-40),(60,-10),(70,20),(75,40),(80,60),(90,80),(999,100)],
        "BTC_PCT_50": [(-20,-80),(-10,-40),(0,-10),(5,40),(15,70),(99,100)],
        "OIL":        [(50,-60),(60,-20),(75,30),(85,10),(95,-30),(110,-60),(999,-90)],
        "GOLD_CHG":   [(-5,30),(-2,15),(2,5),(5,-10),(10,-30),(20,-60),(999,-80)],
    }
    t = tiers.get(iid, [])
    for max_v, score in t:
        if val <= max_v:
            return score
    return t[-1][1] if t else None


@app.get("/api/macro-score")
def get_macro_score():
    now = time.time()
    if _macro_score_cache["data"] and now - _macro_score_cache["ts"] < _MACRO_SCORE_TTL:
        return _macro_score_cache["data"]

    import yfinance as yf

    # FRED dashboard (reuse existing cache if fresh)
    if not (_macro_dash_cache["data"] and now - _macro_dash_cache["ts"] < _MACRO_DASH_TTL):
        try:
            get_macro_dashboard()
        except Exception:
            pass
    dash = _macro_dash_cache.get("data") or {}
    fred_by_id = {ind["id"]: ind.get("current") for ind in dash.get("indicators", [])}

    # VIX from macro strip (reuse existing cache)
    if not (_macro_cache["data"] and now - _macro_cache["ts"] < _MACRO_TTL):
        try:
            get_macro()
        except Exception:
            pass
    vix = (_macro_cache.get("data") or {}).get("vix")

    # Yield curve: 10Y − 2Y
    y10 = fred_by_id.get("10Y")
    y2  = fred_by_id.get("2Y")
    yield_curve = round(y10 - y2, 3) if (y10 is not None and y2 is not None) else None

    # BTC, OIL, GOLD via yfinance
    btc_pct_50 = oil_price = gold_chg = None
    try:
        raw = yf.download("BTC-USD BZ=F GC=F", period="70d", progress=False, auto_adjust=True)
        cl = raw["Close"]
        btc_s = cl["BTC-USD"].dropna()
        if len(btc_s) >= 50:
            ma50 = float(btc_s.iloc[-50:].mean())
            btc_pct_50 = round((float(btc_s.iloc[-1]) - ma50) / ma50 * 100, 2)
        oil_s = cl["BZ=F"].dropna()
        if len(oil_s) >= 1:
            oil_price = round(float(oil_s.iloc[-1]), 2)
        gold_s = cl["GC=F"].dropna()
        if len(gold_s) >= 2:
            n = min(63, len(gold_s) - 1)
            gold_chg = round((float(gold_s.iloc[-1]) / float(gold_s.iloc[-n]) - 1) * 100, 2)
    except Exception:
        pass

    def _fv(v, fmt):
        return "—" if v is None else fmt % v

    cpi  = fred_by_id.get("CPI")
    pce  = fred_by_id.get("PCE")
    ff   = fred_by_id.get("FED_FUNDS")
    cc   = fred_by_id.get("CORE_CPI")
    ur   = fred_by_id.get("UNRATE")
    t10  = fred_by_id.get("10Y")
    cfn  = fred_by_id.get("CFNAI")
    pay  = fred_by_id.get("PAYROLLS")
    sent = fred_by_id.get("SENTIMENT")

    INDICATORS = [
        {"id": "VIX",        "label": "Market Fear Index (VIX)",    "category": "Sentiment",   "weight": 12, "raw": vix,         "value_fmt": _fv(vix,        "%.1f")},
        {"id": "YIELD_CURVE","label": "Yield Curve (10Y−2Y)",  "category": "Rates",       "weight": 12, "raw": yield_curve, "value_fmt": _fv(yield_curve, "%+.2f%%")},
        {"id": "CPI",        "label": "CPI Inflation (YoY)",        "category": "Inflation",   "weight": 12, "raw": cpi,         "value_fmt": _fv(cpi,        "%.1f%%")},
        {"id": "PCE",        "label": "PCE Inflation (YoY)",        "category": "Inflation",   "weight": 9,  "raw": pce,         "value_fmt": _fv(pce,        "%.1f%%")},
        {"id": "FED_FUNDS",  "label": "Fed Funds Rate",             "category": "Rates",       "weight": 9,  "raw": ff,          "value_fmt": _fv(ff,         "%.2f%%")},
        {"id": "CORE_CPI",   "label": "Core CPI (ex Food/Energy)",  "category": "Inflation",   "weight": 7,  "raw": cc,          "value_fmt": _fv(cc,         "%.1f%%")},
        {"id": "UNRATE",     "label": "Unemployment Rate",          "category": "Labor",       "weight": 8,  "raw": ur,          "value_fmt": _fv(ur,         "%.1f%%")},
        {"id": "10Y",        "label": "10Y Treasury Yield",         "category": "Rates",       "weight": 8,  "raw": t10,         "value_fmt": _fv(t10,        "%.2f%%")},
        {"id": "CFNAI",      "label": "Chicago Activity Index",     "category": "Activity",    "weight": 5,  "raw": cfn,         "value_fmt": _fv(cfn,        "%.2f")},
        {"id": "PAYROLLS",   "label": "Non-Farm Payrolls (MoM)",    "category": "Labor",       "weight": 5,  "raw": pay,         "value_fmt": ("—" if pay is None else "%+.0fK" % pay)},
        {"id": "SENTIMENT",  "label": "Consumer Sentiment (UMich)", "category": "Sentiment",   "weight": 5,  "raw": sent,        "value_fmt": _fv(sent,       "%.0f")},
        {"id": "BTC_PCT_50", "label": "Bitcoin vs 50-Day MA",       "category": "Crypto",      "weight": 4,  "raw": btc_pct_50,  "value_fmt": _fv(btc_pct_50, "%+.1f%%")},
        {"id": "OIL",        "label": "Brent Oil Price",            "category": "Commodities", "weight": 2,  "raw": oil_price,   "value_fmt": _fv(oil_price,  "$%.1f")},
        {"id": "GOLD_CHG",   "label": "Gold (3M Change)",           "category": "Commodities", "weight": 2,  "raw": gold_chg,    "value_fmt": _fv(gold_chg,   "%+.1f%%")},
    ]

    scored = []
    weighted_sum = 0
    total_weight = 0
    for m in INDICATORS:
        score = _bull_score_item(m["id"], m["raw"])
        impact = "positive" if (score or 0) >= 20 else "negative" if (score or 0) <= -20 else "neutral"
        scored.append({
            "id": m["id"], "label": m["label"], "category": m["category"],
            "weight": m["weight"], "value_fmt": m["value_fmt"],
            "score": score, "impact": impact,
        })
        if score is not None:
            weighted_sum += score * m["weight"]
            total_weight += m["weight"]

    raw_avg = weighted_sum / total_weight if total_weight else 0
    bull_score = max(0, min(100, round((raw_avg + 100) / 2)))

    if bull_score >= 75:
        regime, regime_label, regime_desc = "strong_bull",   "Strong Bull",   "Strong macro tailwinds — broad-based favorable conditions for equities."
    elif bull_score >= 60:
        regime, regime_label, regime_desc = "moderate_bull", "Moderate Bull", "Moderate tailwinds — most indicators support risk-on positioning."
    elif bull_score >= 45:
        regime, regime_label, regime_desc = "neutral",       "Neutral",       "Mixed signals — proceed selectively with caution."
    elif bull_score >= 30:
        regime, regime_label, regime_desc = "caution",       "Caution",       "Macro headwinds outweigh tailwinds — defensive posture advised."
    else:
        regime, regime_label, regime_desc = "bear",          "Bear Warning",  "Strong macro headwinds — conditions historically unfavorable for equities."

    data = {
        "bull_score": bull_score,
        "regime": regime,
        "regime_label": regime_label,
        "regime_desc": regime_desc,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "indicators": scored,
    }
    _macro_score_cache["ts"] = now
    _macro_score_cache["data"] = data
    return data


# ── Static frontend ──────────────────────────────────────────────────────────

from fastapi.staticfiles import StaticFiles

frontend_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "frontend", "dist")
if os.path.exists(frontend_dir):
    app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")
