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
from zoneinfo import ZoneInfo
from concurrent.futures import ThreadPoolExecutor, TimeoutError as _FuturesTimeout

# self-load api_data.env so FRED_API_KEY is available without EnvironmentFile in systemd
_env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "api_data.env")
if os.path.exists(_env_path):
    with open(_env_path) as _ef:
        for _line in _ef:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _, _v = _line.partition("=")
                os.environ.setdefault(_k.strip(), _v.strip())

import threading
import core_logic
import db
from models import PredictionResult, ScanRequest

IMPORTANCE_DESCRIPTIONS: dict[str, str] = {
    "sma200_dist":  "Distance from 200-day SMA — measures long-term trend positioning.",
    "rsi":          "RSI(14) — momentum oscillator; >70 overbought, <30 oversold.",
    "bb_pos":       "Bollinger Band position — 0 = lower band, 1 = upper band.",
    "ema9_dist":    "% distance from 9-day EMA — positive = price above short-term trend.",
    "ema21_dist":   "% distance from 21-day EMA — crossover dynamics in normalized units.",
    "ema50_dist":   "% distance from 50-day EMA — intermediate trend positioning, scale-invariant.",
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
TASK_ID_RE = re.compile(r'^[a-zA-Z0-9_-]{1,64}$')

_scan_rate_limit: dict[str, float] = {}
SCAN_COOLDOWN = 300  # seconds between scan starts per IP
_SCAN_SEMAPHORE = threading.Semaphore(3)  # max 3 concurrent full market scans

HEALTH_STATUS: str = "ok"  # "ok" | "degraded"

def _send_telegram_alert(text: str) -> None:
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        print(f"[health] Telegram not configured — alert skipped: {text}")
        return
    try:
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        payload = _json.dumps({"chat_id": chat_id, "text": text}).encode()
        req = _req.Request(url, data=payload, headers={"Content-Type": "application/json"})
        _req.urlopen(req, timeout=5)
    except Exception as e:
        print(f"[health] Telegram alert failed: {e}")

def _run_post_deploy_check() -> None:
    global HEALTH_STATUS
    health = core_logic.check_feature_health("NVDA")
    if "error" in health:
        HEALTH_STATUS = "degraded"
        _send_telegram_alert(f"[Stock Predictor] Deploy check FAILED: {health['error']}")
        return
    nan_features = [k for k, v in health.items() if v is None]
    if nan_features:
        HEALTH_STATUS = "degraded"
        _send_telegram_alert(
            f"[Stock Predictor] Deploy check FAILED — NaN in {nan_features}. "
            f"Macro join broken. All scans will return HOLD."
        )
    else:
        HEALTH_STATUS = "ok"
        _send_telegram_alert(
            f"[Stock Predictor] Deploy OK — "
            f"vix={health['vix']:.1f}, "
            f"rel_spy={health['rel_strength_spy']:.3f}, "
            f"rel_sector={health['rel_strength_sector']:.3f}"
        )

@app.on_event("startup")
def startup_event():
    db.init_db()
    threading.Thread(target=_run_post_deploy_check, daemon=True).start()

@app.get("/api/health")
def health_check():
    return {"status": HEALTH_STATUS}

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
    if not TASK_ID_RE.match(task_id):
        raise HTTPException(status_code=400, detail="Invalid task_id.")
    progress = GLOBAL_PROGRESS.get(task_id, {"current": 0, "total": 1, "message": "Initiating connection..."})
    if task_id in GLOBAL_RESULTS:
        progress = {**progress, "done": True, "results": GLOBAL_RESULTS.pop(task_id)}
        GLOBAL_PROGRESS.pop(task_id, None)  # clean up once results are consumed
    return progress

def _cross_sectional_rank(results: list, top_n: int) -> list:
    """Return top_n BUY signals ranked by confidence × precision, plus all non-BUY results."""
    buys = sorted(
        [r for r in results if r.get("signal") == "BUY"],
        key=lambda x: x["confidence"] * x.get("precision", x.get("precision_score", 0)),
        reverse=True,
    )[:top_n]
    others = [r for r in results if r.get("signal") != "BUY"]
    return buys + others


@app.post("/api/scan")
async def scan_market(req: ScanRequest, request: Request):
    client_ip = request.client.host

    # Premium mode uses its own cache key so it never mixes with full-universe results
    cache_key = "premium" if req.premium_only else req.market_id

    # Serve from cache first — no rate limit needed for cache hits
    if not req.force_refresh:
        cached = db.get_latest_scan(cache_key)
        if cached:
            return {"status": "done", "results": _cross_sectional_rank(cached, req.top_n)}

    # Rate limit ALL background scan starts (force_refresh or cold cache miss)
    now = time.time()
    last = _scan_rate_limit.get(client_ip, 0)
    if now - last < SCAN_COOLDOWN:
        remaining = int(SCAN_COOLDOWN - (now - last))
        raise HTTPException(status_code=429, detail=f"Too many requests. Wait {remaining} seconds before refreshing again.")
    _scan_rate_limit[client_ip] = now

    # Prune stale rate-limit entries to prevent unbounded dict growth
    stale = [ip for ip, ts in list(_scan_rate_limit.items()) if now - ts > SCAN_COOLDOWN * 2]
    for ip in stale:
        _scan_rate_limit.pop(ip, None)

    task_id = req.task_id or str(id(req))
    GLOBAL_PROGRESS[task_id] = {"current": 0, "total": 1, "message": "Starting scan..."}

    def background_scan():
        if not _SCAN_SEMAPHORE.acquire(timeout=5):
            GLOBAL_PROGRESS[task_id] = {"current": 0, "total": 1, "message": "Server busy. Try again shortly.", "error": True}
            return
        try:
            def update_progress(current, total, message):
                GLOBAL_PROGRESS[task_id] = {"current": current, "total": total, "message": message}
            results = core_logic.run_market_scan(req.market_id, progress_callback=update_progress, premium_only=req.premium_only)
            db.save_scan_results(cache_key, results)
            GLOBAL_RESULTS[task_id] = _cross_sectional_rank(results, req.top_n)
            GLOBAL_PROGRESS[task_id] = {"current": 1, "total": 1, "message": "Done!", "done": True}
        except Exception:
            GLOBAL_PROGRESS[task_id] = {"current": 0, "total": 1, "message": "Scan failed. Please try again.", "error": True}
        finally:
            _SCAN_SEMAPHORE.release()

    threading.Thread(target=background_scan, daemon=True).start()
    return {"status": "started", "task_id": task_id}


@app.get("/api/recommendations")
def get_recommendations():
    directory = os.path.dirname(os.path.abspath(__file__))
    files = sorted(glob.glob(os.path.join(directory, "stock_recommendations_*.txt")), key=os.path.getmtime, reverse=True)

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

# ── VIX State Machine ────────────────────────────────────────────────────────

_MACRO_STATE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "macro_state.json")
_macro_state_lock = threading.Lock()

_DOWNGRADE_MAP = {
    "BULL_STRONG": "BULL_WEAK",
    "BULL_WEAK":   "CAUTION",
    "NEUTRAL":     "CAUTION",
}
_GRADUATION = {
    "CAUTION":   {"min_days": 5, "vix_below": 22},
    "NEUTRAL":   {"min_days": 3, "vix_below": 20},
    "BULL_WEAK": {"min_days": 3, "vix_below": 18},
}
# Known macro events 2026 — (date, type, window_before, window_after) in trading days
_ECO_CALENDAR = [
    ("2026-06-11", "CPI",  1, 1),
    ("2026-06-18", "FOMC", 2, 1),
    ("2026-07-02", "NFP",  0, 1),
    ("2026-07-09", "CPI",  1, 1),
    ("2026-07-29", "FOMC", 2, 1),
    ("2026-08-06", "NFP",  0, 1),
    ("2026-08-13", "CPI",  1, 1),
    ("2026-09-03", "NFP",  0, 1),
    ("2026-09-10", "CPI",  1, 1),
    ("2026-09-16", "FOMC", 2, 1),
    ("2026-10-01", "NFP",  0, 1),
    ("2026-10-08", "CPI",  1, 1),
    ("2026-10-28", "FOMC", 2, 1),
    ("2026-11-05", "NFP",  0, 1),
    ("2026-11-12", "CPI",  1, 1),
    ("2026-12-03", "NFP",  0, 1),
    ("2026-12-10", "CPI",  1, 1),
    ("2026-12-16", "FOMC", 2, 1),
]

def _load_macro_state():
    try:
        with open(_MACRO_STATE_FILE) as f:
            d = _json.load(f)
            return d.get("state", "BULL_STRONG"), int(d.get("days", 0))
    except Exception:
        return "BULL_STRONG", 0

def _save_macro_state(state: str, days: int):
    try:
        with open(_MACRO_STATE_FILE, "w") as f:
            _json.dump({"state": state, "days": days,
                        "ts": datetime.now(timezone.utc).isoformat()}, f)
    except Exception:
        pass

def _calc_vix_signals(vix_series):
    """Returns (roc3, roc5, scaled_risk). Floor: ignore spikes when VIX < 14."""
    s = vix_series.dropna()
    if len(s) < 6:
        return 0.0, 0.0, 0.0
    vix_now = float(s.iloc[-1])
    if vix_now < 14:
        return 0.0, 0.0, 0.0
    roc3 = (vix_now - float(s.iloc[-4])) / float(s.iloc[-4]) * 100
    roc5 = (vix_now - float(s.iloc[-6])) / float(s.iloc[-6]) * 100
    scaled = roc3 * (vix_now / 15.0)
    return round(roc3, 2), round(roc5, 2), round(scaled, 2)

def _get_event_ctx():
    from datetime import date as _date
    today = _date.today()
    for (dstr, etype, wb, wa) in _ECO_CALENDAR:
        ev = _date.fromisoformat(dstr)
        offset = (today - ev).days
        if -wb <= offset <= wa:
            return {"is_event": True, "type": etype, "offset": offset}
    return {"is_event": False, "type": None, "offset": None}

def _vix_state_step(vix_series, spy_above_sma200: bool):
    """Asymmetric state machine. Returns (state, days, vix_signals_dict)."""
    s = vix_series.dropna()
    vix_now = float(s.iloc[-1]) if len(s) > 0 else 18.0
    roc3, roc5, scaled = _calc_vix_signals(s)

    with _macro_state_lock:
        state, days = _load_macro_state()

        # Hard circuit breaker → CAUTION from any state
        if vix_now >= 35 or roc5 > 25:
            next_s = "CAUTION"
            next_d = 0 if next_s != state else days + 1
            _save_macro_state(next_s, next_d)
            return next_s, next_d, {"roc3": roc3, "roc5": roc5, "scaled": scaled}

        # Soft downgrade — fast fear spike
        if scaled > 20 or (roc3 > 15 and vix_now > 17):
            if state in _DOWNGRADE_MAP:
                next_s = _DOWNGRADE_MAP[state]
                _save_macro_state(next_s, 0)
                return next_s, 0, {"roc3": roc3, "roc5": roc5, "scaled": scaled}

        # VIX Crush: reduce cooldown the day after a macro event
        crush_bonus = 0
        ev = _get_event_ctx()
        if ev["is_event"] and ev["offset"] == 1 and len(s) >= 2:
            crush = (float(s.iloc[-2]) - vix_now) / float(s.iloc[-2])
            if crush > 0.12:
                crush_bonus = 2

        effective_days = days + 1 + crush_bonus

        # Asymmetric graduation (slow recovery)
        g = _GRADUATION.get(state)
        if g and effective_days >= g["min_days"] and vix_now < g["vix_below"]:
            if state == "BULL_WEAK" and not spy_above_sma200:
                pass  # blocked — SPY still below SMA200
            else:
                order = ["CAUTION", "NEUTRAL", "BULL_WEAK", "BULL_STRONG"]
                next_s = order[order.index(state) + 1]
                _save_macro_state(next_s, 0)
                return next_s, 0, {"roc3": roc3, "roc5": roc5, "scaled": scaled}

        _save_macro_state(state, effective_days)
        return state, effective_days, {"roc3": roc3, "roc5": roc5, "scaled": scaled}

_FRED_DISK_CACHE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fred_cache.json")

def _load_fred_disk_cache():
    try:
        with open(_FRED_DISK_CACHE_PATH) as _f:
            _d = _json.load(_f)
        if _d and _d.get("indicators"):
            _macro_dash_cache["data"] = _d
            _macro_dash_cache["ts"]   = time.time() - _MACRO_DASH_TTL + 3600
    except Exception:
        pass

def _save_fred_disk_cache(data):
    try:
        with open(_FRED_DISK_CACHE_PATH, "w") as _f:
            _json.dump(data, _f)
    except Exception:
        pass

_load_fred_disk_cache()

FRED_INDICATOR_META = [
    {"id": "CPI",       "series": "CPIAUCSL", "label": "CPI Inflation",             "unit": "%",  "transform": "yoy", "good": "down"},
    {"id": "CORE_CPI",  "series": "CPILFESL", "label": "Core CPI (ex Food/Energy)", "unit": "%",  "transform": "yoy", "good": "down"},
    {"id": "PCE",       "series": "PCEPI",    "label": "PCE Inflation",             "unit": "%",  "transform": "yoy", "good": "down"},
    {"id": "UNRATE",    "series": "UNRATE",   "label": "Unemployment Rate",         "unit": "%",  "transform": "raw", "good": "down"},
    {"id": "PAYROLLS",  "series": "PAYEMS",   "label": "Non-Farm Payrolls",         "unit": "K",  "transform": "mom", "good": "up"},
    {"id": "10Y",       "series": "DGS10",    "label": "10Y Treasury Yield",        "unit": "%",  "transform": "raw", "good": "neutral", "daily": True},
    {"id": "2Y",        "series": "DGS2",     "label": "2Y Treasury Yield",         "unit": "%",  "transform": "raw", "good": "neutral", "daily": True},
    {"id": "M2",        "series": "M2SL",     "label": "M2 Money Supply",           "unit": "%",  "transform": "yoy", "good": "neutral"},
    {"id": "FED_FUNDS", "series": "FEDFUNDS", "label": "Fed Funds Rate",            "unit": "%",  "transform": "raw", "good": "neutral"},
    {"id": "CFNAI",     "series": "CFNAI",    "label": "Chicago Fed CFNAI",         "unit": "",   "transform": "raw", "good": "up"},
    {"id": "SENTIMENT", "series": "UMCSENT",  "label": "Consumer Sentiment",        "unit": "idx","transform": "raw", "good": "up"},
]


def _fred_obs(series_id: str, limit: int = 14, daily: bool = False) -> list:
    params = "&frequency=m&aggregation_method=avg" if daily else ""
    url = (
        f"https://api.stlouisfed.org/fred/series/observations"
        f"?series_id={series_id}{params}"
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
        tickers = yf.download("^VIX ^TNX SPY", period="15d", progress=False, auto_adjust=True)
        close = tickers["Close"]

        vix_series = close["^VIX"].dropna()
        vix        = float(vix_series.iloc[-1])
        rate_10y   = float(close["^TNX"].dropna().iloc[-1])
        spy_s      = close["SPY"].dropna()
        spy_change = float((spy_s.iloc[-1] / spy_s.iloc[-2] - 1) * 100)

        roc3, roc5, scaled = _calc_vix_signals(vix_series)

        yield_curve = None
        if FRED_API_KEY:
            try:
                obs2 = _fred_obs("DGS2", limit=2)
                rate_2y = float(obs2[-1]["value"]) if obs2 else None
                yield_curve = round(rate_10y - rate_2y, 2) if rate_2y else None
            except Exception:
                pass

        vix_accel = scaled > 20
        if vix < 20 and not vix_accel and (yield_curve is None or yield_curve > 0):
            regime, label, desc = "risk-on",  "Risk On",  "Low volatility, positive yield curve — favorable conditions."
        elif vix > 30 or roc5 > 25 or (yield_curve is not None and yield_curve < -0.5):
            regime, label, desc = "risk-off", "Risk Off", "High volatility or inverted yield curve — defensive posture."
        elif vix_accel:
            regime, label, desc = "caution",  "Caution",  f"VIX accelerating (+{roc3:.0f}% / 3d) — fear momentum rising."
        else:
            regime, label, desc = "caution",  "Caution",  "Mixed signals — elevated uncertainty, proceed selectively."

        data = {
            "vix": round(vix, 2), "yield_curve": yield_curve,
            "rate_10y": round(rate_10y, 2), "spy_change": round(spy_change, 2),
            "regime": regime, "regime_label": label, "regime_desc": desc,
            "vix_roc3": roc3, "vix_scaled": round(scaled, 1),
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

    indicators = []
    for _meta in FRED_INDICATOR_META:
        try:
            indicators.append(_build_indicator(_meta, _fred_obs(_meta["series"], limit=26, daily=_meta.get("daily", False))))
        except Exception:
            indicators.append(_build_indicator(_meta, []))
        time.sleep(0.5)

    valid = sum(1 for ind in indicators if ind.get("current") is not None)
    if valid < 4 and _macro_dash_cache["data"]:
        return _macro_dash_cache["data"]

    data = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "indicators": indicators,
    }
    _macro_dash_cache["ts"] = now
    _macro_dash_cache["data"] = data
    _save_fred_disk_cache(data)
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
        raw = yf.download("BTC-USD CL=F GC=F", period="70d", progress=False, auto_adjust=True)
        cl = raw["Close"]
        btc_s = cl["BTC-USD"].dropna()
        if len(btc_s) >= 50:
            ma50 = float(btc_s.iloc[-50:].mean())
            btc_pct_50 = round((float(btc_s.iloc[-1]) - ma50) / ma50 * 100, 2)
        oil_s = cl["CL=F"].dropna()
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
        {"id": "OIL",        "label": "WTI Oil Price",              "category": "Commodities", "weight": 2,  "raw": oil_price,   "value_fmt": _fv(oil_price,  "$%.1f")},
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


_strategic_ctx_cache: dict = {"ts": 0, "data": None}
_STRATEGIC_CTX_TTL = 1800  # 30 min — SMA regime changes daily, not intraday


@app.get("/api/strategic-context")
async def strategic_context():
    now = time.time()
    if _strategic_ctx_cache["data"] and now - _strategic_ctx_cache["ts"] < _STRATEGIC_CTX_TTL:
        return _strategic_ctx_cache["data"]
    try:
        import yfinance as yf
        # Use list form to guarantee MultiIndex columns
        spy_raw = yf.download(["SPY"],  period="300d", progress=False, auto_adjust=True)
        vix_raw = yf.download(["^VIX"], period="15d",  progress=False, auto_adjust=True)
        spy     = spy_raw["Close"]["SPY"].dropna()
        vix     = vix_raw["Close"]["^VIX"].dropna()
        sma50   = float(spy.rolling(50).mean().iloc[-1])
        sma200  = float(spy.rolling(200).mean().iloc[-1])
        current = float(spy.iloc[-1])
        ret5d   = float(spy.pct_change(5).iloc[-1])
        vix_val = float(vix.iloc[-1])

        spy_above_sma200 = current > sma200

        # Structural trend (SMA-based — unchanged)
        if current > sma50 and current > sma200:
            sma_trend = "BULL_STRONG"
        elif current >= sma200:
            sma_trend = "BULL_WEAK"
        else:
            sma_trend = "BEAR"

        # VIX-aware state machine overrides bullish signals on fast fear spikes
        sm_trend, _, vix_sig = _vix_state_step(vix, spy_above_sma200)
        trend = "BEAR" if sma_trend == "BEAR" else sm_trend

        data = {
            "spy_trend":  trend,
            "vix":        round(vix_val, 1),
            "spy_ret5d":  round(ret5d * 100, 2),
            "spy_price":  round(current, 2),
            "sma50":      round(sma50, 2),
            "sma200":     round(sma200, 2),
            "vix_roc3":   vix_sig["roc3"],
            "vix_scaled": vix_sig["scaled"],
        }
        _strategic_ctx_cache["ts"] = now
        _strategic_ctx_cache["data"] = data
        return data
    except Exception:
        return _strategic_ctx_cache["data"] or {"spy_trend": None, "vix": None, "spy_ret5d": None}


_NAME_OVERRIDES = {
    "AZRG.TA": "עזריאלי",
}

# ── Volume Leaders ───────────────────────────────────────────────────────────

_volume_leaders_cache: dict = {"ts": 0, "data": None}
_VOLUME_LEADERS_TTL = 1800  # 30 min
_BETA_HIGH_THRESHOLD = 1.5  # 6-month rolling beta above this → suppress ML BUY

_market_context_cache: dict = {"ts": 0, "data": None}
_MARKET_CONTEXT_TTL = 120   # 2 min


def _get_market_context() -> dict:
    now = time.time()
    if _market_context_cache["data"] and now - _market_context_cache["ts"] < _MARKET_CONTEXT_TTL:
        return _market_context_cache["data"]
    try:
        import yfinance as yf
        raw = yf.download(["SPY", "QQQ"], period="1d", interval="1m", progress=False, auto_adjust=True)
        result = {}
        for sym in ["SPY", "QQQ"]:
            try:
                close  = raw["Close"][sym].dropna()
                high   = raw["High"][sym].dropna()
                low    = raw["Low"][sym].dropna()
                volume = raw["Volume"][sym].dropna()
                tp   = (high + low + close) / 3
                vwap = (tp * volume).cumsum() / volume.cumsum()
                price    = float(close.iloc[-1])
                vwap_val = float(vwap.iloc[-1])
                result[sym] = {
                    "price":         round(price, 2),
                    "vwap":          round(vwap_val, 2),
                    "above_vwap":    price > vwap_val,
                    "pct_from_vwap": round((price / vwap_val - 1) * 100, 2),
                }
            except Exception:
                result[sym] = None
        spy_up = result.get("SPY") and result["SPY"]["above_vwap"]
        qqq_up = result.get("QQQ") and result["QQQ"]["above_vwap"]
        both_present = bool(result.get("SPY") and result.get("QQQ"))
        context = {
            "SPY":      result.get("SPY"),
            "QQQ":      result.get("QQQ"),
            "tailwind": bool(spy_up and qqq_up),
            "headwind": bool(both_present and not spy_up and not qqq_up),
        }
        _market_context_cache["ts"] = now
        _market_context_cache["data"] = context
        return context
    except Exception:
        return _market_context_cache["data"] or {"SPY": None, "QQQ": None, "tailwind": None, "headwind": None}


def _compute_momentum(rsi, vol_ratio: float, ret_5d: float) -> str:
    """Short-term signal: RSI + volume surge only. No ML."""
    if rsi is not None and rsi > 75:
        return "OVEREXTENDED"
    if rsi is not None and rsi < 35 and vol_ratio >= 2.0:
        return "WATCH"
    if vol_ratio >= 3.0 and (ret_5d or 0) > 1.0:
        return "SURGING"
    if (ret_5d or 0) < -4.0 and vol_ratio >= 2.0:
        return "SELLING OFF"
    return "NEUTRAL"


def _detect_falling_wedge(highs, lows, closes, volumes) -> dict:
    """Detect a falling wedge on daily OHLCV data.
    Tries windows 30 → 45 → 60 days; returns the first valid match.
    Criteria: both lines descend, upper faster (converging), compression >= 25%,
    >= 3 pivot touches on each line within 4% tolerance, volume declining.
    """
    try:
        from scipy.signal import find_peaks
        import numpy as np

        h_all = np.array(highs,   dtype=float).flatten()
        l_all = np.array(lows,    dtype=float).flatten()
        c_all = np.array(closes,  dtype=float).flatten()
        v_all = np.array(volumes, dtype=float).flatten()
        total = len(c_all)

        # Adaptive compression threshold: shorter windows require less convergence
        min_compression = {30: 0.15, 45: 0.20, 60: 0.25}

        for lookback in (30, 45, 60):
            n = min(lookback, total)
            if n < 25:
                continue

            h = h_all[-n:]; l = l_all[-n:]
            c = c_all[-n:]; v = v_all[-n:]
            x = np.arange(n, dtype=float)

            price_range = h.max() - l.min()
            if price_range <= 0:
                continue

            prominence = price_range * 0.03
            peaks,   _ = find_peaks( h, distance=3, prominence=prominence)
            troughs, _ = find_peaks(-l, distance=3, prominence=prominence)

            # Need >= 3 pivots on each side
            if len(peaks) < 3 or len(troughs) < 3:
                continue

            upper_slope, upper_intercept = np.polyfit(peaks,   h[peaks],  1)
            lower_slope, lower_intercept = np.polyfit(troughs, l[troughs], 1)

            # Both must descend; upper must descend faster (lines converge)
            if upper_slope >= -0.001 or lower_slope >= -0.001:
                continue
            if upper_slope >= lower_slope:
                continue

            # Touch count: at least 3 pivots within 4% of the fitted line
            upper_fitted  = upper_slope * peaks   + upper_intercept
            lower_fitted  = lower_slope * troughs + lower_intercept
            upper_touches = int(np.sum(np.abs(h[peaks]   - upper_fitted) / np.abs(upper_fitted) < 0.04))
            lower_touches = int(np.sum(np.abs(l[troughs] - lower_fitted) / np.abs(lower_fitted) < 0.04))
            if upper_touches < 3 or lower_touches < 3:
                continue

            width_start = upper_intercept - lower_intercept
            width_end   = (upper_slope * (n-1) + upper_intercept) - (lower_slope * (n-1) + lower_intercept)
            if width_start <= 0 or width_end < 0:
                continue

            compression = 1.0 - (width_end / width_start)
            if compression < min_compression[lookback]:
                continue

            half = n // 2
            vol_declining = float(v[half:].mean()) < float(v[:half].mean()) * 0.85

            upper_now = float(upper_slope * (n-1) + upper_intercept)
            breakout  = float(c[-1]) > upper_now

            fresh_breakout = False
            if breakout:
                upper_line = upper_slope * x + upper_intercept
                for i in range(max(0, n - 3), n):
                    prev_below = (i == 0) or (c[i-1] <= upper_line[i-1])
                    if c[i] > upper_line[i] and prev_below:
                        fresh_breakout = True
                        break

            return {
                "detected":       True,
                "breakout":       breakout,
                "fresh_breakout": fresh_breakout,
                "compression":    round(float(compression), 2),
                "vol_declining":  vol_declining,
                "lookback_used":  n,
            }

        return {}
    except Exception:
        return {}


def _classify_regime(highs, lows, closes) -> tuple[str, float]:
    """Two-axis regime: ADX-14 (Wilder) trend strength × ATR-14 vol percentile.
    Returns (regime_str, adx_value), e.g. ('ranging_low_vol', 18.3).
    Requires ≥30 bars; returns ('unknown', 0.0) otherwise.

    Wilder's smoothing: alpha = 1/N  →  out[i] = (out[i-1]*(N-1) + arr[i]) / N
    This differs from standard EMA (alpha = 2/(N+1)) and is the correct formula
    for ADX/ATR as defined by Welles Wilder. Using EMA instead would inflate the
    smoothed values and shift the 25/40 thresholds off their intended meaning.
    ATR percentile uses the 100 most-recent bars (requires ≥6mo download).
    """
    import numpy as np
    highs  = np.asarray(highs,  dtype=float).ravel()
    lows   = np.asarray(lows,   dtype=float).ravel()
    closes = np.asarray(closes, dtype=float).ravel()
    if len(closes) < 30:
        return "unknown", 0.0

    h  = highs[1:]
    l  = lows[1:]
    pc = closes[:-1]
    c  = closes[1:]

    tr   = np.maximum(h - l, np.maximum(np.abs(h - pc), np.abs(l - pc)))
    up   = highs[1:] - highs[:-1]
    down = lows[:-1] - lows[1:]
    pdm  = np.where((up > down) & (up > 0),   up,   0.0)
    ndm  = np.where((down > up) & (down > 0), down, 0.0)

    n = 14
    def _wilder(arr):
        # Wilder's: seed = SMA(first N), then (prev*(N-1) + curr) / N
        out = np.full(len(arr), np.nan)
        if len(arr) < n:
            return out
        out[n - 1] = arr[:n].mean()
        for i in range(n, len(arr)):
            out[i] = (out[i - 1] * (n - 1) + arr[i]) / n
        return out

    atr14 = _wilder(tr)
    safe  = np.where(atr14 > 0, atr14, np.nan)
    pdi14 = 100.0 * _wilder(pdm) / safe
    ndi14 = 100.0 * _wilder(ndm) / safe
    di_s  = pdi14 + ndi14
    dx    = np.where(di_s > 0, 100.0 * np.abs(pdi14 - ndi14) / di_s, 0.0)

    dx_valid = dx[n - 1:]
    if len(dx_valid) < n:
        return "unknown", 0.0
    adx_arr = _wilder(dx_valid)
    cur_adx = adx_arr[-1]
    if np.isnan(cur_adx):
        return "unknown", 0.0

    adx_val = round(float(cur_adx), 1)
    trend = "ranging" if cur_adx < 25 else "weak_trend" if cur_adx < 40 else "strong_trend"

    # ATR percentile: rank current ATR/price against last 100 bars (~5 months)
    atr_norm = atr14 / c
    valid    = atr_norm[~np.isnan(atr_norm)]
    if len(valid) < 10:
        return f"{trend}_unknown", adx_val
    window = valid[-100:]                                    # cap at 100 most-recent bars
    rank   = float(np.sum(window < window[-1])) / len(window)
    vol    = "low_vol" if rank < 0.33 else "med_vol" if rank < 0.67 else "high_vol"

    return f"{trend}_{vol}", adx_val


def _compute_verdict(
    ml_signal: str,
    ml_confidence,
    vol_ratio: float = 1.0,
    rsi=None,
    open_price=None,
    current_price=None,
    above_sma50: bool = True,
) -> str:
    """10-day ML outlook with four momentum overlays:
    1. Anti-chasing guard  — don't buy if already up 3%+ from open
    2. Vol Breakout        — pure volume signal, no ML needed (3x vol + RSI < 65)
    3. Trend filter        — suppress BUY when price < SMA50 (downtrend)
    4. Hybrid threshold    — lower BUY bar to 60% when vol_ratio >= 2x
    """
    # 1. Anti-chasing guard — price already ran, don't chase
    if open_price and current_price and open_price > 0:
        if (current_price - open_price) / open_price >= 0.03:
            return "OVEREXTENDED"

    # 2. Vol Breakout — independent of ML
    if vol_ratio >= 3.0 and rsi is not None and rsi < 65:
        return "VOL BREAKOUT"

    # 3. ML BUY — require trend alignment (price > SMA50), volume >= 1.0x, confidence threshold
    threshold = 0.65 if vol_ratio >= 2.0 else 0.70
    if ml_signal == "BUY" and (ml_confidence or 0) >= threshold and vol_ratio >= 1.0 and above_sma50:
        return "BUY"

    if ml_signal == "SELL":
        return "HOLD"
    if ml_signal in (None, "N/A"):
        return "N/A"
    return "HOLD"


_INTRADAY_DB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "intraday_cache.db")


def _get_tod_rvol_cached(symbol: str, time_slot: str, today_str: str):
    """
    Returns (rvol, quality) using median baseline from intraday_cache.db.
    quality: 'full' (>=10d), 'partial' (3-9d), 'insufficient' (<3d).
    Falls back to (None, 'insufficient') if cache unavailable or too little history.
    """
    try:
        if not os.path.exists(_INTRADAY_DB):
            return None, "insufficient"
        import sqlite3, statistics
        con = sqlite3.connect(_INTRADAY_DB, timeout=3)
        row = con.execute(
            "SELECT volume FROM intraday_bars WHERE symbol=? AND date=? AND time_slot=?",
            (symbol, today_str, time_slot)
        ).fetchone()
        if row is None:
            con.close()
            return None, "insufficient"
        current_vol = row[0]
        hist = con.execute(
            "SELECT volume FROM intraday_bars "
            "WHERE symbol=? AND date<? AND time_slot=? ORDER BY date DESC LIMIT 20",
            (symbol, today_str, time_slot)
        ).fetchall()
        con.close()
        days = len(hist)
        if days < 3:
            return None, "insufficient"
        median_vol = statistics.median(r[0] for r in hist)
        if median_vol <= 0:
            return None, "insufficient"
        rvol = round(current_vol / median_vol, 1)
        return rvol, ("full" if days >= 10 else "partial")
    except Exception:
        return None, "insufficient"


def _get_intraday_signals(tickers: list) -> dict:
    """Batch 5-minute intraday analysis: VWAP, time-of-day RVOL, ORB, setups.
    Falls back to last completed trading day when market is closed.
    """
    import yfinance as yf
    import pandas as pd

    ET = ZoneInfo("America/New_York")
    now_et = datetime.now(ET)

    try:
        raw5m = yf.download(
            tickers, period="10d", interval="5m",
            group_by="ticker", auto_adjust=True, progress=False
        )
    except Exception:
        return {}

    is_multi = len(tickers) > 1
    in_session = (
        now_et.replace(hour=9, minute=30, second=0, microsecond=0)
        <= now_et <=
        now_et.replace(hour=16, minute=0, second=0, microsecond=0)
    )

    out = {}
    for sym in tickers:
        try:
            # group_by='ticker' → raw5m[sym] gives OHLCV DataFrame directly
            df = (raw5m[sym] if is_multi else raw5m)[["Open", "High", "Low", "Close", "Volume"]].copy()

            df.dropna(subset=["Close", "Volume"], inplace=True)
            if df.empty:
                continue

            if df.index.tz is None:
                df.index = df.index.tz_localize("UTC").tz_convert(ET)
            else:
                df.index = df.index.tz_convert(ET)

            today = now_et.date()
            unique_dates = sorted({d.date() for d in df.index})

            if in_session and today in unique_dates:
                target_date = today
                is_live = True
            else:
                completed = [d for d in unique_dates if d < today]
                target_date = completed[-1] if completed else unique_dates[-1]
                is_live = False

            day_df = df[df.index.date == target_date].copy()
            if day_df.empty:
                continue

            # VWAP — resets at 9:30 ET each day
            day_df["tp"] = (day_df["High"] + day_df["Low"] + day_df["Close"]) / 3
            day_df["tp_vol"] = day_df["tp"] * day_df["Volume"]
            day_df["vwap"] = day_df["tp_vol"].cumsum() / day_df["Volume"].cumsum()
            current_vwap  = float(day_df["vwap"].iloc[-1])
            current_price = float(day_df["Close"].iloc[-1])

            # ORB — opening range = candles from 9:30 to 10:00 ET
            orb_end  = day_df.index[0].replace(hour=10, minute=0, second=0, microsecond=0)
            orb_df   = day_df[day_df.index <= orb_end]
            orb_high = float(orb_df["High"].max()) if not orb_df.empty else None
            orb_low  = float(orb_df["Low"].min())  if not orb_df.empty else None
            orb_breakout = bool(orb_high and current_price > orb_high)

            # Find the first candle after ORB period where price crossed above orb_high
            breakout_time = None
            if orb_breakout:
                post_orb = day_df[day_df.index > orb_end]
                first_break = post_orb[post_orb["Close"] > orb_high]
                if not first_break.empty:
                    breakout_time = first_break.index[0].isoformat()

            # Time-of-Day RVOL — prefer SQLite median cache, fall back to 5m-window mean
            ref_df = day_df.iloc[:-1] if (is_live and len(day_df) > 1) else day_df
            rvol = None
            rvol_quality = "insufficient"
            if not ref_df.empty:
                import numpy as np
                ref_time = ref_df.index[-1].time()
                slot_str = ref_time.strftime("%H:%M")
                today_str = str(target_date)
                rvol, rvol_quality = _get_tod_rvol_cached(sym, slot_str, today_str)
                if rvol is None:
                    ref_vol   = float(ref_df["Volume"].iloc[-1])
                    mask      = (np.array(df.index.date) != target_date) & (np.array(df.index.time) == ref_time)
                    hist_same = df[mask]["Volume"]
                    if len(hist_same) >= 3:
                        avg_vol = float(hist_same.mean())
                        if avg_vol > 0:
                            rvol = round(ref_vol / avg_vol, 1)
                            rvol_quality = "legacy"

            # VWAP Bounce — prev candle touched VWAP from below, curr candle green + higher vol
            vwap_bounce = False
            if len(day_df) >= 2:
                prev      = day_df.iloc[-2]
                curr      = day_df.iloc[-1]
                prev_vwap = float(day_df["vwap"].iloc[-2])
                vwap_bounce = (
                    float(prev["Low"]) <= prev_vwap <= float(prev["Close"])
                    and float(curr["Close"]) > float(curr["Open"])
                    and float(curr["Volume"]) > float(prev["Volume"])
                )


            # Base detection: no new low + range compression (Reversion Hunter)
            base_forming = None
            if len(day_df) >= 10:
                last5_lows   = day_df["Low"].iloc[-5:].values
                prior5_lows  = day_df["Low"].iloc[-10:-5].values
                last5_range  = float((day_df["High"].iloc[-5:] - day_df["Low"].iloc[-5:]).mean())
                prior5_range = float((day_df["High"].iloc[-10:-5] - day_df["Low"].iloc[-10:-5]).mean())
                no_new_low       = float(last5_lows.min()) >= float(prior5_lows.min()) * 0.998
                range_compressed = prior5_range > 0 and last5_range < prior5_range * 0.75
                base_forming = bool(no_new_low and range_compressed)

            # Setup priority: ORB BREAKOUT > LIQUID SURGE > VWAP BOUNCE
            setup = None
            if orb_breakout and (rvol or 0) >= 2.0:
                setup = "ORB BREAKOUT"
            elif (rvol or 0) >= 3.0 and current_price > current_vwap:
                setup = "LIQUID SURGE"
            elif vwap_bounce:
                setup = "VWAP BOUNCE"

            out[sym] = {
                "vwap":         round(current_vwap, 2),
                "rvol":         rvol,
                "rvol_quality": rvol_quality,
                "orb_high":     round(orb_high, 2) if orb_high else None,
                "orb_low":      round(orb_low,  2) if orb_low  else None,
                "orb_breakout": orb_breakout,
                "above_vwap":   current_price > current_vwap,
                "vwap_bounce":  vwap_bounce,
                "setup":          setup,
                "breakout_time":  breakout_time,
                "is_live":        is_live,
                "analysis_date":  str(target_date),
                "base_forming":   base_forming,
            }
        except Exception:
            pass

    return out


@app.get("/api/volume-leaders")
def get_volume_leaders(min_market_cap: int = 200_000_000, force: bool = False):
    now = time.time()
    if not force and _volume_leaders_cache["data"] and now - _volume_leaders_cache["ts"] < _VOLUME_LEADERS_TTL:
        return _volume_leaders_cache["data"]

    try:
        url = (
            "https://query1.finance.yahoo.com/v1/finance/screener/predefined/saved"
            "?formatted=false&scrIds=most_actives&count=50&corsDomain=finance.yahoo.com"
        )
        headers = {"User-Agent": "Mozilla/5.0 (compatible; StockPredictor/1.0)"}
        request = _req.Request(url, headers=headers)
        with _req.urlopen(request, timeout=15) as resp:
            raw = _json.loads(resp.read())
        quotes = raw["finance"]["result"][0]["quotes"]
    except Exception as e:
        if _volume_leaders_cache["data"]:
            return _volume_leaders_cache["data"]  # serve stale cache rather than failing
        raise HTTPException(status_code=502, detail=f"Failed to fetch volume leaders: {e}")

    _24h_ago = now - 86400
    filtered = [
        q for q in quotes
        if q.get("marketCap", 0) >= min_market_cap
        and q.get("regularMarketVolume", 0) > 0
        and (q.get("regularMarketTime") or 0) >= _24h_ago
        and q.get("averageDailyVolume3Month", 0) * q.get("regularMarketPrice", 0) >= 2_000_000
    ][:20]
    if not filtered:
        return _volume_leaders_cache["data"] or []

    sp500_map = {r["symbol"]: r for r in db.get_latest_scan("sp500")}
    nasdaq_map = {r["symbol"]: r for r in db.get_latest_scan("nasdaq100")}

    # On-demand ML for stocks not covered by the daily pre-scan
    uncached = [q["symbol"] for q in filtered
                if q["symbol"] not in sp500_map and q["symbol"] not in nasdaq_map]
    live_predictions: dict = {}
    if uncached:
        def _predict_live(sym):
            try:
                result = core_logic.get_prediction(sym, light_mode=True)
                if result and result.get("signal") not in ("EXCLUDED", None):
                    return sym, {"signal": result["signal"], "confidence": result["confidence"]}
            except Exception:
                pass
            return sym, None

        with ThreadPoolExecutor(max_workers=5) as executor:
            try:
                for sym, pred in executor.map(_predict_live, uncached, timeout=25):
                    if pred:
                        live_predictions[sym] = pred
            except _FuturesTimeout:
                pass  # return partial results; remaining stocks get N/A

    import yfinance as yf
    import numpy as np
    import pandas as pd
    tickers = [q["symbol"] for q in filtered]
    hist     = yf.download(tickers, period="6mo", interval="1d", progress=False, auto_adjust=True)
    intraday = _get_intraday_signals(tickers)

    # SPY returns for rolling beta — download once, reuse per symbol
    spy_returns = None
    try:
        _spy_raw = yf.download("SPY", period="6mo", interval="1d", progress=False, auto_adjust=True)
        if isinstance(_spy_raw.columns, pd.MultiIndex):
            _spy_raw.columns = _spy_raw.columns.get_level_values(0)
        _spy_close = _spy_raw["Close"].dropna()
        _spy_close.index = pd.to_datetime(_spy_close.index).tz_localize(None)
        spy_returns = _spy_close.pct_change().dropna()
    except Exception:
        pass

    results = []
    for quote in filtered:
        sym = quote["symbol"]
        ml = sp500_map.get(sym) or nasdaq_map.get(sym)
        ml_live = False
        if not ml and sym in live_predictions:
            ml = live_predictions[sym]
            ml_live = True
        ml_signal = ml["signal"] if ml else "N/A"
        ml_conf = ml["confidence"] if ml else None

        vol_today = quote.get("regularMarketVolume", 0)
        vol_avg3m = quote.get("averageDailyVolume3Month", 0)
        vol_ratio = round(vol_today / vol_avg3m, 1) if vol_avg3m > 0 else 1.0

        rsi = None
        ret_5d = None
        sma50  = None
        wedge  = {}
        regime  = "unknown"
        adx_val = 0.0
        try:
            multi = len(tickers) > 1
            closes  = hist["Close"][sym].dropna()  if multi else hist["Close"].dropna()
            highs   = hist["High"][sym].dropna()   if multi else hist["High"].dropna()
            lows    = hist["Low"][sym].dropna()    if multi else hist["Low"].dropna()
            volumes = hist["Volume"][sym].dropna() if multi else hist["Volume"].dropna()
            if len(closes) >= 15:
                delta = closes.diff()
                gain = delta.clip(lower=0).rolling(14).mean()
                loss = (-delta.clip(upper=0)).rolling(14).mean()
                rs = gain.iloc[-1] / loss.iloc[-1]
                rsi = round(float(100 - 100 / (1 + rs)), 1)
            if len(closes) >= 6:
                ret_5d = round(float((closes.iloc[-1] / closes.iloc[-6] - 1) * 100), 1)
            if len(closes) >= 50:
                sma50 = round(float(closes.iloc[-50:].mean()), 2)
            regime, adx_val = _classify_regime(highs.values, lows.values, closes.values)
            wedge = _detect_falling_wedge(highs.values, lows.values, closes.values, volumes.values)
        except Exception:
            pass

        # 6-month rolling beta vs SPY
        beta = None
        if spy_returns is not None and len(closes) >= 60:
            try:
                stock_ret = closes.pct_change().dropna()
                stock_ret.index = pd.to_datetime(stock_ret.index).tz_localize(None)
                s, m = stock_ret.align(spy_returns, join="inner")
                if len(s) >= 60:
                    cov = float(np.cov(s, m)[0, 1])
                    var = float(np.var(m, ddof=1))
                    beta = round(cov / var, 2) if var > 0 else None
            except Exception:
                pass

        current_price = quote.get("regularMarketPrice")
        above_sma50 = (current_price is not None and sma50 is not None and current_price > sma50)

        momentum = _compute_momentum(rsi, vol_ratio, ret_5d or 0.0)
        verdict  = _compute_verdict(
            ml_signal, ml_conf,
            vol_ratio=vol_ratio,
            rsi=rsi,
            open_price=quote.get("regularMarketOpen"),
            current_price=current_price,
            above_sma50=above_sma50,
        )
        # Beta gate: suppress ML BUY on high-beta stocks (momentum chasers, not mean-reversion)
        beta_blocked = False
        if beta is not None and beta > _BETA_HIGH_THRESHOLD and verdict == "BUY":
            verdict = "HIGH-BETA"
            beta_blocked = True

        intra = intraday.get(sym, {})
        results.append({
            "symbol": sym,
            "name": _NAME_OVERRIDES.get(sym, quote.get("shortName", sym)),
            "price": quote.get("regularMarketPrice"),
            "change_pct": quote.get("regularMarketChangePercent"),
            "volume": vol_today,
            "vol_ratio": vol_ratio,
            "ret_5d": ret_5d,
            "rsi": rsi,
            "ml_signal": ml_signal,
            "ml_confidence": round(ml_conf * 100, 1) if ml_conf else None,
            "ml_live": ml_live,
            "momentum": momentum,
            "verdict": verdict,
            "vwap":          intra.get("vwap"),
            "rvol":          intra.get("rvol"),
            "rvol_quality":  intra.get("rvol_quality", "legacy"),
            "orb_high":      intra.get("orb_high"),
            "orb_low":       intra.get("orb_low"),
            "orb_breakout":  intra.get("orb_breakout"),
            "above_vwap":    intra.get("above_vwap"),
            "vwap_bounce":   intra.get("vwap_bounce"),
            "setup":          intra.get("setup"),
            "breakout_time":  intra.get("breakout_time"),
            "is_live":        intra.get("is_live", False),
            "analysis_date": intra.get("analysis_date"),
            "wedge":          wedge.get("detected", False),
            "wedge_breakout": wedge.get("breakout", False),
            "wedge_fresh":    wedge.get("fresh_breakout", False),
            "wedge_compression": wedge.get("compression"),
            "wedge_vol_declining": wedge.get("vol_declining", False),
            "regime": regime,
            "adx":    adx_val,
            "beta":         beta,
            "beta_blocked": beta_blocked,
        })
        # Log all non-trivial verdicts for outcome tracking (not just BUY)
        if verdict not in ("HOLD", "N/A"):
            _setup_log_event("volume_leaders", results[-1])

    results.sort(key=lambda x: x["volume"] or 0, reverse=True)

    def _clean(obj):
        if isinstance(obj, float):
            import math
            return None if (math.isnan(obj) or math.isinf(obj)) else obj
        if isinstance(obj, dict):
            return {k: _clean(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_clean(v) for v in obj]
        return obj

    payload = _clean({
        "results": results,
        "market_context": _get_market_context(),
        "fetched_at": datetime.fromtimestamp(now, tz=timezone.utc).isoformat(),
    })
    _volume_leaders_cache["ts"] = now
    _volume_leaders_cache["data"] = payload
    return payload


_WEDGE_CACHE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "wedge_cache.json")

@app.get("/api/wedge-scan")
def get_wedge_scan():
    if not os.path.exists(_WEDGE_CACHE_PATH):
        return {"scan_date": None, "scan_ts": None, "results": []}
    try:
        with open(_WEDGE_CACHE_PATH) as f:
            return _json.load(f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read wedge cache: {e}")


@app.get("/api/wedge-live")
def get_wedge_live():
    """Return today's intraday % change for every symbol in the wedge cache."""
    if not os.path.exists(_WEDGE_CACHE_PATH):
        return {}
    try:
        with open(_WEDGE_CACHE_PATH) as f:
            cache = _json.load(f)
        symbols = [r["symbol"] for r in cache.get("results", [])]
        if not symbols:
            return {}

        import yfinance as yf
        df = yf.download(
            symbols, period="2d", interval="1d",
            group_by="ticker", auto_adjust=True,
            progress=False, threads=True,
        )

        result = {}
        for sym in symbols:
            try:
                closes = (df["Close"] if len(symbols) == 1 else df[sym]["Close"]).dropna()
                if len(closes) < 2:
                    continue
                prev  = float(closes.iloc[-2])
                curr  = float(closes.iloc[-1])
                result[sym] = {
                    "price":      round(curr, 2),
                    "change_pct": round((curr / prev - 1) * 100, 2),
                }
            except Exception:
                pass
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Earnings Calendar ────────────────────────────────────────────────────────

_earnings_cache: dict[str, dict] = {}  # symbol → {"result": dict|None, "ts": float}
_EARNINGS_CACHE_TTL = 6 * 3600  # 6 hours


def _get_next_earnings(symbol: str) -> dict | None:
    """Return next upcoming earnings for a symbol, or None if not found/past."""
    try:
        import yfinance as yf
        info = yf.Ticker(symbol).info
        ts = info.get("earningsTimestamp")
        if ts:
            dt = datetime.fromtimestamp(ts, tz=timezone.utc).date()
            today = datetime.now(timezone.utc).date()
            if dt >= today:
                return {"date": str(dt), "days_until": (dt - today).days}
    except Exception:
        pass
    return None


class EarningsRequest(BaseModel):
    symbols: List[str]


@app.post("/api/earnings-calendar")
def get_earnings_calendar(req: EarningsRequest):
    symbols = [s.upper() for s in req.symbols if TICKER_RE.match(s.upper())][:300]
    if not symbols:
        return {}

    now = time.time()
    missing = [s for s in symbols
               if s not in _earnings_cache
               or now - _earnings_cache[s]["ts"] > _EARNINGS_CACHE_TTL]

    if missing:
        with ThreadPoolExecutor(max_workers=10) as ex:
            futures = {ex.submit(_get_next_earnings, sym): sym for sym in missing}
            for fut, sym in futures.items():
                try:
                    result = fut.result(timeout=8)
                except Exception:
                    result = None
                _earnings_cache[sym] = {"result": result, "ts": now}

    return {
        s: _earnings_cache[s]["result"]
        for s in symbols
        if _earnings_cache.get(s, {}).get("result") is not None
    }



# ── Reversion Leaders (Day Losers / Mean-Reversion Engine) ───────────────────

_reversion_cache: dict = {"ts": 0, "data": None}
_REVERSION_TTL = 900  # 15 min


@app.get("/api/reversion-leaders")
def get_reversion_leaders(min_market_cap: int = 500_000_000, force: bool = False):
    now = time.time()
    if not force and _reversion_cache["data"] and now - _reversion_cache["ts"] < _REVERSION_TTL:
        return _reversion_cache["data"]

    try:
        url = (
            "https://query1.finance.yahoo.com/v1/finance/screener/predefined/saved"
            "?formatted=false&scrIds=day_losers&count=50&corsDomain=finance.yahoo.com"
        )
        headers = {"User-Agent": "Mozilla/5.0 (compatible; StockPredictor/1.0)"}
        request = _req.Request(url, headers=headers)
        with _req.urlopen(request, timeout=15) as resp:
            raw = _json.loads(resp.read())
        quotes = raw["finance"]["result"][0]["quotes"]
    except Exception as e:
        if _reversion_cache["data"]:
            return _reversion_cache["data"]
        raise HTTPException(status_code=502, detail=f"Failed to fetch day losers: {e}")

    _24h_ago = now - 86400
    filtered = [
        q for q in quotes
        if q.get("marketCap", 0) >= min_market_cap
        and q.get("regularMarketVolume", 0) >= 2_000_000
        and (q.get("regularMarketTime") or 0) >= _24h_ago
        and q.get("regularMarketChangePercent", 0) <= -5.0
    ][:25]

    if not filtered:
        return _reversion_cache["data"] or {"results": [], "market_context": _get_market_context(), "fetched_at": ""}

    sp500_map  = {r["symbol"]: r for r in db.get_latest_scan("sp500")}
    nasdaq_map = {r["symbol"]: r for r in db.get_latest_scan("nasdaq100")}

    uncached = [q["symbol"] for q in filtered
                if q["symbol"] not in sp500_map and q["symbol"] not in nasdaq_map]
    live_predictions: dict = {}
    if uncached:
        def _predict_reversion(sym):
            try:
                result = core_logic.get_prediction(sym, light_mode=True)
                if result and result.get("signal") not in ("EXCLUDED", None):
                    return sym, {"signal": result["signal"], "confidence": result["confidence"]}
            except Exception:
                pass
            return sym, None

        with ThreadPoolExecutor(max_workers=5) as executor:
            try:
                for sym, pred in executor.map(_predict_reversion, uncached, timeout=30):
                    if pred:
                        live_predictions[sym] = pred
            except _FuturesTimeout:
                pass

    import yfinance as yf
    import numpy as np
    import pandas as pd
    tickers = [q["symbol"] for q in filtered]
    hist     = yf.download(tickers, period="6mo", interval="1d", progress=False, auto_adjust=True)
    intraday = _get_intraday_signals(tickers)

    results = []
    for quote in filtered:
        sym      = quote["symbol"]
        ml       = sp500_map.get(sym) or nasdaq_map.get(sym)
        ml_live  = False
        if not ml and sym in live_predictions:
            ml      = live_predictions[sym]
            ml_live = True
        ml_signal = ml["signal"]     if ml else "N/A"
        ml_conf   = ml["confidence"] if ml else None

        rsi     = None
        regime  = "unknown"
        adx_val = 0.0
        try:
            multi  = len(tickers) > 1
            closes = hist["Close"][sym].dropna() if multi else hist["Close"].dropna()
            highs  = hist["High"][sym].dropna()  if multi else hist["High"].dropna()
            lows   = hist["Low"][sym].dropna()   if multi else hist["Low"].dropna()
            if len(closes) >= 15:
                delta = closes.diff()
                gain  = delta.clip(lower=0).rolling(14).mean()
                loss  = (-delta.clip(upper=0)).rolling(14).mean()
                rs    = gain.iloc[-1] / loss.iloc[-1]
                rsi   = round(float(100 - 100 / (1 + rs)), 1)
            regime, adx_val = _classify_regime(highs.values, lows.values, closes.values)
        except Exception:
            pass

        intra        = intraday.get(sym, {})
        price        = quote.get("regularMarketPrice")
        vwap         = intra.get("vwap")
        vwap_gap_pct = None
        if price and vwap and vwap > 0:
            vwap_gap_pct = round((price - vwap) / vwap * 100, 2)

        is_oversold = rsi is not None and rsi < 35
        below_vwap  = vwap_gap_pct is not None and vwap_gap_pct < -2.0
        is_buy      = ml_signal == "BUY"

        if is_buy and is_oversold and below_vwap:
            reversion_verdict = "DEEP BUY"
        elif is_buy and (is_oversold or below_vwap):
            reversion_verdict = "POTENTIAL BOUNCE"
        elif is_oversold and below_vwap:
            reversion_verdict = "OVERSOLD"
        else:
            reversion_verdict = "WATCH"

        # Downgrade to FALLING KNIFE if no base detected (explicit False only)
        if reversion_verdict != "WATCH" and intra.get("base_forming") is False:
            reversion_verdict = "FALLING KNIFE"

        results.append({
            "symbol":            sym,
            "name":              _NAME_OVERRIDES.get(sym, quote.get("shortName", sym)),
            "price":             price,
            "change_pct":        quote.get("regularMarketChangePercent"),
            "volume":            quote.get("regularMarketVolume", 0),
            "market_cap":        quote.get("marketCap"),
            "rsi":               rsi,
            "ml_signal":         ml_signal,
            "ml_confidence":     round(ml_conf * 100, 1) if ml_conf else None,
            "ml_live":           ml_live,
            "vwap":              vwap,
            "vwap_gap_pct":      vwap_gap_pct,
            "above_vwap":        intra.get("above_vwap"),
            "is_live":           intra.get("is_live", False),
            "rvol":               intra.get("rvol"),
            "rvol_quality":      intra.get("rvol_quality", "insufficient"),
            "rvol_alert":        bool((intra.get("rvol") or 0) > 5.0),
            "reversion_verdict": reversion_verdict,
            "regime":            regime,
            "adx":               adx_val,
            "base_forming":      intra.get("base_forming"),
        })

        if reversion_verdict in ("DEEP BUY", "POTENTIAL BOUNCE"):
            _setup_log_event("reversion_hunter", results[-1])

    _vord = {"DEEP BUY": 0, "POTENTIAL BOUNCE": 1, "OVERSOLD": 2, "FALLING KNIFE": 3, "WATCH": 4}
    results.sort(key=lambda x: (
        _vord.get(x["reversion_verdict"], 9),
        x["vwap_gap_pct"] if x["vwap_gap_pct"] is not None else 0,
    ))

    def _clean_rev(obj):
        if isinstance(obj, float):
            import math
            return None if (math.isnan(obj) or math.isinf(obj)) else obj
        if isinstance(obj, dict):
            return {k: _clean_rev(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_clean_rev(v) for v in obj]
        return obj

    payload = _clean_rev({
        "results":        results,
        "market_context": _get_market_context(),
        "fetched_at":     datetime.fromtimestamp(now, tz=timezone.utc).isoformat(),
    })
    _reversion_cache["ts"]   = now
    _reversion_cache["data"] = payload
    return payload



# -- Falling Knife Logger -------------------------------------------------------

import sqlite3 as _sqlite3

_FK_LOG_DB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "falling_knife_log.db")

def _fk_db_init():
    con = _sqlite3.connect(_FK_LOG_DB)
    con.execute("""CREATE TABLE IF NOT EXISTS fk_events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        sym TEXT, date TEXT, price REAL, change_pct REAL,
        rsi REAL, rvol REAL, vwap_gap_pct REAL,
        close_next REAL, ph_return REAL, resolved INTEGER DEFAULT 0
    )""")
    con.execute("""CREATE TABLE IF NOT EXISTS fk_milestones (key TEXT PRIMARY KEY, value TEXT)""")
    con.commit(); con.close()

try:
    _fk_db_init()
except Exception:
    pass


def _fk_log_event(sym, price, change_pct, rsi, rvol, vwap_gap_pct):
    try:
        from datetime import date as _fk_date
        today = _fk_date.today().isoformat()
        con = _sqlite3.connect(_FK_LOG_DB)
        existing = con.execute("SELECT id FROM fk_events WHERE symbol=? AND date=?", (sym, today)).fetchone()
        if not existing:
            con.execute(
                "INSERT INTO fk_events (symbol, date, classify_ts, price, change_pct, rsi, rvol, vwap_gap_pct) VALUES (?,?,?,?,?,?,?,?)",
                (sym, today, __import__("datetime").datetime.utcnow().isoformat(), price, change_pct, rsi, rvol, vwap_gap_pct)
            )
            con.commit()
        con.close()
    except Exception:
        pass


def _fk_resolve_yesterday():
    try:
        import yfinance as _yf_fkr
        from datetime import date as _fkr_date, timedelta as _td
        con = _sqlite3.connect(_FK_LOG_DB)
        rows = con.execute("SELECT id, symbol, date FROM fk_events WHERE resolved=0").fetchall()
        today = _fkr_date.today()
        for row_id, sym, date_str in rows:
            try:
                event_date = _fkr_date.fromisoformat(date_str)
                if (today - event_date).days < 1:
                    continue
                hist = _yf_fkr.download(sym, start=date_str,
                                         end=str(event_date + _td(days=5)),
                                         interval="1d", auto_adjust=True, progress=False)
                if hist.empty or len(hist) < 2:
                    continue
                closes_arr = hist["Close"].values.flatten()
                entry_price = float(closes_arr[0])
                close_next  = float(closes_arr[1]) if len(closes_arr) > 1 else None
                ph_return   = round((close_next / entry_price - 1) * 100, 2) if close_next else None
                con.execute("UPDATE fk_events SET price_close=?, ph_return=?, resolved=1 WHERE id=?",
                            (close_next, ph_return, row_id))
            except Exception:
                pass
        con.commit()
        total_resolved = con.execute("SELECT COUNT(*) FROM fk_events WHERE resolved=1").fetchone()[0]
        alerted = con.execute("SELECT value FROM fk_milestones WHERE key='n30_alerted'").fetchone()
        if total_resolved >= 30 and not alerted:
            try:
                import requests as _rq
                tg_token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
                tg_chat  = os.environ.get("TELEGRAM_CHAT_ID", "")
                if tg_token and tg_chat:
                    _rq.post(f"https://api.telegram.org/bot{tg_token}/sendMessage",
                             json={"chat_id": tg_chat, "text": f"Falling Knife milestone: {total_resolved} resolved events!"}, timeout=5)
            except Exception:
                pass
            con.execute("INSERT OR IGNORE INTO fk_milestones (key, ts) VALUES ('n30_alerted', CURRENT_TIMESTAMP)")
            con.commit()
        con.close()
    except Exception:
        pass


@app.get("/api/falling-knife-stats")
def get_falling_knife_stats():
    _fk_resolve_yesterday()
    try:
        con = _sqlite3.connect(_FK_LOG_DB)
        rows = con.execute(
            "SELECT id,symbol,date,price,change_pct,rsi,rvol,vwap_gap_pct,price_close,ph_return,resolved "
            "FROM fk_events ORDER BY date DESC LIMIT 100"
        ).fetchall()
        cols = ["id","sym","date","price","change_pct","rsi","rvol","vwap_gap_pct","close_next","ph_return","resolved"]
        events   = [dict(zip(cols, row)) for row in rows]
        resolved = [e for e in events if e["resolved"]]
        mean_ph  = round(sum(e["ph_return"] for e in resolved if e["ph_return"] is not None) / len(resolved), 2) if resolved else None
        con.close()
        return {"events": events, "total": len(events), "resolved": len(resolved), "mean_ph_return": mean_ph}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



# ── Setup Outcome Logger ────────────────────────────────────────────────────
# Tracks Volume Leaders + Reversion Hunter signals to measure gate effectiveness.
# Pattern mirrors falling_knife_log.db — one row per symbol per day per source.

_SETUP_LOG_DB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "setup_log.db")

def _setup_db_init():
    con = _sqlite3.connect(_SETUP_LOG_DB)
    con.execute("""CREATE TABLE IF NOT EXISTS setup_log (
        id            INTEGER PRIMARY KEY AUTOINCREMENT,
        source        TEXT,    -- 'volume_leaders' | 'reversion_hunter'
        symbol        TEXT,
        date          TEXT,    -- YYYY-MM-DD
        log_ts        TEXT,
        price         REAL,
        verdict       TEXT,    -- final verdict (BUY / HIGH-BETA / OVEREXTENDED / VOL BREAKOUT / HOLD)
        ml_signal     TEXT,
        ml_confidence REAL,
        vol_ratio     REAL,
        rsi           REAL,
        beta          REAL,
        beta_blocked  INTEGER DEFAULT 0,
        above_sma50   INTEGER DEFAULT 1,
        regime        TEXT,
        reversion_verdict TEXT,
        vwap_gap_pct  REAL,
        rvol_val      REAL,
        rvol_alert    INTEGER DEFAULT 0,
        -- resolution
        resolved      INTEGER DEFAULT 0,
        close_1d      REAL,
        close_5d      REAL,
        ret_1d        REAL,
        ret_5d        REAL
    )""")
    con.commit(); con.close()

try:
    _setup_db_init()
except Exception:
    pass


def _setup_log_event(source: str, row: dict):
    """Log a signal for outcome tracking. One log per source+symbol+day."""
    try:
        from datetime import date as _d
        today = _d.today().isoformat()
        con = _sqlite3.connect(_SETUP_LOG_DB)
        exists = con.execute(
            "SELECT id FROM setup_log WHERE source=? AND symbol=? AND date=?",
            (source, row.get("symbol"), today)
        ).fetchone()
        if not exists:
            con.execute("""
                INSERT INTO setup_log (
                    source, symbol, date, log_ts, price,
                    verdict, ml_signal, ml_confidence,
                    vol_ratio, rsi, beta, beta_blocked, above_sma50,
                    regime, reversion_verdict, vwap_gap_pct, rvol_val, rvol_alert
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, (
                source, row.get("symbol"), today,
                datetime.utcnow().isoformat(),
                row.get("price"),
                row.get("verdict"),
                row.get("ml_signal"),
                row.get("ml_confidence"),
                row.get("vol_ratio"),
                row.get("rsi"),
                row.get("beta"),
                1 if row.get("beta_blocked") else 0,
                1 if row.get("above_sma50", True) else 0,
                row.get("regime"),
                row.get("reversion_verdict"),
                row.get("vwap_gap_pct"),
                row.get("rvol"),
                1 if row.get("rvol_alert") else 0,
            ))
            con.commit()
        con.close()
    except Exception:
        pass


def _setup_resolve():
    """Fetch 1d and 5d forward returns for unresolved setup_log rows."""
    try:
        import yfinance as _yf_sr
        from datetime import date as _d, timedelta as _td
        con = _sqlite3.connect(_SETUP_LOG_DB)
        rows = con.execute(
            "SELECT id, symbol, date, price FROM setup_log WHERE resolved=0"
        ).fetchall()
        today = _d.today()
        for row_id, sym, date_str, entry_price in rows:
            try:
                event_date = _d.fromisoformat(date_str)
                if (today - event_date).days < 1:
                    continue
                hist = _yf_sr.download(
                    sym,
                    start=date_str,
                    end=str(event_date + _td(days=10)),
                    interval="1d", auto_adjust=True, progress=False
                )
                if hist.empty or len(hist) < 2 or entry_price is None:
                    continue
                closes = hist["Close"].values.flatten()
                c1 = float(closes[1]) if len(closes) > 1 else None
                c5 = float(closes[5]) if len(closes) > 5 else None
                r1 = round((c1 / entry_price - 1) * 100, 2) if c1 else None
                r5 = round((c5 / entry_price - 1) * 100, 2) if c5 else None
                resolved = 1 if (today - event_date).days >= 5 else 0
                con.execute(
                    "UPDATE setup_log SET close_1d=?, close_5d=?, ret_1d=?, ret_5d=?, resolved=? WHERE id=?",
                    (c1, c5, r1, r5, resolved, row_id)
                )
            except Exception:
                pass
        con.commit()
        con.close()
    except Exception:
        pass


@app.get("/api/setup-stats")
def get_setup_stats():
    _setup_resolve()
    try:
        con = _sqlite3.connect(_SETUP_LOG_DB)
        rows = con.execute("""
            SELECT source, verdict, beta_blocked, COUNT(*) as n,
                   AVG(ret_1d) as mean_1d, AVG(ret_5d) as mean_5d,
                   SUM(CASE WHEN ret_5d > 0 THEN 1 ELSE 0 END) as wins_5d
            FROM setup_log WHERE resolved=1
            GROUP BY source, verdict, beta_blocked
            ORDER BY source, mean_5d DESC
        """).fetchall()
        cols = ["source", "verdict", "beta_blocked", "n", "mean_1d", "mean_5d", "wins_5d"]
        breakdown = [dict(zip(cols, r)) for r in rows]
        for b in breakdown:
            b["win_rate_5d"] = round(b["wins_5d"] / b["n"] * 100, 1) if b["n"] else None
            b["mean_1d"] = round(b["mean_1d"], 2) if b["mean_1d"] is not None else None
            b["mean_5d"] = round(b["mean_5d"], 2) if b["mean_5d"] is not None else None

        total = con.execute("SELECT COUNT(*) FROM setup_log").fetchone()[0]
        resolved = con.execute("SELECT COUNT(*) FROM setup_log WHERE resolved=1").fetchone()[0]
        con.close()
        return {"total_logged": total, "resolved": resolved, "breakdown": breakdown}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Gainers / Momentum Hunter ─────────────────────────────────────────────────

_gainers_cache: dict       = {"ts": 0, "data": None}
_GAINERS_TTL               = 300          # 5 min
_gainers_daily_cache: dict = {}           # sym → {date, sma200, nearest_resist, atr14, …}
_vaccel_cache: dict        = {}           # sym → {ts, vaccel}
_VACCEL_TTL                = 300          # 5 min


def _get_overhead_supply(sym: str, price: float) -> dict:
    """Lazy per-symbol daily cache (1 trading day TTL).
    Returns overhead supply data: nearest resistance, ATR-14, overhead_blocked flag."""
    from datetime import date as _date
    today_str = _date.today().isoformat()
    cached = _gainers_daily_cache.get(sym)
    if cached and cached.get("date") == today_str:
        return cached
    try:
        import yfinance as _yf2
        import numpy as _np2
        hist = _yf2.download(sym, period="6mo", interval="1d", auto_adjust=True, progress=False)
        if hist.empty or len(hist) < 20:
            return {}
        closes = _np2.asarray(hist["Close"].values, dtype=float).ravel()
        highs  = _np2.asarray(hist["High"].values,  dtype=float).ravel()
        lows   = _np2.asarray(hist["Low"].values,   dtype=float).ravel()
        n      = len(closes)

        sma200 = float(_np2.mean(closes[-200:])) if n >= 200 else float(_np2.mean(closes))

        def _wilder_g(arr, period):
            out = _np2.full(len(arr), _np2.nan)
            if len(arr) < period:
                return out
            out[period - 1] = _np2.mean(arr[:period])
            for i in range(period, len(arr)):
                out[i] = (out[i - 1] * (period - 1) + arr[i]) / period
            return out

        trs = _np2.maximum(
            highs[1:] - lows[1:],
            _np2.maximum(
                _np2.abs(highs[1:] - closes[:-1]),
                _np2.abs(lows[1:]  - closes[:-1]),
            ),
        )
        atr14_arr = _wilder_g(trs, 14)
        atr14 = float(atr14_arr[-1]) if not _np2.isnan(atr14_arr[-1]) else None

        pivot_highs = []
        window = min(60, n - 2)
        for i in range(1, window + 1):
            idx = n - 1 - i
            if idx > 0 and highs[idx] > highs[idx - 1] and highs[idx] > highs[idx + 1]:
                if highs[idx] > price * 1.001:
                    pivot_highs.append(float(highs[idx]))

        candidates = [r for r in ([sma200] + pivot_highs) if r > price * 1.001]
        nearest_resist = min(candidates) if candidates else None

        overhead_blocked   = False
        dist_to_resist_pct = None
        if nearest_resist and atr14 and atr14 > 0:
            dist_abs           = nearest_resist - price
            dist_to_resist_pct = round(dist_abs / price * 100, 2)
            overhead_blocked   = dist_abs < 0.5 * atr14

        result = {
            "date":               today_str,
            "sma200":             round(sma200, 2),
            "nearest_resist":     round(nearest_resist, 2) if nearest_resist else None,
            "atr14":              round(atr14, 2) if atr14 else None,
            "dist_to_resist_pct": dist_to_resist_pct,
            "overhead_blocked":   overhead_blocked,
        }
        _gainers_daily_cache[sym] = result
        return result
    except Exception:
        return {}


def _get_vaccel(sym: str) -> float | None:
    """V_accel = mean(last 3 bars vol) / mean(last 15 bars vol) for today 5m bars."""
    now = time.time()
    cached = _vaccel_cache.get(sym)
    if cached and now - cached["ts"] < _VACCEL_TTL:
        return cached["vaccel"]
    try:
        import yfinance as _yf3
        import numpy as _np3
        intra = _yf3.download(sym, period="1d", interval="5m", auto_adjust=True, progress=False)
        if intra.empty or len(intra) < 15:
            return None
        vols   = _np3.asarray(intra["Volume"].values, dtype=float).ravel()
        v_long = float(_np3.mean(vols[-15:]))
        if v_long == 0:
            return None
        v_short = float(_np3.mean(vols[-3:]))
        vaccel  = round(v_short / v_long, 2)
        _vaccel_cache[sym] = {"ts": now, "vaccel": vaccel}
        return vaccel
    except Exception:
        return None


def _gainers_verdict(v_accel, overhead_blocked: bool, vwap_gap_pct) -> str:
    above_vwap = vwap_gap_pct is not None and vwap_gap_pct > 0
    if overhead_blocked:
        return "OVERHEAD WALL"
    if v_accel is not None and v_accel < 1.0:
        return "FADE RISK"
    if v_accel is not None and v_accel >= 1.5 and above_vwap:
        return "BREAKOUT CONFIRMED"
    if v_accel is not None and v_accel >= 1.0:
        return "DEVELOPING"
    return "WATCH"


@app.get("/api/gainers")
def get_gainers(force: bool = False):
    now = time.time()
    if not force and _gainers_cache["data"] and now - _gainers_cache["ts"] < _GAINERS_TTL:
        return _gainers_cache["data"]

    try:
        url = (
            "https://query1.finance.yahoo.com/v1/finance/screener/predefined/saved"
            "?formatted=false&scrIds=day_gainers&count=50&corsDomain=finance.yahoo.com"
        )
        headers = {"User-Agent": "Mozilla/5.0 (compatible; StockPredictor/1.0)"}
        request = _req.Request(url, headers=headers)
        with _req.urlopen(request, timeout=15) as resp:
            raw = _json.loads(resp.read())
        quotes = raw["finance"]["result"][0]["quotes"]
    except Exception as e:
        if _gainers_cache["data"]:
            return _gainers_cache["data"]
        raise HTTPException(status_code=502, detail=f"Failed to fetch day gainers: {e}")

    _24h_ago = now - 86400
    filtered = [
        q for q in quotes
        if q.get("marketCap", 0) >= 500_000_000
        and q.get("regularMarketPrice", 0) >= 5.0
        and q.get("regularMarketVolume", 0) > 0
        and (q.get("regularMarketTime") or 0) >= _24h_ago
        and q.get("regularMarketChangePercent", 0) >= 5.0
        and q.get("regularMarketVolume", 0) * q.get("regularMarketPrice", 0) >= 5_000_000
    ][:25]

    if not filtered:
        return _gainers_cache["data"] or {
            "results": [], "market_context": _get_market_context(),
            "fetched_at": datetime.fromtimestamp(now, tz=timezone.utc).isoformat(),
        }

    tickers  = [q["symbol"] for q in filtered]
    intraday = _get_intraday_signals(tickers)

    price_map = {q["symbol"]: q.get("regularMarketPrice") for q in filtered}

    vaccel_map:   dict = {}
    overhead_map: dict = {}

    def _fetch_sym_data(sym):
        price = price_map.get(sym)
        va = _get_vaccel(sym)
        oh = _get_overhead_supply(sym, price) if price else {}
        return sym, va, oh

    with ThreadPoolExecutor(max_workers=5) as executor:
        try:
            for sym, va, oh in executor.map(_fetch_sym_data, tickers, timeout=60):
                vaccel_map[sym]   = va
                overhead_map[sym] = oh
        except _FuturesTimeout:
            pass

    results = []
    for quote in filtered:
        sym   = quote["symbol"]
        price = quote.get("regularMarketPrice")
        intra = intraday.get(sym, {})
        vwap  = intra.get("vwap")
        vwap_gap_pct = None
        if price and vwap and vwap > 0:
            vwap_gap_pct = round((price - vwap) / vwap * 100, 2)

        v_accel          = vaccel_map.get(sym)
        overhead         = overhead_map.get(sym, {})
        overhead_blocked = overhead.get("overhead_blocked", False)
        verdict          = _gainers_verdict(v_accel, overhead_blocked, vwap_gap_pct)

        results.append({
            "symbol":            sym,
            "name":              _NAME_OVERRIDES.get(sym, quote.get("shortName", sym)),
            "price":             price,
            "change_pct":        quote.get("regularMarketChangePercent"),
            "volume":            quote.get("regularMarketVolume", 0),
            "market_cap":        quote.get("marketCap"),
            "vwap":              vwap,
            "vwap_gap_pct":      vwap_gap_pct,
            "above_vwap":        intra.get("above_vwap"),
            "is_live":           intra.get("is_live", False),
            "rvol":              intra.get("rvol"),
            "rvol_quality":      intra.get("rvol_quality", "insufficient"),
            "v_accel":           v_accel,
            "nearest_resist":    overhead.get("nearest_resist"),
            "dist_to_resist_pct": overhead.get("dist_to_resist_pct"),
            "overhead_blocked":  overhead_blocked,
            "atr14":             overhead.get("atr14"),
            "verdict":           verdict,
        })
        if verdict not in ("WATCH",):
            _setup_log_event("gainers", results[-1])

    _VORD = {"BREAKOUT CONFIRMED": 0, "DEVELOPING": 1, "WATCH": 2, "FADE RISK": 3, "OVERHEAD WALL": 4}
    results.sort(key=lambda x: _VORD.get(x["verdict"], 9))

    def _clean_g(obj):
        if isinstance(obj, float):
            import math
            return None if (math.isnan(obj) or math.isinf(obj)) else obj
        if isinstance(obj, dict):
            return {k: _clean_g(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_clean_g(v) for v in obj]
        return obj

    payload = _clean_g({
        "results":        results,
        "market_context": _get_market_context(),
        "fetched_at":     datetime.fromtimestamp(now, tz=timezone.utc).isoformat(),
    })
    _gainers_cache["ts"]   = now
    _gainers_cache["data"] = payload
    return payload

# ── Static frontend ──────────────────────────────────────────────────────────

from fastapi.staticfiles import StaticFiles

frontend_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "frontend", "dist")
if os.path.exists(frontend_dir):
    app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")
