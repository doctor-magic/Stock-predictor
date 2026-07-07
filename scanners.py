"""Scanner helper functions — pure computation and intraday enrichment.

Imported by api.py. No imports from api.py (prevents circular imports).
DB logging calls use db.py directly.
"""
import logging
import os
import time
import sqlite3
import statistics
from datetime import datetime, date
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)

_INTRADAY_DB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "intraday_cache.db")

# ── Caches owned by helpers ───────────────────────────────────────────────────

_market_context_cache: dict = {"ts": 0, "data": None}
_MARKET_CONTEXT_TTL = 120   # 2 min

_lev_sentiment_cache: dict = {"ts": 0, "data": None}
_LEV_SENTIMENT_TTL = 300    # 5 min
_LEV_PAIRS = {"semis": ("SOXS", "SOXL"), "qqq": ("SQQQ", "TQQQ")}   # (short, long)

_gainers_daily_cache: dict = {}           # sym → {date, sma200, nearest_resist, atr14, …}
_vaccel_cache: dict        = {}           # sym → {ts, vaccel}
_VACCEL_TTL                = 300          # 5 min

# ── Pure computation helpers ──────────────────────────────────────────────────

def compute_momentum(rsi, vol_ratio: float, ret_5d: float) -> str:
    if rsi is not None and rsi > 75:
        return "OVEREXTENDED"
    if rsi is not None and rsi < 35 and vol_ratio >= 2.0:
        return "WATCH"
    if vol_ratio >= 3.0 and (ret_5d or 0) > 1.0:
        return "SURGING"
    if (ret_5d or 0) < -4.0 and vol_ratio >= 2.0:
        return "SELLING OFF"
    return "NEUTRAL"


def compute_verdict(
    ml_signal: str,
    ml_confidence,
    vol_ratio: float = 1.0,
    rsi=None,
    open_price=None,
    current_price=None,
    above_sma50: bool = True,
) -> str:
    """10-day ML outlook with four momentum overlays."""
    if open_price and current_price and open_price > 0:
        if (current_price - open_price) / open_price >= 0.03:
            return "OVEREXTENDED"
    if vol_ratio >= 3.0 and rsi is not None and rsi < 65:
        return "VOL BREAKOUT"
    threshold = 0.65 if vol_ratio >= 2.0 else 0.70
    if ml_signal == "BUY" and (ml_confidence or 0) >= threshold and vol_ratio >= 1.0 and above_sma50:
        return "BUY"
    if ml_signal == "SELL":
        return "HOLD"
    if ml_signal in (None, "N/A"):
        return "N/A"
    return "HOLD"


def gainers_verdict(v_accel, overhead_blocked: bool, vwap_gap_pct) -> str:
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


def detect_falling_wedge(highs, lows, closes, volumes) -> dict:
    """Falling wedge detector. Returns {} when no pattern found."""
    try:
        from scipy.signal import find_peaks
        import numpy as np

        h_all = np.array(highs,   dtype=float).flatten()
        l_all = np.array(lows,    dtype=float).flatten()
        c_all = np.array(closes,  dtype=float).flatten()
        v_all = np.array(volumes, dtype=float).flatten()
        total = len(c_all)

        # Raised May 15 2026 (0.15/0.20/0.25 → too many false signals; NIO case
        # study validated these). Regressed silently in the Jun 7 refactor —
        # restored Jul 5 2026. Must match the pre_scan.py copy.
        min_compression = {30: 0.40, 45: 0.45, 60: 0.50}

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

            if len(peaks) < 3 or len(troughs) < 3:
                continue

            upper_slope, upper_intercept = np.polyfit(peaks,   h[peaks],  1)
            lower_slope, lower_intercept = np.polyfit(troughs, l[troughs], 1)

            if upper_slope >= -0.001 or lower_slope >= -0.001:
                continue
            if upper_slope >= lower_slope:
                continue

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


def classify_regime(highs, lows, closes) -> tuple[str, float]:
    """ADX-14 (Wilder) × ATR-14 percentile → regime string + ADX value.

    Uses Wilder's smoothing (alpha=1/N), NOT pandas .ewm(). Do not replace.
    Requires ≥30 bars; returns ('unknown', 0.0) otherwise.
    """
    import numpy as np
    highs  = np.asarray(highs,  dtype=float).ravel()
    lows   = np.asarray(lows,   dtype=float).ravel()
    closes = np.asarray(closes, dtype=float).ravel()
    if len(closes) < 30:
        return "unknown", 0.0

    h  = highs[1:];  l = lows[1:];  pc = closes[:-1];  c = closes[1:]
    tr   = np.maximum(h - l, np.maximum(np.abs(h - pc), np.abs(l - pc)))
    up   = highs[1:] - highs[:-1]
    down = lows[:-1] - lows[1:]
    pdm  = np.where((up > down) & (up > 0),   up,   0.0)
    ndm  = np.where((down > up) & (down > 0), down, 0.0)

    n = 14
    def _wilder(arr):
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

    atr_norm = atr14 / c
    valid    = atr_norm[~np.isnan(atr_norm)]
    if len(valid) < 10:
        return f"{trend}_unknown", adx_val
    window = valid[-100:]
    rank   = float(np.sum(window < window[-1])) / len(window)
    vol    = "low_vol" if rank < 0.33 else "med_vol" if rank < 0.67 else "high_vol"
    return f"{trend}_{vol}", adx_val


# ── Intraday helpers ──────────────────────────────────────────────────────────

def get_tod_rvol_cached(symbol: str, time_slot: str, today_str: str):
    """Returns (rvol, quality) from intraday_cache.db median baseline."""
    try:
        if not os.path.exists(_INTRADAY_DB):
            return None, "insufficient"
        con = sqlite3.connect(_INTRADAY_DB, timeout=30)
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
    except Exception as e:
        logger.warning("RVOL calc failed for %s: %s", symbol, e)
        return None, "insufficient"


def get_intraday_signals(tickers: list) -> dict:
    """Batch 5-minute intraday analysis: VWAP, RVOL, ORB, setups."""
    import yfinance as yf
    import pandas as pd
    import numpy as np

    ET = ZoneInfo("America/New_York")
    now_et = datetime.now(ET)

    try:
        raw5m = yf.download(
            tickers, period="10d", interval="5m",
            group_by="ticker", auto_adjust=True, progress=False
        )
    except Exception as e:
        logger.error("yfinance batch download failed: %s", e)
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

            day_df["tp"]     = (day_df["High"] + day_df["Low"] + day_df["Close"]) / 3
            day_df["tp_vol"] = day_df["tp"] * day_df["Volume"]
            day_df["vwap"]   = day_df["tp_vol"].cumsum() / day_df["Volume"].cumsum()
            current_vwap  = float(day_df["vwap"].iloc[-1])
            current_price = float(day_df["Close"].iloc[-1])

            orb_end      = day_df.index[0].replace(hour=10, minute=0, second=0, microsecond=0)
            orb_df       = day_df[day_df.index <= orb_end]
            orb_high     = float(orb_df["High"].max()) if not orb_df.empty else None
            orb_low      = float(orb_df["Low"].min())  if not orb_df.empty else None
            orb_breakout = bool(orb_high and current_price > orb_high)

            breakout_time = None
            if orb_breakout:
                post_orb    = day_df[day_df.index > orb_end]
                first_break = post_orb[post_orb["Close"] > orb_high]
                if not first_break.empty:
                    breakout_time = first_break.index[0].isoformat()

            ref_df       = day_df.iloc[:-1] if (is_live and len(day_df) > 1) else day_df
            rvol         = None
            rvol_quality = "insufficient"
            if not ref_df.empty:
                ref_time  = ref_df.index[-1].time()
                slot_str  = ref_time.strftime("%H:%M")
                today_str = str(target_date)
                rvol, rvol_quality = get_tod_rvol_cached(sym, slot_str, today_str)
                if rvol is None:
                    ref_vol   = float(ref_df["Volume"].iloc[-1])
                    mask      = (np.array(df.index.date) != target_date) & (np.array(df.index.time) == ref_time)
                    hist_same = df[mask]["Volume"]
                    if len(hist_same) >= 3:
                        avg_vol = float(hist_same.mean())
                        if avg_vol > 0:
                            rvol = round(ref_vol / avg_vol, 1)
                            rvol_quality = "legacy"

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

            base_forming = None
            if len(day_df) >= 10:
                last5_lows   = day_df["Low"].iloc[-5:].values
                prior5_lows  = day_df["Low"].iloc[-10:-5].values
                last5_range  = float((day_df["High"].iloc[-5:] - day_df["Low"].iloc[-5:]).mean())
                prior5_range = float((day_df["High"].iloc[-10:-5] - day_df["Low"].iloc[-10:-5]).mean())
                no_new_low       = float(last5_lows.min()) >= float(prior5_lows.min()) * 0.998
                range_compressed = prior5_range > 0 and last5_range < prior5_range * 0.75
                base_forming = bool(no_new_low and range_compressed)

            setup = None
            if orb_breakout and (rvol or 0) >= 2.0:
                setup = "ORB BREAKOUT"
            elif (rvol or 0) >= 3.0 and current_price > current_vwap:
                setup = "LIQUID SURGE"
            elif vwap_bounce:
                setup = "VWAP BOUNCE"

            out[sym] = {
                "vwap":          round(current_vwap, 2),
                "rvol":          rvol,
                "rvol_quality":  rvol_quality,
                "orb_high":      round(orb_high, 2) if orb_high else None,
                "orb_low":       round(orb_low,  2) if orb_low  else None,
                "orb_breakout":  orb_breakout,
                "above_vwap":    current_price > current_vwap,
                "vwap_bounce":   vwap_bounce,
                "setup":         setup,
                "breakout_time": breakout_time,
                "is_live":       is_live,
                "analysis_date": str(target_date),
                "base_forming":  base_forming,
            }
        except Exception:
            pass

    return out


def _compute_lev_ratios(dollar_vol: dict) -> dict:
    """Pure math: short/long dollar-volume ratio per _LEV_PAIRS. None on missing/zero denominator."""
    out = {}
    for name, (short, long_) in _LEV_PAIRS.items():
        dv_s, dv_l = dollar_vol.get(short), dollar_vol.get(long_)
        out[name] = round(dv_s / dv_l, 2) if dv_s and dv_l else None
    return out


def get_lev_sentiment():
    """Leveraged-ETF sentiment: SOXS:SOXL + SQQQ:TQQQ dollar-volume ratios (short/long).
    OBSERVATIONAL ONLY (spec Jul 7 2026) — feeds the header strip + setup_log lev_sent_*
    columns for the pre-registered N>=50 question. NOT a gate; must never block a scan.
    Dollar volume (vol x price), not share volume — the pairs trade at different unit prices."""
    now = time.time()
    if _lev_sentiment_cache["data"] is not None and now - _lev_sentiment_cache["ts"] < _LEV_SENTIMENT_TTL:
        return _lev_sentiment_cache["data"]
    try:
        import yfinance as yf
        syms = [s for pair in _LEV_PAIRS.values() for s in pair]
        raw = yf.download(syms, period="1d", interval="1d", progress=False, auto_adjust=True)
        dollar_vol = {}
        for sym in syms:
            try:
                close = float(raw["Close"][sym].dropna().iloc[-1])
                vol   = float(raw["Volume"][sym].dropna().iloc[-1])
                dollar_vol[sym] = close * vol
            except Exception:
                dollar_vol[sym] = None
        out = _compute_lev_ratios(dollar_vol)
        _lev_sentiment_cache["ts"]   = now
        _lev_sentiment_cache["data"] = out
        return out
    except Exception:
        return _lev_sentiment_cache["data"]


def get_market_context() -> dict:
    now = time.time()
    if _market_context_cache["data"] and now - _market_context_cache["ts"] < _MARKET_CONTEXT_TTL:
        ctx = _market_context_cache["data"]
        ctx["lev"] = get_lev_sentiment()
        return ctx
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
        _market_context_cache["ts"]   = now
        _market_context_cache["data"] = context
        context["lev"] = get_lev_sentiment()
        return context
    except Exception:
        return _market_context_cache["data"] or {"SPY": None, "QQQ": None, "tailwind": None, "headwind": None, "lev": None}


def get_overhead_supply(sym: str, price: float) -> dict:
    """Lazy per-symbol daily cache. Returns nearest resistance, ATR-14, overhead_blocked."""
    today_str = date.today().isoformat()
    cached = _gainers_daily_cache.get(sym)
    if cached and cached.get("date") == today_str:
        return cached
    try:
        import yfinance as yf
        import numpy as np
        hist = yf.download(sym, period="6mo", interval="1d", auto_adjust=True, progress=False)
        if hist.empty or len(hist) < 20:
            return {}
        closes = np.asarray(hist["Close"].values, dtype=float).ravel()
        highs  = np.asarray(hist["High"].values,  dtype=float).ravel()
        lows   = np.asarray(hist["Low"].values,   dtype=float).ravel()
        n      = len(closes)

        sma200 = float(np.mean(closes[-200:])) if n >= 200 else float(np.mean(closes))

        def _wilder_local(arr, period):
            out = np.full(len(arr), np.nan)
            if len(arr) < period:
                return out
            out[period - 1] = np.mean(arr[:period])
            for i in range(period, len(arr)):
                out[i] = (out[i - 1] * (period - 1) + arr[i]) / period
            return out

        trs = np.maximum(
            highs[1:] - lows[1:],
            np.maximum(np.abs(highs[1:] - closes[:-1]), np.abs(lows[1:] - closes[:-1])),
        )
        atr14_arr = _wilder_local(trs, 14)
        atr14 = float(atr14_arr[-1]) if not np.isnan(atr14_arr[-1]) else None

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


def get_vaccel(sym: str) -> float | None:
    """V_accel = mean(last 3 bars vol) / mean(last 15 bars vol) for today 5m bars."""
    now = time.time()
    cached = _vaccel_cache.get(sym)
    if cached and now - cached["ts"] < _VACCEL_TTL:
        return cached["vaccel"]
    try:
        import yfinance as yf
        import numpy as np
        intra = yf.download(sym, period="1d", interval="5m", auto_adjust=True, progress=False)
        if intra.empty or len(intra) < 15:
            return None
        vols    = np.asarray(intra["Volume"].values, dtype=float).ravel()
        v_long  = float(np.mean(vols[-15:]))
        if v_long == 0:
            return None
        v_short = float(np.mean(vols[-3:]))
        vaccel  = round(v_short / v_long, 2)
        _vaccel_cache[sym] = {"ts": now, "vaccel": vaccel}
        return vaccel
    except Exception:
        return None
