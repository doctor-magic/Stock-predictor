"""
live_tracker.py — Daily signal logger and outcome resolver for Long-only Expert (>=0.70)

Usage:
  python3 live_tracker.py --log              # log today's BUY signals + resolve past outcomes
  python3 live_tracker.py --report           # print precision report only
  python3 live_tracker.py --log --no-telegram
"""

import argparse
import json
import os
import sqlite3
import sys
import time
import urllib.request
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

# ── Config ─────────────────────────────────────────────────────────────────────
API_BASE      = "https://stock-predictor.online"
CONF_MIN      = 0.70
FORWARD_DAYS  = 10   # trading days
HIT_THRESHOLD = 0.03
MARKETS       = ["sp500", "nasdaq100"]
DB_PATH       = Path(__file__).parent / "tracker.db"

# Bump this whenever a material model change ships:
# features, threshold, universe, label scheme, SELL suppression, etc.
# Format: YYYY-MM_short_description
MODEL_VERSION = "2026-05_ema_dist_regime"  # normalized EMA features + regime tagging + 0.70 threshold
ENV_FILE      = Path(__file__).parent / ".env"

# Load .env (Telegram credentials)
if ENV_FILE.exists():
    for _line in ENV_FILE.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _, _v = _line.partition("=")
            os.environ.setdefault(_k.strip(), _v.strip())

TELEGRAM_TOKEN   = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")


# ── DB ──────────────────────────────────────────────────────────────────────────
def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript("""
    CREATE TABLE IF NOT EXISTS signals (
        id          INTEGER PRIMARY KEY,
        sym         TEXT    NOT NULL,
        date_logged TEXT    NOT NULL,
        entry_price REAL,
        confidence  REAL,
        resolved    INTEGER DEFAULT 0,
        UNIQUE(sym, date_logged)
    );
    CREATE TABLE IF NOT EXISTS outcomes (
        id            INTEGER PRIMARY KEY,
        signal_id     INTEGER UNIQUE,
        date_resolved TEXT,
        exit_price    REAL,
        fwd_ret       REAL,
        hit           INTEGER
    );
    """)
    for col, typedef in [
        ("regime",          "TEXT"),
        ("model_version",   "TEXT"),
        ("beta",            "REAL"),
        ("entry_vix",       "REAL"),
        ("spy_trend",       "TEXT"),
        ("entry_spy_ret5d", "REAL"),
    ]:
        try:
            conn.execute(f"ALTER TABLE signals ADD COLUMN {col} {typedef}")
            conn.commit()
        except Exception:
            pass  # column already exists
    conn.commit()


def _classify_regime(highs, lows, closes) -> tuple[str, float]:
    """ADX-14 (Wilder) × ATR-14 vol percentile → regime string + ADX value.
    Wilder's: seed=SMA(N), then out[i]=(out[i-1]*(N-1)+arr[i])/N  (alpha=1/N, not 2/(N+1)).
    ATR percentile: last 100 bars (requires 6mo download).
    """
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
    window = valid[-100:]                                    # last 100 bars ~= 5 months
    rank   = float(np.sum(window < window[-1])) / len(window)
    vol    = "low_vol" if rank < 0.33 else "med_vol" if rank < 0.67 else "high_vol"
    return f"{trend}_{vol}", adx_val


def _batch_regimes(symbols: list[str]) -> dict[str, dict]:
    """Download 6mo OHLCV for symbols + SPY; return per-symbol regime and beta.
    6mo required for 100-bar ATR percentile window in _classify_regime().
    SPY added once for batch beta computation — not re-downloaded per symbol.
    """
    if not symbols:
        return {}
    unique = list(dict.fromkeys(symbols))
    tickers = unique + ([] if "SPY" in unique else ["SPY"])
    try:
        hist = yf.download(tickers, period="6mo", interval="1d", progress=False, auto_adjust=True)
        if hist.empty:
            return {}
        multi = len(tickers) > 1

        spy_returns = None
        try:
            spy_close = (hist["Close"]["SPY"] if multi else hist["Close"]).dropna()
            spy_returns = spy_close.pct_change().dropna()
        except Exception:
            pass

        result = {}
        for sym in unique:
            if sym == "SPY":
                continue
            try:
                closes = (hist["Close"][sym] if multi else hist["Close"]).dropna()
                highs  = (hist["High"][sym]  if multi else hist["High"]).dropna()
                lows   = (hist["Low"][sym]   if multi else hist["Low"]).dropna()
                regime, _ = _classify_regime(highs.values, lows.values, closes.values)

                beta = None
                if spy_returns is not None and len(closes) >= 60:
                    stock_ret = closes.pct_change().dropna()
                    s, m = stock_ret.align(spy_returns, join="inner")
                    if len(s) >= 60:
                        cov = float(np.cov(s, m)[0, 1])
                        var = float(np.var(m, ddof=1))
                        beta = round(cov / var, 2) if var > 0 else None

                result[sym] = {"regime": regime, "beta": beta}
            except Exception:
                result[sym] = {"regime": "unknown", "beta": None}
        return result
    except Exception:
        return {}


def _get_market_context() -> tuple[float | None, str | None, float | None]:
    """Return (entry_vix, spy_trend, spy_ret5d) at time of signal logging.
    spy_trend: BULL_STRONG (>SMA50 & >SMA200) / BULL_WEAK (>SMA200 only) / BEAR (<SMA200).
    Requires period="300d" — 200 calendar days gives only ~140 trading days, not enough for SMA200.
    """
    spy  = yf.Ticker("SPY").history(period="300d")["Close"]
    vix  = float(yf.Ticker("^VIX").history(period="5d")["Close"].iloc[-1])
    sma50   = float(spy.rolling(50).mean().iloc[-1])
    sma200  = float(spy.rolling(200).mean().iloc[-1])
    current = float(spy.iloc[-1])
    ret5d   = float(spy.pct_change(5).iloc[-1])
    if current > sma50 and current > sma200:
        trend = "BULL_STRONG"
    elif current >= sma200:
        trend = "BULL_WEAK"
    else:
        trend = "BEAR"
    return round(vix, 2), trend, round(ret5d, 4)


# ── Helpers ─────────────────────────────────────────────────────────────────────
def trading_days_elapsed(from_date: date) -> int:
    return int(np.busday_count(str(from_date), str(date.today())))


def fetch_close_on(sym: str, target_date: date) -> float | None:
    """Return close price on or just before target_date."""
    try:
        end = target_date + timedelta(days=5)
        raw = yf.download(sym, start=str(target_date - timedelta(days=5)),
                          end=str(end), progress=False, auto_adjust=True)
        if raw.empty:
            return None
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        closes = raw["Close"].dropna()
        closes.index = [i.date() for i in closes.index]
        valid = closes[closes.index <= target_date]
        return float(valid.iloc[-1]) if not valid.empty else None
    except Exception:
        return None


def fetch_scan(market_id: str) -> list[dict]:
    """Pull BUY signals from GCP scan cache (no force-refresh)."""
    payload = json.dumps({
        "market_id": market_id,
        "min_confidence": CONF_MIN,
        "top_n": 200,
        "force_refresh": False,
    }).encode()
    req = urllib.request.Request(
        f"{API_BASE}/api/scan",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        data = json.loads(r.read())

    if data.get("status") == "done":
        return data.get("results", [])

    # Cache miss — poll progress endpoint (rare; pre-scan cron runs at 05:00)
    task_id = data.get("task_id")
    if not task_id:
        return []
    for _ in range(90):
        time.sleep(2)
        with urllib.request.urlopen(
            f"{API_BASE}/api/scan/progress/{task_id}", timeout=10
        ) as r:
            prog = json.loads(r.read())
        if prog.get("done"):
            return prog.get("results", [])
    print(f"  [{market_id}] scan timed out waiting for results")
    return []


def send_telegram(text: str) -> None:
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("[Telegram] not configured — skipping")
        return
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        payload = json.dumps({
            "chat_id": TELEGRAM_CHAT_ID,
            "text": text,
            "parse_mode": "HTML",
        }).encode()
        req = urllib.request.Request(
            url, data=payload, headers={"Content-Type": "application/json"}
        )
        urllib.request.urlopen(req, timeout=5)
    except Exception as e:
        print(f"[Telegram] failed: {e}")


# ── Core operations ─────────────────────────────────────────────────────────────
def log_signals(conn: sqlite3.Connection) -> list[dict]:
    """Fetch today's BUY>=0.70 signals from GCP and persist new ones (with regime tag)."""
    today = str(date.today())
    seen: set[str] = set()
    to_insert: list[tuple] = []  # (sym, conf, price)

    for market in MARKETS:
        try:
            results = fetch_scan(market)
        except Exception as e:
            print(f"  [{market}] fetch error: {e}")
            continue

        for r in results:
            if r.get("signal") != "BUY":
                continue
            sym = r["symbol"]
            if sym in seen:
                continue
            seen.add(sym)

            prev = conn.execute(
                "SELECT id FROM signals WHERE sym=? AND date_logged=?", (sym, today)
            ).fetchone()
            if prev:
                continue

            to_insert.append((sym, r.get("confidence", 0), r.get("last_price")))

    if not to_insert:
        return []

    # Batch compute regime — one yfinance download for all new BUY symbols
    print(f"  Computing regime for {len(to_insert)} new signal(s)...")
    regime_map = _batch_regimes([t[0] for t in to_insert])

    vix_val, spy_trend_val, spy_ret5d_val = None, None, None
    try:
        vix_val, spy_trend_val, spy_ret5d_val = _get_market_context()
        print(f"  Market context: {spy_trend_val}, VIX={vix_val}, SPY 5d={spy_ret5d_val:+.2%}")
    except Exception as e:
        print(f"  ⚠️ market context unavailable: {e}")

    new_signals: list[dict] = []
    for sym, conf, price in to_insert:
        entry  = regime_map.get(sym, {})
        regime = entry.get("regime", "unknown")
        beta   = entry.get("beta")
        conn.execute(
            "INSERT INTO signals (sym, date_logged, entry_price, confidence, regime, model_version, beta,"
            " entry_vix, spy_trend, entry_spy_ret5d)"
            " VALUES (?,?,?,?,?,?,?,?,?,?)",
            (sym, today, price, conf, regime, MODEL_VERSION, beta, vix_val, spy_trend_val, spy_ret5d_val),
        )
        new_signals.append({
            "sym": sym, "conf": conf, "price": price, "regime": regime, "beta": beta,
            "entry_vix": vix_val, "spy_trend": spy_trend_val, "entry_spy_ret5d": spy_ret5d_val,
        })

    conn.commit()
    return new_signals


def resolve_outcomes(conn: sqlite3.Connection) -> list[dict]:
    """Check pending signals that are 10+ trading days old and record actual returns."""
    today = date.today()
    pending = conn.execute(
        "SELECT id, sym, date_logged, entry_price FROM signals WHERE resolved=0"
    ).fetchall()

    resolved: list[dict] = []
    for sig_id, sym, date_str, entry_price in pending:
        logged_date = date.fromisoformat(date_str)
        if trading_days_elapsed(logged_date) < FORWARD_DAYS:
            continue

        exit_np   = np.busday_offset(str(logged_date), FORWARD_DAYS, roll="forward")
        exit_date = date.fromisoformat(str(exit_np))
        exit_price = fetch_close_on(sym, exit_date)

        if exit_price is None or entry_price is None:
            continue

        fwd_ret = (exit_price - entry_price) / entry_price
        hit     = 1 if fwd_ret >= HIT_THRESHOLD else 0

        conn.execute(
            "INSERT OR REPLACE INTO outcomes (signal_id, date_resolved, exit_price, fwd_ret, hit)"
            " VALUES (?,?,?,?,?)",
            (sig_id, str(today), exit_price, fwd_ret, hit),
        )
        conn.execute("UPDATE signals SET resolved=1 WHERE id=?", (sig_id,))
        resolved.append({"sym": sym, "fwd_ret": fwd_ret * 100, "hit": hit})

    conn.commit()
    return resolved


# ── Report ──────────────────────────────────────────────────────────────────────
def print_report(conn: sqlite3.Connection) -> dict:
    rows = conn.execute("""
        SELECT s.sym, s.date_logged, s.confidence, s.entry_price, s.resolved,
               o.fwd_ret, o.hit
        FROM signals s
        LEFT JOIN outcomes o ON o.signal_id = s.id
        ORDER BY s.date_logged DESC
    """).fetchall()

    resolved = [r for r in rows if r[4] == 1]
    pending  = [r for r in rows if r[4] == 0]

    print(f"\n{'='*62}")
    print(f"  LIVE TRACKER  —  Long-only Expert (>={CONF_MIN:.0%})")
    print(f"{'='*62}")

    if resolved:
        hits      = sum(r[6] for r in resolved)
        precision = hits / len(resolved) * 100
        avg_ret   = sum(r[5] for r in resolved) / len(resolved)
        print(f"\n  Resolved: {len(resolved)}  |  Precision: {precision:.1f}%  |  Avg ret: {avg_ret:+.2f}%")
        print(f"\n  {'Sym':<7} {'Date':<12} {'Conf':>5} {'Entry':>8} {'Ret':>8}  Hit")
        print(f"  {'-'*48}")
        for sym, dl, conf, entry, _, fwd_ret, hit in sorted(resolved, key=lambda x: x[1], reverse=True):
            marker = "Y" if hit else "N"
            print(f"  {sym:<7} {dl:<12} {conf:>4.0%} {entry:>8.2f} {fwd_ret:>+7.2f}%   {marker}")
    else:
        print("\n  No resolved signals yet.")

    if pending:
        print(f"\n  Pending ({len(pending)}):")
        print(f"  {'Sym':<7} {'Date':<12} {'Conf':>5} {'Entry':>8}  Days")
        print(f"  {'-'*44}")
        for sym, dl, conf, entry, *_ in sorted(pending, key=lambda x: x[1], reverse=True):
            td = trading_days_elapsed(date.fromisoformat(dl))
            print(f"  {sym:<7} {dl:<12} {conf:>4.0%} {entry:>8.2f}  {td}/{FORWARD_DAYS}")

    total_res = len(resolved)
    return {
        "n_resolved": total_res,
        "n_pending":  len(pending),
        "precision":  sum(r[6] for r in resolved) / total_res * 100 if total_res else None,
        "avg_ret":    sum(r[5] for r in resolved) / total_res       if total_res else None,
    }


def build_telegram_msg(new_signals: list, resolved: list, stats: dict) -> str:
    today_str = date.today().strftime("%-d/%m/%Y")
    lines = [f"<b>Long-only Expert — {today_str}</b>"]

    if new_signals:
        s0 = new_signals[0]
        spy_t = s0.get("spy_trend")
        vix   = s0.get("entry_vix")
        ret5d = s0.get("entry_spy_ret5d")
        if spy_t or vix is not None:
            parts = []
            if spy_t:            parts.append(spy_t)
            if vix  is not None: parts.append(f"VIX {vix:.1f}")
            if ret5d is not None: parts.append(f"SPY 5d {ret5d*100:+.1f}%")
            lines.append(f"🌡 {' | '.join(parts)}")

        lines.append(f"\n<b>New BUY signals ({len(new_signals)}):</b>")
        for s in sorted(new_signals, key=lambda x: -x["conf"]):
            regime = s.get("regime", "")
            regime_tag = f"  [{regime}]" if regime and regime != "unknown" else ""
            beta = s.get("beta")
            beta_tag = f" ⚠β{beta:.1f}" if beta is not None and beta > 1.5 else ""
            lines.append(f"  • {s['sym']:<6} {s['conf']:.0%}  ${s['price']:.2f}{regime_tag}{beta_tag}")
    else:
        lines.append("\nNo new BUY signals today.")

    if resolved:
        lines.append(f"\n<b>Resolved today ({len(resolved)}):</b>")
        for r in resolved:
            icon = "+" if r["hit"] else "-"
            lines.append(f"  [{icon}] {r['sym']:<6} {r['fwd_ret']:+.2f}%")

    if stats["n_resolved"] > 0:
        lines.append(f"\n<b>Validation ({stats['n_resolved']} signals resolved):</b>")
        lines.append(f"  Precision : {stats['precision']:.1f}%")
        lines.append(f"  Avg ret   : {stats['avg_ret']:+.2f}%")
        lines.append(f"  Pending   : {stats['n_pending']} signals")

    return "\n".join(lines)


# ── CLI ─────────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="Live signal tracker for Long-only Expert")
    parser.add_argument("--log",          action="store_true", help="Log today's signals + resolve past")
    parser.add_argument("--report",       action="store_true", help="Print report only")
    parser.add_argument("--no-telegram",  action="store_true", help="Skip Telegram digest")
    args = parser.parse_args()

    if not args.log and not args.report:
        parser.print_help()
        sys.exit(0)

    conn = sqlite3.connect(DB_PATH)
    init_db(conn)

    new_signals: list[dict] = []
    resolved:    list[dict] = []

    if args.log:
        print("Fetching BUY signals from GCP API...")
        new_signals = log_signals(conn)
        print(f"  New signals logged: {len(new_signals)}")

        print("Resolving past outcomes...")
        resolved = resolve_outcomes(conn)
        print(f"  Outcomes resolved:  {len(resolved)}")

    stats = print_report(conn)
    conn.close()

    if args.log and not args.no_telegram:
        msg = build_telegram_msg(new_signals, resolved, stats)
        send_telegram(msg)
        print("\nTelegram digest sent.")


if __name__ == "__main__":
    main()
