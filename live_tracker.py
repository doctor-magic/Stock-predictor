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
    conn.commit()


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
    """Fetch today's BUY>=0.70 signals from GCP and persist new ones."""
    today = str(date.today())
    new_signals: list[dict] = []
    seen: set[str] = set()

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

            conf  = r.get("confidence", 0)
            price = r.get("last_price")

            prev = conn.execute(
                "SELECT id FROM signals WHERE sym=? AND date_logged=?", (sym, today)
            ).fetchone()
            if prev:
                continue

            conn.execute(
                "INSERT INTO signals (sym, date_logged, entry_price, confidence) VALUES (?,?,?,?)",
                (sym, today, price, conf),
            )
            new_signals.append({"sym": sym, "conf": conf, "price": price})

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
        today = date.today()
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
        lines.append(f"\n<b>New BUY signals ({len(new_signals)}):</b>")
        for s in sorted(new_signals, key=lambda x: -x["conf"]):
            lines.append(f"  • {s['sym']:<6} {s['conf']:.0%}  ${s['price']:.2f}")
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
