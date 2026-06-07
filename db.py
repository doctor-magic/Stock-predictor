import sqlite3
import os
from datetime import date, datetime

DB_PATH = os.path.join(os.path.dirname(__file__), "scanner_cache.db")

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS scan_results (
            market_id TEXT,
            symbol TEXT,
            symbol_name TEXT,
            signal TEXT,
            confidence REAL,
            precision_score REAL,
            last_price REAL,
            scan_date DATE,
            PRIMARY KEY (market_id, symbol, scan_date)
        )
    """)
    conn.commit()
    conn.close()

def save_scan_results(market_id: str, results: list):
    """
    results is a list of dicts: 
    { 'symbol': ..., 'symbol_name': ..., 'signal': ..., 'confidence': ..., 'precision': ..., 'last_price': ... }
    """
    today = date.today().isoformat()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Optional: Delete old scans for this market to save space, or just keep history.
    # We will keep history but only query the latest.
    cursor.execute("DELETE FROM scan_results WHERE market_id = ? AND scan_date = ?", (market_id, today))
    
    for row in results:
        cursor.execute("""
            INSERT INTO scan_results (market_id, symbol, symbol_name, signal, confidence, precision_score, last_price, scan_date)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            market_id, row['symbol'], row['symbol_name'], row['signal'], 
            row['confidence'], row['precision'], row['last_price'], today
        ))
    conn.commit()
    conn.close()

def get_latest_scan(market_id: str):
    """Return all cached results for market_id from today, or empty list."""
    today = date.today().isoformat()
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT * FROM scan_results 
        WHERE market_id = ? AND scan_date = ?
        ORDER BY confidence DESC
    """, (market_id, today))
    
    rows = cursor.fetchall()
    conn.close()
    
    return [{
        "symbol": r["symbol"],
        "symbol_name": r["symbol_name"],
        "signal": r["signal"],
        "confidence": r["confidence"],
        "precision": r["precision_score"],
        "last_price": r["last_price"]
    } for r in rows]


# ── Falling Knife Log ─────────────────────────────────────────────────────────

_FK_LOG_DB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "falling_knife_log.db")


def fk_db_init():
    con = sqlite3.connect(_FK_LOG_DB, timeout=30)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("""CREATE TABLE IF NOT EXISTS fk_events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        symbol TEXT, date TEXT, classify_ts TEXT, price REAL, change_pct REAL,
        rsi REAL, rvol REAL, vwap_gap_pct REAL,
        price_close REAL, ph_return REAL, resolved INTEGER DEFAULT 0
    )""")
    con.execute("CREATE TABLE IF NOT EXISTS fk_milestones (key TEXT PRIMARY KEY, value TEXT)")
    con.commit()
    con.close()


def fk_log_event(sym, price, change_pct, rsi, rvol, vwap_gap_pct):
    try:
        today = date.today().isoformat()
        con = sqlite3.connect(_FK_LOG_DB, timeout=30)
        existing = con.execute("SELECT id FROM fk_events WHERE symbol=? AND date=?", (sym, today)).fetchone()
        if not existing:
            con.execute(
                "INSERT INTO fk_events (symbol, date, classify_ts, price, change_pct, rsi, rvol, vwap_gap_pct) VALUES (?,?,?,?,?,?,?,?)",
                (sym, today, datetime.utcnow().isoformat(), price, change_pct, rsi, rvol, vwap_gap_pct)
            )
            con.commit()
        con.close()
    except Exception:
        pass


def fk_resolve_yesterday():
    try:
        import yfinance as _yf
        from datetime import timedelta
        con = sqlite3.connect(_FK_LOG_DB, timeout=30)
        rows = con.execute("SELECT id, symbol, date FROM fk_events WHERE resolved=0").fetchall()
        today = date.today()
        for row_id, sym, date_str in rows:
            try:
                event_date = date.fromisoformat(date_str)
                if (today - event_date).days < 1:
                    continue
                hist = _yf.download(sym, start=date_str,
                                    end=str(event_date + timedelta(days=5)),
                                    interval="1d", auto_adjust=True, progress=False)
                if hist.empty or len(hist) < 2:
                    continue
                closes_arr = hist["Close"].values.flatten()
                entry_price = float(closes_arr[0])
                price_close = float(closes_arr[1]) if len(closes_arr) > 1 else None
                ph_return   = round((price_close / entry_price - 1) * 100, 2) if price_close else None
                con.execute("UPDATE fk_events SET price_close=?, ph_return=?, resolved=1 WHERE id=?",
                            (price_close, ph_return, row_id))
            except Exception:
                pass
        con.commit()
        total_resolved = con.execute("SELECT COUNT(*) FROM fk_events WHERE resolved=1").fetchone()[0]
        alerted = con.execute("SELECT value FROM fk_milestones WHERE key='n30_alerted'").fetchone()
        if total_resolved >= 30 and not alerted:
            try:
                import urllib.request as _req
                import json as _json
                token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
                chat  = os.environ.get("TELEGRAM_CHAT_ID", "")
                if token and chat:
                    url  = f"https://api.telegram.org/bot{token}/sendMessage"
                    data = _json.dumps({"chat_id": chat, "text": f"Falling Knife milestone: {total_resolved} resolved events!"}).encode()
                    _req.urlopen(_req.Request(url, data=data, headers={"Content-Type": "application/json"}), timeout=5)
            except Exception:
                pass
            con.execute("INSERT OR IGNORE INTO fk_milestones (key, value) VALUES ('n30_alerted', 'yes')")
            con.commit()
        con.close()
    except Exception:
        pass


def get_fk_stats():
    fk_resolve_yesterday()
    con = sqlite3.connect(_FK_LOG_DB, timeout=30)
    rows = con.execute(
        "SELECT id,symbol,date,price,change_pct,rsi,rvol,vwap_gap_pct,price_close,ph_return,resolved "
        "FROM fk_events ORDER BY date DESC LIMIT 100"
    ).fetchall()
    cols = ["id","symbol","date","price","change_pct","rsi","rvol","vwap_gap_pct","price_close","ph_return","resolved"]
    events   = [dict(zip(cols, row)) for row in rows]
    resolved = [e for e in events if e["resolved"]]
    mean_ph  = round(sum(e["ph_return"] for e in resolved if e["ph_return"] is not None) / len(resolved), 2) if resolved else None
    con.close()
    return {"events": events, "total": len(events), "resolved": len(resolved), "mean_ph_return": mean_ph}


# ── Setup Outcome Logger ──────────────────────────────────────────────────────

_SETUP_LOG_DB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "setup_log.db")


def setup_db_init():
    con = sqlite3.connect(_SETUP_LOG_DB, timeout=30)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("""CREATE TABLE IF NOT EXISTS setup_log (
        id            INTEGER PRIMARY KEY AUTOINCREMENT,
        source        TEXT,
        symbol        TEXT,
        date          TEXT,
        log_ts        TEXT,
        price         REAL,
        verdict       TEXT,
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
        resolved      INTEGER DEFAULT 0,
        close_1d      REAL,
        close_5d      REAL,
        ret_1d        REAL,
        ret_5d        REAL
    )""")
    con.commit()
    con.close()


def setup_log_event(source: str, row: dict):
    try:
        today = date.today().isoformat()
        con = sqlite3.connect(_SETUP_LOG_DB, timeout=30)
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


def setup_resolve():
    try:
        import yfinance as _yf
        from datetime import timedelta
        con = sqlite3.connect(_SETUP_LOG_DB, timeout=30)
        rows = con.execute(
            "SELECT id, symbol, date, price FROM setup_log WHERE resolved=0"
        ).fetchall()
        today = date.today()
        for row_id, sym, date_str, entry_price in rows:
            try:
                event_date = date.fromisoformat(date_str)
                if (today - event_date).days < 1:
                    continue
                hist = _yf.download(sym, start=date_str,
                                    end=str(event_date + timedelta(days=10)),
                                    interval="1d", auto_adjust=True, progress=False)
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


def get_setup_breakdown():
    setup_resolve()
    con = sqlite3.connect(_SETUP_LOG_DB, timeout=30)
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
    total    = con.execute("SELECT COUNT(*) FROM setup_log").fetchone()[0]
    resolved = con.execute("SELECT COUNT(*) FROM setup_log WHERE resolved=1").fetchone()[0]
    con.close()
    return {"total_logged": total, "resolved": resolved, "breakdown": breakdown}


try:
    fk_db_init()
    setup_db_init()
except Exception:
    pass

