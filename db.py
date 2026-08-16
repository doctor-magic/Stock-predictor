import sqlite3
import os
from datetime import date, datetime
from zoneinfo import ZoneInfo

from market_calendar import has_session_opened

DB_PATH = os.path.join(os.path.dirname(__file__), "scanner_cache.db")

_ET = ZoneInfo("America/New_York")


def _signal_date() -> str:
    """US market trading date (ET) as YYYY-MM-DD.

    setup_log / fk_events rows describe US-stock signals, so their `date` must
    match yfinance daily bars (ET) and the resolver's forward-return windows.
    The server runs in Asia/Jerusalem (UTC+3), so `date.today()` rolled to
    tomorrow for any scan logged 21:00-24:00 UTC — mis-stamping evening signals
    a day ahead of their UTC `log_ts`. Anchoring on ET fixes it at the source.
    """
    return datetime.now(_ET).date().isoformat()

def init_db():
    conn = sqlite3.connect(DB_PATH, timeout=30)
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
    # RAW P(BUY) alongside the winning-class confidence (Jul 31 2026) — diagnostics
    # only. Without it the reversion funnel can only measure p(BUY) for the minority
    # of symbols that need a live prediction, i.e. exactly the biased subset.
    try:
        cursor.execute("ALTER TABLE scan_results ADD COLUMN proba_buy REAL")
    except Exception:
        pass  # column already exists
    conn.commit()
    conn.close()

def save_scan_results(market_id: str, results: list):
    """
    results is a list of dicts: 
    { 'symbol': ..., 'symbol_name': ..., 'signal': ..., 'confidence': ..., 'precision': ..., 'last_price': ... }
    """
    today = date.today().isoformat()
    conn = sqlite3.connect(DB_PATH, timeout=30)
    cursor = conn.cursor()

    # Optional: Delete old scans for this market to save space, or just keep history.
    # We will keep history but only query the latest.
    cursor.execute("DELETE FROM scan_results WHERE market_id = ? AND scan_date = ?", (market_id, today))
    
    for row in results:
        cursor.execute("""
            INSERT INTO scan_results (market_id, symbol, symbol_name, signal, confidence, precision_score, last_price, scan_date, proba_buy)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            market_id, row['symbol'], row['symbol_name'], row['signal'],
            row['confidence'], row['precision'], row['last_price'], today,
            row.get('proba_buy')
        ))
    conn.commit()
    conn.close()

def get_latest_scan(market_id: str):
    """Return all cached results for market_id from today, or empty list."""
    today = date.today().isoformat()
    conn = sqlite3.connect(DB_PATH, timeout=30)
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
        "proba_buy": (r["proba_buy"] if "proba_buy" in r.keys() else None),
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
        today = _signal_date()
        # Same session choke point as setup_log_event — endpoint-driven writes
        # have no cron entrypoint to guard (holiday/weekend AND pre-open).
        if not has_session_opened():
            return
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
        dist_from_sma50 REAL,
        resolved      INTEGER DEFAULT 0,
        close_1d      REAL,
        close_5d      REAL,
        ret_1d        REAL,
        ret_5d        REAL
    )""")
    for _mig in (
        "ALTER TABLE setup_log ADD COLUMN dist_from_sma50 REAL",
        # Restoration instrumentation (Jul 5 2026): ALL gate evaluations per row
        # (JSON array, no short-circuit) + market context at signal time — so the
        # next N>=50 can answer per-gate and per-regime questions.
        "ALTER TABLE setup_log ADD COLUMN blocked_reasons TEXT",
        "ALTER TABLE setup_log ADD COLUMN market_state TEXT",
        "ALTER TABLE setup_log ADD COLUMN vix_state TEXT",
        # Leveraged-ETF sentiment at signal time (Jul 7 2026, observational only):
        # RAW short/long dollar-volume ratios (SOXS:SOXL, SQQQ:TQQQ) — labels live in
        # the display layer. Pre-registered N>=50 question keys off lev_sent_semis > 2.
        "ALTER TABLE setup_log ADD COLUMN lev_sent_semis REAL",
        "ALTER TABLE setup_log ADD COLUMN lev_sent_qqq REAL",
        # Forward-SPY outcome columns (Jul 30 2026): market-adjusted basis captured
        # live at resolution, aligned to the STOCK's own bar dates over the same
        # window as ret_1d/ret_5d. Outcome data written at resolution time — the
        # forward-only rule (signal-time features) is untouched. Units: percent.
        "ALTER TABLE setup_log ADD COLUMN spy_ret_1d REAL",
        "ALTER TABLE setup_log ADD COLUMN spy_ret_5d REAL",
        # Gate DENOMINATORS (Aug 5 2026) — blocked_reasons alone cannot answer the
        # pre-registered per-gate questions, because an empty list is stored as
        # NULL and NULL therefore means three different things at once: the row
        # predates Jul 5, or the gate was evaluated and did not fire, or the gate
        # could not be evaluated at all (off-hours for HOD, fewer than three
        # readings for the RVOL slope). Without separating "did not fire" from
        # "never ran", the control group is a mix of market conditions and any
        # gate comparison is confounded.
        # Both values are already computed per row and were simply not persisted:
        #   rvol_trend    "up"/"flat"/"down"; NULL = still warming up
        #   hod_gap_ratio non-NULL = the gate ran; NULL = outside its window
        # Signal-time features, additive, NULL for existing rows — analyses filter
        # WHERE <col> IS NOT NULL. No gate behaviour and no logging coverage
        # changes, so sample composition is untouched and this needs no amendment.
        "ALTER TABLE setup_log ADD COLUMN rvol_trend TEXT",
        "ALTER TABLE setup_log ADD COLUMN hod_gap_ratio REAL",
    ):
        try:
            con.execute(_mig)
        except Exception:
            pass
    con.commit()
    con.close()


def setup_log_event(source: str, row: dict):
    try:
        today = _signal_date()
        # Session choke point: never log endpoint-driven setup rows on a
        # non-trading day OR in the pre-open window (midnight→09:30 ET), when
        # screeners still serve yesterday's quotes under today's ET date
        # (Jul-7-2026 QNT incident). This is the ONLY guard that covers the
        # */25 warm cron and any open browser — no cron entrypoint to wrap.
        if not has_session_opened():
            return
        con = sqlite3.connect(_SETUP_LOG_DB, timeout=30)
        exists = con.execute(
            "SELECT id FROM setup_log WHERE source=? AND symbol=? AND date=?",
            (source, row.get("symbol"), today)
        ).fetchone()
        if not exists:
            _reasons = row.get("blocked_reasons")
            if isinstance(_reasons, (list, tuple)):
                import json as _json
                _reasons = _json.dumps(list(_reasons)) if _reasons else None
            con.execute("""
                INSERT INTO setup_log (
                    source, symbol, date, log_ts, price,
                    verdict, ml_signal, ml_confidence,
                    vol_ratio, rsi, beta, beta_blocked, above_sma50,
                    regime, reversion_verdict, vwap_gap_pct, rvol_val, rvol_alert,
                    dist_from_sma50, blocked_reasons, market_state, vix_state,
                    lev_sent_semis, lev_sent_qqq, rvol_trend, hod_gap_ratio
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
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
                row.get("dist_from_sma50"),
                _reasons,
                row.get("market_state"),
                row.get("vix_state"),
                row.get("lev_sent_semis"),
                row.get("lev_sent_qqq"),
                row.get("rvol_trend"),
                row.get("hod_gap_ratio"),
            ))
            con.commit()
        con.close()
    except Exception:
        pass


# ── DEVELOPING display breaker (pre-registered safety rule, Jul 3 2026) ───────
# IF at N>=20 resolved gainers/DEVELOPING rows the mean ret_5d is worse than -5%
# → the DISPLAY layer demotes DEVELOPING to WATCH. Logging coverage is NOT
# narrowed — events keep being logged with their true verdict. Research
# conclusions still wait for N>=50; this is a stop-loss, not a finding.
_DEV_BREAKER_MIN_N    = 20
_DEV_BREAKER_MEAN_PCT = -5.0
_dev_breaker_cache: dict = {"ts": 0.0, "tripped": False}


def developing_breaker_tripped() -> bool:
    import time as _time
    now = _time.time()
    if now - _dev_breaker_cache["ts"] < 3600:
        return _dev_breaker_cache["tripped"]
    tripped = False
    try:
        con = sqlite3.connect(_SETUP_LOG_DB, timeout=30)
        n, mean5 = con.execute(
            "SELECT COUNT(*), AVG(ret_5d) FROM setup_log "
            "WHERE source='gainers' AND verdict='DEVELOPING' AND resolved=1 AND ret_5d IS NOT NULL"
        ).fetchone()
        con.close()
        tripped = bool(n is not None and n >= _DEV_BREAKER_MIN_N
                       and mean5 is not None and mean5 < _DEV_BREAKER_MEAN_PCT)
        if tripped:
            print(f"[developing_breaker] TRIPPED: n={n} mean_5d={mean5:.2f}% — "
                  f"DEVELOPING demoted to WATCH in display", flush=True)
    except Exception:
        pass
    _dev_breaker_cache["ts"] = now
    _dev_breaker_cache["tripped"] = tripped
    return tripped


def setup_resolve(max_rows: int = 50):
    """Resolve a BOUNDED, THROTTLED batch of mature setup_log rows.

    The old version selected ALL unresolved rows and fired one yf.download each
    in a tight loop (200+), which tripped Yahoo's 429 IP rate-limit and then
    swallowed it via a silent `except: pass` (printed success with 0 resolved).
    Rewritten to be active/restrained/distributed:
      • only rows aged >= 7 calendar days (>= 5 trading-day closes available);
      • capped at `max_rows` per run — the nightly cron drains any backlog over
        a few runs instead of bursting;
      • time.sleep(0.5) between calls + per-row commit (partial progress survives
        a crash) + real logging instead of silent pass.
    Resolution is driven by resolve_setups.py (03:00 UTC cron), NOT the endpoint.

    Download window is event_date + 16 CALENDAR days: resolution needs closes[5]
    (6 bars), and 10 days left only 5 bars whenever a market holiday fell inside
    a Friday-dated row's window (Jun-12/Juneteenth, Jun-26/Jul-4th cohorts sat
    unresolvable at the head of the ORDER BY date queue and starved the
    max_rows budget — found Jul 9 2026). 16 days guarantees >= 6 bars through
    two weekends plus a holiday.

    Stale tombstone: a row still unresolvable 30+ calendar days after its event
    (delisted / renamed symbol) is marked resolved=-1 — final, excluded both
    from future candidate scans (resolved=0) and from all stats (resolved=1) —
    so it can never clog the queue head.
    """
    try:
        import yfinance as _yf
        import time as _time
        from datetime import timedelta
        con = sqlite3.connect(_SETUP_LOG_DB, timeout=30)
        cutoff = (date.today() - timedelta(days=7)).isoformat()
        stale_cutoff = (date.today() - timedelta(days=30)).isoformat()
        rows = con.execute(
            "SELECT id, symbol, date, price FROM setup_log "
            "WHERE resolved=0 AND date <= ? ORDER BY date LIMIT ?",
            (cutoff, max_rows)
        ).fetchall()
        # ONE SPY download spanning the whole batch (Jul 30 2026) — per-row SPY
        # returns are then indexed by each stock's own bar dates, so the SPY
        # window matches ret_1d/ret_5d exactly (holidays/halts align naturally).
        spy_closes = None
        if rows:
            try:
                d_min = date.fromisoformat(rows[0][2])   # ORDER BY date → earliest first
                d_max = date.fromisoformat(rows[-1][2])
                spy_hist = _yf.download("SPY", start=str(d_min),
                                        end=str(d_max + timedelta(days=16)),
                                        interval="1d", auto_adjust=True, progress=False)
                if not spy_hist.empty:
                    sc = spy_hist["Close"]
                    if hasattr(sc, "columns"):  # yfinance 2.x MultiIndex frame
                        sc = sc.iloc[:, 0]
                    spy_closes = {i.date(): float(v) for i, v in sc.dropna().items()}
            except Exception as e:
                print(f"[setup_resolve] SPY batch download failed: {e}", flush=True)
        if rows and not spy_closes:
            # Defer the whole batch one night rather than resolving rows with a
            # permanent NULL spy basis — resolved rows are written exactly once.
            print("[setup_resolve] no SPY data — deferring batch to next run", flush=True)
            rows = []
        attempted = resolved_n = failed = stale_n = 0
        for row_id, sym, date_str, entry_price in rows:
            try:
                event_date = date.fromisoformat(date_str)
                is_stale = date_str <= stale_cutoff
                hist = _yf.download(sym, start=date_str,
                                    end=str(event_date + timedelta(days=16)),
                                    interval="1d", auto_adjust=True, progress=False)
                attempted += 1
                if hist.empty or len(hist) < 2:
                    if is_stale:
                        con.execute("UPDATE setup_log SET resolved=-1 WHERE id=?", (row_id,))
                        con.commit()
                        stale_n += 1
                        print(f"[setup_resolve] {sym} {date_str} tombstoned (no data, 30+ days)", flush=True)
                    _time.sleep(0.5)
                    continue
                closes = hist["Close"].values.flatten()
                # Adjusted day-0 close as the baseline — SAME auto_adjust basis as the
                # forward closes, so splits/dividends cancel instead of poisoning the
                # return. Using the raw stored `price` as the denominator (adjusted
                # numerator / raw denominator) injects a fake return on any corporate
                # action inside the window. `price` is kept in the row for reference.
                base = float(closes[0])
                c1 = float(closes[1]) if len(closes) > 1 else None
                c5 = float(closes[5]) if len(closes) > 5 else None
                r1 = round((c1 / base - 1) * 100, 2) if c1 else None
                r5 = round((c5 / base - 1) * 100, 2) if c5 else None
                # SPY over the SAME bar dates as the stock's window (see batch
                # download above). SPY trades every NYSE session, so a missing
                # key here is a genuine data hole — resolve anyway, NULL spy.
                bar_dates = [i.date() for i in hist.index]
                spy0 = spy_closes.get(bar_dates[0])
                spy1 = spy_closes.get(bar_dates[1]) if len(bar_dates) > 1 else None
                spy5 = spy_closes.get(bar_dates[5]) if len(bar_dates) > 5 else None
                spy_r1 = round((spy1 / spy0 - 1) * 100, 2) if spy0 and spy1 else None
                spy_r5 = round((spy5 / spy0 - 1) * 100, 2) if spy0 and spy5 else None
                # 30+ days old with data but still no 6th bar (halted mid-window
                # etc.) — tombstone rather than retry forever; keep partial c1.
                is_res = 1 if c5 is not None else (-1 if is_stale else 0)
                con.execute(
                    "UPDATE setup_log SET close_1d=?, close_5d=?, ret_1d=?, ret_5d=?, "
                    "spy_ret_1d=?, spy_ret_5d=?, resolved=? WHERE id=?",
                    (c1, c5, r1, r5, spy_r1, spy_r5, is_res, row_id)
                )
                con.commit()
                if is_res == 1:
                    resolved_n += 1
                elif is_res == -1:
                    stale_n += 1
                    print(f"[setup_resolve] {sym} {date_str} tombstoned (no 6th bar, 30+ days)", flush=True)
                _time.sleep(0.5)
            except Exception as e:
                failed += 1
                print(f"[setup_resolve] {sym} {date_str} failed: {e}", flush=True)
                _time.sleep(0.5)
        con.close()
        print(f"[setup_resolve] candidates={len(rows)} attempted={attempted} "
              f"resolved={resolved_n} stale={stale_n} failed={failed}", flush=True)
    except Exception as e:
        print(f"[setup_resolve] fatal: {e}", flush=True)


def get_setup_breakdown():
    # Read-only. Resolution is decoupled to the nightly resolve_setups.py cron
    # (03:00 UTC) so the yfinance burst never hits this web-request path.
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


# ── Positions layer (Aug 13 2026) ─────────────────────────────────────────────
# The system measures fixed-horizon population alpha; it has NO concept of a held
# position (the CCL case: 6 BUY signals in 8 days at ever-worse entries, and the
# holder got nothing). This layer closes that gap for the USER, not the research:
# it READS signals and never creates them. Its own state lives in positions.db;
# the only research DB it touches is tracker.db, opened mode=ro so this code
# physically cannot write it. R1 is untouched by construction.

_POSITIONS_DB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "positions.db")
_TRACKER_DB_RO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tracker.db")

COMMISSION_PCT_PER_SIDE = 0.08   # broker rate per trade — net P&L subtracts BOTH sides
POSITION_HORIZON_TDAYS  = 10     # the tracker's own measurement horizon


def positions_db_init(db_path=None):
    con = sqlite3.connect(db_path or _POSITIONS_DB, timeout=30)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("""CREATE TABLE IF NOT EXISTS positions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        symbol      TEXT NOT NULL,
        entry_price REAL NOT NULL,
        entry_date  TEXT NOT NULL,
        stop_pct    REAL,
        notes       TEXT,
        status      TEXT NOT NULL DEFAULT 'open',
        opened_ts   TEXT NOT NULL,
        closed_ts   TEXT,
        exit_price  REAL,
        exit_ret_net_pct REAL
    )""")
    # Share count (Aug 14 2026) — without it the P&L is an abstract percentage.
    # Idempotent ADD COLUMN, NULL for rows entered before it existed; every
    # dollar figure below returns None when shares is NULL rather than guessing 1.
    try:
        con.execute("ALTER TABLE positions ADD COLUMN shares REAL")
    except Exception:
        pass  # column already exists
    # Ownership (Aug 16 2026) — the table was built with no owner concept, so a
    # second entry in BASIC_AUTH_USERS would have shown everyone every position,
    # and sequential ids made another user's row editable by guessing a number.
    # Every function below now takes `owner` as a REQUIRED first argument: a call
    # site that forgets it raises TypeError instead of silently leaking.
    try:
        con.execute("ALTER TABLE positions ADD COLUMN owner TEXT")
    except Exception:
        pass  # column already exists
    con.execute("CREATE INDEX IF NOT EXISTS idx_positions_owner ON positions(owner)")
    con.commit()
    con.close()


def positions_backfill_owner(owner, db_path=None):
    """Assign pre-ownership rows to `owner`. Caller decides whether this is safe
    — api.py only calls it when exactly ONE account is configured, so there is
    no ambiguity about whose rows these are. Rows left NULL stay invisible to
    every reader (fail closed), never visible to all."""
    con = sqlite3.connect(db_path or _POSITIONS_DB, timeout=30)
    cur = con.execute("UPDATE positions SET owner=? WHERE owner IS NULL", (owner,))
    con.commit()
    n = cur.rowcount
    con.close()
    return n


def _position_net_pct_raw(entry_price, current_price, commission_per_side):
    """UNROUNDED net percentage. Rounding belongs at the edges only: rounding
    here first and then multiplying by the basis inflates the error by the size
    of the position (a 0.005pp rounding on a $63k position is $3, and the user
    spotted exactly that — a $20.54 loss displayed as $20.66)."""
    if not entry_price or not current_price or entry_price <= 0:
        return None
    return (current_price / entry_price - 1) * 100 - 2 * commission_per_side


def position_net_pnl_pct(entry_price, current_price,
                         commission_per_side=COMMISSION_PCT_PER_SIDE):
    """Gross move minus commission on BOTH sides — the number the user banks.
    Pure math, unit-tested."""
    raw = _position_net_pct_raw(entry_price, current_price, commission_per_side)
    return None if raw is None else round(raw, 2)


def position_cost_basis(entry_price, shares):
    """What the position cost to open, before commission. None if unknown."""
    if not entry_price or not shares:
        return None
    return round(entry_price * shares, 2)


def position_net_pnl_usd(entry_price, current_price, shares,
                         commission_per_side=COMMISSION_PCT_PER_SIDE):
    """Dollar P&L, DERIVED from the percentage so the two can never disagree
    on screen. (A per-side charge on each leg's actual value would be a hair
    more exact — it differs by <0.01pp on a 50% move, far below the spread the
    user actually pays — and two subtly different commission models shown side
    by side is the worse bug.)"""
    raw = _position_net_pct_raw(entry_price, current_price, commission_per_side)
    basis = position_cost_basis(entry_price, shares)
    if raw is None or basis is None:
        return None
    return round(basis * raw / 100, 2)


def positions_totals(rows):
    """Portfolio roll-up over OPEN rows that have both shares and a live price.
    Reports how many rows were skipped so a partial total is never read as
    complete. Pure, unit-tested."""
    basis = pnl = counted = skipped = 0
    for r in rows:
        if r.get("status") != "open":
            continue
        b = position_cost_basis(r.get("entry_price"), r.get("shares"))
        u = r.get("net_pnl_usd")
        if b is None or u is None:
            skipped += 1
            continue
        basis += b
        pnl += u
        counted += 1
    if not counted:
        return {"cost_basis": None, "net_pnl_usd": None, "net_pnl_pct": None,
                "counted": 0, "skipped": skipped}
    return {"cost_basis": round(basis, 2),
            "net_pnl_usd": round(pnl, 2),
            "net_pnl_pct": round(pnl / basis * 100, 2) if basis else None,
            "counted": counted, "skipped": skipped}


def position_trading_days_held(entry_date_str, today=None):
    """NYSE sessions strictly AFTER entry_date, up to and including today.
    Entry day itself is day 0 — matches how the tracker counts its horizon.
    Pure calendar walk, unit-tested."""
    from market_calendar import NYSE_HOLIDAYS
    from datetime import timedelta
    try:
        d = date.fromisoformat(entry_date_str)
    except Exception:
        return None
    today = today or datetime.now(_ET).date()
    if d >= today:
        return 0
    held = 0
    while d < today:
        d += timedelta(days=1)
        if d.weekday() < 5 and d.isoformat() not in NYSE_HOLIDAYS:
            held += 1
    return held


def position_alerts(entry_price, current_price, stop_pct, days_held,
                    horizon=POSITION_HORIZON_TDAYS):
    """Rule evaluation → list of alert dicts. Pure, no I/O, unit-tested.
    STOP embodies the user's own logged rule: first entry going wrong is an
    exit signal, not an add-more signal."""
    alerts = []
    if entry_price and current_price and entry_price > 0:
        gross = (current_price / entry_price - 1) * 100
        if stop_pct is not None and gross <= -abs(stop_pct):
            alerts.append({"kind": "STOP",
                           "detail": f"{gross:.1f}% ≤ -{abs(stop_pct):.1f}%"})
    if days_held is not None and days_held >= horizon:
        alerts.append({"kind": "HORIZON",
                       "detail": f"{days_held} ≥ {horizon} trading days — past the system's measured horizon"})
    return alerts


def position_open_row(owner, symbol, entry_price, entry_date=None, stop_pct=None,
                      notes=None, shares=None, db_path=None):
    con = sqlite3.connect(db_path or _POSITIONS_DB, timeout=30)
    cur = con.execute(
        "INSERT INTO positions (owner, symbol, entry_price, entry_date, stop_pct, notes, shares, status, opened_ts) "
        "VALUES (?,?,?,?,?,?,?,'open',?)",
        (owner, symbol.upper(), float(entry_price),
         entry_date or datetime.now(_ET).date().isoformat(),
         stop_pct, notes, (float(shares) if shares else None),
         datetime.utcnow().isoformat()))
    con.commit()
    rid = cur.lastrowid
    con.close()
    return rid


_POSITION_EDITABLE = ("entry_price", "entry_date", "stop_pct", "notes", "shares")


def position_update_row(pos_id, owner, db_path=None, **fields):
    """Partial update of an OPEN position. Only the keys actually passed are
    written — the caller (api layer) derives them from pydantic's
    model_fields_set, so `stop_pct=None` means CLEAR THE STOP and an absent
    stop_pct means leave it alone. Conflating those two is how an edit form
    silently wipes a stop the user still wanted."""
    unknown = set(fields) - set(_POSITION_EDITABLE)
    if unknown:
        raise ValueError(f"not editable: {sorted(unknown)}")
    if not fields:
        return None
    con = sqlite3.connect(db_path or _POSITIONS_DB, timeout=30)
    # owner is part of the lookup, not a check after it — another user's row is
    # simply not found, so a guessed id reveals nothing about whether it exists.
    row = con.execute("SELECT status FROM positions WHERE id=? AND owner=?",
                      (pos_id, owner)).fetchone()
    if not row or row[0] != "open":
        con.close()
        return None
    if "entry_price" in fields:
        fields["entry_price"] = float(fields["entry_price"])
    if fields.get("shares") is not None:
        fields["shares"] = float(fields["shares"])
    if "symbol" in fields:
        fields["symbol"] = str(fields["symbol"]).upper()
    sets = ", ".join(f"{k}=?" for k in fields)
    con.execute(f"UPDATE positions SET {sets} WHERE id=? AND owner=?",
                (*fields.values(), pos_id, owner))
    con.commit()
    con.row_factory = sqlite3.Row
    out = dict(con.execute("SELECT * FROM positions WHERE id=? AND owner=?",
                           (pos_id, owner)).fetchone())
    con.close()
    return out


def position_close_row(pos_id, owner, exit_price, db_path=None):
    con = sqlite3.connect(db_path or _POSITIONS_DB, timeout=30)
    row = con.execute("SELECT entry_price, status FROM positions WHERE id=? AND owner=?",
                      (pos_id, owner)).fetchone()
    if not row or row[1] != "open":
        con.close()
        return None
    net = position_net_pnl_pct(row[0], float(exit_price))
    con.execute("UPDATE positions SET status='closed', closed_ts=?, exit_price=?, "
                "exit_ret_net_pct=? WHERE id=? AND owner=?",
                (datetime.utcnow().isoformat(), float(exit_price), net, pos_id, owner))
    con.commit()
    con.close()
    return {"id": pos_id, "exit_price": float(exit_price), "exit_ret_net_pct": net}


def positions_list(owner, status="open", db_path=None):
    con = sqlite3.connect(db_path or _POSITIONS_DB, timeout=30)
    con.row_factory = sqlite3.Row
    if status == "all":
        rows = con.execute("SELECT * FROM positions WHERE owner=? ORDER BY id DESC",
                           (owner,)).fetchall()
    else:
        rows = con.execute("SELECT * FROM positions WHERE owner=? AND status=? "
                           "ORDER BY id DESC", (owner, status)).fetchall()
    out = [dict(r) for r in rows]
    con.close()
    return out


def position_signal_history(symbol, tracker_path=None, lookback_days=14, limit=5):
    """READ-ONLY look at tracker.db for one symbol: current signal pressure and
    how past resolved signals actually ended. mode=ro is the guarantee — this
    layer cannot write the research DB even by bug."""
    path = tracker_path or _TRACKER_DB_RO
    try:
        con = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=30)
    except Exception:
        return None
    try:
        recent = con.execute(
            "SELECT COUNT(*) FROM signals WHERE sym=? AND date_logged >= date('now', ?)",
            (symbol.upper(), f"-{int(lookback_days)} day")).fetchone()[0]
        resolved = con.execute(
            "SELECT s.date_logged, s.confidence, o.fwd_ret FROM signals s "
            "JOIN outcomes o ON o.signal_id = s.id "
            "WHERE s.sym=? ORDER BY s.date_logged DESC LIMIT ?",
            (symbol.upper(), int(limit))).fetchall()
        hist = [{"date": r[0],
                 "confidence": round(r[1], 3) if r[1] is not None else None,
                 "fwd_ret_pct": round(r[2] * 100, 1) if r[2] is not None else None}
                for r in resolved]
        rets = [h["fwd_ret_pct"] for h in hist if h["fwd_ret_pct"] is not None]
        return {"recent_signals": recent,
                "lookback_days": lookback_days,
                "resolved": hist,
                "resolved_mean_pct": round(sum(rets) / len(rets), 1) if rets else None}
    except Exception:
        return None
    finally:
        con.close()


try:
    fk_db_init()
    setup_db_init()
    positions_db_init()
except Exception:
    pass

