#!/usr/bin/env python3
"""
Daily intraday bar cache — runs at 16:30 ET after market close (20:30 UTC).
Downloads 1m bars for most_actives UNION all currently tracked symbols.
Resamples to 5-min slots, fills gaps with 0, stores in SQLite.
Deletes records older than 20 trading days.
"""
import sqlite3, os, json
import urllib.request as _req
import pandas as pd
import yfinance as yf
from datetime import date, timedelta, datetime
from zoneinfo import ZoneInfo

from market_calendar import is_us_market_session

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "intraday_cache.db")
MAX_DAYS = 20
ET = ZoneInfo("America/New_York")


def init_db(con):
    con.execute("""
        CREATE TABLE IF NOT EXISTS intraday_bars (
            symbol    TEXT    NOT NULL,
            date      TEXT    NOT NULL,
            time_slot TEXT    NOT NULL,
            volume    INTEGER NOT NULL DEFAULT 0,
            PRIMARY KEY (symbol, date, time_slot)
        )
    """)
    con.execute("CREATE INDEX IF NOT EXISTS idx_sym_slot ON intraday_bars(symbol, time_slot)")
    con.commit()


def get_most_actives():
    url = (
        "https://query1.finance.yahoo.com/v1/finance/screener/predefined/saved"
        "?formatted=false&scrIds=most_actives&count=50&corsDomain=finance.yahoo.com"
    )
    req = _req.Request(url, headers={"User-Agent": "Mozilla/5.0 (compatible; StockPredictor/1.0)"})
    with _req.urlopen(req, timeout=15) as resp:
        raw = json.loads(resp.read())
    return [q["symbol"] for q in raw["finance"]["result"][0]["quotes"]]


def get_tracked_symbols(con):
    cutoff = (date.today() - timedelta(days=MAX_DAYS)).isoformat()
    cur = con.execute(
        "SELECT DISTINCT symbol FROM intraday_bars WHERE date >= ?", (cutoff,)
    )
    return [r[0] for r in cur.fetchall()]


def fetch_and_store(symbols, today_str, con):
    if not symbols:
        return
    print(f"  Downloading 1m bars for {len(symbols)} symbols...")
    raw = yf.download(symbols, period="1d", interval="1m", progress=False, auto_adjust=True)
    multi = isinstance(raw.columns, pd.MultiIndex)
    rows = []
    for sym in symbols:
        try:
            vol = raw["Volume"][sym].dropna() if multi else raw["Volume"].dropna()
            if vol.empty:
                continue
            if vol.index.tz is None:
                vol.index = vol.index.tz_localize("UTC").tz_convert(ET)
            else:
                vol.index = vol.index.tz_convert(ET)
            vol = vol.between_time("09:30", "15:55")
            if vol.empty:
                continue
            vol_5m = vol.resample("5min").sum().fillna(0)
            for ts, v in vol_5m.items():
                rows.append((sym, today_str, ts.strftime("%H:%M"), int(v)))
        except Exception as e:
            print(f"  [{sym}] error: {e}")
    con.executemany(
        "INSERT OR REPLACE INTO intraday_bars (symbol, date, time_slot, volume) VALUES (?,?,?,?)",
        rows
    )
    con.commit()
    print(f"  Stored {len(rows)} rows for {len(symbols)} symbols")


def purge_old(con):
    cutoff = (date.today() - timedelta(days=MAX_DAYS + 1)).isoformat()
    cur = con.execute("DELETE FROM intraday_bars WHERE date < ?", (cutoff,))
    con.commit()
    if cur.rowcount:
        print(f"  Purged {cur.rowcount} old rows")


def main():
    et_now = datetime.now(ET)
    today_str = et_now.strftime("%Y-%m-%d")
    print(f"[{et_now.strftime('%H:%M ET')}] Intraday cache update — {today_str}")
    if not is_us_market_session(et_now.date()):
        print(f"  US market closed {today_str} (weekend/holiday) — skipping fetch.")
        return
    con = sqlite3.connect(DB_PATH)
    try:
        init_db(con)
        active  = get_most_actives()
        tracked = get_tracked_symbols(con)
        symbols = list(set(active) | set(tracked))
        print(f"  Universe: {len(symbols)} symbols (actives={len(active)}, tracked={len(tracked)})")
        fetch_and_store(symbols, today_str, con)
        purge_old(con)
        print("  Done.")
    except Exception as e:
        print(f"  ERROR: {e}")
        raise
    finally:
        con.close()


if __name__ == "__main__":
    main()
